#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


def _pick_min_mid_max(vals: List[float]) -> List[float]:
    if not vals:
        return []
    vals = sorted(vals)
    picks: List[float] = []
    picks.append(vals[0])
    mid = vals[len(vals) // 2]
    if mid not in picks:
        picks.append(mid)
    if vals[-1] not in picks:
        picks.append(vals[-1])
    return picks


def _df_to_markdown_table(df: pd.DataFrame, max_rows: int = 200) -> str:
    # Keep it simple, avoid fancy formatting.
    df2 = df.head(max_rows).copy()
    return df2.to_markdown(index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Export report-ready assets (tables + copy-paste snippets) for one RUN_DIR.")
    ap.add_argument("run_dir", type=Path, help="results/<RUN_ID> directory")
    args = ap.parse_args()

    run_dir = args.run_dir.resolve()
    agg_csv = run_dir / "agg" / "ann_policy_agg.csv"
    sla_csv = run_dir / "agg" / "sla_reachable_recall.csv"
    fig_dir = run_dir / "figures"

    if not agg_csv.exists():
        raise SystemExit(f"[export_assets] Missing: {agg_csv}")

    out_dir = run_dir / "report_assets"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(agg_csv)

    required = [
        "policy", "dram_fraction", "max_iops", "nprobe",
        "p95_latency_us_mean", "recall_at_k_mean",
        "avg_io_amplification_mean", "migration_bytes_per_query_mean",
        "num_seeds",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"[export_assets] Missing columns in agg CSV: {missing}\nAll columns: {list(df.columns)}")

    # --- 1) figures list ---
    figs = []
    if fig_dir.exists():
        for p in sorted(fig_dir.glob("*.png")):
            figs.append(p.relative_to(run_dir).as_posix())
    (out_dir / "figures_list.md").write_text(
        "# Figures in this run\n\n" + "\n".join(f"- `{f}`" for f in figs) + "\n",
        encoding="utf-8",
    )

    # --- 2) representative key-points table (min/mid/max on each axis) ---
    nprobes = sorted(df["nprobe"].unique().tolist())
    drams = sorted(df["dram_fraction"].unique().tolist())
    iops = sorted(df["max_iops"].unique().tolist())

    pick_nprobe = _pick_min_mid_max([float(x) for x in nprobes])
    pick_dram = _pick_min_mid_max([float(x) for x in drams])
    pick_iops = _pick_min_mid_max([float(x) for x in iops])

    sub = df[
        df["nprobe"].isin(pick_nprobe)
        & df["dram_fraction"].isin(pick_dram)
        & df["max_iops"].isin(pick_iops)
    ].copy()

    # nicer units for report
    sub["dram_pct"] = (sub["dram_fraction"] * 100.0).round(1)
    sub["max_iops_m"] = (sub["max_iops"] / 1e6).round(1)

    cols = [
        "dram_pct", "max_iops_m", "nprobe", "policy",
        "p95_latency_us_mean", "recall_at_k_mean",
        "avg_io_amplification_mean", "migration_bytes_per_query_mean",
        "num_seeds",
    ]
    sub = sub[cols].sort_values(["dram_pct", "max_iops_m", "nprobe", "policy"])

    sub.to_csv(out_dir / "table_key_points.csv", index=False)
    (out_dir / "table_key_points.md").write_text(
        "# Table: key points (min / mid / max across DRAM, IOPS, nprobe)\n\n" + _df_to_markdown_table(sub) + "\n",
        encoding="utf-8",
    )

    # --- 3) SLA table + "best policy" per SLA group ---
    if sla_csv.exists():
        sla = pd.read_csv(sla_csv)
        need2 = ["dram_fraction", "max_iops", "policy", "sla_us", "best_nprobe", "reachable_recall_at_k", "best_p95_latency_us"]
        miss2 = [c for c in need2 if c not in sla.columns]
        if miss2:
            raise SystemExit(f"[export_assets] Missing columns in SLA CSV: {miss2}\nAll columns: {list(sla.columns)}")

        sla2 = sla.copy()
        sla2["dram_pct"] = (sla2["dram_fraction"] * 100.0).round(1)
        sla2["max_iops_m"] = (sla2["max_iops"] / 1e6).round(1)
        sla2 = sla2[
            ["dram_pct", "max_iops_m", "sla_us", "policy", "best_nprobe", "reachable_recall_at_k", "best_p95_latency_us"]
        ].sort_values(["dram_pct", "max_iops_m", "sla_us", "policy"])

        sla2.to_csv(out_dir / "table_sla.csv", index=False)
        (out_dir / "table_sla.md").write_text(
            "# Table: SLA → reachable recall@k\n\n" + _df_to_markdown_table(sla2) + "\n",
            encoding="utf-8",
        )

        # Best policy per (dram_pct, max_iops_m, sla_us)
        best_rows = []
        for keys, g in sla2.groupby(["dram_pct", "max_iops_m", "sla_us"]):
            gg = g.sort_values(["reachable_recall_at_k", "best_nprobe"], ascending=[False, False])
            best = gg.iloc[0].to_dict()
            best_rows.append(best)
        best_df = pd.DataFrame(best_rows).sort_values(["dram_pct", "max_iops_m", "sla_us"])
        best_df.to_csv(out_dir / "table_sla_best.csv", index=False)
        (out_dir / "table_sla_best.md").write_text(
            "# Table: best policy under each SLA\n\n" + _df_to_markdown_table(best_df) + "\n",
            encoding="utf-8",
        )
    else:
        (out_dir / "table_sla.md").write_text(
            "# Table: SLA → reachable recall@k\n\n(sla_reachable_recall.csv not found in this run)\n",
            encoding="utf-8",
        )

    # --- 4) short copy-paste snippet (fill into report Results section) ---
    snippet = []
    snippet.append("# Copy-paste snippet for report\n")
    snippet.append(f"- RUN_DIR: `{run_dir}`")
    snippet.append(f"- agg CSV: `{agg_csv.relative_to(run_dir)}`")
    snippet.append(f"- sla CSV: `{sla_csv.relative_to(run_dir)}`" if sla_csv.exists() else "- sla CSV: (not found)")
    snippet.append(f"- figures dir: `{fig_dir.relative_to(run_dir)}`" if fig_dir.exists() else "- figures dir: (not found)")
    snippet.append("")
    snippet.append("## Recommended files to paste")
    snippet.append(f"- `{(out_dir / 'table_key_points.md').relative_to(run_dir)}`")
    if (out_dir / "table_sla_best.md").exists():
        snippet.append(f"- `{(out_dir / 'table_sla_best.md').relative_to(run_dir)}`")
    snippet.append(f"- `{(out_dir / 'figures_list.md').relative_to(run_dir)}`")
    snippet.append("")
    (out_dir / "report_snippet.md").write_text("\n".join(snippet) + "\n", encoding="utf-8")

    print("============================================================")
    print("[export_assets] DONE")
    print("RUN_DIR:", run_dir)
    print("Wrote:", out_dir)
    print("Key files:")
    for f in [
        "report_snippet.md",
        "figures_list.md",
        "table_key_points.md",
        "table_sla.md",
        "table_sla_best.md",
    ]:
        p = out_dir / f
        if p.exists():
            print(" -", p)
    print("============================================================")


if __name__ == "__main__":
    main()
