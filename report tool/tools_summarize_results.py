#!/usr/bin/env python3
"""
Summarize tiered ANN experiment results for report writing.

This script is designed to be robust to different column names by using heuristics.

Usage (from repo root):
    python tools_summarize_results.py results/tiered_ann_seconds_rule_20251215_193936

Expected inputs inside RUN_DIR:
    agg/ann_policy_agg.csv
    agg/sla_reachable_recall.csv   (optional)

Outputs:
    - Prints a concise summary to stdout
    - Writes report_snippet.md (ready to paste into FINAL_REPORT.md)

Notes:
- Edit the POLICY_NAME_HINTS list if your policy names differ.
- If column auto-detection fails, the script prints columns and tells you what to change.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd


def pick_col(cols: Sequence[str], must_have: Sequence[str], any_of: Sequence[str] = ()) -> Optional[str]:
    """
    Heuristic: choose the first column whose lowercase name contains all tokens in must_have
    and (if provided) contains at least one token from any_of.
    """
    cols_l = [(c, c.lower()) for c in cols]
    for c, cl in cols_l:
        if all(tok in cl for tok in must_have):
            if any_of:
                if any(tok in cl for tok in any_of):
                    return c
            else:
                return c
    return None


def detect_columns(df: pd.DataFrame) -> Dict[str, str]:
    cols = list(df.columns)

    mapping: Dict[str, Optional[str]] = {
        "policy": pick_col(cols, must_have=["policy"]),
        "dram": pick_col(cols, must_have=["dram"], any_of=["frac", "ratio", "budget", "pct", "percent"]),
        "p95": pick_col(cols, must_have=["p95"]),
        "recall": pick_col(cols, must_have=["recall"]),
        "io_amp": pick_col(cols, must_have=["io"], any_of=["amp", "ampl"]),
        "iops": pick_col(cols, must_have=["iops"]),
        "migration": pick_col(cols, must_have=["migr"], any_of=["byte", "bytes", "mb", "gb"]),
    }

    # Fallbacks for DRAM column (some use "cache" or "mem")
    if mapping["dram"] is None:
        mapping["dram"] = pick_col(cols, must_have=["cache"], any_of=["frac", "ratio", "budget", "pct", "percent"]) \
                          or pick_col(cols, must_have=["mem"], any_of=["frac", "ratio", "budget", "pct", "percent"])

    # Convert Nones to empty for easier printing
    return {k: (v or "") for k, v in mapping.items()}


def require(mapping: Dict[str, str], key: str) -> str:
    v = mapping.get(key, "")
    if not v:
        raise KeyError(key)
    return v


def monotonicity_violations(df: pd.DataFrame, policy_col: str, x_col: str, y_col: str) -> pd.DataFrame:
    import numpy as np

    rows = []
    for pol, g in df.groupby(policy_col):
        g = g.sort_values(x_col)
        y = g[y_col].to_numpy()
        if len(y) <= 1:
            viol = 0
            total = 0
        else:
            viol = int(np.sum(np.diff(y) > 0))
            total = len(y) - 1
        rows.append({"policy": pol, "violations": viol, "comparisons": total,
                     "violation_rate": (viol / total) if total else 0.0})
    return pd.DataFrame(rows).sort_values(["violation_rate", "violations"], ascending=[False, False])


def find_policy_name(df: pd.DataFrame, policy_col: str, hints: Sequence[str]) -> Optional[str]:
    policies = sorted(set(df[policy_col].astype(str)))
    lower_map = {p.lower(): p for p in policies}
    for h in hints:
        # exact match
        if h.lower() in lower_map:
            return lower_map[h.lower()]
    # substring match
    for p in policies:
        pl = p.lower()
        for h in hints:
            if h.lower() in pl:
                return p
    return None


def gap_to_oracle(df: pd.DataFrame, policy_col: str, x_col: str, y_col: str,
                  oracle_policy: str) -> pd.DataFrame:
    """
    Compute relative gap (policy/oracle - 1) at each x for each policy.
    """
    pivot = df.pivot_table(index=x_col, columns=policy_col, values=y_col, aggfunc="mean")
    if oracle_policy not in pivot.columns:
        raise ValueError(f"Oracle policy '{oracle_policy}' not found in policies: {list(pivot.columns)}")
    oracle = pivot[oracle_policy]
    gaps = pivot.div(oracle, axis=0) - 1.0
    # average gap per policy
    avg = gaps.mean(axis=0, skipna=True).sort_values()
    out = avg.reset_index()
    out.columns = ["policy", "avg_gap_vs_oracle"]
    return out


def write_report_snippet(out_path: Path, run_dir: str, summary_lines: List[str]) -> None:
    md = []
    md.append(f"## Auto-generated results snippet for `{run_dir}`\n\n")
    md.extend([line + "\n" for line in summary_lines])
    out_path.write_text("".join(md), encoding="utf-8")


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python tools_summarize_results.py <RUN_DIR>", file=sys.stderr)
        sys.exit(2)

    run_dir = Path(sys.argv[1]).resolve()
    agg_csv = run_dir / "agg" / "ann_policy_agg.csv"
    sla_csv = run_dir / "agg" / "sla_reachable_recall.csv"

    if not agg_csv.exists():
        print(f"ERROR: missing {agg_csv}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(agg_csv)
    mapping = detect_columns(df)

    print("=== Detected columns ===")
    for k, v in mapping.items():
        print(f"{k:10s}: {v!r}")
    print()

    try:
        policy_col = require(mapping, "policy")
        dram_col = require(mapping, "dram")
        p95_col = require(mapping, "p95")
    except KeyError as e:
        print("ERROR: failed to auto-detect required columns. Here are all columns:", file=sys.stderr)
        print(list(df.columns), file=sys.stderr)
        print("Fix: edit detect_columns() rules or hardcode column names in the script.", file=sys.stderr)
        sys.exit(1)

    summary_lines: List[str] = []

    # Basic dataset size
    policies = sorted(set(df[policy_col].astype(str)))
    budgets = sorted(set(df[dram_col]))
    summary_lines.append(f"- Policies: {len(policies)} ({', '.join(policies[:10])}{' ...' if len(policies)>10 else ''})")
    summary_lines.append(f"- DRAM budget points: {len(budgets)} (min={min(budgets)}, max={max(budgets)})")
    summary_lines.append(f"- Rows in agg CSV: {len(df)}")
    summary_lines.append("")

    # Monotonicity (p95 vs dram)
    try:
        mono = monotonicity_violations(df, policy_col, dram_col, p95_col)
        summary_lines.append("### Monotonicity deviation: p95 should decrease as DRAM increases")
        for _, r in mono.head(10).iterrows():
            summary_lines.append(f"- {r['policy']}: violations {int(r['violations'])}/{int(r['comparisons'])} "
                                 f"({r['violation_rate']:.1%})")
        summary_lines.append("")
    except Exception as e:
        summary_lines.append(f"- (monotonicity check failed: {type(e).__name__})\n")

    # Gap vs All-DRAM (if exists)
    ORACLE_HINTS = ["all-dram", "all_dram", "alldram", "dram_only", "dram"]
    oracle = find_policy_name(df, policy_col, ORACLE_HINTS)
    if oracle:
        try:
            gap = gap_to_oracle(df, policy_col, dram_col, p95_col, oracle)
            summary_lines.append(f"### Gap-to-oracle (oracle policy: {oracle})")
            for _, r in gap.head(10).iterrows():
                summary_lines.append(f"- {r['policy']}: avg p95 gap vs oracle = {r['avg_gap_vs_oracle']:.2%}")
            summary_lines.append("")
        except Exception as e:
            summary_lines.append(f"- (gap-to-oracle failed: {type(e).__name__}: {e})\n")
    else:
        summary_lines.append("### Gap-to-oracle")
        summary_lines.append("- Could not detect an All-DRAM oracle policy name automatically.")
        summary_lines.append("  Fix: edit ORACLE_HINTS in the script to match your policy names.\n")

    # SLA reachable recall (optional)
    if sla_csv.exists():
        try:
            sla = pd.read_csv(sla_csv)
            summary_lines.append("### SLA reachable recall (from sla_reachable_recall.csv)")
            summary_lines.append(f"- Rows: {len(sla)}; Columns: {list(sla.columns)}")
            summary_lines.append("- (Open this CSV to report: max recall per policy under SLA)\n")
        except Exception as e:
            summary_lines.append(f"- (failed to read sla_reachable_recall.csv: {type(e).__name__})\n")
    else:
        summary_lines.append("### SLA reachable recall")
        summary_lines.append("- sla_reachable_recall.csv not found for this run (optional artifact).")
        summary_lines.append("")

    # Write snippet
    out_path = Path("report_snippet.md")
    write_report_snippet(out_path, run_dir.as_posix(), summary_lines)
    print(f"Wrote {out_path.resolve()}")


if __name__ == "__main__":
    main()
