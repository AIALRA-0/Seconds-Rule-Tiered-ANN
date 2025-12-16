#!/usr/bin/env python3
"""
dump_results.py

One-click printing of the key information from this experiment, so you can copy it directly into ChatGPT for analysis.

Usage (run at the repo root):

    # Automatically find the latest tiered_ann_seconds_rule_* directory
    python scripts/dump_results.py

    # Or manually specify a particular run directory
    python scripts/dump_results.py results/tiered_ann_seconds_rule_20251215_193936
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Sequence, Optional

import pandas as pd


# ---------- Some small helper functions ----------

def pick_col(cols: Sequence[str], must_have: Sequence[str], any_of: Sequence[str] = ()) -> Optional[str]:
    """
    Simple column-name auto matching:
    - all tokens in must_have must appear in the column name (lowercased)
    - at least one token in any_of must appear in the column name (if provided)
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
        "policy": pick_col(cols, ["policy"]),
        "dram": pick_col(cols, ["dram"], ["frac", "ratio", "budget", "pct", "percent"]),
        "p95": pick_col(cols, ["p95"]),
        "recall": pick_col(cols, ["recall"]),
        "io_amp": pick_col(cols, ["io"], ["amp", "ampl"]),
        "migration": pick_col(cols, ["migr"], ["byte", "bytes", "mb", "gb"]),
    }

    # DRAM fallback: some projects may call it cache_xxx or mem_xxx
    if mapping["dram"] is None:
        mapping["dram"] = pick_col(cols, ["cache"], ["frac", "ratio", "budget", "pct", "percent"]) \
                          or pick_col(cols, ["mem"], ["frac", "ratio", "budget", "pct", "percent"])

    # None → "" to make printing easier
    return {k: (v or "") for k, v in mapping.items()}


def main() -> None:
    root = Path(".").resolve()
    results_root = root / "results"

    if not results_root.exists():
        print(f"[ERROR] Cannot find results directory: {results_root}", file=sys.stderr)
        sys.exit(1)

    # 1) Parse command-line args: use the given run directory if provided; otherwise automatically pick the latest
    if len(sys.argv) >= 2:
        run_dir = Path(sys.argv[1]).resolve()
        if not run_dir.exists():
            print(f"[ERROR] Specified RUN_DIR does not exist: {run_dir}", file=sys.stderr)
            sys.exit(1)
    else:
        candidates = sorted(results_root.glob("tiered_ann_seconds_rule_*"))
        if not candidates:
            print("[ERROR] Cannot find any tiered_ann_seconds_rule_* directory under results/", file=sys.stderr)
            sys.exit(1)
        run_dir = candidates[-1]

    print("============================================================")
    print("[dump_results] RUN_DIR =", run_dir.as_posix())
    print("============================================================\n")

    agg_csv = run_dir / "agg" / "ann_policy_agg.csv"
    sla_csv = run_dir / "agg" / "sla_reachable_recall.csv"

    if not agg_csv.exists():
        print(f"[ERROR] Cannot find agg/ann_policy_agg.csv: {agg_csv}", file=sys.stderr)
        sys.exit(1)

    # 2) Read agg CSV
    df = pd.read_csv(agg_csv)
    print("------------------------------------------------------------")
    print("1) Basic information of ann_policy_agg.csv")
    print("------------------------------------------------------------")
    print("File path:", agg_csv.as_posix())
    print("Total rows:", len(df))
    print("Columns:", list(df.columns))
    print()

    # Automatically detect several key columns
    mapping = detect_columns(df)
    print("Auto-detected columns (you may want to double-check):")
    for k, v in mapping.items():
        print(f"  {k:10s} -> {v!r}")
    print()

    policy_col = mapping.get("policy") or "policy"
    dram_col = mapping.get("dram") or ""

    # Print some information about the policy column and DRAM column
    if policy_col not in df.columns:
        print(f"[WARN] Column '{policy_col}' not found in the table. Many analyses below will fail; "
              f"please inspect df.columns and modify the script accordingly.")
    else:
        policies = sorted(set(df[policy_col].astype(str)))
        print("All policy names (use these names when writing conclusions):")
        for p in policies:
            print("  -", p)
        print()

    if dram_col and dram_col in df.columns:
        budgets = sorted(set(df[dram_col]))
        print(f"Unique values of {dram_col} (DRAM configurations):", budgets)
        print()
    else:
        budgets = []
        print("[WARN] Failed to automatically detect the DRAM column; only printing head below.")
        print()

    print("Preview of the first 10 rows (so I can understand the table schema):")
    print(df.head(10).to_string(index=False))
    print()

    # 3) For the minimum/median/maximum DRAM, print key rows for each policy
    if dram_col and dram_col in df.columns and policy_col in df.columns and len(budgets) >= 1:
        print("------------------------------------------------------------")
        print("2) Metrics for each policy at key DRAM points (min / median / max)")
        print("------------------------------------------------------------")

        # Pick three representative DRAM points
        import math
        unique_b = sorted(budgets)
        picks: List[float] = []
        picks.append(unique_b[0])
        mid = unique_b[len(unique_b) // 2]
        if mid not in picks:
            picks.append(mid)
        if unique_b[-1] not in picks:
            picks.append(unique_b[-1])

        for b in picks:
            g = df[df[dram_col] == b]
            print(f"\n=== Rows for each policy when {dram_col} = {b} ===")
            # Try to sort by p95 if possible
            p95_col = mapping.get("p95")
            if p95_col and p95_col in g.columns:
                g = g.sort_values(p95_col)
            print(g.to_string(index=False))
    else:
        print("(Due to failure in detecting the DRAM column / policy column, DRAM-based grouping is not printed.)")

    # 4) Try to read SLA reachable recall
    print("\n------------------------------------------------------------")
    print("3) sla_reachable_recall.csv (if present)")
    print("------------------------------------------------------------")

    if sla_csv.exists():
        sla = pd.read_csv(sla_csv)
        print("File path:", sla_csv.as_posix())
        print("Total rows:", len(sla))
        print("Columns:", list(sla.columns))
        print("\nFirst 20 rows:")
        print(sla.head(20).to_string(index=False))
    else:
        print(f"Did not find {sla_csv.as_posix()} (that's fine; this file is optional)")

    print("\n============================================================")
    print("Copy all the output above into ChatGPT, and I can directly help you see which one is better, "
          "by how much, and how to write the conclusions.")
    print("============================================================")


if __name__ == "__main__":
    main()
