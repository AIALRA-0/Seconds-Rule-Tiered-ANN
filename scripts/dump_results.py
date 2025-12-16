#!/usr/bin/env python3
"""
dump_results.py (multi-run compact version, English only)

One-click printing of key information for ALL experiments under ./results,
with a COMPACT view that is easy to paste into ChatGPT.

Convention:
  - Each experiment directory looks like:
        results/sr_sift_default_fast_20251216_090724/
    and contains:
        agg/ann_policy_agg.csv
        (optional) agg/sla_reachable_recall.csv

Usage (run at the repo root):

    # 1) Automatically scan ./results for all run dirs
    python scripts/dump_results.py

    # 2) Only process ONE specific run directory
    python scripts/dump_results.py results/sr_sift_default_fast_20251216_090724

    # 3) Use a custom results root (scan all run dirs under it)
    python scripts/dump_results.py path/to/your/results
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Sequence, Optional

import pandas as pd


# ---------- Small helper functions ----------

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
        mapping["dram"] = (
            pick_col(cols, ["cache"], ["frac", "ratio", "budget", "pct", "percent"])
            or pick_col(cols, ["mem"], ["frac", "ratio", "budget", "pct", "percent"])
        )

    # None → "" to make printing easier
    return {k: (v or "") for k, v in mapping.items()}


def print_compact_summary(df: pd.DataFrame, mapping: Dict[str, str], policy_col: str, dram_col: str) -> None:
    """
    Print a compact view for copy-paste:
      - only key columns
      - only a few DRAM points (min / median / max)
      - limit rows per DRAM
    """
    if not dram_col or dram_col not in df.columns or policy_col not in df.columns:
        print("(Compact summary skipped: DRAM or policy column missing)")
        return

    budgets = sorted(set(df[dram_col]))
    if not budgets:
        print("(Compact summary skipped: DRAM column has no values)")
        return

    # Pick representative DRAM points: min / median / max (deduplicated)
    picks: List[float] = []
    picks.append(budgets[0])
    mid = budgets[len(budgets) // 2]
    if mid not in picks:
        picks.append(mid)
    if budgets[-1] not in picks:
        picks.append(budgets[-1])

    print("------------------------------------------------------------")
    print("2) COMPACT metrics at key DRAM points (min / median / max)")
    print("   (only key columns, at most 40 rows per DRAM)")
    print("------------------------------------------------------------")

    # Decide which columns to show
    cols_to_show: List[str] = []

    def _add(col_name: Optional[str]):
        if col_name and col_name in df.columns and col_name not in cols_to_show:
            cols_to_show.append(col_name)

    # Basic configuration columns
    _add("nprobe")
    _add(dram_col)
    _add("max_iops")
    _add(policy_col)

    # Key metric columns (mapped automatically if possible)
    _add(mapping.get("p95", ""))
    _add(mapping.get("recall", ""))
    _add(mapping.get("io_amp", ""))
    _add(mapping.get("migration", ""))

    # Fallback: keep at least some columns even if mapping failed
    if not cols_to_show:
        cols_to_show = list(df.columns)[:8]

    for b in picks:
        g = df[df[dram_col] == b]
        if g.empty:
            continue
        print(f"\n=== COMPACT: {dram_col} = {b} (showing first 40 rows, key metrics only) ===")

        # Sort by p95 if available
        p95_col = mapping.get("p95")
        if p95_col and p95_col in g.columns:
            g = g.sort_values(p95_col)

        # Only show selected columns and limit rows
        print(g[cols_to_show].head(40).to_string(index=False))


# ---------- Per-run processing ----------

def process_run_dir(run_dir: Path) -> None:
    """
    Process a single run directory (containing agg/ann_policy_agg.csv) and print
    a compact summary that is easy to copy into ChatGPT.
    """
    agg_csv = run_dir / "agg" / "ann_policy_agg.csv"
    sla_csv = run_dir / "agg" / "sla_reachable_recall.csv"

    print("============================================================")
    print("[dump_results] RUN_DIR =", run_dir.as_posix())
    print("============================================================\n")

    if not agg_csv.exists():
        print(f"[ERROR] Cannot find agg/ann_policy_agg.csv: {agg_csv}")
        return

    # 1) Read agg CSV
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
        print(
            f"[WARN] Column '{policy_col}' not found in the table. Many analyses below "
            f"may fail; please inspect df.columns and modify the script accordingly."
        )
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
        print("[WARN] Failed to automatically detect the DRAM column; only printing head() below.")
        print()

    print("Preview of the first 10 rows (schema overview):")
    print(df.head(10).to_string(index=False))
    print()

    # 2) COMPACT summary at key DRAM points
    print_compact_summary(df, mapping, policy_col, dram_col)

    # 3) Try to read SLA reachable recall
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
        print(f"Did not find {sla_csv.as_posix()} (this file is optional)")

    print()  # blank line between runs


# ---------- Discovering run dirs ----------

def list_run_dirs(results_root: Path) -> List[Path]:
    """
    Under the given results root, find all subdirectories that contain
    agg/ann_policy_agg.csv. Each such subdirectory is treated as one run.
    """
    if not results_root.exists():
        return []

    run_dirs: List[Path] = []
    for p in sorted(results_root.iterdir()):
        if p.is_dir() and (p / "agg" / "ann_policy_agg.csv").exists():
            run_dirs.append(p)
    return run_dirs


def main() -> None:
    repo_root = Path(".").resolve()

    # CLI behavior:
    #   - no argument: use ./results as the root, scan all run dirs
    #   - one argument:
    #       * if it is a run dir (contains agg/ann_policy_agg.csv), process ONLY that
    #       * else, treat it as a results root and scan for all run dirs inside
    run_dirs: List[Path] = []

    if len(sys.argv) >= 2:
        user_path = Path(sys.argv[1]).resolve()
        if not user_path.exists():
            print(f"[ERROR] Path does not exist: {user_path}", file=sys.stderr)
            sys.exit(1)

        if user_path.is_dir() and (user_path / "agg" / "ann_policy_agg.csv").exists():
            # This is a specific run directory
            run_dirs = [user_path]
        elif user_path.is_dir():
            # Treat as results root
            run_dirs = list_run_dirs(user_path)
            if not run_dirs:
                print(
                    f"[ERROR] No run directories (with agg/ann_policy_agg.csv) found under: {user_path}",
                    file=sys.stderr,
                )
                sys.exit(1)
        else:
            print(
                f"[ERROR] Please pass a directory (run dir or results root), not a file: {user_path}",
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        # Default: ./results
        results_root = repo_root / "results"
        run_dirs = list_run_dirs(results_root)
        if not run_dirs:
            print(
                f"[ERROR] Cannot find any run directories under {results_root} "
                f"(need subdirs with agg/ann_policy_agg.csv)",
                file=sys.stderr,
            )
            sys.exit(1)

    print("============================================================")
    print("[dump_results] Found the following run directories:")
    for rd in run_dirs:
        print("  -", rd.as_posix())
    print("============================================================")
    print()

    # Process each run
    for idx, rd in enumerate(run_dirs, 1):
        print(f"\n#################### RUN {idx} / {len(run_dirs)} ####################\n")
        process_run_dir(rd)

    print("\n============================================================")
    print("Copy the COMPACT summaries above into ChatGPT (include the")
    print("RUN_DIR names). Then I can compare policies and help you")
    print("write the final analysis/report.")
    print("============================================================")


if __name__ == "__main__":
    main()
