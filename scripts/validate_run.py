#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


def _require_cols(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f"[validate_run] Missing columns: {missing}\nAll columns: {list(df.columns)}")


def _check_monotonic(
    df: pd.DataFrame,
    *,
    group_keys: List[str],
    x_col: str,
    y_col: str,
    should_be: str,  # "non_increasing" or "non_decreasing"
) -> List[Tuple[Tuple, int, int]]:
    """
    Count monotonicity violations by adjacent comparisons after sorting by x.

    Returns: list of (group_key_tuple, violations, comparisons)
    """
    out: List[Tuple[Tuple, int, int]] = []
    for keys, sub in df.groupby(group_keys):
        sub = sub.sort_values(x_col)
        y = sub[y_col].to_numpy(dtype=float)
        if len(y) <= 1:
            continue
        diff = np.diff(y)

        if should_be == "non_increasing":
            viol = int(np.sum(diff > 1e-12))
        elif should_be == "non_decreasing":
            viol = int(np.sum(diff < -1e-12))
        else:
            raise ValueError(should_be)

        total = int(len(diff))
        if total > 0:
            out.append((keys, viol, total))
    return out


def _print_top_violations(title: str, items: List[Tuple[Tuple, int, int]], limit: int = 10) -> None:
    bad = [(k, v, t) for (k, v, t) in items if v > 0]
    if not bad:
        print(f"[OK] {title}: no violations")
        return

    bad.sort(key=lambda x: (x[1] / max(x[2], 1), x[1]), reverse=True)
    print(f"[WARN] {title}: {len(bad)} groups have violations")
    for k, v, t in bad[:limit]:
        print(f"  - keys={k} violations={v}/{t}")


def _find_policy(df: pd.DataFrame, name: str) -> Optional[str]:
    policies = sorted(set(df["policy"].astype(str)))
    for p in policies:
        if p == name:
            return p
    return None


def _check_oracle_min_latency(df: pd.DataFrame) -> None:
    """
    Check that all_dram has the minimum p95 latency at each (nprobe, dram_fraction, max_iops).
    This should hold (it never reads SSD), and is a strong sanity check.
    """
    if _find_policy(df, "all_dram") is None:
        print("[SKIP] oracle check: policy 'all_dram' not found")
        return

    keys = ["nprobe", "dram_fraction", "max_iops"]
    bad = 0
    total = 0
    for k, sub in df.groupby(keys):
        total += 1
        sub = sub.copy()
        oracle = sub[sub["policy"] == "all_dram"]
        if oracle.empty:
            bad += 1
            continue
        oracle_p95 = float(oracle["p95_latency_us_mean"].iloc[0])
        min_p95 = float(sub["p95_latency_us_mean"].min())
        # allow equality (e.g. dram_fraction=1.0 makes others all-in-dram too)
        if oracle_p95 > min_p95 + 1e-9:
            bad += 1

    if bad == 0:
        print("[OK] oracle min-latency check: all_dram is minimal (or tied) in every group")
    else:
        print(f"[WARN] oracle min-latency check: {bad}/{total} groups violated")


def _check_all_dram_zero_io(df: pd.DataFrame) -> None:
    if _find_policy(df, "all_dram") is None:
        print("[SKIP] all_dram IO check: policy not found")
        return
    sub = df[df["policy"] == "all_dram"]
    if sub.empty:
        print("[SKIP] all_dram IO check: no rows")
        return
    max_abs = float(np.max(np.abs(sub["avg_io_amplification_mean"].to_numpy(dtype=float))))
    if max_abs <= 1e-12:
        print("[OK] all_dram IO amp check: avg_io_amplification_mean is 0 (as expected)")
    else:
        print(f"[WARN] all_dram IO amp check: expected 0 but max_abs={max_abs}")


def _check_nprobe_recall_monotonic(df: pd.DataFrame) -> None:
    # recall depends only on nprobe; still we check monotonicity within each (dram_fraction,max_iops,policy)
    items = _check_monotonic(
        df,
        group_keys=["policy", "dram_fraction", "max_iops"],
        x_col="nprobe",
        y_col="recall_at_k_mean",
        should_be="non_decreasing",
    )
    _print_top_violations("Recall should be non-decreasing when nprobe increases", items)


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python scripts/validate_run.py <RUN_DIR>", file=sys.stderr)
        sys.exit(2)

    run_dir = Path(sys.argv[1]).resolve()
    agg_csv = run_dir / "agg" / "ann_policy_agg.csv"

    if not agg_csv.exists():
        raise SystemExit(f"[validate_run] Missing {agg_csv}")

    df = pd.read_csv(agg_csv)

    _require_cols(df, [
        "policy",
        "dram_fraction",
        "max_iops",
        "nprobe",
        "p95_latency_us_mean",
        "avg_io_amplification_mean",
        "recall_at_k_mean",
    ])

    print("============================================================")
    print(f"[validate_run] RUN_DIR = {run_dir}")
    print(f"[validate_run] agg_csv = {agg_csv}")
    print("============================================================")

    # 1) DRAM ↑ => p95 ↓ (non-increasing)
    mono_lat_dram = _check_monotonic(
        df,
        group_keys=["policy", "max_iops", "nprobe"],
        x_col="dram_fraction",
        y_col="p95_latency_us_mean",
        should_be="non_increasing",
    )
    _print_top_violations("DRAM fraction ↑ => p95 latency should be non-increasing", mono_lat_dram)

    # 2) DRAM ↑ => IO amp ↓
    mono_io_dram = _check_monotonic(
        df,
        group_keys=["policy", "max_iops", "nprobe"],
        x_col="dram_fraction",
        y_col="avg_io_amplification_mean",
        should_be="non_increasing",
    )
    _print_top_violations("DRAM fraction ↑ => IO amplification should be non-increasing", mono_io_dram)

    # 3) IOPS ↑ => p95 ↓ (for any policy; for all_dram it's constant so still OK)
    mono_lat_iops = _check_monotonic(
        df,
        group_keys=["policy", "dram_fraction", "nprobe"],
        x_col="max_iops",
        y_col="p95_latency_us_mean",
        should_be="non_increasing",
    )
    _print_top_violations("max_iops ↑ => p95 latency should be non-increasing", mono_lat_iops)

    # 4) nprobe ↑ => recall ↑ (usually; can be violated rarely by tie effects, so only WARN)
    _check_nprobe_recall_monotonic(df)

    # 5) Strong sanity checks
    _check_oracle_min_latency(df)
    _check_all_dram_zero_io(df)

    # Exit code: fail if any DRAM monotonicity violations exist (these should never happen unless bug/noise)
    viol_lat = sum(v for _, v, _ in mono_lat_dram)
    viol_io = sum(v for _, v, _ in mono_io_dram)

    if viol_lat > 0 or viol_io > 0:
        print("[validate_run] FAIL: found DRAM-related monotonicity violations (likely bug or randomness)")
        sys.exit(1)

    print("[validate_run] PASS")
    sys.exit(0)


if __name__ == "__main__":
    main()
