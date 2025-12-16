from __future__ import annotations

import csv
from pathlib import Path
from typing import List

import pandas as pd

from .ann_engine import build_context, eval_ivf_for_nprobe
from .config import FullConfig
from .experiment_runner import run_policies_on_queries
from .latency_model import TierModelParams
from .utils import SummaryStats, ensure_dir


def run_ann_policy_sweep(cfg: FullConfig, run_dir: Path) -> Path:
    """
    Generate raw CSV: each row = (seed, nprobe, dram_fraction, max_iops, policy, metrics...)
    """
    raw_dir = ensure_dir(run_dir / "raw")
    out_csv = raw_dir / "ann_policy_raw.csv"

    fieldnames = [
        "seed",
        "nprobe",
        "dram_fraction",
        "max_iops",
        "policy",
        "avg_ann_us",
        "recall_at_k",
        "avg_latency_us",
        "p50_latency_us",
        "p95_latency_us",
        "p99_latency_us",
        "qps_from_avg_latency",
        "avg_ssd_pages",
        "avg_ssd_bytes",
        "avg_io_amplification",
        "total_migration_bytes",
        "migration_bytes_per_query",
        "avg_migrated_clusters_per_rebalance",
        "total_migration_time_us",
        "migration_time_us_per_query",
    ]

    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for seed in cfg.experiment.seeds:
            ctx = build_context(cfg, seed)

            for nprobe in cfg.index.nprobe_candidates:
                avg_ann_us, rec, q_lists = eval_ivf_for_nprobe(ctx, cfg, nprobe)

                for dram_fraction in cfg.tier.dram_fraction_list:
                    for max_iops in cfg.tier.max_iops_list:
                        tier = TierModelParams(
                            page_size_bytes=cfg.tier.page_size_bytes,
                            dram_access_us_per_list=cfg.tier.dram_access_us_per_list,
                            ssd_base_lat_us_per_page=cfg.tier.ssd_base_lat_us_per_page,
                            max_iops=float(max_iops),
                            migration_count_eviction=cfg.tier.migration_count_eviction,
                        )

                        metrics_by_policy = run_policies_on_queries(
                            cfg,
                            query_list_ids=q_lists,
                            avg_ann_us=avg_ann_us,
                            recall_at_k=rec,
                            list_bytes=ctx.list_bytes,
                            bytes_per_vec=ctx.bytes_per_vec,
                            tier=tier,
                            dram_fraction=float(dram_fraction),
                        )

                        for policy, m in metrics_by_policy.items():
                            row = {
                                "seed": seed,
                                "nprobe": nprobe,
                                "dram_fraction": float(dram_fraction),
                                "max_iops": float(max_iops),
                                "policy": policy,
                            }
                            row.update(m)
                            w.writerow(row)

    return out_csv


def aggregate_raw(raw_csv: Path, out_csv: Path) -> Path:
    """
    Aggregate across multiple seeds: mean/std/ci95
    Grouping keys: (nprobe, dram_fraction, max_iops, policy)
    """
    df = pd.read_csv(raw_csv)

    group_cols = ["nprobe", "dram_fraction", "max_iops", "policy"]

    metric_cols = [c for c in df.columns if c not in group_cols + ["seed"]]
    rows = []

    for keys, sub in df.groupby(group_cols):
        row = dict(zip(group_cols, keys))
        for mc in metric_cols:
            vals = sub[mc].astype(float).tolist()
            st = SummaryStats.from_values(vals)
            row[f"{mc}_mean"] = st.mean
            row[f"{mc}_std"] = st.std
            row[f"{mc}_ci95"] = st.ci95
        row["num_seeds"] = int(len(sub))
        rows.append(row)

    out_df = pd.DataFrame(rows).sort_values(group_cols)
    out_df.to_csv(out_csv, index=False)
    return out_csv


def compute_sla_reachable_recall(
    agg_csv: Path,
    slas_us: List[float],
    out_csv: Path,
) -> Path:
    """
    SLA → reachable recall@k:
    For each (dram_fraction, max_iops, policy, SLA), pick the largest nprobe such that p95_mean <= SLA,
    and output best_nprobe and the corresponding recall_mean.
    """
    df = pd.read_csv(agg_csv)

    rows = []
    keys = ["dram_fraction", "max_iops", "policy"]

    for (dram_fraction, max_iops, policy), sub in df.groupby(keys):
        # Per policy: we have multiple nprobe values
        sub = sub.sort_values("nprobe")
        for sla in slas_us:
            feasible = sub[sub["p95_latency_us_mean"] <= float(sla)]
            if len(feasible) == 0:
                best_nprobe = None
                best_recall = 0.0
                best_p95 = float("inf")
            else:
                best = feasible.iloc[-1]  # largest nprobe
                best_nprobe = int(best["nprobe"])
                best_recall = float(best["recall_at_k_mean"])
                best_p95 = float(best["p95_latency_us_mean"])

            rows.append(
                {
                    "dram_fraction": float(dram_fraction),
                    "max_iops": float(max_iops),
                    "policy": str(policy),
                    "sla_us": float(sla),
                    "best_nprobe": best_nprobe if best_nprobe is not None else -1,
                    "reachable_recall_at_k": best_recall,
                    "best_p95_latency_us": best_p95,
                }
            )

    out = pd.DataFrame(rows).sort_values(keys + ["sla_us"])
    out.to_csv(out_csv, index=False)
    return out_csv
