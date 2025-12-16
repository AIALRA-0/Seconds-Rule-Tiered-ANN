from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .config import load_config
from .sweeps import aggregate_raw, compute_sla_reachable_recall, run_ann_policy_sweep
from .plotting import (
    plot_io_amp_vs_dram,
    plot_migration_vs_dram,
    plot_p95_vs_dram,
    plot_p95_vs_iops,
    plot_recall_latency_frontier,
    plot_sla_reachable_recall,
)
from .utils import ensure_dir, now_timestamp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument("--run_id", type=str, default="")
    args = ap.parse_args()

    cfg = load_config(args.config)

    run_id = args.run_id or f"{cfg.project.run_name}_{now_timestamp()}"
    run_dir = ensure_dir(Path(cfg.project.results_root) / run_id)
    fig_dir = ensure_dir(run_dir / "figures")
    agg_dir = ensure_dir(run_dir / "agg")

    # 1) run raw sweep
    raw_csv = run_ann_policy_sweep(cfg, run_dir)

    # 2) aggregate
    agg_csv = aggregate_raw(raw_csv, agg_dir / "ann_policy_agg.csv")

    # 3) SLA reachable recall
    sla_csv = compute_sla_reachable_recall(
        agg_csv=agg_csv,
        slas_us=cfg.experiment.slas_us,
        out_csv=agg_dir / "sla_reachable_recall.csv",
    )

    if not cfg.plot.enable:
        print(f"[run_all] Done. Results in {run_dir}")
        return

    # 4) plot
    agg_df = pd.read_csv(agg_csv)
    sla_df = pd.read_csv(sla_csv)

    # You can also narrow down the set of policies to plot here
    policy_filter = cfg.policy.enabled_policies

    if "p95_vs_dram" in cfg.plot.figures:
        plot_p95_vs_dram(agg_df, fig_dir, cfg.plot.errorbar, policy_filter=policy_filter)
    if "p95_vs_iops" in cfg.plot.figures:
        plot_p95_vs_iops(agg_df, fig_dir, cfg.plot.errorbar, policy_filter=policy_filter)
    if "io_amp_vs_dram" in cfg.plot.figures:
        plot_io_amp_vs_dram(agg_df, fig_dir, cfg.plot.errorbar, policy_filter=policy_filter)
    if "migration_vs_dram" in cfg.plot.figures:
        plot_migration_vs_dram(agg_df, fig_dir, cfg.plot.errorbar, policy_filter=policy_filter)
    if "sla_reachable_recall" in cfg.plot.figures:
        plot_sla_reachable_recall(sla_df, fig_dir, policy_filter=policy_filter)
    if "recall_latency_frontier" in cfg.plot.figures:
        plot_recall_latency_frontier(agg_df, fig_dir, policy_filter=policy_filter)

    print(f"[run_all] Done. Results in {run_dir}")


if __name__ == "__main__":
    main()
