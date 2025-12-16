from __future__ import annotations

from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import pandas as pd


# -----------------------------
# Helper
# -----------------------------

def _yerr(df: pd.DataFrame, col_base: str, mode: Literal["std", "ci95"]):
    if mode == "std":
        return df[f"{col_base}_std"]
    return df[f"{col_base}_ci95"]


def _finalize_and_save(fig, ax, out_path: Path):
    """
    统一处理：
    - legend 放到图外右侧
    - 给图腾空间
    - 保存并关闭
    """
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=9,
    )
    fig.subplots_adjust(right=0.75)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# -----------------------------
# Plots
# -----------------------------

def plot_p95_vs_dram(agg_df: pd.DataFrame, out_dir: Path, errorbar: str, *, policy_filter=None):
    df = agg_df.copy()
    if policy_filter is not None:
        df = df[df["policy"].isin(policy_filter)]

    fig, ax = plt.subplots(figsize=(6, 4))

    for (policy, max_iops), sub in df.groupby(["policy", "max_iops"]):
        sub = sub.sort_values("dram_fraction")
        x = sub["dram_fraction"] * 100.0
        y = sub["p95_latency_us_mean"]
        yerr = _yerr(sub, "p95_latency_us", errorbar)

        ax.errorbar(
            x,
            y,
            yerr=yerr,
            marker="o",
            linestyle="-",
            capsize=3,
            label=f"{policy}, IOPS={max_iops/1e6:.0f}M",
        )

    ax.set_xlabel("DRAM fraction (%)")
    ax.set_ylabel("p95 latency (µs)")
    ax.set_title("p95 latency vs DRAM fraction")
    ax.grid(True, linestyle=":")

    _finalize_and_save(fig, ax, out_dir / "p95_vs_dram.png")


def plot_p95_vs_iops(agg_df: pd.DataFrame, out_dir: Path, errorbar: str, *, policy_filter=None):
    df = agg_df.copy()
    if policy_filter is not None:
        df = df[df["policy"].isin(policy_filter)]

    fig, ax = plt.subplots(figsize=(6, 4))

    for (policy, dram_fraction), sub in df.groupby(["policy", "dram_fraction"]):
        sub = sub.sort_values("max_iops")
        x = sub["max_iops"] / 1e6
        y = sub["p95_latency_us_mean"]
        yerr = _yerr(sub, "p95_latency_us", errorbar)

        ax.errorbar(
            x,
            y,
            yerr=yerr,
            marker="o",
            linestyle="-",
            capsize=3,
            label=f"{policy}, DRAM={dram_fraction*100:.0f}%",
        )

    ax.set_xlabel("max IOPS (million)")
    ax.set_ylabel("p95 latency (µs)")
    ax.set_title("p95 latency vs max IOPS")
    ax.grid(True, linestyle=":")

    _finalize_and_save(fig, ax, out_dir / "p95_vs_iops.png")


def plot_io_amp_vs_dram(agg_df: pd.DataFrame, out_dir: Path, errorbar: str, *, policy_filter=None):
    df = agg_df.copy()
    if policy_filter is not None:
        df = df[df["policy"].isin(policy_filter)]

    fig, ax = plt.subplots(figsize=(6, 4))

    for (policy, max_iops), sub in df.groupby(["policy", "max_iops"]):
        sub = sub.sort_values("dram_fraction")
        x = sub["dram_fraction"] * 100.0
        y = sub["avg_io_amplification_mean"]
        yerr = _yerr(sub, "avg_io_amplification", errorbar)

        ax.errorbar(
            x,
            y,
            yerr=yerr,
            marker="o",
            linestyle="-",
            capsize=3,
            label=f"{policy}, IOPS={max_iops/1e6:.0f}M",
        )

    ax.set_xlabel("DRAM fraction (%)")
    ax.set_ylabel("avg I/O amplification")
    ax.set_title("I/O amplification vs DRAM fraction")
    ax.grid(True, linestyle=":")

    _finalize_and_save(fig, ax, out_dir / "io_amp_vs_dram.png")


def plot_migration_vs_dram(agg_df: pd.DataFrame, out_dir: Path, errorbar: str, *, policy_filter=None):
    df = agg_df.copy()
    if policy_filter is not None:
        df = df[df["policy"].isin(policy_filter)]

    fig, ax = plt.subplots(figsize=(6, 4))

    for (policy, max_iops), sub in df.groupby(["policy", "max_iops"]):
        sub = sub.sort_values("dram_fraction")
        x = sub["dram_fraction"] * 100.0
        y = sub["migration_bytes_per_query_mean"]
        yerr = _yerr(sub, "migration_bytes_per_query", errorbar)

        ax.errorbar(
            x,
            y,
            yerr=yerr,
            marker="o",
            linestyle="-",
            capsize=3,
            label=f"{policy}, IOPS={max_iops/1e6:.0f}M",
        )

    ax.set_xlabel("DRAM fraction (%)")
    ax.set_ylabel("migration bytes per query")
    ax.set_title("Migration overhead vs DRAM fraction")
    ax.grid(True, linestyle=":")

    _finalize_and_save(fig, ax, out_dir / "migration_vs_dram.png")


def plot_sla_reachable_recall(sla_df: pd.DataFrame, out_dir: Path, *, policy_filter=None):
    df = sla_df.copy()
    if policy_filter is not None:
        df = df[df["policy"].isin(policy_filter)]

    fig, ax = plt.subplots(figsize=(6, 4))

    for policy, sub in df.groupby("policy"):
        sub = sub.sort_values("sla_us")
        x = sub["sla_us"]
        y = sub["reachable_recall_at_k"]

        ax.plot(
            x,
            y,
            marker="o",
            linestyle="-",
            label=policy,
        )

    ax.set_xlabel("p95 latency SLA (µs)")
    ax.set_ylabel("reachable recall@k")
    ax.set_title("SLA → reachable recall@k")
    ax.grid(True, linestyle=":")

    _finalize_and_save(fig, ax, out_dir / "sla_reachable_recall.png")


def plot_recall_latency_frontier(agg_df: pd.DataFrame, out_dir: Path, *, policy_filter=None):
    """
    Recall–latency frontier:
    x = p95 latency, y = recall@k; points correspond to different nprobe.
    """
    df = agg_df.copy()
    if policy_filter is not None:
        df = df[df["policy"].isin(policy_filter)]

    fig, ax = plt.subplots(figsize=(6, 4))

    for policy, sub in df.groupby("policy"):
        sub = sub.sort_values("nprobe")
        x = sub["p95_latency_us_mean"]
        y = sub["recall_at_k_mean"]

        ax.plot(
            x,
            y,
            marker="o",
            linestyle="-",
            label=policy,
        )

    ax.set_xlabel("p95 latency (µs)")
    ax.set_ylabel("recall@k")
    ax.set_title("Recall–Latency frontier")
    ax.grid(True, linestyle=":")

    _finalize_and_save(fig, ax, out_dir / "recall_latency_frontier.png")
