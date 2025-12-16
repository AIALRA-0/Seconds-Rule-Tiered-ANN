from __future__ import annotations

from pathlib import Path
from typing import Literal, Iterable, Optional, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import matplotlib as mpl
# mpl.rcParams["figure.autolayout"] = True



# -----------------------------
# Helper
# -----------------------------

def _mean_and_err(vals: np.ndarray, mode: Literal["std", "ci95"]) -> tuple[float, float]:
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 0.0
    m = float(vals.mean())
    if vals.size <= 1:
        return m, 0.0
    sd = float(vals.std(ddof=1))
    if mode == "std":
        return m, sd
    # ci95 half-width
    return m, 1.96 * sd / np.sqrt(vals.size)


def _set_recall_ylim(ax, ys: np.ndarray, *, pad_frac: float = 0.10):
    ys = np.asarray(ys, dtype=float)
    lo, hi = _finite_min_max(ys)
    lo, hi = _pad_limits(lo, hi, pad_frac, min_span=0.05)

    lo = min(lo, -0.02, 0.0)
    hi = max(hi, 1.02, 1.0)

    lo = max(lo, -0.05)
    hi = min(hi, 1.05)
    ax.set_ylim(lo, hi)


def _maybe_yerr(df: pd.DataFrame, col_base: str, mode: Literal["std", "ci95"]) -> Optional[np.ndarray]:
    primary = f"{col_base}_std" if mode == "std" else f"{col_base}_ci95"
    backup  = f"{col_base}_ci95" if mode == "std" else f"{col_base}_std"

    for col in (primary, backup):
        if col in df.columns:
            arr = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            return arr
    return None



def _yerr(df: pd.DataFrame, col_base: str, mode: Literal["std", "ci95"]):
    if mode == "std":
        return df[f"{col_base}_std"]
    return df[f"{col_base}_ci95"]


def _finite_min_max(arr: np.ndarray) -> tuple[float, float]:
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0, 1.0
    return float(arr.min()), float(arr.max())


def _pad_limits(lo: float, hi: float, pad_frac: float, *, min_span: float = 1e-9) -> tuple[float, float]:
    """
    Robust axis padding that does NOT explode small ranges (e.g. values in [0, 1]).

    - If span is ~0, we expand the span based on the magnitude of the values:
        span = max(min_span, 0.1 * max(|lo|, |hi|), 1e-3)
      so for recall around 0.8 we get ~0.08 span, not 1.0 span.
    """
    if not np.isfinite(lo) or not np.isfinite(hi):
        return 0.0, 1.0
    if hi < lo:
        lo, hi = hi, lo

    span = hi - lo
    if span < min_span:
        scale = max(abs(lo), abs(hi))
        span = max(min_span, 0.1 * scale, 1e-3)
        mid = 0.5 * (lo + hi)
        lo = mid - 0.5 * span
        hi = mid + 0.5 * span

    pad = pad_frac * span
    return lo - pad, hi + pad


def _union_legend_handles(axes: np.ndarray) -> Tuple[list, list]:
    """
    Collect a UNION of legend entries across subplots (instead of taking only the first axis).
    This avoids missing legend labels when some subplots have fewer lines.
    """
    handles: list = []
    labels: list = []
    seen: set[str] = set()
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if not ll:
                continue
            if ll in seen:
                continue
            seen.add(ll)
            handles.append(hh)
            labels.append(ll)
    return handles, labels


def _finalize_grid_figure(
    fig,
    axes: np.ndarray,
    out_path: Path,
    *,
    suptitle: str,
    xlabel: str,
    ylabel: str,
    legend_loc: Literal["right", "bottom"] = "right",
    legend_ncol_max: int = 6,
):
    fig.suptitle(suptitle, y=0.99)

    if hasattr(fig, "supxlabel"):
        fig.supxlabel(xlabel)
    else:
        fig.text(0.5, 0.04, xlabel, ha="center")

    if hasattr(fig, "supylabel"):
        fig.supylabel(ylabel)
    else:
        fig.text(0.04, 0.5, ylabel, va="center", rotation="vertical")

    handles, labels = _union_legend_handles(axes)

    legend = None
    if handles:
        if legend_loc == "right":
            legend = fig.legend(
                handles,
                labels,
                loc="center left",
                bbox_to_anchor=(1.01, 0.5),  
                frameon=False,
                fontsize=9,
            )
            rect = (0.06, 0.06, 0.82, 0.92)  
        else:
            legend = fig.legend(
                handles,
                labels,
                loc="lower center",
                bbox_to_anchor=(0.5, 0.01),
                ncol=min(len(handles), legend_ncol_max),
                frameon=False,
                fontsize=9,
            )
            rect = (0.06, 0.08, 0.98, 0.92)
    else:
        rect = (0.06, 0.06, 0.98, 0.92)

    fig.tight_layout(rect=rect)

    extra = [legend] if legend is not None else None
    fig.savefig(out_path, dpi=250, bbox_inches="tight", bbox_extra_artists=extra)
    plt.close(fig)


def _finalize_and_save(fig, ax, out_path: Path):
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        fontsize=9,
    )
    fig.tight_layout(rect=(0.0, 0.0, 0.82, 1.0))
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Small-multiples helpers
# -----------------------------

def _iter_policies(df: pd.DataFrame, policy_filter: Optional[Iterable[str]] = None) -> List[str]:
    policies = list(df["policy"].unique())
    if policy_filter is not None:
        # Keep user-provided order (important for consistency across figures)
        allowed = [p for p in policy_filter if p in policies]
        return allowed
    return sorted(policies)


def _iter_nprobes(df: pd.DataFrame, nprobe_filter: Optional[Iterable[int]] = None) -> List[int]:
    nprobes = sorted(df["nprobe"].unique())
    if nprobe_filter is not None:
        allowed_set = set(nprobe_filter)
        # Keep user-provided order
        allowed = [n for n in nprobe_filter if n in allowed_set and n in nprobes]
        return allowed
    return nprobes


def _make_grid(n_items: int):
    import math
    if n_items <= 0:
        return 1, 1
    ncols = 3 if n_items >= 3 else (2 if n_items > 1 else 1)
    nrows = math.ceil(n_items / ncols)
    return nrows, ncols


def _set_ylim_from_series(ax, y: np.ndarray, yerr: Optional[np.ndarray] = None, *, pad_frac: float = 0.10):
    y = np.asarray(y, dtype=float)
    if yerr is not None:
        yerr = np.asarray(yerr, dtype=float)
        lo, hi = _finite_min_max(np.concatenate([y - yerr, y + yerr]))
    else:
        lo, hi = _finite_min_max(y)
    lo, hi = _pad_limits(lo, hi, pad_frac)
    ax.set_ylim(lo, hi)


def _set_xlim_from_series(ax, x: np.ndarray, *, pad_frac: float = 0.05, min_span: float = 1.0):
    x = np.asarray(x, dtype=float)
    lo, hi = _finite_min_max(x)
    lo, hi = _pad_limits(lo, hi, pad_frac, min_span=min_span)
    ax.set_xlim(lo, hi)


# -----------------------------
# Plots: p95 vs DRAM (faceted by policy, per max_iops)
# -----------------------------

def plot_p95_vs_dram(agg_df: pd.DataFrame, out_dir: Path, errorbar: str, *, policy_filter=None, nprobe_filter=None):
    df = agg_df.copy()
    nprobes = _iter_nprobes(df, nprobe_filter)

    for max_iops, df_iops in df.groupby("max_iops"):
        policies = _iter_policies(df_iops, policy_filter)
        if not policies:
            continue

        nrows, ncols = _make_grid(len(policies))
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(5.8 * ncols, 3.2 * nrows),
            sharex=True,
            sharey=False,
        )
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])
        axes = axes.reshape(-1)

        # dynamic x range based on existing DRAM points
        x_all = (df_iops["dram_fraction"] * 100.0).to_numpy()
        for ax in axes:
            _set_xlim_from_series(ax, x_all, pad_frac=0.06, min_span=5.0)

        for ax, policy in zip(axes, policies):
            sub_all = df_iops[df_iops["policy"] == policy]

            y_all: list[np.ndarray] = []
            yerr_all: list[np.ndarray] = []

            for nprobe in nprobes:
                sub = sub_all[sub_all["nprobe"] == nprobe].sort_values("dram_fraction")
                if sub.empty:
                    continue

                x = (sub["dram_fraction"] * 100.0).to_numpy()
                y = sub["p95_latency_us_mean"].to_numpy()
                yerr = _yerr(sub, "p95_latency_us", errorbar).to_numpy()

                ax.errorbar(
                    x,
                    y,
                    yerr=yerr,
                    marker="o",
                    linestyle="-",
                    capsize=3,
                    elinewidth=1,
                    linewidth=1.5,
                    markersize=4,
                    label=f"nprobe={nprobe}",
                )
                y_all.append(y)
                yerr_all.append(yerr)

            ax.set_title(policy, fontsize=10)
            ax.grid(True, linestyle=":")
            ax.margins(x=0.02, y=0.10)

            if y_all:
                y_cat = np.concatenate(y_all)
                ye_cat = np.concatenate(yerr_all) if yerr_all else None
                _set_ylim_from_series(ax, y_cat, ye_cat, pad_frac=0.18)

        for ax in axes[len(policies):]:
            ax.axis("off")

        out_path = out_dir / f"p95_vs_dram_iops{int(max_iops/1e6)}M.png"
        _finalize_grid_figure(
            fig,
            axes,
            out_path,
            suptitle=f"p95 latency vs DRAM (max_iops={max_iops/1e6:.0f}M)",
            xlabel="DRAM fraction (%)",
            ylabel="p95 latency (µs)",
            legend_loc="right",
        )


# -----------------------------
# Plots: p95 vs max_iops (faceted by policy, per DRAM)
# -----------------------------

def plot_p95_vs_iops(agg_df: pd.DataFrame, out_dir: Path, errorbar: str, *, policy_filter=None, nprobe_filter=None):
    df = agg_df.copy()
    nprobes = _iter_nprobes(df, nprobe_filter)

    for dram_fraction, df_dram in df.groupby("dram_fraction"):
        policies = _iter_policies(df_dram, policy_filter)
        if not policies:
            continue

        nrows, ncols = _make_grid(len(policies))
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(5.8 * ncols, 3.2 * nrows),
            sharex=True,
            sharey=False,
        )
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])
        axes = axes.reshape(-1)

        x_all = (df_dram["max_iops"] / 1e6).to_numpy()
        for ax in axes:
            _set_xlim_from_series(ax, x_all, pad_frac=0.06, min_span=0.2)

        for ax, policy in zip(axes, policies):
            sub_all = df_dram[df_dram["policy"] == policy]

            y_all: list[np.ndarray] = []
            yerr_all: list[np.ndarray] = []

            for nprobe in nprobes:
                sub = sub_all[sub_all["nprobe"] == nprobe].sort_values("max_iops")
                if sub.empty:
                    continue

                x = (sub["max_iops"] / 1e6).to_numpy()
                y = sub["p95_latency_us_mean"].to_numpy()
                yerr = _yerr(sub, "p95_latency_us", errorbar).to_numpy()

                ax.errorbar(
                    x,
                    y,
                    yerr=yerr,
                    marker="o",
                    linestyle="-",
                    capsize=3,
                    elinewidth=1,
                    linewidth=1.5,
                    markersize=4,
                    label=f"nprobe={nprobe}",
                )
                y_all.append(y)
                yerr_all.append(yerr)

            ax.set_title(policy, fontsize=10)
            ax.grid(True, linestyle=":")
            ax.margins(x=0.02, y=0.10)

            if y_all:
                y_cat = np.concatenate(y_all)
                ye_cat = np.concatenate(yerr_all) if yerr_all else None
                _set_ylim_from_series(ax, y_cat, ye_cat, pad_frac=0.18)

        for ax in axes[len(policies):]:
            ax.axis("off")

        out_path = out_dir / f"p95_vs_iops_dram{int(dram_fraction*100)}.png"
        _finalize_grid_figure(
            fig,
            axes,
            out_path,
            suptitle=f"p95 latency vs max IOPS (DRAM={dram_fraction*100:.0f}%)",
            xlabel="max IOPS (million)",
            ylabel="p95 latency (µs)",
            legend_loc="right",
        )


# -----------------------------
# Plots: IO amplification vs DRAM (faceted by policy, per max_iops)
# -----------------------------

def plot_io_amp_vs_dram(agg_df: pd.DataFrame, out_dir: Path, errorbar: str, *, policy_filter=None, nprobe_filter=None):
    df = agg_df.copy()

    df = df[df["policy"] != "all_dram"]

    nprobes = _iter_nprobes(df, nprobe_filter)

    for max_iops, df_iops in df.groupby("max_iops"):
        policies = _iter_policies(df_iops, policy_filter)
        if not policies:
            continue

        nrows, ncols = _make_grid(len(policies))
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(5.8 * ncols, 3.6 * nrows),
            sharex=True,
            sharey=False,
        )
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])
        axes = axes.reshape(-1)

        x_all = (df_iops["dram_fraction"] * 100.0).to_numpy()
        for ax in axes:
            _set_xlim_from_series(ax, x_all, pad_frac=0.06, min_span=5.0)

        for ax, policy in zip(axes, policies):
            sub_all = df_iops[df_iops["policy"] == policy]

            y_all: list[np.ndarray] = []
            yerr_all: list[np.ndarray] = []

            for nprobe in nprobes:
                sub = sub_all[sub_all["nprobe"] == nprobe].sort_values("dram_fraction")
                if sub.empty:
                    continue

                x = (sub["dram_fraction"] * 100.0).to_numpy()
                y = sub["avg_io_amplification_mean"].to_numpy()
                yerr = _yerr(sub, "avg_io_amplification", errorbar).to_numpy()

                ax.errorbar(
                    x,
                    y,
                    yerr=yerr,
                    marker="o",
                    linestyle="-",
                    capsize=3,
                    elinewidth=1,
                    linewidth=1.5,
                    markersize=4,
                    label=f"nprobe={nprobe}",
                )
                y_all.append(y)
                yerr_all.append(yerr)

            ax.set_title(policy, fontsize=10)
            ax.grid(True, linestyle=":")
            ax.margins(x=0.02, y=0.12)

            if y_all:
                y_cat = np.concatenate(y_all)
                ye_cat = np.concatenate(yerr_all) if yerr_all else None
                _set_ylim_from_series(ax, y_cat, ye_cat, pad_frac=0.15)

        for ax in axes[len(policies):]:
            ax.axis("off")

        out_path = out_dir / f"io_amp_vs_dram_iops{int(max_iops/1e6)}M.png"
        _finalize_grid_figure(
            fig,
            axes,
            out_path,
            suptitle=f"I/O amplification vs DRAM (max_iops={max_iops/1e6:.0f}M)",
            xlabel="DRAM fraction (%)",
            ylabel="avg I/O amplification",
            legend_loc="right",
        )


# -----------------------------
# Plots: migration bytes vs DRAM (faceted by policy, per max_iops)
# -----------------------------

def plot_migration_vs_dram(agg_df: pd.DataFrame, out_dir: Path, errorbar: str, *, policy_filter=None, nprobe_filter=None):
    df = agg_df.copy()
    nprobes = _iter_nprobes(df, nprobe_filter)

    for max_iops, df_iops in df.groupby("max_iops"):
        policies = _iter_policies(df_iops, policy_filter)
        if not policies:
            continue

        nrows, ncols = _make_grid(len(policies))
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(5.8 * ncols, 3.2 * nrows),
            sharex=True,
            sharey=False,
        )
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])
        axes = axes.reshape(-1)

        x_all = (df_iops["dram_fraction"] * 100.0).to_numpy()
        for ax in axes:
            _set_xlim_from_series(ax, x_all, pad_frac=0.06, min_span=5.0)

        for ax, policy in zip(axes, policies):
            sub_all = df_iops[df_iops["policy"] == policy]

            y_all: list[np.ndarray] = []
            yerr_all: list[np.ndarray] = []

            for nprobe in nprobes:
                sub = sub_all[sub_all["nprobe"] == nprobe].sort_values("dram_fraction")
                if sub.empty:
                    continue

                x = (sub["dram_fraction"] * 100.0).to_numpy()
                y = sub["migration_bytes_per_query_mean"].to_numpy()
                yerr = _yerr(sub, "migration_bytes_per_query", errorbar).to_numpy()

                ax.errorbar(
                    x,
                    y,
                    yerr=yerr,
                    marker="o",
                    linestyle="-",
                    capsize=3,
                    elinewidth=1,
                    linewidth=1.5,
                    markersize=4,
                    label=f"nprobe={nprobe}",
                )
                y_all.append(y)
                yerr_all.append(yerr)

            ax.set_title(policy, fontsize=10)
            ax.grid(True, linestyle=":")
            ax.margins(x=0.02, y=0.12)

            if y_all:
                y_cat = np.concatenate(y_all)
                ye_cat = np.concatenate(yerr_all) if yerr_all else None
                _set_ylim_from_series(ax, y_cat, ye_cat, pad_frac=0.15)

        for ax in axes[len(policies):]:
            ax.axis("off")

        out_path = out_dir / f"migration_vs_dram_iops{int(max_iops/1e6)}M.png"
        _finalize_grid_figure(
            fig,
            axes,
            out_path,
            suptitle=f"Migration bytes per query vs DRAM (max_iops={max_iops/1e6:.0f}M)",
            xlabel="DRAM fraction (%)",
            ylabel="migration bytes / query",
            legend_loc="right",
        )


# -----------------------------
# SLA reachable recall (one figure per dram_fraction & max_iops)
# -----------------------------

def plot_sla_reachable_recall(sla_df: pd.DataFrame, out_dir: Path, errorbar: str = "ci95", *, policy_filter=None):
    df = sla_df.copy()
    if policy_filter is not None:
        df = df[df["policy"].isin(policy_filter)]

    for (dram_fraction, max_iops), sub in df.groupby(["dram_fraction", "max_iops"]):
        if sub.empty:
            continue

        fig, ax = plt.subplots(figsize=(5.8, 3.8))

        for policy, g0 in sub.groupby("policy"):
            g0 = g0.sort_values("sla_us")

            if g0["sla_us"].duplicated().any():
                rows = []
                for sla_us, gg in g0.groupby("sla_us"):
                    m, e = _mean_and_err(gg["reachable_recall_at_k"].to_numpy(dtype=float), errorbar)
                    rows.append((float(sla_us), m, e))
                rows.sort(key=lambda t: t[0])
                x = np.array([r[0] for r in rows], dtype=float)
                y = np.array([r[1] for r in rows], dtype=float)
                yerr = np.array([r[2] for r in rows], dtype=float)

                ax.errorbar(
                    x, y, yerr=yerr,
                    marker="o", linestyle="-",
                    linewidth=1.5, markersize=4,
                    capsize=4, elinewidth=1.2, capthick=1.2,
                    label=policy,
                )
            else:
                x = g0["sla_us"].to_numpy(dtype=float)
                y = g0["reachable_recall_at_k"].to_numpy(dtype=float)
                yerr = _maybe_yerr(g0, "reachable_recall_at_k", errorbar)

                if yerr is not None:
                    ax.errorbar(
                        x, y, yerr=yerr,
                        marker="o", linestyle="-",
                        linewidth=1.5, markersize=4,
                        capsize=4, elinewidth=1.2, capthick=1.2,
                        label=policy,
                    )
                else:
                    ax.plot(x, y, marker="o", linestyle="-", linewidth=1.5, markersize=4, label=policy)

        _set_recall_ylim(ax, sub["reachable_recall_at_k"].to_numpy(dtype=float), pad_frac=0.10)

        ax.set_xlabel("p95 latency SLA (µs)")
        ax.set_ylabel("reachable recall@k")
        ax.set_title(f"SLA → reachable recall@k (DRAM={dram_fraction*100:.0f}%, max_iops={max_iops/1e6:.0f}M)")
        ax.grid(True, linestyle=":")
        ax.margins(x=0.02, y=0.08)

        _finalize_and_save(
            fig,
            ax,
            out_dir / f"sla_reachable_recall_dram{int(dram_fraction*100)}_iops{int(max_iops/1e6)}M.png",
        )





# -----------------------------
# Recall–latency frontier (one figure per dram_fraction & max_iops)
# -----------------------------

def plot_recall_latency_frontier(agg_df: pd.DataFrame, out_dir: Path, errorbar: str = "ci95", *, policy_filter=None):
    df = agg_df.copy()
    if policy_filter is not None:
        df = df[df["policy"].isin(policy_filter)]

    for (dram_fraction, max_iops), sub in df.groupby(["dram_fraction", "max_iops"]):
        if sub.empty:
            continue

        fig, ax = plt.subplots(figsize=(5.8, 3.8))
        policies = _iter_policies(sub, policy_filter)

        for policy in policies:
            g0 = sub[sub["policy"] == policy].sort_values("nprobe")

            if g0["nprobe"].duplicated().any():
                rows = []
                for nprobe, gg in g0.groupby("nprobe"):
                    xm, xe = _mean_and_err(gg["p95_latency_us_mean"].to_numpy(dtype=float), errorbar)
                    ym, ye = _mean_and_err(gg["recall_at_k_mean"].to_numpy(dtype=float), errorbar)
                    rows.append((int(nprobe), xm, xe, ym, ye))
                rows.sort(key=lambda t: t[0])

                x = np.array([r[1] for r in rows], dtype=float)
                y = np.array([r[3] for r in rows], dtype=float)
                xerr = np.array([r[2] for r in rows], dtype=float)
                yerr = np.array([r[4] for r in rows], dtype=float)

                ax.errorbar(
                    x, y, xerr=xerr, yerr=yerr,
                    marker="o", linestyle="-",
                    linewidth=1.5, markersize=4,
                    capsize=4, elinewidth=1.2, capthick=1.2,
                    label=policy,
                )
            else:
                x = g0["p95_latency_us_mean"].to_numpy(dtype=float)
                y = g0["recall_at_k_mean"].to_numpy(dtype=float)
                xerr = _maybe_yerr(g0, "p95_latency_us", errorbar)
                yerr = _maybe_yerr(g0, "recall_at_k", errorbar)

                if (xerr is not None) or (yerr is not None):
                    ax.errorbar(
                        x, y, xerr=xerr, yerr=yerr,
                        marker="o", linestyle="-",
                        linewidth=1.5, markersize=4,
                        capsize=4, elinewidth=1.2, capthick=1.2,
                        label=policy,
                    )
                else:
                    ax.plot(x, y, marker="o", linestyle="-", linewidth=1.5, markersize=4, label=policy)

        xs = sub["p95_latency_us_mean"].to_numpy(dtype=float)
        x_lo, x_hi = _finite_min_max(xs)
        x_lo, x_hi = _pad_limits(x_lo, x_hi, 0.08, min_span=1.0)
        ax.set_xlim(x_lo, x_hi)

        _set_recall_ylim(ax, sub["recall_at_k_mean"].to_numpy(dtype=float), pad_frac=0.10)

        ax.set_xlabel("p95 latency (µs)")
        ax.set_ylabel("recall@k")
        ax.set_title(f"Recall–Latency frontier (DRAM={dram_fraction*100:.0f}%, max_iops={max_iops/1e6:.0f}M)")
        ax.grid(True, linestyle=":")
        ax.margins(x=0.02, y=0.06)

        _finalize_and_save(
            fig,
            ax,
            out_dir / f"recall_latency_frontier_dram{int(dram_fraction*100)}_iops{int(max_iops/1e6)}M.png",
        )

# -----------------------------
# CLI entry
# -----------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Plot Seconds-Rule tiered ANN results.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Specific results/tiered_ann_seconds_rule_* directory; if not provided, the latest one is selected automatically.",
    )
    parser.add_argument(
        "--errorbar",
        choices=["std", "ci95"],
        default="ci95",
        help="Use std or ci95 for error bars.",
    )
    parser.add_argument(
        "--policy",
        action="append",
        default=None,
        help="Plot only the specified policies; can be repeated, e.g., --policy seconds_rule --policy naive_lfu",
    )
    parser.add_argument(
        "--nprobe",
        type=int,
        action="append",
        default=None,
        help="Plot only the specified nprobe values; can be repeated, e.g., --nprobe 1 --nprobe 4 --nprobe 32; default is all.",
    )

    args = parser.parse_args()

    root = Path(".").resolve()
    results_root = root / "results"

    if args.run_dir is not None:
        run_dir = args.run_dir.resolve()
    else:
        candidates = sorted(results_root.glob("tiered_ann_seconds_rule_*"))
        if not candidates:
            raise SystemExit(f"[ERROR] Cannot find any tiered_ann_seconds_rule_* under {results_root}")
        run_dir = candidates[-1]

    agg_csv = run_dir / "agg" / "ann_policy_agg.csv"
    sla_csv = run_dir / "agg" / "sla_reachable_recall.csv"

    if not agg_csv.exists():
        raise SystemExit(f"[ERROR] Cannot find {agg_csv}")

    agg_df = pd.read_csv(agg_csv)
    sla_df = pd.read_csv(sla_csv) if sla_csv.exists() else None

    # Standardize: always write into run_dir/figures (same as src/run_all.py)
    out_dir = run_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[plotting] RUN_DIR = {run_dir}")
    print(f"[plotting] Save figures into {out_dir}")

    plot_p95_vs_dram(agg_df, out_dir, args.errorbar, policy_filter=args.policy, nprobe_filter=args.nprobe)
    plot_p95_vs_iops(agg_df, out_dir, args.errorbar, policy_filter=args.policy, nprobe_filter=args.nprobe)
    plot_io_amp_vs_dram(agg_df, out_dir, args.errorbar, policy_filter=args.policy, nprobe_filter=args.nprobe)
    plot_migration_vs_dram(agg_df, out_dir, args.errorbar, policy_filter=args.policy, nprobe_filter=args.nprobe)

    if sla_df is not None:
        plot_sla_reachable_recall(sla_df, out_dir, args.errorbar, policy_filter=args.policy)

    plot_recall_latency_frontier(agg_df, out_dir, args.errorbar, policy_filter=args.policy)


if __name__ == "__main__":
    main()
