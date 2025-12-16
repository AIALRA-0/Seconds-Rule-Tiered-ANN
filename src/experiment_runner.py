from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .config import FullConfig
from .latency_model import TierModelParams, migration_overhead_us, query_latency_us
from .policies import build_policy
from .utils import percentile


@dataclass
class PolicyRunResult:
    avg_latency_us: float
    p50_latency_us: float
    p95_latency_us: float
    p99_latency_us: float
    qps_from_avg_latency: float

    avg_ssd_pages: float
    avg_ssd_bytes: float
    avg_io_amplification: float

    total_migration_bytes: float
    migration_bytes_per_query: float
    avg_migrated_clusters_per_rebalance: float

    total_migration_time_us: float
    migration_time_us_per_query: float


def _pages_for_list_bytes(list_bytes: np.ndarray, page_size: int) -> np.ndarray:
    # pages[i] = ceil(list_bytes[i] / page_size)
    pages = np.ceil(list_bytes / float(page_size)).astype(np.int64)
    return pages


def run_policies_on_queries(
    cfg: FullConfig,
    *,
    query_list_ids: np.ndarray,   # (Q, nprobe)
    avg_ann_us: float,
    recall_at_k: float,
    list_bytes: np.ndarray,       # (nlist,)
    bytes_per_vec: int,
    tier: TierModelParams,
    dram_fraction: float,
) -> Dict[str, Dict[str, float]]:
    """
    For a fixed (nprobe, query_list_ids, avg_ann_us, tier params):
      - run multiple policies in parallel
      - output the metrics for each policy
    """
    Q, nprobe = query_list_ids.shape
    k = cfg.index.k

    pages_per_list = _pages_for_list_bytes(list_bytes, tier.page_size_bytes)
    bytes_read_per_list = pages_per_list.astype(np.float64) * float(tier.page_size_bytes)
    list_bytes_list = list(list_bytes.astype(float))

    enabled = cfg.policy.enabled_policies
    policies = {
        name: build_policy(
            name,
            num_clusters=cfg.index.nlist,
            dram_fraction=dram_fraction,
            window_size=cfg.policy.window_lfu.window_size_queries,
            seconds_alpha=cfg.policy.seconds_rule.alpha,
            seconds_recency_weight=cfg.policy.seconds_rule.recency_weight,
            t_star_queries=cfg.policy.seconds_rule.t_star_queries,
        )
        for name in enabled
    }

    rebalance_interval = int(cfg.policy.rebalance_interval)
    count_eviction = bool(cfg.tier.migration_count_eviction)

    # per-policy tracking
    latencies: Dict[str, List[float]] = {p: [] for p in policies}
    ssd_pages: Dict[str, List[int]] = {p: [] for p in policies}
    ssd_bytes: Dict[str, List[float]] = {p: [] for p in policies}

    # migration totals
    migrated_bytes: Dict[str, float] = {p: 0.0 for p in policies}
    migrated_clusters: Dict[str, int] = {p: 0 for p in policies}
    migrated_time_us: Dict[str, float] = {p: 0.0 for p in policies}
    num_rebalances = 0

    for qid in range(Q):
        lists = query_list_ids[qid]

        # (Optional) de-duplicate: nprobe is small, a simple set is enough
        unique_lists = []
        seen = set()
        for cid in lists:
            c = int(cid)
            if c not in seen:
                seen.add(c)
                unique_lists.append(c)

        for pname, policy in policies.items():
            # update access stats
            for cid in unique_lists:
                policy.on_access(cid, qid)

            dram_ops = 0
            ssd_page_ops = 0
            ssd_byte = 0.0

            for cid in unique_lists:
                if policy.is_in_dram(cid):
                    dram_ops += 1
                else:
                    ssd_page_ops += int(pages_per_list[cid])
                    ssd_byte += float(bytes_read_per_list[cid])

            total_us = query_latency_us(
                cpu_us=avg_ann_us,
                dram_list_ops=dram_ops,
                ssd_page_ops=ssd_page_ops,
                tier=tier,
            )
            latencies[pname].append(total_us)
            ssd_pages[pname].append(ssd_page_ops)
            ssd_bytes[pname].append(ssd_byte)

        # rebalance + migration
        if (qid + 1) % rebalance_interval == 0:
            num_rebalances += 1
            for pname, policy in policies.items():
                ms = policy.rebalance(list_bytes_list, count_eviction=count_eviction)
                migrated_bytes[pname] += float(ms.moved_bytes)
                migrated_clusters[pname] += int(ms.moved_in + ms.moved_out)

                _, t_us = migration_overhead_us(ms.moved_bytes, tier)
                migrated_time_us[pname] += float(t_us)

    # summarize
    out: Dict[str, Dict[str, float]] = {}
    denom_useful = float(Q * k * bytes_per_vec)  # amount of data returned for top-k results

    for pname in policies:
        ls = latencies[pname]
        avg_lat = float(sum(ls) / len(ls))
        p50 = percentile(ls, 50.0)
        p95 = percentile(ls, 95.0)
        p99 = percentile(ls, 99.0)

        qps = 1e6 / avg_lat if avg_lat > 0 else 0.0

        avg_pages = float(sum(ssd_pages[pname]) / len(ssd_pages[pname]))
        avg_b = float(sum(ssd_bytes[pname]) / len(ssd_bytes[pname]))

        total_ssd_bytes = float(sum(ssd_bytes[pname]))
        io_amp = total_ssd_bytes / denom_useful if denom_useful > 0 else 0.0

        total_mig_b = float(migrated_bytes[pname])
        mig_b_per_q = total_mig_b / float(Q)
        avg_mig_clusters = (float(migrated_clusters[pname]) / float(num_rebalances)) if num_rebalances > 0 else 0.0

        total_mig_t = float(migrated_time_us[pname])
        mig_t_per_q = total_mig_t / float(Q)

        out[pname] = {
            "avg_ann_us": float(avg_ann_us),
            "recall_at_k": float(recall_at_k),
            "avg_latency_us": avg_lat,
            "p50_latency_us": p50,
            "p95_latency_us": p95,
            "p99_latency_us": p99,
            "qps_from_avg_latency": qps,
            "avg_ssd_pages": avg_pages,
            "avg_ssd_bytes": avg_b,
            "avg_io_amplification": io_amp,
            "total_migration_bytes": total_mig_b,
            "migration_bytes_per_query": mig_b_per_q,
            "avg_migrated_clusters_per_rebalance": avg_mig_clusters,
            "total_migration_time_us": total_mig_t,
            "migration_time_us_per_query": mig_t_per_q,
        }

    return out
