from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .config import FullConfig
from .latency_model import TierModelParams, migration_overhead_us, query_latency_us
from .policies import build_policy
from .utils import percentile
from .workload import make_workload


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
    # Here it is still interpreted as “the access results of the unique queries”, 
    # and the request sequence is later generated using workload
    query_list_ids: np.ndarray,   # (Q_unique, nprobe)
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

    Now supports cfg.experiment.workload:
        - experiment.workload: "uniform" / "zipf" / "hotspot_shift"
        - experiment.workload_zipf_s
        - experiment.workload_hot_frac
        - experiment.workload_hot_prob
        - experiment.workload_shift_interval
        - experiment.workload_num_requests
    """

    # ------------------------------------------------
    # 1) Basic info & workload configuration
    # ------------------------------------------------
    base_Q, nprobe = query_list_ids.shape
    k = cfg.index.k

    exp_cfg = cfg.experiment

    workload_kind = getattr(exp_cfg, "workload", "uniform").lower()
    zipf_s = getattr(exp_cfg, "workload_zipf_s", 1.2)
    hot_frac = getattr(exp_cfg, "workload_hot_frac", 0.05)
    hot_prob = getattr(exp_cfg, "workload_hot_prob", 0.8)
    shift_interval = getattr(exp_cfg, "workload_shift_interval", 10_000)
    workload_num_requests = getattr(exp_cfg, "workload_num_requests", None)

    # Generate request sequence: each element is a query index in [0, base_Q)
    if workload_kind in ("uniform", "random", "iid"):
        # uniform: default behavior same as older version, each query used once
        if isinstance(workload_num_requests, int) and workload_num_requests > 0:
            # If workload_num_requests is specified, simply reuse with round-robin
            qids = np.arange(base_Q, dtype=np.int64)
            if workload_num_requests <= base_Q:
                request_query_ids = qids[:workload_num_requests]
            else:
                # np.resize will cycle-fill
                request_query_ids = np.resize(qids, workload_num_requests)
        else:
            # If not specified, each query once
            request_query_ids = np.arange(base_Q, dtype=np.int64)
    else:
        # Non-uniform: use Zipf / HotspotShiftWorkload to generate
        num_requests = (
            workload_num_requests
            if isinstance(workload_num_requests, int) and workload_num_requests > 0
            else base_Q
        )

        rng_seed = exp_cfg.seeds[0] if getattr(exp_cfg, "seeds", None) else 0
        rng = np.random.default_rng(rng_seed)

        workload = make_workload(
            kind=workload_kind,
            zipf_s=zipf_s,
            hot_frac=hot_frac,
            hot_prob=hot_prob,
            shift_interval=shift_interval,
        )
        request_query_ids = workload.generate(
            num_requests=num_requests,
            num_queries=base_Q,
            rng=rng,
        )

    # Actual number of requests participating in the policy simulation
    Q = int(len(request_query_ids))
    progress_step = max(Q // 10, 1)

    # ------------------------------------------------
    # 2) Pre-compute page / bytes for each IVF list
    # ------------------------------------------------
    pages_per_list = _pages_for_list_bytes(list_bytes, tier.page_size_bytes)
    bytes_read_per_list = pages_per_list.astype(np.float64) * float(tier.page_size_bytes)
    list_bytes_list = list(list_bytes.astype(float))

    # ------------------------------------------------
    # 3) Build policy instances
    # ------------------------------------------------
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

    # ------------------------------------------------
    # 4) Main loop: send queries according to workload order
    #    t = “time step”, Seconds-Rule uses t for temporal logic
    # ------------------------------------------------
    for t in range(Q):
        # Progress hint (print every 1/10 progress)
        if (t + 1) % progress_step == 0 or (t + 1) == Q:
            print(
                f"[run_policies] progress {t+1}/{Q} "
                f"({(t+1)/Q*100:.1f}%)"
            )
        qid = int(request_query_ids[t])   # The query index for this request
        lists = query_list_ids[qid]

        # (Optional) de-duplicate: nprobe is small, so set is enough
        unique_lists: List[int] = []
        seen = set()
        for cid in lists:
            c = int(cid)
            if c not in seen:
                seen.add(c)
                unique_lists.append(c)

        for pname, policy in policies.items():
            # Update access statistics: use time step t as "current time"
            for cid in unique_lists:
                policy.on_access(cid, t)

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

        # Periodic rebalance + record migrations
        if (t + 1) % rebalance_interval == 0:
            num_rebalances += 1
            for pname, policy in policies.items():
                ms = policy.rebalance(list_bytes_list, count_eviction=count_eviction)
                migrated_bytes[pname] += float(ms.moved_bytes)
                migrated_clusters[pname] += int(ms.moved_in + ms.moved_out)

                _, t_us = migration_overhead_us(ms.moved_bytes, tier)
                migrated_time_us[pname] += float(t_us)

    # ------------------------------------------------
    # 5) Aggregate metrics
    # ------------------------------------------------
    out: Dict[str, Dict[str, float]] = {}
    # This round has Q requests, each returns k vectors, each vector has bytes_per_vec bytes
    denom_useful = float(Q * k * bytes_per_vec)

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
        avg_mig_clusters = (
            float(migrated_clusters[pname]) / float(num_rebalances)
            if num_rebalances > 0 else 0.0
        )

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
