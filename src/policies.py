from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Set


def dram_budget(num_clusters: int, dram_fraction: float) -> int:
    if dram_fraction < 0.0 or dram_fraction > 1.0:
        raise ValueError("dram_fraction must be in [0, 1]")
    return int(math.floor(num_clusters * dram_fraction))


@dataclass
class MigrationStats:
    moved_in: int = 0
    moved_out: int = 0
    moved_bytes: float = 0.0


class TierPolicy:
    """
    Cluster/List level DRAM/SSD placement policy interface.
    """

    name: str = "base"

    def __init__(self, num_clusters: int, dram_fraction: float):
        self.num_clusters = num_clusters
        self.dram_fraction = dram_fraction
        self.budget = dram_budget(num_clusters, dram_fraction)
        self.in_dram: List[bool] = [False] * num_clusters
        self.current_qid: int = -1

    def on_access(self, cluster_id: int, qid: int) -> None:
        self.current_qid = qid

    def rebalance(self, cluster_bytes: List[float], count_eviction: bool = False) -> MigrationStats:
        """
        Base interface: subclasses should implement their own rebalance logic.
        cluster_bytes: size of each cluster/list (used to compute migrated bytes)
        count_eviction: if True, bytes from evictions are also counted into moved_bytes
        """
        # By default, do not perform any migration and return 0 (this implementation is rarely used directly)
        return MigrationStats()

    def is_in_dram(self, cluster_id: int) -> bool:
        return bool(self.in_dram[cluster_id])

    def dram_set(self) -> Set[int]:
        return {i for i, v in enumerate(self.in_dram) if v}

    def _apply_new_dram_set(self, new_set: Set[int], cluster_bytes: List[float], count_eviction: bool) -> MigrationStats:
        old_set = self.dram_set()
        moved_in = new_set - old_set
        moved_out = old_set - new_set

        moved_bytes = 0.0
        for cid in moved_in:
            moved_bytes += float(cluster_bytes[cid])
        if count_eviction:
            for cid in moved_out:
                moved_bytes += float(cluster_bytes[cid])

        self.in_dram = [False] * self.num_clusters
        for cid in new_set:
            self.in_dram[cid] = True

        return MigrationStats(moved_in=len(moved_in), moved_out=len(moved_out), moved_bytes=moved_bytes)


class AllDRAMPolicy(TierPolicy):
    name = "all_dram"

    def __init__(self, num_clusters: int, dram_fraction: float):
        super().__init__(num_clusters, dram_fraction)
        self.in_dram = [True] * num_clusters

    def rebalance(self, cluster_bytes: List[float], count_eviction: bool = False) -> MigrationStats:
        """
        All-DRAM policy: no migration happens; count_eviction is accepted only to keep the interface consistent.
        """
        return MigrationStats(moved_in=0, moved_out=0, moved_bytes=0.0)


class AllSSDPolicy(TierPolicy):
    name = "all_ssd"

    def __init__(self, num_clusters: int, dram_fraction: float):
        super().__init__(num_clusters, dram_fraction)
        # All-SSD: at initialization all clusters are on SSD
        self.in_dram = [False] * num_clusters

    def rebalance(self, cluster_bytes: List[float], count_eviction: bool = False) -> MigrationStats:
        """
        All-SSD policy: likewise no migration (always on SSD); only kept for interface consistency.
        """
        return MigrationStats(moved_in=0, moved_out=0, moved_bytes=0.0)


class NaiveLFUPolicy(TierPolicy):
    """
    Frequency accumulation (LFU-like): select top-K by total access count into DRAM.
    Fix: initialization must satisfy the DRAM budget; disallow “everything in DRAM by default”.
    """

    name = "naive_lfu"

    def __init__(self, num_clusters: int, dram_fraction: float):
        super().__init__(num_clusters, dram_fraction)
        self.access_count = [0] * num_clusters

        # FIX: strictly satisfy the budget at initialization (first budget clusters in DRAM, the rest on SSD)
        for i in range(min(self.budget, num_clusters)):
            self.in_dram[i] = True

    def on_access(self, cluster_id: int, qid: int) -> None:
        super().on_access(cluster_id, qid)
        self.access_count[cluster_id] += 1

    def rebalance(self, cluster_bytes: List[float], count_eviction: bool = True) -> MigrationStats:
        if self.budget <= 0:
            return self._apply_new_dram_set(set(), cluster_bytes, count_eviction)

        idxs = list(range(self.num_clusters))
        idxs.sort(key=lambda i: self.access_count[i], reverse=True)
        new_set = set(idxs[: self.budget])
        return self._apply_new_dram_set(new_set, cluster_bytes, count_eviction)


class LRUPolicy(TierPolicy):
    """
    LRU: select top-K by most recent access time.
    """

    name = "lru"

    def __init__(self, num_clusters: int, dram_fraction: float):
        super().__init__(num_clusters, dram_fraction)
        self.last_access = [-1] * num_clusters

        for i in range(min(self.budget, num_clusters)):
            self.in_dram[i] = True

    def on_access(self, cluster_id: int, qid: int) -> None:
        super().on_access(cluster_id, qid)
        self.last_access[cluster_id] = qid

    def rebalance(self, cluster_bytes: List[float], count_eviction: bool = True) -> MigrationStats:
        if self.budget <= 0:
            return self._apply_new_dram_set(set(), cluster_bytes, count_eviction)

        idxs = list(range(self.num_clusters))
        idxs.sort(key=lambda i: self.last_access[i], reverse=True)  # more recently accessed ones come first
        new_set = set(idxs[: self.budget])
        return self._apply_new_dram_set(new_set, cluster_bytes, count_eviction)


class WindowLFUPolicy(TierPolicy):
    """
    Window-LFU: count access frequency only within the most recent W queries (sliding window).
    Meaning: better adapts to drifting hotspots than naive LFU, but is more robust to noise than LRU.
    """

    name = "window_lfu"

    def __init__(self, num_clusters: int, dram_fraction: float, window_size: int):
        super().__init__(num_clusters, dram_fraction)
        self.window_size = int(window_size)
        if self.window_size <= 0:
            raise ValueError("window_size must be > 0")

        self.window: List[int] = []
        self.counts = [0] * num_clusters

        for i in range(min(self.budget, num_clusters)):
            self.in_dram[i] = True

    def on_access(self, cluster_id: int, qid: int) -> None:
        super().on_access(cluster_id, qid)

        self.window.append(cluster_id)
        self.counts[cluster_id] += 1

        if len(self.window) > self.window_size:
            old = self.window.pop(0)
            self.counts[old] -= 1

    def rebalance(self, cluster_bytes: List[float], count_eviction: bool = True) -> MigrationStats:
        if self.budget <= 0:
            return self._apply_new_dram_set(set(), cluster_bytes, count_eviction)

        idxs = list(range(self.num_clusters))
        idxs.sort(key=lambda i: self.counts[i], reverse=True)
        new_set = set(idxs[: self.budget])
        return self._apply_new_dram_set(new_set, cluster_bytes, count_eviction)


class SecondsRulePolicy(TierPolicy):
    """
    Seconds-rule (interval + idle):
      - avg_interval: EWMA of access interval
      - idle: time elapsed since the last access
      - effective = avg_interval + recency_weight * idle

    Smaller effective means hotter; select top-K at rebalance.

    t_star_queries: break-even interval in seconds mapped to “query intervals”
      - you can use it for interpretation/thresholding (here mainly as a conceptual aid + a debug metric)
    """

    name = "seconds_rule"

    def __init__(
        self,
        num_clusters: int,
        dram_fraction: float,
        t_star_queries: float,
        alpha: float,
        recency_weight: float,
    ):
        super().__init__(num_clusters, dram_fraction)

        self.t_star_queries = float(t_star_queries)
        self.alpha = float(alpha)
        self.recency_weight = float(recency_weight)

        self.last_access = [-1] * num_clusters
        self.avg_interval = [self.t_star_queries] * num_clusters

        for i in range(min(self.budget, num_clusters)):
            self.in_dram[i] = True

    def on_access(self, cluster_id: int, qid: int) -> None:
        super().on_access(cluster_id, qid)

        last = self.last_access[cluster_id]
        if last >= 0:
            interval = qid - last
            old = self.avg_interval[cluster_id]
            self.avg_interval[cluster_id] = self.alpha * float(interval) + (1.0 - self.alpha) * old
        self.last_access[cluster_id] = qid

    def rebalance(self, cluster_bytes: List[float], count_eviction: bool = True) -> MigrationStats:
        if self.budget <= 0:
            return self._apply_new_dram_set(set(), cluster_bytes, count_eviction)

        now = self.current_qid if self.current_qid >= 0 else 0

        scored: List[tuple[float, int]] = []
        for cid in range(self.num_clusters):
            if self.last_access[cid] < 0:
                eff = 1e18
            else:
                idle = now - self.last_access[cid]
                eff = self.avg_interval[cid] + self.recency_weight * float(idle)
            scored.append((eff, cid))

        scored.sort(key=lambda x: x[0])  # smaller effective interval => hotter
        new_set = set(cid for _, cid in scored[: self.budget])
        return self._apply_new_dram_set(new_set, cluster_bytes, count_eviction)


def build_policy(
    name: str,
    num_clusters: int,
    dram_fraction: float,
    *,
    window_size: int,
    seconds_alpha: float,
    seconds_recency_weight: float,
    t_star_queries: float,
):
    if name == "all_dram":
        return AllDRAMPolicy(num_clusters, dram_fraction)
    if name == "all_ssd":
        return AllSSDPolicy(num_clusters, dram_fraction)
    if name == "naive_lfu":
        return NaiveLFUPolicy(num_clusters, dram_fraction)
    if name == "lru":
        return LRUPolicy(num_clusters, dram_fraction)
    if name == "window_lfu":
        return WindowLFUPolicy(num_clusters, dram_fraction, window_size=window_size)
    if name == "seconds_rule":
        return SecondsRulePolicy(
            num_clusters,
            dram_fraction,
            t_star_queries=t_star_queries,
            alpha=seconds_alpha,
            recency_weight=seconds_recency_weight,
        )
    raise ValueError(f"Unknown policy name: {name}")
