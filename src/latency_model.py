from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TierModelParams:
    page_size_bytes: int
    dram_access_us_per_list: float
    ssd_base_lat_us_per_page: float
    max_iops: float
    migration_count_eviction: bool = True

    def ssd_service_us_per_page(self) -> float:
        # Key point: let max_iops have a real impact on latency
        # IOPS = ops/sec => average service time per op (seconds) = 1/IOPS => microseconds = 1e6/IOPS
        if self.max_iops <= 0:
            raise ValueError("max_iops must be > 0")
        return 1e6 / float(self.max_iops)

    def ssd_us_per_page(self) -> float:
        return float(self.ssd_base_lat_us_per_page + self.ssd_service_us_per_page())


def query_latency_us(
    cpu_us: float,
    dram_list_ops: int,
    ssd_page_ops: int,
    tier: TierModelParams,
) -> float:
    """
    Total latency (simplified model):
      latency = cpu + DRAM_cost + SSD_cost

    - dram_list_ops: how many of the lists accessed by this query are in DRAM (small overhead per list)
    - ssd_page_ops: how many pages this query needs to read from SSD (one I/O per page)
    """
    dram_cost = float(dram_list_ops) * tier.dram_access_us_per_list
    ssd_cost = float(ssd_page_ops) * tier.ssd_us_per_page()
    return float(cpu_us + dram_cost + ssd_cost)


def migration_overhead_us(bytes_moved: float, tier: TierModelParams) -> tuple[int, float]:
    """
    Make migration cost explicit: bytes migrated => pages => time
    Returns (pages, time_us)
    """
    if bytes_moved <= 0:
        return 0, 0.0
    pages = int(math.ceil(bytes_moved / float(tier.page_size_bytes)))
    time_us = float(pages) * tier.ssd_us_per_page()
    return pages, time_us
