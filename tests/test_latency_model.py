from src.latency_model import TierModelParams, query_latency_us


def test_iops_affects_latency():
    tier_slow = TierModelParams(
        page_size_bytes=4096,
        dram_access_us_per_list=0.2,
        ssd_base_lat_us_per_page=20.0,
        max_iops=1_000_000.0,
        migration_count_eviction=True,
    )
    tier_fast = TierModelParams(
        page_size_bytes=4096,
        dram_access_us_per_list=0.2,
        ssd_base_lat_us_per_page=20.0,
        max_iops=10_000_000.0,
        migration_count_eviction=True,
    )

    cpu = 10.0
    dram_ops = 0
    ssd_pages = 10

    lat_slow = query_latency_us(cpu, dram_ops, ssd_pages, tier_slow)
    lat_fast = query_latency_us(cpu, dram_ops, ssd_pages, tier_fast)

    assert lat_fast < lat_slow, "Higher IOPS should reduce SSD service time and total latency"
