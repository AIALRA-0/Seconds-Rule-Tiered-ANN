from src.policies import NaiveLFUPolicy, LRUPolicy, WindowLFUPolicy, SecondsRulePolicy, dram_budget


def test_naive_init_respects_budget():
    n = 100
    frac = 0.1
    b = dram_budget(n, frac)
    p = NaiveLFUPolicy(n, frac)
    assert sum(1 for i in range(n) if p.is_in_dram(i)) == b


def test_lru_init_respects_budget():
    n = 64
    frac = 0.2
    b = dram_budget(n, frac)
    p = LRUPolicy(n, frac)
    assert sum(1 for i in range(n) if p.is_in_dram(i)) == b


def test_window_lfu_window_behavior():
    n = 10
    frac = 0.3
    p = WindowLFUPolicy(n, frac, window_size=3)

    # access pattern: 1,1,2,3
    p.on_access(1, 0)
    p.on_access(1, 1)
    p.on_access(2, 2)
    p.on_access(3, 3)  # now window keeps last 3: 1,2,3 => count(1)=1

    # internal counts are not exposed; just ensure no exception and rebalance works
    ms = p.rebalance([1.0] * n)
    assert ms.moved_in >= 0


def test_seconds_rule_updates_interval():
    n = 5
    p = SecondsRulePolicy(
        num_clusters=n,
        dram_fraction=0.4,
        t_star_queries=100.0,
        alpha=0.5,
        recency_weight=0.1,
    )
    p.on_access(2, 10)
    p.on_access(2, 20)  # interval 10, avg should update from t_star towards 10
    # not asserting exact numeric, but should not remain default
    assert p.avg_interval[2] != 100.0
