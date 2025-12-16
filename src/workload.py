from __future__ import annotations

import random
from typing import List

import numpy as np


def generate_uniform_clusters(num_queries: int, num_clusters: int, seed: int) -> np.ndarray:
    rng = random.Random(seed)
    return np.array([rng.randrange(num_clusters) for _ in range(num_queries)], dtype=np.int64)


def generate_phased_hotspot_clusters(
    num_queries: int,
    num_clusters: int,
    num_phases: int,
    hot_fraction: float,
    hot_prob: float,
    seed: int,
) -> np.ndarray:
    """
    Split the query sequence into phases; each phase has different hotspot clusters (hotspot drift).
    """
    rng = random.Random(seed)
    phase_len = max(1, num_queries // num_phases)
    seq: List[int] = []

    all_ids = list(range(num_clusters))
    for _ in range(num_phases):
        hot_size = max(1, int(num_clusters * hot_fraction))
        hot = rng.sample(all_ids, hot_size)
        cold = [c for c in all_ids if c not in hot] or hot

        for _ in range(phase_len):
            if len(seq) >= num_queries:
                break
            if rng.random() < hot_prob:
                seq.append(rng.choice(hot))
            else:
                seq.append(rng.choice(cold))

    while len(seq) < num_queries:
        seq.append(seq[-1])
    return np.array(seq, dtype=np.int64)
