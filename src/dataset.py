from __future__ import annotations

import numpy as np


def generate_vectors(
    num_base: int,
    num_queries: int,
    dim: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    xb = rng.normal(size=(num_base, dim)).astype("float32")
    xq = rng.normal(size=(num_queries, dim)).astype("float32")
    return xb, xq
