from __future__ import annotations

from pathlib import Path
from typing import Literal, Iterable, Optional, List, Protocol
from dataclasses import dataclass

import numpy as np


class Workload(Protocol):
    """Interface for generating a query sequence.

    The return value is an ndarray of length num_requests, where each element is
    an integer in [0, num_queries), representing the query_id to issue.
    """

    def generate(self, num_requests: int, num_queries: int, rng: np.random.Generator) -> np.ndarray:
        ...


# -----------------------------
# Specific workload types
# -----------------------------

@dataclass
class UniformRandomWorkload:
    """Completely uniform random access, used as a baseline."""

    def generate(self, num_requests: int, num_queries: int, rng: np.random.Generator) -> np.ndarray:
        return rng.integers(low=0, high=num_queries, size=num_requests, endpoint=False)


@dataclass
class ZipfWorkload:
    """Zipf-distributed access: a fixed “hot” set but with a long tail over time.

    Args:
        s: Zipf parameter (>1 means more skew toward the head)
    """

    s: float = 1.2

    def generate(self, num_requests: int, num_queries: int, rng: np.random.Generator) -> np.ndarray:
        # numpy's zipf() returns values starting from 1 with theoretically no upper bound.
        # Here we wrap by num_queries.
        raw = rng.zipf(a=self.s, size=num_requests)
        idx = (raw - 1) % num_queries
        return idx.astype(np.int64)


@dataclass
class HotspotShiftWorkload:
    """A workload with temporal locality and shifting “hot spots”.

    Concept:
    - For a period, there is a hot set occupying hot_frac of all queries (e.g., 5%).
    - During this period, each request falls into the hot set with probability hot_prob,
      otherwise it goes to cold data.
    - Every shift_interval requests, re-sample a new hot set,
      simulating a “moving hot window” over time.
    """

    hot_frac: float = 0.05       # Proportion of queries belonging to the hot set
    hot_prob: float = 0.8        # Probability that a request hits the hot set
    shift_interval: int = 10_000 # How many requests before rotating the hot set

    def generate(self, num_requests: int, num_queries: int, rng: np.random.Generator) -> np.ndarray:
        if num_queries <= 0:
            raise ValueError("num_queries must be > 0")

        num_hot = max(1, int(num_queries * self.hot_frac))
        indices = np.empty(num_requests, dtype=np.int64)

        # Initialize the first hot set
        hot_set = rng.choice(num_queries, size=num_hot, replace=False)

        for t in range(num_requests):
            # Periodically rotate the hot set
            if t > 0 and (t % self.shift_interval) == 0:
                hot_set = rng.choice(num_queries, size=num_hot, replace=False)

            if rng.random() < self.hot_prob:
                indices[t] = rng.choice(hot_set)
            else:
                indices[t] = rng.integers(low=0, high=num_queries)

        return indices


# -----------------------------
# Factory function: given a string, return a workload instance
# -----------------------------

def make_workload(
    kind: str,
    *,
    zipf_s: float = 1.2,
    hot_frac: float = 0.05,
    hot_prob: float = 0.8,
    shift_interval: int = 10_000,
) -> Workload:
    kind = kind.lower()
    if kind in ("uniform", "random", "iid"):
        return UniformRandomWorkload()
    if kind in ("zipf", "zipfian"):
        return ZipfWorkload(s=zipf_s)
    if kind in ("hotspot", "hotspot_shift", "temporal"):
        return HotspotShiftWorkload(
            hot_frac=hot_frac,
            hot_prob=hot_prob,
            shift_interval=shift_interval,
        )
    raise ValueError(f"Unknown workload kind: {kind}")
