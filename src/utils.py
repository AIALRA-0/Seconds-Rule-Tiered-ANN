from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np


def now_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def percentile(values: List[float], p: float) -> float:
    """Simple percentile with linear interpolation."""
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(s[int(k)])
    d0 = s[f] * (c - k)
    d1 = s[c] * (k - f)
    return float(d0 + d1)


@dataclass(frozen=True)
class SummaryStats:
    mean: float
    std: float
    ci95: float  # half-width

    @staticmethod
    def from_values(values: List[float]) -> "SummaryStats":
        if not values:
            return SummaryStats(mean=0.0, std=0.0, ci95=0.0)
        arr = np.asarray(values, dtype=np.float64)
        mean = float(arr.mean())
        std = float(arr.std(ddof=1)) if len(arr) >= 2 else 0.0
        ci95 = float(1.96 * std / math.sqrt(len(arr))) if len(arr) >= 2 else 0.0
        return SummaryStats(mean=mean, std=std, ci95=ci95)


def human_bytes(n: float) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(n)
    for u in units:
        if x < 1024.0:
            return f"{x:.2f}{u}"
        x /= 1024.0
    return f"{x:.2f}PB"
