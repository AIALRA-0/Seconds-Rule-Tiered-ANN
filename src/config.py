from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal

import yaml


MetricType = Literal["l2"]


@dataclass(frozen=True)
class ProjectConfig:
    results_root: str
    run_name: str


@dataclass(frozen=True)
class DataConfig:
    dataset: Literal["sift1m", "synthetic"]
    sift1m_dir: str
    num_base: int
    num_queries: int
    dim: int
    query_shuffle: bool
    query_sample_without_replacement: bool


@dataclass(frozen=True)
class IndexConfig:
    type: Literal["ivf_flat"]
    metric: MetricType
    k: int
    nlist: int
    nprobe_candidates: List[int]
    faiss_num_threads: int
    train_seed: int


@dataclass(frozen=True)
class TierConfig:
    page_size_bytes: int
    dram_access_us_per_list: float
    ssd_base_lat_us_per_page: float
    dram_fraction_list: List[float]
    max_iops_list: List[float]
    migration_count_eviction: bool


@dataclass(frozen=True)
class SecondsRuleConfig:
    alpha: float
    recency_weight: float
    t_star_seconds: float
    assumed_qps_for_tstar: float

    @property
    def t_star_queries(self) -> float:
        return float(self.t_star_seconds * self.assumed_qps_for_tstar)


@dataclass(frozen=True)
class WindowLFUConfig:
    window_size_queries: int


@dataclass(frozen=True)
class PolicyConfig:
    enabled_policies: List[str]
    rebalance_interval: int
    window_lfu: WindowLFUConfig
    seconds_rule: SecondsRuleConfig


@dataclass(frozen=True)
class ExperimentConfig:
    seeds: List[int]
    slas_us: List[float]
    export_metrics: List[str]


@dataclass(frozen=True)
class PlotConfig:
    enable: bool
    errorbar: Literal["std", "ci95"]
    figures: List[str]


@dataclass(frozen=True)
class FullConfig:
    project: ProjectConfig
    data: DataConfig
    index: IndexConfig
    tier: TierConfig
    policy: PolicyConfig
    experiment: ExperimentConfig
    plot: PlotConfig

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "FullConfig":
        project = ProjectConfig(**d["project"])
        data = DataConfig(**d["data"])
        index = IndexConfig(**d["index"])

        tier = TierConfig(**d["tier"])

        seconds_rule = SecondsRuleConfig(**d["policy"]["seconds_rule"])
        window_lfu = WindowLFUConfig(**d["policy"]["window_lfu"])
        policy = PolicyConfig(
            enabled_policies=list(d["policy"]["enabled_policies"]),
            rebalance_interval=int(d["policy"]["rebalance_interval"]),
            window_lfu=window_lfu,
            seconds_rule=seconds_rule,
        )

        experiment = ExperimentConfig(**d["experiment"])
        plot = PlotConfig(**d["plot"])
        return FullConfig(
            project=project,
            data=data,
            index=index,
            tier=tier,
            policy=policy,
            experiment=experiment,
            plot=plot,
        )


def load_config(path: str | Path) -> FullConfig:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    raw = yaml.safe_load(path.read_text())
    cfg = FullConfig.from_dict(raw)

    # minimal validation
    if cfg.index.nlist <= 0:
        raise ValueError("index.nlist must be > 0")
    if any(n <= 0 for n in cfg.index.nprobe_candidates):
        raise ValueError("All nprobe_candidates must be > 0")
    if cfg.index.k <= 0:
        raise ValueError("index.k must be > 0")
    if cfg.data.num_base <= 0 or cfg.data.num_queries <= 0:
        raise ValueError("num_base/num_queries must be > 0")
    if cfg.tier.page_size_bytes not in (4096, 8192, 16384, 32768):
        # allow but warn by raising? no, just allow; user might want weird page sizes
        pass

    return cfg
