from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Tuple

import faiss
import numpy as np

from .config import FullConfig
from .dataset import generate_vectors
from .dataset_sift1m import load_sift1m
from .groundtruth import compute_groundtruth_I, recall_at_k
from .utils import set_global_seed


@dataclass
class AnnContext:
    xb: np.ndarray
    xq: np.ndarray
    gt_I: np.ndarray
    index_ivf: faiss.IndexIVFFlat
    dim: int
    bytes_per_vec: int
    list_sizes: np.ndarray  # (nlist,)
    list_bytes: np.ndarray  # (nlist,)


def _set_faiss_threads(num_threads: int) -> None:
    try:
        faiss.omp_set_num_threads(int(num_threads))
    except Exception:
        pass


def load_dataset_from_cfg(cfg: FullConfig, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Key to introducing variance across multiple seeds: sample/shuffle queries
    """
    set_global_seed(seed)

    if cfg.data.dataset == "synthetic":
        xb, xq = generate_vectors(
            num_base=cfg.data.num_base,
            num_queries=cfg.data.num_queries,
            dim=cfg.data.dim,
            seed=seed,
        )
        return xb, xq

    xb_full, xq_full = load_sift1m(cfg.data.sift1m_dir)

    # base: sample or take prefix
    if cfg.data.query_sample_without_replacement:
        rng = np.random.default_rng(seed)
        base_idx = rng.choice(len(xb_full), size=cfg.data.num_base, replace=False)
        query_idx = rng.choice(len(xq_full), size=cfg.data.num_queries, replace=False)
        xb = xb_full[base_idx]
        xq = xq_full[query_idx]
    else:
        xb = xb_full[: cfg.data.num_base]
        xq = xq_full[: cfg.data.num_queries]

    if cfg.data.query_shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(xq, axis=0)

    xb = np.ascontiguousarray(xb.astype("float32"))
    xq = np.ascontiguousarray(xq.astype("float32"))
    return xb, xq


def build_ivf_flat(cfg: FullConfig, xb: np.ndarray) -> faiss.IndexIVFFlat:
    _set_faiss_threads(cfg.index.faiss_num_threads)

    dim = xb.shape[1]
    quantizer = faiss.IndexFlatL2(dim)
    index_ivf = faiss.IndexIVFFlat(quantizer, dim, cfg.index.nlist, faiss.METRIC_L2)

    # train + add
    index_ivf.train(xb)
    index_ivf.add(xb)
    return index_ivf


def compute_list_sizes_from_quantizer(index_ivf: faiss.IndexIVFFlat, xb: np.ndarray, nlist: int) -> np.ndarray:
    """
    Key point: list sizes must come from IVF quantizer assignments, not from a separate k-means run.
    """
    _, I = index_ivf.quantizer.search(xb, 1)
    list_ids = I.reshape(-1).astype(np.int64)
    sizes = np.bincount(list_ids, minlength=nlist).astype(np.int64)
    return sizes


def compute_query_list_ids(index_ivf: faiss.IndexIVFFlat, xq: np.ndarray, nprobe: int) -> np.ndarray:
    """
    Key fix: the lists accessed by each query must be obtained via IVF quantizer.search to get list IDs
    """
    _, I = index_ivf.quantizer.search(xq, int(nprobe))
    return I.astype(np.int64)


def build_context(cfg: FullConfig, seed: int) -> AnnContext:
    xb, xq = load_dataset_from_cfg(cfg, seed)
    dim = xb.shape[1]
    bytes_per_vec = dim * 4  # float32

    gt_I = compute_groundtruth_I(xb, xq, k=cfg.index.k)

    index_ivf = build_ivf_flat(cfg, xb)
    list_sizes = compute_list_sizes_from_quantizer(index_ivf, xb, nlist=cfg.index.nlist)
    list_bytes = (list_sizes.astype(np.float64) * float(bytes_per_vec)).astype(np.float64)

    return AnnContext(
        xb=xb,
        xq=xq,
        gt_I=gt_I,
        index_ivf=index_ivf,
        dim=dim,
        bytes_per_vec=bytes_per_vec,
        list_sizes=list_sizes,
        list_bytes=list_bytes,
    )


def eval_ivf_for_nprobe(ctx: AnnContext, cfg: FullConfig, nprobe: int) -> tuple[float, float, np.ndarray]:
    """
    Returns:
      - avg_ann_us
      - recall@k
      - query_list_ids (Q, nprobe)
    """
    ctx.index_ivf.nprobe = int(nprobe)

    t0 = time.perf_counter()
    _, I_approx = ctx.index_ivf.search(ctx.xq, cfg.index.k)
    t1 = time.perf_counter()

    total_us = (t1 - t0) * 1e6
    avg_ann_us = float(total_us / ctx.xq.shape[0])

    rec = recall_at_k(ctx.gt_I, I_approx, k=cfg.index.k)
    q_lists = compute_query_list_ids(ctx.index_ivf, ctx.xq, nprobe=nprobe)
    return avg_ann_us, float(rec), q_lists
