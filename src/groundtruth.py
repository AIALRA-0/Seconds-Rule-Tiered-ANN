from __future__ import annotations

import faiss
import numpy as np


def compute_groundtruth_I(xb: np.ndarray, xq: np.ndarray, k: int) -> np.ndarray:
    xb = np.ascontiguousarray(xb.astype("float32"))
    xq = np.ascontiguousarray(xq.astype("float32"))
    d = xb.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(xb)
    _, I = index.search(xq, k)
    return I


def recall_at_k(gt_I: np.ndarray, approx_I: np.ndarray, k: int) -> float:
    assert gt_I.shape == approx_I.shape
    Q = gt_I.shape[0]
    total = 0.0
    for i in range(Q):
        gt = set(int(x) for x in gt_I[i])
        hit = sum(1 for x in approx_I[i] if int(x) in gt)
        total += hit / float(k)
    return total / float(Q)
