from __future__ import annotations

from pathlib import Path
import numpy as np


def read_fvecs(path: str | Path) -> np.ndarray:
    path = Path(path)
    data = np.fromfile(path, dtype="int32")
    if data.size == 0:
        raise ValueError(f"Empty fvecs file: {path}")

    d = int(data[0])
    if d <= 0:
        raise ValueError(f"Invalid dimension {d} in {path}")

    data = data.reshape(-1, d + 1)
    vecs = data[:, 1:].view("float32")
    return np.ascontiguousarray(vecs)


def load_sift1m(base_dir: str | Path) -> tuple[np.ndarray, np.ndarray]:
    base_dir = Path(base_dir)
    base_path = base_dir / "sift_base.fvecs"
    query_path = base_dir / "sift_query.fvecs"

    if not base_path.exists() or not query_path.exists():
        raise FileNotFoundError(
            f"Cannot find sift_base.fvecs or sift_query.fvecs under {base_dir}"
        )

    xb = read_fvecs(base_path)
    xq = read_fvecs(query_path)
    return xb, xq
