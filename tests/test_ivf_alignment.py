import numpy as np
import faiss

from src.ann_engine import compute_query_list_ids


def test_quantizer_list_ids_shape_and_range():
    # tiny synthetic IVF
    d = 8
    nlist = 16
    xb = np.random.randn(2000, d).astype("float32")
    xq = np.random.randn(10, d).astype("float32")

    quantizer = faiss.IndexFlatL2(d)
    index_ivf = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
    index_ivf.train(xb)
    index_ivf.add(xb)

    nprobe = 4
    q_lists = compute_query_list_ids(index_ivf, xq, nprobe=nprobe)

    assert q_lists.shape == (xq.shape[0], nprobe)
    assert q_lists.min() >= 0
    assert q_lists.max() < nlist
