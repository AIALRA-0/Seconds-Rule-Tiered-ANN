import numpy as np
import faiss

def main():
    d = 4      # vector dimension
    nb = 1000  # database size
    nq = 10    # number of queries

    # Create some random vectors
    rng = np.random.default_rng(42)
    xb = rng.normal(size=(nb, d)).astype("float32")
    xq = rng.normal(size=(nq, d)).astype("float32")

    # Build a simple L2 index in memory
    index = faiss.IndexFlatL2(d)
    index.add(xb)

    # Search
    k = 5
    D, I = index.search(xq, k)

    print("Faiss index type:", type(index))
    print("Distances shape:", D.shape)
    print("Indices shape:", I.shape)
    print("First query top-5 neighbors' indices:", I[0])

if __name__ == "__main__":
    main()
