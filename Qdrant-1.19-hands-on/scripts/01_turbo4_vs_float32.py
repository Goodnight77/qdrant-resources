"""
01_turbo4_vs_float32.py

The headline feature of Qdrant 1.19: the `turbo4` DATATYPE.

In 1.18, TurboQuant was a quantization layer: Qdrant kept the original
float32 vectors on disk PLUS a 4-bit compressed copy (36 bits per coordinate
in total). In 1.19, `turbo4` makes the 4-bit representation the ONLY copy:
4 bits per coordinate, ~9x less storage, no full-precision original to
rescore against.

This script creates two collections with the exact same 5000 vectors:
  - `demo_float32`  -> classic full-precision storage
  - `demo_turbo4`   -> datatype="turbo4", one line of difference
then compares recall@10 of each collection vs an exact numpy brute-force search.
"""

import time

import numpy as np
from qdrant_client import models

from common import get_client, make_vectors, brute_force_topk, recall_at_k, DIM

N_QUERIES = 50
K = 10


def create_collection(client, name: str, datatype: models.Datatype):
    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(
        collection_name=name,
        vectors_config=models.VectorParams(
            size=DIM,
            distance=models.Distance.COSINE,
            datatype=datatype,  # <-- the only line that changes
        ),
    )


def upload(client, name: str, vectors: np.ndarray, max_wait_s: float = 60.0):
    client.upload_collection(
        collection_name=name,
        vectors=vectors,
        ids=list(range(len(vectors))),
        batch_size=512,
    )
    # wait until the collection is fully indexed / optimized
    deadline = time.perf_counter() + max_wait_s
    while client.get_collection(name).status != models.CollectionStatus.GREEN:
        if time.perf_counter() > deadline:
            raise TimeoutError(f"'{name}' did not turn GREEN within {max_wait_s:.0f}s")
        time.sleep(0.5)


def measure_recall(client, name: str, vectors: np.ndarray, queries: np.ndarray) -> float:
    recalls = []
    for q in queries:
        truth = brute_force_topk(vectors, q, k=K)
        hits = client.query_points(collection_name=name, query=q.tolist(), limit=K).points
        retrieved = [h.id for h in hits]
        recalls.append(recall_at_k(truth, retrieved))
    return float(np.mean(recalls))


def main():
    client = get_client()
    vectors = make_vectors()
    queries = make_vectors(n=N_QUERIES, seed=123)

    print(f"dataset: {len(vectors)} vectors x {DIM} dims (float32 raw = "
          f"{vectors.nbytes / 1024 / 1024:.1f} MB in numpy)\n")

    results = {}
    for name, dtype in [
        ("demo_float32", models.Datatype.FLOAT32),
        ("demo_turbo4", models.Datatype.TURBO4),
    ]:
        print(f"-> creating '{name}' (datatype={dtype.value}) ...")
        create_collection(client, name, dtype)
        upload(client, name, vectors)

        t0 = time.perf_counter()
        recall = measure_recall(client, name, vectors, queries)
        elapsed = time.perf_counter() - t0

        results[name] = (recall, elapsed)
        print(f"   recall@{K} = {recall:.4f}   ({N_QUERIES} queries in {elapsed:.2f}s)\n")

    # theoretical storage math (what the release blog describes):
    float32_bits = 32
    turbo4_bits = 4
    print("storage per coordinate:")
    print(f"   float32 storage           : {float32_bits} bits")
    print(f"   1.18 TQ quantization      : {float32_bits} + {turbo4_bits} = 36 bits "
          f"(original on disk + 4-bit copy)")
    print(f"   1.19 turbo4 datatype      : {turbo4_bits} bits only  "
          f"-> {36 / turbo4_bits:.0f}x smaller than the two-copy model")

    r32, _ = results["demo_float32"]
    rt4, _ = results["demo_turbo4"]
    print(f"\nrecall cost of dropping the full-precision copy: "
          f"{(r32 - rt4) * 100:.2f} recall points on this dataset")
    print("if that loss is acceptable for your app -> turbo4 gives you ~9x storage back.")
    print("if you need max recall -> keep float32 + TurboQuant *quantization* instead.")
    print("\ncaveat: random gaussian vectors are the WORST case for any quantizer.")
    print("real embedding models produce structured vectors, expect a smaller recall")
    print("gap on your actual data. always benchmark with your own embeddings.")


if __name__ == "__main__":
    main()
