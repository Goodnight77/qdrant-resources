"""Evaluation suite. Query each collection, compute Recall@K vs exact-NN ground truth,
log per-config metrics to W&B, dump results.json

One pass per collection captures BOTH retrieved ids (recall) and latency (no double query).
"""
import json
import time

import numpy as np
from qdrant_client import QdrantClient

from qfo.config import QDRANT_URL, COLLECTIONS, K, results_path
from qfo.data.dbpedia import load_dbpedia
from qfo.data.ground_truth import GT_PATH
from qfo.telemetry import container_rss_mb, pctiles, log_run


def search_collect(client, name, queries, k):
    """Returns (ids[N,k] int64, latencies_ms[N]). Missing slots = -1."""
    ids = np.full((len(queries), k), -1, dtype=np.int64)
    lats = np.empty(len(queries), dtype=np.float64)
    for i, q in enumerate(queries):
        t = time.perf_counter()
        res = client.query_points(collection_name=name, query=q.tolist(), limit=k)
        lats[i] = (time.perf_counter() - t) * 1000.0
        got = [p.id for p in res.points]
        ids[i, : len(got)] = got
    return ids, lats


def recall_at_k(retrieved, gt, k):
    """mean |retrieved_k ∩ gt_k| / k. gt holds top-GT_DEPTH; compare against gt[:, :k]."""
    hits = 0
    for r, g in zip(retrieved, gt):
        hits += len(set(r.tolist()) & set(g[:k].tolist()))
    return hits / (len(retrieved) * k)


def main(n_queries=None):
    _base, _payload, queries = load_dbpedia()
    with open(GT_PATH) as f:
        gt = np.array(json.load(f), dtype=np.int64)
    if n_queries:
        queries, gt = queries[:n_queries], gt[:n_queries]

    client = QdrantClient(url=QDRANT_URL, timeout=120)
    results = {}
    for name in COLLECTIONS:
        search_collect(client, name, queries[:200], K)        # warm cache, discard
        ids, lats = search_collect(client, name, queries, K)  # measured pass
        m = {
            "recall_at_k": recall_at_k(ids, gt, K),
            "k": K,
            "rss_mb": container_rss_mb(),
            "n_queries": len(queries),
            **pctiles(lats),
        }
        results[name] = m
        log_run(name, m, group="faceoff")
        print(name, m)

    with open(results_path("results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("eval done -> results/results.json")


if __name__ == "__main__":
    # money-path check
    assert abs(recall_at_k(np.array([[1, 2, 3]]), np.array([[1, 2, 9, 8]]), 3) - 2 / 3) < 1e-9
    main()
