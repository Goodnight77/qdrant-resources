"""Throughput (QPS) under concurrency per config — not single-query latency.
Fires N queries across CONCURRENCY worker threads, measures queries/second + mean
latency under load. Read-only on the collections. Logs to W&B group 'throughput'.
"""

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from qdrant_client import QdrantClient

from qfo.config import QDRANT_URL, COLLECTIONS, K, results_path
from qfo.data.dbpedia import load_dbpedia
from qfo.telemetry import log_run

CONCURRENCY = 16
N = 2000

# one client per worker thread — a shared client's socket isn't thread-safe (WinError 10038)
_local = threading.local()


def _client():
    if not hasattr(_local, "c"):
        _local.c = QdrantClient(url=QDRANT_URL, timeout=120)
    return _local.c


def main():
    _b, _p, queries = load_dbpedia()
    base = [v.tolist() for v in queries]
    qs = [base[i % len(base)] for i in range(N)]  # cycle to reach N

    results = {}
    for name in COLLECTIONS:
        _client().query_points(name, query=qs[0], limit=K)  # warm

        def one(vec, _name=name):
            t = time.perf_counter()
            _client().query_points(_name, query=vec, limit=K)
            return (time.perf_counter() - t) * 1000.0

        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
            lats = list(ex.map(one, qs))
        elapsed = time.perf_counter() - t0

        m = {
            "qps": N / elapsed,
            "concurrency": CONCURRENCY,
            "n": N,
            "mean_latency_ms": float(np.mean(lats)),
            "p95_latency_ms": float(np.percentile(lats, 95)),
        }
        results[name] = m
        log_run(f"{name}-qps", m, group="throughput")
        print(
            f"{name:9} {m['qps']:.0f} qps  mean {m['mean_latency_ms']:.1f}ms  (conc={CONCURRENCY})"
        )

    with open(results_path("throughput.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("throughput -> results/throughput.json")


if __name__ == "__main__":
    main()
