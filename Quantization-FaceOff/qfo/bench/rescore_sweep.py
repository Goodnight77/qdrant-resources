"""Rescoring / oversampling sweep on the PQ collection.

PQ search on quantized codes is fast but lossy. Qdrant can oversample (pull N x more
candidates with the cheap quantized vectors) then rescore them with the full-precision
originals on disk. This sweep shows PQ recall climbing back toward baseline as we
oversample — the production lever that makes 16x compression usable.

Read-only on the PQ collection. No drops.
"""

import json
import sys
import time

import numpy as np
from qdrant_client import QdrantClient, models

from qfo.config import QDRANT_URL, K, results_path
from qfo.data.dbpedia import load_dbpedia
from qfo.data.ground_truth import GT_PATH
from qfo.bench.eval import recall_at_k
from qfo.telemetry import pctiles, log_run

COLL = sys.argv[1] if len(sys.argv) > 1 else "pq"

# (label, rescore, oversampling)
SWEEP = [
    ("no rescore", False, 1.0),
    ("rescore 1x", True, 1.0),
    ("rescore 2x", True, 2.0),
    ("rescore 4x", True, 4.0),
    ("rescore 8x", True, 8.0),
]


def run_one(client, queries, k, rescore, oversampling):
    params = models.SearchParams(
        quantization=models.QuantizationSearchParams(
            rescore=rescore, oversampling=oversampling
        )
    )
    ids = np.full((len(queries), k), -1, dtype=np.int64)
    lats = np.empty(len(queries))
    for i, q in enumerate(queries):
        t = time.perf_counter()
        res = client.query_points(COLL, query=q.tolist(), limit=k, search_params=params)
        lats[i] = (time.perf_counter() - t) * 1000.0
        got = [p.id for p in res.points]
        ids[i, : len(got)] = got
    return ids, lats


def main():
    _b, _p, queries = load_dbpedia()
    with open(GT_PATH) as f:
        gt = np.array(json.load(f), dtype=np.int64)
    client = QdrantClient(url=QDRANT_URL, timeout=120)

    rows = []
    for label, rescore, oversampling in SWEEP:
        ids, lats = run_one(client, queries, K, rescore, oversampling)
        m = {
            "label": label,
            "rescore": rescore,
            "oversampling": oversampling,
            "recall_at_k": recall_at_k(ids, gt, K),
            **pctiles(lats),
        }
        rows.append(m)
        print(
            f"{label:12} recall={m['recall_at_k']:.3f}  p50={m['latency_p50_ms']:.1f}ms"
        )
        log_run(f"{COLL}-{label.replace(' ', '_')}", m, group=f"rescore-{COLL}")

    out = results_path(f"rescore_sweep_{COLL}.json")
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"rescore sweep done -> {out}")
    return rows


if __name__ == "__main__":
    main()
