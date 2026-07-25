"""
ACORN vs vanilla filtered HNSW in Qdrant 1.16
Sweeps filter selectivity from ~0.1% to 90% and measures recall@10 + latency.

Design:
- 200,000 random vectors, dim=96, cosine (same dim as deep-image-96)
- single segment, so results are not blurred by per-segment planning
- full_scan_threshold set very low, so the planner always uses the graph
  (we want to see the raw graph behavior, not the brute-force safety net)
- selectivity levels built from combined keyword filters:
  two independent uniform fields with V values each -> combined selectivity 1/V^2
- ground truth computed with exact=True on the same filter
"""

import time

import numpy as np
from qdrant_client import QdrantClient, models

rng = np.random.default_rng(42)

N = 200_000
DIM = 96
K = 10
N_QUERIES = 50
COLL = "acorn_bench"

client = QdrantClient(url="http://localhost:6333", timeout=600)

# selectivity levels: (label, fields spec)
# combined: two fields with V values each -> 1/V^2
# single:  one field with given probability of the target value
LEVELS = [
    ("0.1%", ("combined", 32)),  # 1/1024 ~ 0.098%
    ("1%", ("combined", 10)),  # 1/100
    ("4%", ("combined", 5)),  # 1/25
    ("11%", ("combined", 3)),  # 1/9
    ("25%", ("combined", 2)),  # 1/4
    ("50%", ("single", 0.50)),
    ("90%", ("single", 0.90)),
]


def build_payloads():
    """One payload dict per point, with all fields for all levels."""
    payload_fields = {}
    for label, (kind, v) in LEVELS:
        if kind == "combined":
            payload_fields[f"a{v}"] = rng.integers(0, v, N)
            payload_fields[f"b{v}"] = rng.integers(0, v, N)
        else:
            p = v
            payload_fields[f"s{int(p*100)}"] = (rng.random(N) < p).astype(int)
    return payload_fields


N_CLUSTERS = 1024
CENTERS = rng.standard_normal((N_CLUSTERS, DIM)).astype(np.float32)


def make_vectors():
    """Clustered data: a mixture of Gaussians, like real embeddings."""
    assign = rng.integers(0, N_CLUSTERS, N)
    v = CENTERS[assign] + 0.35 * rng.standard_normal((N, DIM)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v.astype(np.float32)


def make_queries(n):
    """Queries near real clusters, like real user queries."""
    assign = rng.integers(0, N_CLUSTERS, n)
    q = CENTERS[assign] + 0.35 * rng.standard_normal((n, DIM)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return q.astype(np.float32)


def make_filter(level):
    _label, (kind, v) = level
    if kind == "combined":
        return models.Filter(
            must=[
                models.FieldCondition(key=f"a{v}", match=models.MatchValue(value=0)),
                models.FieldCondition(key=f"b{v}", match=models.MatchValue(value=0)),
            ]
        )
    else:
        return models.Filter(
            must=[
                models.FieldCondition(
                    key=f"s{int(v*100)}", match=models.MatchValue(value=1)
                ),
            ]
        )


def setup():
    if client.collection_exists(COLL):
        client.delete_collection(COLL)
    client.create_collection(
        collection_name=COLL,
        vectors_config=models.VectorParams(size=DIM, distance=models.Distance.COSINE),
        hnsw_config=models.HnswConfigDiff(
            m=16,
            ef_construct=100,
            full_scan_threshold=10,  # force graph traversal, no brute-force fallback
        ),
        optimizers_config=models.OptimizersConfigDiff(
            default_segment_number=1,
            max_segment_size=800_000,
        ),
    )

    payload_fields = build_payloads()
    for field in payload_fields:
        client.create_payload_index(
            collection_name=COLL,
            field_name=field,
            field_schema=models.PayloadSchemaType.INTEGER,
        )

    print("uploading vectors ...")
    vectors = make_vectors()
    payloads = (
        {f: int(payload_fields[f][i]) for f in payload_fields} for i in range(N)
    )
    client.upload_collection(
        collection_name=COLL,
        vectors=vectors,
        payload=payloads,
        ids=range(N),
        batch_size=2048,
    )

    print("waiting for indexing ...")
    while True:
        info = client.get_collection(COLL)
        if info.status == models.CollectionStatus.GREEN:
            break
        time.sleep(3)
    print("indexed. segments ready.")


def run_queries(queries, flt, search_params, limit=K):
    results, times = [], []
    for q in queries:
        t0 = time.perf_counter()
        res = client.query_points(
            collection_name=COLL,
            query=q.tolist(),
            query_filter=flt,
            search_params=search_params,
            limit=limit,
        )
        times.append((time.perf_counter() - t0) * 1000)
        results.append([p.id for p in res.points])
    return results, times


def recall(approx_ids, exact_ids):
    vals = []
    for a, e in zip(approx_ids, exact_ids):
        if not e:
            continue
        vals.append(len(set(a) & set(e)) / len(e))
    return vals


def main():
    setup()
    queries = make_queries(N_QUERIES)

    # unfiltered baseline: how good is HNSW on this data with no filter at all
    exact_ids, _ = run_queries(queries, None, models.SearchParams(exact=True))
    b_ids, b_t = run_queries(queries, None, models.SearchParams(hnsw_ef=64))
    base_recall = float(np.mean(recall(b_ids, exact_ids)))
    base_ms = float(np.median(b_t))
    print(f"unfiltered baseline: recall={base_recall:.3f} lat={base_ms:.1f}ms")

    rows = []
    for level in LEVELS:
        label = level[0]
        flt = make_filter(level)
        est = client.count(COLL, count_filter=flt, exact=True).count
        sel = est / N

        # ground truth
        exact_ids, _ = run_queries(queries, flt, models.SearchParams(exact=True))

        # warmup
        run_queries(queries[:5], flt, models.SearchParams(hnsw_ef=64))

        # vanilla filtered HNSW
        v_ids, v_t = run_queries(queries, flt, models.SearchParams(hnsw_ef=64))
        v_rec = recall(v_ids, exact_ids)

        # ACORN, forced on regardless of estimated selectivity
        a_params = models.SearchParams(
            hnsw_ef=64,
            acorn=models.AcornSearchParams(enable=True, max_selectivity=1.0),
        )
        run_queries(queries[:5], flt, a_params)
        a_ids, a_t = run_queries(queries, flt, a_params)
        a_rec = recall(a_ids, exact_ids)

        # ACORN with default threshold: only activates below 40% selectivity
        d_params = models.SearchParams(
            hnsw_ef=64, acorn=models.AcornSearchParams(enable=True)
        )
        d_ids, d_t = run_queries(queries, flt, d_params)
        d_rec = recall(d_ids, exact_ids)

        row = {
            "label": label,
            "selectivity": sel,
            "vanilla_recall": float(np.mean(v_rec)),
            "acorn_recall": float(np.mean(a_rec)),
            "vanilla_recall_p10": float(np.percentile(v_rec, 10)),
            "acorn_recall_p10": float(np.percentile(a_rec, 10)),
            "vanilla_ms": float(np.median(v_t)),
            "acorn_ms": float(np.median(a_t)),
            "vanilla_ms_p95": float(np.percentile(v_t, 95)),
            "acorn_ms_p95": float(np.percentile(a_t, 95)),
            "default_recall": float(np.mean(d_rec)),
            "default_ms": float(np.median(d_t)),
        }
        rows.append(row)
        print(
            f"{label:>5}  sel={sel:.4f}  "
            f"recall v={row['vanilla_recall']:.3f} a={row['acorn_recall']:.3f}  "
            f"lat v={row['vanilla_ms']:.1f}ms a={row['acorn_ms']:.1f}ms"
        )

    # ef sweep at the ~1% level: the classic recall/latency tradeoff curve
    level = LEVELS[1]
    flt = make_filter(level)
    exact_ids, _ = run_queries(queries, flt, models.SearchParams(exact=True))
    sweep = []
    for ef in [16, 32, 64, 128, 256, 512]:
        v_ids, v_t = run_queries(queries, flt, models.SearchParams(hnsw_ef=ef))
        a_params = models.SearchParams(
            hnsw_ef=ef, acorn=models.AcornSearchParams(enable=True, max_selectivity=1.0)
        )
        a_ids, a_t = run_queries(queries, flt, a_params)
        sweep.append(
            {
                "ef": ef,
                "vanilla_recall": float(np.mean(recall(v_ids, exact_ids))),
                "acorn_recall": float(np.mean(recall(a_ids, exact_ids))),
                "vanilla_ms": float(np.median(v_t)),
                "acorn_ms": float(np.median(a_t)),
            }
        )
        print(
            f"ef={ef:>3}  v: r={sweep[-1]['vanilla_recall']:.3f} {sweep[-1]['vanilla_ms']:.1f}ms | "
            f"a: r={sweep[-1]['acorn_recall']:.3f} {sweep[-1]['acorn_ms']:.1f}ms"
        )

    import json

    with open("results.json", "w") as f:
        json.dump(
            {
                "rows": rows,
                "sweep": sweep,
                "n": N,
                "dim": DIM,
                "k": K,
                "base_recall": base_recall,
                "base_ms": base_ms,
            },
            f,
            indent=2,
        )
    print("saved results.json")


if __name__ == "__main__":
    main()
