"""Experiment 2: tiered multitenancy for a whale + long-tail distribution.

Real multi-tenant apps are rarely made of equal-sized tenants: a handful of
big accounts hold most of the data, and a long tail of small tenants share
the rest. Qdrant's tiered multitenancy (1.16+) handles this by giving large
tenants their own dedicated shard and letting small tenants share a
"fallback" shard together, instead of everyone (whale included) sharing
one collection-wide index.

We reproduce the two shapes with hnswlib graphs, same as experiment 1:

  "naive"  = one shared HNSW graph across the whale and every minnow tenant,
             each queried via a tenant filter. This is what you get if you
             dump every tenant into one collection with no partitioning.
  "tiered" = the whale gets its own dedicated graph; all the minnows share a
             second graph sized to just their own data (the "fallback
             shard"), decoupled from the whale entirely.

The story here isn't about is_tenant's per-tenant HNSW links (experiment 1
already covers that). It's specifically about what happens to the *small*
tenants when a single disproportionately large tenant shares their graph.

Same real embedding pool as experiment 1 (dbpedia-entities-openai-1M), just
re-split whale/minnow instead of into even tenants.
"""

import time

import numpy as np

from mtp import config
from mtp.data import assign_tenants_whale_and_minnows, brute_force_topk, build_hnsw_index
from mtp.real_data import load_dbpedia_vectors


def run():
    rng = np.random.default_rng(config.SEED)
    dim = config.E2_DIM
    k = config.E2_K

    print("loading real embeddings (dbpedia-entities-openai-1M) ...")
    vectors, query_pool = load_dbpedia_vectors(config.E2_N_QUERY_POOL, config.SEED)
    n = len(vectors)
    minnow_count = config.E2_MINNOW_COUNT
    whale_n = int(n * config.E2_WHALE_FRACTION)
    minnow_n_each = (n - whale_n) // minnow_count
    whale_n = n - minnow_n_each * minnow_count  # absorb the rounding remainder into the whale

    tenant_ids = assign_tenants_whale_and_minnows(n, whale_n, minnow_count, minnow_n_each)
    all_ids = np.arange(n)
    print(f"  {n} base vectors ({whale_n} whale + {minnow_count} x {minnow_n_each} minnows), {len(query_pool)} held-out queries")

    whale_mask = tenant_ids == 0
    minnow_mask = ~whale_mask

    print(f"naive: building one shared graph over all {n} vectors (whale is {whale_mask.mean()*100:.0f}% of it) ...")
    t0 = time.time()
    naive_index = build_hnsw_index(vectors, all_ids, dim, config.E2_HNSW_M, config.E2_EF_CONSTRUCTION)
    naive_build_s = time.time() - t0
    naive_index.set_ef(config.E2_EF_SEARCH)
    print(f"  done in {naive_build_s:.1f}s")

    print(f"tiered: building a dedicated whale graph ({whale_mask.sum()} vectors) ...")
    t0 = time.time()
    whale_ids_local = np.arange(whale_mask.sum())
    whale_index = build_hnsw_index(vectors[whale_mask], whale_ids_local, dim, config.E2_HNSW_M, config.E2_EF_CONSTRUCTION)
    whale_build_s = time.time() - t0
    whale_index.set_ef(config.E2_EF_SEARCH)
    whale_global_ids = all_ids[whale_mask]
    print(f"  done in {whale_build_s:.1f}s")

    print(f"tiered: building a shared fallback graph for the {minnow_count} minnows ({minnow_mask.sum()} vectors) ...")
    t0 = time.time()
    minnow_ids_local = np.arange(minnow_mask.sum())
    fallback_index = build_hnsw_index(
        vectors[minnow_mask], minnow_ids_local, dim, config.E2_HNSW_M, config.E2_EF_CONSTRUCTION
    )
    fallback_build_s = time.time() - t0
    fallback_index.set_ef(config.E2_EF_SEARCH)
    minnow_global_ids = all_ids[minnow_mask]
    print(f"  done in {fallback_build_s:.1f}s")

    def eval_minnow(tenant, n_queries):
        tenant_mask_global = tenant_ids == tenant
        tenant_vec_ids = all_ids[tenant_mask_global]
        tenant_vectors = vectors[tenant_mask_global]
        if len(tenant_vec_ids) < k:
            return None

        start = (tenant * n_queries) % len(query_pool)
        queries = query_pool[np.arange(start, start + n_queries) % len(query_pool)]
        naive_recalls, naive_lat = [], []
        tiered_recalls, tiered_lat = [], []

        tenant_membership_full = np.zeros(n, dtype=bool)
        tenant_membership_full[tenant_vec_ids] = True

        tenant_mask_in_fallback = tenant_ids[minnow_global_ids] == tenant

        for q in queries:
            gt = tenant_vec_ids[brute_force_topk(q, tenant_vectors, k)]

            t0 = time.time()
            lbl, _ = naive_index.knn_query(q, k=k, filter=lambda l: tenant_membership_full[l])
            naive_lat.append(time.time() - t0)
            naive_recalls.append(len(set(lbl[0]) & set(gt)) / k)

            t0 = time.time()
            lbl_f, _ = fallback_index.knn_query(q, k=k, filter=lambda l: tenant_mask_in_fallback[l])
            tiered_lat.append(time.time() - t0)
            tiered_ids = minnow_global_ids[lbl_f[0]]
            tiered_recalls.append(len(set(tiered_ids) & set(gt)) / k)

        return {
            "naive_recall_at_10": float(np.mean(naive_recalls)),
            "naive_latency_ms": float(np.mean(naive_lat) * 1000),
            "tiered_recall_at_10": float(np.mean(tiered_recalls)),
            "tiered_latency_ms": float(np.mean(tiered_lat) * 1000),
        }

    def eval_whale(n_queries):
        queries = query_pool[np.arange(n_queries) % len(query_pool)]
        whale_vectors = vectors[whale_mask]
        naive_recalls, naive_lat = [], []
        tiered_recalls, tiered_lat = [], []

        for q in queries:
            gt = whale_global_ids[brute_force_topk(q, whale_vectors, k)]

            t0 = time.time()
            lbl, _ = naive_index.knn_query(q, k=k, filter=lambda l: whale_mask[l])
            naive_lat.append(time.time() - t0)
            naive_recalls.append(len(set(lbl[0]) & set(gt)) / k)

            t0 = time.time()
            lbl_w, _ = whale_index.knn_query(q, k=k)
            tiered_lat.append(time.time() - t0)
            tiered_ids = whale_global_ids[lbl_w[0]]
            tiered_recalls.append(len(set(tiered_ids) & set(gt)) / k)

        return {
            "naive_recall_at_10": float(np.mean(naive_recalls)),
            "naive_latency_ms": float(np.mean(naive_lat) * 1000),
            "tiered_recall_at_10": float(np.mean(tiered_recalls)),
            "tiered_latency_ms": float(np.mean(tiered_lat) * 1000),
        }

    sample_minnows = rng.choice(np.arange(1, minnow_count + 1), size=config.E2_SAMPLE_MINNOWS, replace=False)
    minnow_rows = []
    for tenant in sample_minnows:
        r = eval_minnow(int(tenant), config.E2_QUERIES_PER_TENANT)
        if r:
            minnow_rows.append(r)

    minnow_summary = {
        "naive_recall_at_10": float(np.mean([r["naive_recall_at_10"] for r in minnow_rows])),
        "naive_latency_ms": float(np.mean([r["naive_latency_ms"] for r in minnow_rows])),
        "tiered_recall_at_10": float(np.mean([r["tiered_recall_at_10"] for r in minnow_rows])),
        "tiered_latency_ms": float(np.mean([r["tiered_latency_ms"] for r in minnow_rows])),
    }
    whale_summary = eval_whale(config.E2_QUERIES_PER_TENANT * 3)

    print("minnows (avg over sampled tenants):", minnow_summary)
    print("whale:", whale_summary)

    return {
        "n_total": n,
        "whale_share_pct": float(whale_mask.mean() * 100),
        "naive_build_s": naive_build_s,
        "whale_build_s": whale_build_s,
        "fallback_build_s": fallback_build_s,
        "minnow_summary": minnow_summary,
        "minnow_rows": minnow_rows,
        "whale_summary": whale_summary,
    }


if __name__ == "__main__":
    import json
    result = run()
    with open(config.results_path("exp2_tiered.json"), "w") as f:
        json.dump(result, f, indent=2)
