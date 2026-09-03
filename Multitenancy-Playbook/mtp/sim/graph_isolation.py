"""Experiment 1: payload partitioning vs. dedicated (is_tenant) graphs.

Qdrant's own docs describe `is_tenant=true` and dedicated per-tenant shards as
two ways of getting the same underlying benefit: instead of one HNSW graph
that links every tenant's vectors together and gets filtered post-hoc, each
tenant gets its own graph (or its own tenant-restricted links, via
`payload_m` + `m: 0`). We reproduce that mechanism directly with hnswlib,
the same graph algorithm Qdrant's segments use internally, so we can
measure the effect in isolation, without a live cluster's network/disk noise
in the way.

  "shared"    = one HNSW graph over every tenant's vectors, tenant matched
                with hnswlib's `filter` callback at query time. This stands
                in for a multi-tenant collection *without* is_tenant, or
                without any partitioning at all: tenant is just another
                payload field, and the graph doesn't know about it.
  "dedicated" = tenant gets its own graph containing only its own vectors.
                This stands in for `is_tenant=true` (tenant-restricted HNSW
                links) and for collection-per-tenant / a dedicated shard:
                structurally, all three give a tenant its own graph.

Vectors are real OpenAI ada-002 embeddings (dbpedia-entities-openai-1M, see
mtp/real_data.py): real semantic clustering, not a synthetic Gaussian
mixture. Tenant assignment is independent of that clustering (a tenant is a
business/access boundary, not a topic), so filtering by tenant in the shared
graph is a genuine low-selectivity needle-in-a-haystack search, not an easy
case where the tenant's vectors happen to already sit together.
"""

import time

import numpy as np

from mtp import config
from mtp.data import brute_force_topk, build_hnsw_index
from mtp.real_data import load_dbpedia_vectors


def run():
    rng = np.random.default_rng(config.SEED)
    dim = config.E1_DIM
    k = config.E1_K

    print("loading real embeddings (dbpedia-entities-openai-1M) ...")
    vectors, query_pool = load_dbpedia_vectors(config.E1_N_QUERY_POOL, config.SEED)
    n = len(vectors)
    all_ids = np.arange(n)
    print(f"  {n} base vectors, {len(query_pool)} held-out queries")

    print(f"building shared graph over {n} vectors (dim={dim}) ...")
    t0 = time.time()
    shared_index = build_hnsw_index(vectors, all_ids, dim, config.E1_HNSW_M, config.E1_EF_CONSTRUCTION)
    shared_build_s = time.time() - t0
    shared_index.set_ef(config.E1_EF_SEARCH)
    print(f"  done in {shared_build_s:.1f}s")

    rows = []
    for n_tenants in config.E1_TENANT_COUNTS:
        tenant_ids = rng.integers(0, n_tenants, n)
        selectivity = 1.0 / n_tenants
        sample_tenants = rng.choice(n_tenants, size=min(config.E1_SAMPLE_TENANTS, n_tenants), replace=False)

        shared_recalls, shared_latencies = [], []
        dedicated_recalls, dedicated_latencies = [], []
        dedicated_build_times = []

        q_i = 0
        for tenant in sample_tenants:
            tenant_mask = tenant_ids == tenant
            tenant_vec_ids = all_ids[tenant_mask]
            tenant_vectors = vectors[tenant_mask]
            if len(tenant_vec_ids) < k:
                q_i += config.E1_QUERIES_PER_TENANT
                continue

            # dedicated per-tenant graph: build once, query E1_QUERIES_PER_TENANT times
            t0 = time.time()
            local_ids = np.arange(len(tenant_vec_ids))
            dedicated_index = build_hnsw_index(
                tenant_vectors, local_ids, dim, config.E1_HNSW_M, config.E1_EF_CONSTRUCTION
            )
            dedicated_build_times.append(time.time() - t0)
            dedicated_index.set_ef(config.E1_EF_SEARCH)

            tenant_membership = np.zeros(n, dtype=bool)
            tenant_membership[tenant_vec_ids] = True

            for _ in range(config.E1_QUERIES_PER_TENANT):
                q = query_pool[q_i % len(query_pool)]
                q_i += 1
                gt = tenant_vec_ids[brute_force_topk(q, tenant_vectors, k)]

                t0 = time.time()
                lbl, _ = shared_index.knn_query(q, k=k, filter=lambda l: tenant_membership[l])
                shared_latencies.append(time.time() - t0)
                shared_recalls.append(len(set(lbl[0]) & set(gt)) / k)

                t0 = time.time()
                lbl_d, _ = dedicated_index.knn_query(q, k=min(k, len(tenant_vec_ids)))
                dedicated_latencies.append(time.time() - t0)
                dedicated_ids = tenant_vec_ids[lbl_d[0]]
                dedicated_recalls.append(len(set(dedicated_ids) & set(gt)) / k)

        rows.append({
            "n_tenants": n_tenants,
            "selectivity_pct": selectivity * 100,
            "shared_recall_at_10": float(np.mean(shared_recalls)),
            "shared_latency_ms": float(np.mean(shared_latencies) * 1000),
            "shared_latency_p95_ms": float(np.percentile(shared_latencies, 95) * 1000),
            "dedicated_recall_at_10": float(np.mean(dedicated_recalls)),
            "dedicated_latency_ms": float(np.mean(dedicated_latencies) * 1000),
            "dedicated_latency_p95_ms": float(np.percentile(dedicated_latencies, 95) * 1000),
            "dedicated_build_ms_per_tenant": float(np.mean(dedicated_build_times) * 1000),
        })
        print(
            f"tenants={n_tenants:>5} (selectivity {selectivity*100:5.2f}%)  "
            f"shared recall@10={rows[-1]['shared_recall_at_10']:.3f} "
            f"lat={rows[-1]['shared_latency_ms']:.2f}ms   |   "
            f"dedicated recall@10={rows[-1]['dedicated_recall_at_10']:.3f} "
            f"lat={rows[-1]['dedicated_latency_ms']:.2f}ms"
        )

    return {"n_total": n, "shared_build_s": shared_build_s, "rows": rows}


if __name__ == "__main__":
    import json
    result = run()
    with open(config.results_path("exp1_graph_isolation.json"), "w") as f:
        json.dump(result, f, indent=2)
