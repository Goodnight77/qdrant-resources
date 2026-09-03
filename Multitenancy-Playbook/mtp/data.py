"""Shared HNSW/tenant-assignment helpers for the dense-vector experiments (1 and
2). The vectors themselves are real embeddings (see mtp/real_data.py); tenant
assignment here is independent of their content, exactly like a real SaaS
app, where "which customer does this belong to" has nothing to do with what
the embedding is about. That's what makes tenant filtering on an unpartitioned
index a genuinely hard case: the matching vectors are scattered evenly across
the whole graph, not sitting in one neighborhood.
"""

import hnswlib
import numpy as np


def assign_tenants_whale_and_minnows(n, whale_n, minnow_count, minnow_n_each):
    """tenant_id 0 = whale, 1..minnow_count = minnows. Returns a tenant_ids array
    sized whale_n + minnow_count * minnow_n_each. Vector cluster membership is
    generated independently of position, so block order here doesn't bias anything."""
    assert n == whale_n + minnow_count * minnow_n_each
    tenant_ids = np.empty(n, dtype=np.int64)
    tenant_ids[:whale_n] = 0
    for i in range(minnow_count):
        start = whale_n + i * minnow_n_each
        tenant_ids[start:start + minnow_n_each] = i + 1
    return tenant_ids


def build_hnsw_index(vectors, ids, dim, m, ef_construction, num_threads=4):
    idx = hnswlib.Index(space="cosine", dim=dim)
    idx.init_index(max_elements=len(ids), ef_construction=ef_construction, M=m)
    idx.add_items(vectors, ids, num_threads=num_threads)
    return idx


def brute_force_topk(query, vectors, k):
    """Exact cosine-sim top-k indices (ground truth for recall)."""
    v = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12)
    q = query / (np.linalg.norm(query) + 1e-12)
    sims = v @ q
    idx = np.argpartition(-sims, min(k, len(sims) - 1))[:k]
    return idx[np.argsort(-sims[idx])]
