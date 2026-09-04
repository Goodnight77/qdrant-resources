"""Experiment 3: per-tenant IDF statistics (Qdrant 1.19).

BM25-style sparse search weights a term by how rare it is across the corpus
(IDF). In a multi-tenant collection, "the corpus" is ambiguous: by default
Qdrant computes IDF across the whole shard: every tenant's vocabulary
blended together. 1.19 lets you scope that computation to a payload filter
(`search_params.idf.corpus`), typically the same tenant filter you're already
applying to the results.

Two controlled demonstrations use generated two-term queries and binary
relevance defined by rare-term presence. AG News supplies real documents;
the synthetic corpus illustrates the mechanism with constructed vocabulary.
Neither estimates typical production search quality. Document vectors use
raw term counts, without BM25 saturation or document-length normalization.

The default backend is qdrant-client's Python local implementation. Set
QDRANT_URL / QDRANT_API_KEY to compare against a Qdrant server. Each case
creates a uniquely named temporary collection and removes it when finished.
"""

import os
from contextlib import closing, contextmanager
from datetime import datetime, timezone
from importlib.metadata import version
from uuid import uuid4

import numpy as np
from qdrant_client import QdrantClient, models

from mtp import config
from mtp.real_text_data import CATEGORIES, discover_word_pairs, load_ag_news_sample
from mtp.real_text_data import tokens_to_bm25
from mtp.real_text_data import tokens_to_sparse as real_tokens_to_sparse
from mtp.text_data import build_vocab_and_tenants, generate_tenant_docs, tokens_to_sparse


def _make_client():
    url = os.environ.get("QDRANT_URL")
    if url:
        return QdrantClient(url=url, api_key=os.environ.get("QDRANT_API_KEY") or None, timeout=60)
    return QdrantClient(":memory:")


def _backend_metadata():
    metadata = {
        "mode": "server" if os.environ.get("QDRANT_URL") else "local_memory",
        "qdrant_client_version": version("qdrant-client"),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if metadata["mode"] == "server":
        with closing(_make_client()) as client:
            metadata["server"] = client.info().model_dump()
    return metadata


@contextmanager
def _experiment_collection():
    client = _make_client()
    collection = f"mtp_per_tenant_idf_{uuid4().hex}"
    created = False
    try:
        client.create_collection(
            collection,
            vectors_config={},
            sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )
        created = True
        client.create_payload_index(
            collection,
            field_name="tenant",
            field_schema=models.KeywordIndexParams(type=models.KeywordIndexType.KEYWORD, is_tenant=True),
        )
        yield client, collection
    finally:
        try:
            if created:
                client.delete_collection(collection)
        finally:
            client.close()


def _ndcg_at_k(ranked_relevant, k, n_relevant):
    """Binary NDCG normalized by all relevant documents in the tenant corpus."""
    dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ranked_relevant[:k]))
    idcg = sum(1 / np.log2(i + 2) for i in range(min(k, n_relevant)))
    return dcg / idcg if idcg > 0 else 0.0


def _relevant_count(client, collection, tenant, word, field):
    return client.count(
        collection,
        count_filter=models.Filter(must=[
            models.FieldCondition(key="tenant", match=models.MatchValue(value=tenant)),
            models.FieldCondition(key=field, match=models.MatchValue(value=word)),
        ]),
        exact=True,
    ).count


def _setup_collection(client, collection, tenants, n_common, n_domain_common, n_domain_rare, docs_per_tenant, seed):
    word_to_id, common_words, tenant_vocab = build_vocab_and_tenants(
        tenants, n_common, n_domain_common, n_domain_rare, seed
    )

    all_docs = []
    point_id = 0
    points = []
    for i, tenant in enumerate(tenants):
        docs = generate_tenant_docs(
            tenant,
            common_words,
            tenant_vocab[tenant]["domain_common"],
            tenant_vocab[tenant]["domain_rare"],
            docs_per_tenant,
            config.E3_DOC_LEN,
            config.E3_DOMAIN_COMMON_PROB,
            config.E3_DOMAIN_RARE_PROB,
            seed=seed + i,
        )
        for d in docs:
            sparse = tokens_to_sparse(d["tokens"], word_to_id)
            points.append(
                models.PointStruct(
                    id=point_id,
                    payload={"tenant": tenant, "rare_hits": list(d["rare_hits"])},
                    vector={"bm25": models.SparseVector(indices=list(sparse.keys()), values=[float(v) for v in sparse.values()])},
                )
            )
            all_docs.append(d)
            point_id += 1

    for i in range(0, len(points), 500):
        client.upsert(collection, points=points[i:i + 500])

    return word_to_id, tenant_vocab, all_docs


def _run_trials(client, collection, word_to_id, tenant_vocab, tenants, n_trials, k, seed):
    rng = np.random.default_rng(seed)
    global_scores, scoped_scores = [], []
    per_tenant = {t: {"global": [], "scoped": []} for t in tenants}

    for _ in range(n_trials):
        tenant = tenants[rng.integers(0, len(tenants))]
        w_rare = tenant_vocab[tenant]["domain_rare"][rng.integers(0, len(tenant_vocab[tenant]["domain_rare"]))]
        w_common = tenant_vocab[tenant]["domain_common"][rng.integers(0, len(tenant_vocab[tenant]["domain_common"]))]
        query_vec = models.SparseVector(
            indices=[word_to_id[w_common], word_to_id[w_rare]], values=[1.0, 1.0]
        )
        tenant_filter = models.Filter(must=[models.FieldCondition(key="tenant", match=models.MatchValue(value=tenant))])

        def relevance(hit):
            return 1 if w_rare in (hit.payload or {}).get("rare_hits", []) else 0

        global_hits = client.query_points(
            collection, query=query_vec, using="bm25", query_filter=tenant_filter, limit=k, with_payload=True
        ).points
        scoped_hits = client.query_points(
            collection,
            query=query_vec,
            using="bm25",
            query_filter=tenant_filter,
            search_params=models.SearchParams(idf=models.IdfCorpusParams(corpus=tenant_filter)),
            limit=k,
            with_payload=True,
        ).points

        n_relevant = _relevant_count(client, collection, tenant, w_rare, "rare_hits")
        g = _ndcg_at_k([relevance(h) for h in global_hits], k, n_relevant)
        s = _ndcg_at_k([relevance(h) for h in scoped_hits], k, n_relevant)
        global_scores.append(g)
        scoped_scores.append(s)
        per_tenant[tenant]["global"].append(g)
        per_tenant[tenant]["scoped"].append(s)

    return global_scores, scoped_scores, per_tenant


def run_synthetic_flagship():
    """3 named industries, constructed vocabulary tuned to make the failure
    mode unmistakable. See the module docstring for why this isn't the
    headline number."""
    with _experiment_collection() as (client, collection):
        word_to_id, tenant_vocab, _ = _setup_collection(
            client, collection,
            config.E3_TENANTS,
            config.E3_N_COMMON_WORDS,
            config.E3_N_DOMAIN_COMMON,
            config.E3_N_DOMAIN_RARE,
            config.E3_DOCS_PER_TENANT,
            config.SEED,
        )
        global_scores, scoped_scores, per_tenant = _run_trials(
            client, collection, word_to_id, tenant_vocab, config.E3_TENANTS, config.E3_TRIALS, config.E3_K, config.SEED + 1
        )
        result = {
            "n_tenants": len(config.E3_TENANTS),
            "docs_per_tenant": config.E3_DOCS_PER_TENANT,
            "global_ndcg_mean": float(np.mean(global_scores)),
            "global_ndcg_std": float(np.std(global_scores)),
            "scoped_ndcg_mean": float(np.mean(scoped_scores)),
            "scoped_ndcg_std": float(np.std(scoped_scores)),
            "per_tenant": {
                t: {
                    "global_ndcg_mean": float(np.mean(v["global"])),
                    "scoped_ndcg_mean": float(np.mean(v["scoped"])),
                }
                for t, v in per_tenant.items()
            },
        }
        print(
            f"synthetic worst case (3 tenants): global NDCG@10={result['global_ndcg_mean']:.3f} "
            f"vs per-tenant NDCG@10={result['scoped_ndcg_mean']:.3f}"
        )
        return result


def run_synthetic_tenant_count_sweep(tenant_counts, trials_per_point):
    """Measure how the global-versus-per-tenant IDF gap changes with tenant count.
    Uses generic tenant_0..tenant_{T-1}
    instead of the three named industries, so it scales to any T."""
    rows = []
    for t_count in tenant_counts:
        tenants = [f"tenant_{i}" for i in range(t_count)]
        with _experiment_collection() as (client, collection):
            word_to_id, tenant_vocab, _ = _setup_collection(
                client, collection,
                tenants,
                config.E3_N_COMMON_WORDS,
                config.E3_N_DOMAIN_COMMON,
                config.E3_N_DOMAIN_RARE,
                config.E3_DOCS_PER_TENANT,
                config.SEED + 100 + t_count,
            )
            global_scores, scoped_scores, _ = _run_trials(
                client, collection, word_to_id, tenant_vocab, tenants, trials_per_point, config.E3_K, config.SEED + 200 + t_count
            )
            rows.append({
                "n_tenants": t_count,
                "global_ndcg_mean": float(np.mean(global_scores)),
                "scoped_ndcg_mean": float(np.mean(scoped_scores)),
            })
            print(f"n_tenants={t_count:>3}  global NDCG@10={rows[-1]['global_ndcg_mean']:.3f}  per-tenant NDCG@10={rows[-1]['scoped_ndcg_mean']:.3f}")
    return rows


def _index_real_sample(client, collection, sample, word_to_id, bm25_params=None):
    points = []
    for i, (category, tokens) in enumerate(zip(sample["category"], sample["tokens"])):
        sparse = (
            real_tokens_to_sparse(tokens, word_to_id)
            if bm25_params is None
            else tokens_to_bm25(tokens, word_to_id, **bm25_params)
        )
        if not sparse:
            continue
        points.append(
            models.PointStruct(
                id=i,
                payload={"tenant": category, "tokens_present": list(set(tokens))},
                vector={"bm25": models.SparseVector(indices=list(sparse.keys()), values=[float(v) for v in sparse.values()])},
            )
        )
    for i in range(0, len(points), 500):
        client.upsert(collection, points=points[i:i + 500])
    return len(points)


def _measure_real_pairs(client, collection, categories, pairs, word_to_id, k):
    """Exhaustive, deterministic: every discovered (common, rare) pair per
    category gets exactly one query each way. No resampling: the documents
    and their term frequencies are real and fixed, so there's no randomness
    to average over beyond "which real word pair," and we use every one we
    found rather than a random subset of them."""
    per_category = {}
    all_global, all_scoped = [], []
    for category in categories:
        common_words = [w for w in pairs[category]["domain_common"] if w in word_to_id]
        rare_words = [w for w in pairs[category]["domain_rare"] if w in word_to_id]
        tenant_filter = models.Filter(must=[models.FieldCondition(key="tenant", match=models.MatchValue(value=category))])
        gs, ss = [], []
        for w_common in common_words:
            for w_rare in rare_words:
                query_vec = models.SparseVector(indices=[word_to_id[w_common], word_to_id[w_rare]], values=[1.0, 1.0])
                global_hits = client.query_points(
                    collection, query=query_vec, using="bm25", query_filter=tenant_filter, limit=k, with_payload=True
                ).points
                scoped_hits = client.query_points(
                    collection,
                    query=query_vec,
                    using="bm25",
                    query_filter=tenant_filter,
                    search_params=models.SearchParams(idf=models.IdfCorpusParams(corpus=tenant_filter)),
                    limit=k,
                    with_payload=True,
                ).points

                def relevance(hit, w_rare=w_rare):
                    return 1 if w_rare in (hit.payload or {}).get("tokens_present", []) else 0

                n_relevant = _relevant_count(client, collection, category, w_rare, "tokens_present")
                gs.append(_ndcg_at_k([relevance(h) for h in global_hits], k, n_relevant))
                ss.append(_ndcg_at_k([relevance(h) for h in scoped_hits], k, n_relevant))
        if gs:
            per_category[category] = {
                "n_pairs": len(gs),
                "global_ndcg_mean": float(np.mean(gs)),
                "scoped_ndcg_mean": float(np.mean(ss)),
            }
            all_global += gs
            all_scoped += ss
    return per_category, all_global, all_scoped


def run_real_ag_news():
    """4 real AG News categories (world/sports/business/sci-tech) as tenants.
    Every discovered (domain-common, domain-rare) word pair is measured and
    averaged in under the generated-query, rare-term relevance protocol."""
    with _experiment_collection() as (client, collection):
        sample = load_ag_news_sample(config.E3_AG_NEWS_DOCS_PER_CATEGORY, config.SEED)
        pairs = discover_word_pairs(sample, CATEGORIES, top_k=config.E3_AG_NEWS_TOP_K)
        vocab = set()
        for toks in sample["tokens"]:
            vocab.update(toks)
        word_to_id = {w: i for i, w in enumerate(sorted(vocab))}
        n_indexed = _index_real_sample(client, collection, sample, word_to_id)

        per_category, all_global, all_scoped = _measure_real_pairs(client, collection, CATEGORIES, pairs, word_to_id, config.E3_K)
        result = {
            "n_docs_indexed": n_indexed,
            "n_pairs_measured": len(all_global),
            "global_ndcg_mean": float(np.mean(all_global)),
            "scoped_ndcg_mean": float(np.mean(all_scoped)),
            "per_category": per_category,
            "discovered_pairs": pairs,
        }
        print(
            f"real AG News ({len(CATEGORIES)} categories, {len(all_global)} word pairs): "
            f"global NDCG@10={result['global_ndcg_mean']:.3f} vs per-tenant NDCG@10={result['scoped_ndcg_mean']:.3f}"
        )
        return result


def run_real_category_sweep():
    """Same real AG News data, but only the first T categories share the
    collection, for T = 2..4 (AG News has exactly 4). Re-discovers word pairs
    and re-indexes for each T, since which words count as "tenant-exclusive"
    depends on which other tenants are in the collection."""
    sample = load_ag_news_sample(config.E3_AG_NEWS_DOCS_PER_CATEGORY, config.SEED)
    vocab = set()
    for toks in sample["tokens"]:
        vocab.update(toks)
    word_to_id = {w: i for i, w in enumerate(sorted(vocab))}

    rows = []
    for t_count in range(2, len(CATEGORIES) + 1):
        categories = CATEGORIES[:t_count]
        with _experiment_collection() as (client, collection):
            subset = sample[sample["category"].isin(categories)]
            _index_real_sample(client, collection, subset, word_to_id)
            pairs = discover_word_pairs(subset, categories, top_k=config.E3_AG_NEWS_TOP_K)
            _, all_global, all_scoped = _measure_real_pairs(client, collection, categories, pairs, word_to_id, config.E3_K)
            rows.append({
                "n_tenants": t_count,
                "n_pairs_measured": len(all_global),
                "global_ndcg_mean": float(np.mean(all_global)) if all_global else None,
                "scoped_ndcg_mean": float(np.mean(all_scoped)) if all_scoped else None,
            })
            print(f"real categories={t_count}  global NDCG@10={rows[-1]['global_ndcg_mean']:.3f}  per-tenant NDCG@10={rows[-1]['scoped_ndcg_mean']:.3f}")
    return rows


def run():
    backend = _backend_metadata()
    real_flagship = run_real_ag_news()
    real_sweep = run_real_category_sweep()
    synthetic_flagship = run_synthetic_flagship()
    synthetic_sweep = run_synthetic_tenant_count_sweep(config.E3_SWEEP_TENANT_COUNTS, trials_per_point=config.E3_SWEEP_TRIALS)
    return {
        "backend": backend,
        "metric": "binary_ndcg_at_k_corpus_relevance",
        "k": config.E3_K,
        "real_flagship": real_flagship,
        "real_category_sweep": real_sweep,
        "synthetic_flagship": synthetic_flagship,
        "synthetic_tenant_count_sweep": synthetic_sweep,
    }


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(config.results_path("exp3_per_tenant_idf.json")))
    args = parser.parse_args()
    result = run()
    with args.output.open("w") as f:
        json.dump(result, f, indent=2, allow_nan=False)
