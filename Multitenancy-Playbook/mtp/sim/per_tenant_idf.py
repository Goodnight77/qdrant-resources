"""Experiment 3: per-tenant IDF statistics (Qdrant 1.19).

BM25-style sparse search weights a term by how rare it is across the corpus
(IDF). In a multi-tenant collection, "the corpus" is ambiguous: by default
Qdrant computes IDF across the whole shard: every tenant's vocabulary
blended together. 1.19 lets you scope that computation to a payload filter
(`search_params.idf.corpus`), typically the same tenant filter you're already
applying to the results.

Two measurements, in order of how much you should trust them:

  `run_real_ag_news` / `run_real_category_sweep`: real AG News articles
  (mtp/real_text_data.py), 4 real categories as tenants. Word pairs are
  *discovered* from real term-frequency stats, not constructed, and every
  (common, rare) pair found is measured and averaged in, including the
  ones where global and per-tenant IDF come out identical. This is the
  honest, unfiltered picture: the distortion is real but modest on average,
  and concentrated in specific categories/word pairs rather than uniform.

  `run_synthetic_worst_case`: a constructed vocabulary (mtp/text_data.py)
  tuned so the failure mode is unmistakable (repeated tenant-common jargon,
  no genuinely rare term in the document). This is *not* a claim about how
  bad the effect is in practice. It exists to make the mechanism legible:
  here is what global IDF does at its worst, and why.

Runs against qdrant-client's embedded local mode by default (no server
needed, this is real Qdrant scoring code, just not a real network/disk
path). Point QDRANT_URL / QDRANT_API_KEY at a real deployment to reproduce
against a live server instead; the query and indexing code is unchanged
either way.
"""

import os

import numpy as np
from qdrant_client import QdrantClient, models

from mtp import config
from mtp.real_text_data import CATEGORIES, discover_word_pairs, load_ag_news_sample
from mtp.real_text_data import tokens_to_sparse as real_tokens_to_sparse
from mtp.text_data import build_vocab_and_tenants, generate_tenant_docs, tokens_to_sparse

COLLECTION = "mtp_per_tenant_idf"


def _make_client():
    url = os.environ.get("QDRANT_URL")
    if url:
        return QdrantClient(url=url, api_key=os.environ.get("QDRANT_API_KEY"), timeout=60)
    return QdrantClient(":memory:")


def _ndcg_at_k(ranked_relevant, k):
    """ranked_relevant: list of 0/1 in ranked order."""
    dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ranked_relevant[:k]))
    ideal = sorted(ranked_relevant, reverse=True)
    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal[:k]))
    return dcg / idcg if idcg > 0 else 0.0


def _setup_collection(client, tenants, n_common, n_domain_common, n_domain_rare, docs_per_tenant, seed):
    word_to_id, common_words, tenant_vocab = build_vocab_and_tenants(
        tenants, n_common, n_domain_common, n_domain_rare, seed
    )

    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    client.create_collection(
        COLLECTION,
        vectors_config={},
        sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )
    client.create_payload_index(
        COLLECTION,
        field_name="tenant",
        field_schema=models.KeywordIndexParams(type=models.KeywordIndexType.KEYWORD, is_tenant=True),
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
        client.upsert(COLLECTION, points=points[i:i + 500])

    return word_to_id, tenant_vocab, all_docs


def _run_trials(client, word_to_id, tenant_vocab, tenants, n_trials, k, seed):
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
            COLLECTION, query=query_vec, using="bm25", query_filter=tenant_filter, limit=k, with_payload=True
        ).points
        scoped_hits = client.query_points(
            COLLECTION,
            query=query_vec,
            using="bm25",
            query_filter=tenant_filter,
            search_params=models.SearchParams(idf=models.IdfCorpusParams(corpus=tenant_filter)),
            limit=k,
            with_payload=True,
        ).points

        g = _ndcg_at_k([relevance(h) for h in global_hits], k)
        s = _ndcg_at_k([relevance(h) for h in scoped_hits], k)
        global_scores.append(g)
        scoped_scores.append(s)
        per_tenant[tenant]["global"].append(g)
        per_tenant[tenant]["scoped"].append(s)

    return global_scores, scoped_scores, per_tenant


def run_synthetic_flagship():
    """3 named industries, constructed vocabulary tuned to make the failure
    mode unmistakable. See the module docstring for why this isn't the
    headline number."""
    client = _make_client()
    word_to_id, tenant_vocab, _ = _setup_collection(
        client,
        config.E3_TENANTS,
        config.E3_N_COMMON_WORDS,
        config.E3_N_DOMAIN_COMMON,
        config.E3_N_DOMAIN_RARE,
        config.E3_DOCS_PER_TENANT,
        config.SEED,
    )
    global_scores, scoped_scores, per_tenant = _run_trials(
        client, word_to_id, tenant_vocab, config.E3_TENANTS, config.E3_TRIALS, config.E3_K, config.SEED + 1
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
    """More tenants sharing a collection -> more 'other' volume diluting global
    stats -> the distortion should get worse. Uses generic tenant_0..tenant_{T-1}
    instead of the three named industries, so it scales to any T."""
    rows = []
    for t_count in tenant_counts:
        tenants = [f"tenant_{i}" for i in range(t_count)]
        client = _make_client()
        word_to_id, tenant_vocab, _ = _setup_collection(
            client,
            tenants,
            config.E3_N_COMMON_WORDS,
            config.E3_N_DOMAIN_COMMON,
            config.E3_N_DOMAIN_RARE,
            config.E3_DOCS_PER_TENANT,
            config.SEED + 100 + t_count,
        )
        global_scores, scoped_scores, _ = _run_trials(
            client, word_to_id, tenant_vocab, tenants, trials_per_point, config.E3_K, config.SEED + 200 + t_count
        )
        rows.append({
            "n_tenants": t_count,
            "global_ndcg_mean": float(np.mean(global_scores)),
            "scoped_ndcg_mean": float(np.mean(scoped_scores)),
        })
        print(f"n_tenants={t_count:>3}  global NDCG@10={rows[-1]['global_ndcg_mean']:.3f}  per-tenant NDCG@10={rows[-1]['scoped_ndcg_mean']:.3f}")
    return rows


def _index_real_sample(client, sample, word_to_id):
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    client.create_collection(
        COLLECTION,
        vectors_config={},
        sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )
    client.create_payload_index(
        COLLECTION,
        field_name="tenant",
        field_schema=models.KeywordIndexParams(type=models.KeywordIndexType.KEYWORD, is_tenant=True),
    )
    points = []
    for i, (category, tokens) in enumerate(zip(sample["category"], sample["tokens"])):
        sparse = real_tokens_to_sparse(tokens, word_to_id)
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
        client.upsert(COLLECTION, points=points[i:i + 500])
    return len(points)


def _measure_real_pairs(client, categories, pairs, word_to_id, k):
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
                    COLLECTION, query=query_vec, using="bm25", query_filter=tenant_filter, limit=k, with_payload=True
                ).points
                scoped_hits = client.query_points(
                    COLLECTION,
                    query=query_vec,
                    using="bm25",
                    query_filter=tenant_filter,
                    search_params=models.SearchParams(idf=models.IdfCorpusParams(corpus=tenant_filter)),
                    limit=k,
                    with_payload=True,
                ).points

                def relevance(hit, w_rare=w_rare):
                    return 1 if w_rare in (hit.payload or {}).get("tokens_present", []) else 0

                gs.append(_ndcg_at_k([relevance(h) for h in global_hits], k))
                ss.append(_ndcg_at_k([relevance(h) for h in scoped_hits], k))
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
    averaged in: this is the actual, unfiltered effect size on real text."""
    client = _make_client()
    sample = load_ag_news_sample(config.E3_AG_NEWS_DOCS_PER_CATEGORY, config.SEED)
    pairs = discover_word_pairs(sample, CATEGORIES, top_k=config.E3_AG_NEWS_TOP_K)
    vocab = set()
    for toks in sample["tokens"]:
        vocab.update(toks)
    word_to_id = {w: i for i, w in enumerate(sorted(vocab))}
    n_indexed = _index_real_sample(client, sample, word_to_id)

    per_category, all_global, all_scoped = _measure_real_pairs(client, CATEGORIES, pairs, word_to_id, config.E3_K)
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
        client = _make_client()
        subset = sample[sample["category"].isin(categories)]
        _index_real_sample(client, subset, word_to_id)
        pairs = discover_word_pairs(subset, categories, top_k=config.E3_AG_NEWS_TOP_K)
        _, all_global, all_scoped = _measure_real_pairs(client, categories, pairs, word_to_id, config.E3_K)
        rows.append({
            "n_tenants": t_count,
            "n_pairs_measured": len(all_global),
            "global_ndcg_mean": float(np.mean(all_global)) if all_global else None,
            "scoped_ndcg_mean": float(np.mean(all_scoped)) if all_scoped else None,
        })
        print(f"real categories={t_count}  global NDCG@10={rows[-1]['global_ndcg_mean']:.3f}  per-tenant NDCG@10={rows[-1]['scoped_ndcg_mean']:.3f}")
    return rows


def run():
    real_flagship = run_real_ag_news()
    real_sweep = run_real_category_sweep()
    synthetic_flagship = run_synthetic_flagship()
    synthetic_sweep = run_synthetic_tenant_count_sweep(config.E3_SWEEP_TENANT_COUNTS, trials_per_point=config.E3_SWEEP_TRIALS)
    return {
        "real_flagship": real_flagship,
        "real_category_sweep": real_sweep,
        "synthetic_flagship": synthetic_flagship,
        "synthetic_tenant_count_sweep": synthetic_sweep,
    }


if __name__ == "__main__":
    import json
    result = run()
    with open(config.results_path("exp3_per_tenant_idf.json"), "w") as f:
        json.dump(result, f, indent=2)
