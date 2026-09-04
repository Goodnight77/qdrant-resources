"""Paired raw-count versus BM25 comparison on the same AG News sample.

Documents, tokenization, query pairs, relevance, and evaluation order are
identical for both variants. Only document term weights change. Parameters
are fixed, not selected against these evaluation queries. This retains the
rare-term proxy labels; it is not a human-judged search benchmark.
"""

import hashlib
import json

import numpy as np

from mtp import config
from mtp.real_text_data import CATEGORIES, discover_word_pairs, load_ag_news_sample
from mtp.sim.per_tenant_idf import _backend_metadata, _experiment_collection, _index_real_sample, _measure_real_pairs


def run():
    backend = _backend_metadata()
    sample = load_ag_news_sample(config.E3_AG_NEWS_DOCS_PER_CATEGORY, config.SEED)
    pairs = discover_word_pairs(sample, CATEGORIES, top_k=config.E3_AG_NEWS_TOP_K)
    vocab = sorted({word for tokens in sample["tokens"] for word in tokens})
    word_to_id = {word: i for i, word in enumerate(vocab)}
    params = {
        "avg_len": float(sample["tokens"].map(len).mean()),
        "k1": config.E3_BM25_K1,
        "b": config.E3_BM25_B,
    }
    corpus = list(zip(sample["category"], sample["tokens"]))
    result = {
        "backend": backend,
        "metric": "binary_ndcg_at_k_corpus_relevance",
        "k": config.E3_K,
        "seed": config.SEED,
        "corpus_sha256": hashlib.sha256(json.dumps(corpus).encode()).hexdigest(),
        "bm25_params": params,
        "avg_len_scope": "all indexed documents, fixed for both IDF modes",
        "discovered_pairs": pairs,
        "variants": {},
    }
    scores = {}
    for variant, weights in [("raw_tf", None), ("bm25", params)]:
        with _experiment_collection() as (client, collection):
            n_indexed = _index_real_sample(client, collection, sample, word_to_id, weights)
            per_category, gs, ss = _measure_real_pairs(
                client, collection, CATEGORIES, pairs, word_to_id, config.E3_K
            )
        scores[variant] = {"global": np.array(gs), "scoped": np.array(ss)}
        result["variants"][variant] = {
            "n_docs_indexed": n_indexed,
            "n_pairs_measured": len(gs),
            "global_ndcg_mean": float(np.mean(gs)),
            "scoped_ndcg_mean": float(np.mean(ss)),
            "per_category": per_category,
        }
        print(f"{variant}: global NDCG@10={np.mean(gs):.3f}, scoped NDCG@10={np.mean(ss):.3f}", flush=True)
    result["paired_changes"] = {}
    for mode in ["global", "scoped"]:
        delta = scores["bm25"][mode] - scores["raw_tf"][mode]
        tied = np.isclose(delta, 0, rtol=0, atol=1e-12)
        result["paired_changes"][mode] = {
            "mean_delta": float(np.mean(delta)),
            "improved": int(np.sum((delta > 0) & ~tied)),
            "unchanged": int(np.sum(tied)),
            "worsened": int(np.sum((delta < 0) & ~tied)),
        }
    return result


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(config.results_path("exp3_bm25_comparison.json")))
    args = parser.parse_args()
    result = run()
    with args.output.open("w") as f:
        json.dump(result, f, indent=2, allow_nan=False)
