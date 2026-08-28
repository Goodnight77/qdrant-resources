"""Real-data benchmark: FastEmbed + Qdrant's local inference on NFCorpus
(BeIR/nfcorpus), a public biomedical IR benchmark from Hugging Face -
3,633 real PubMed abstracts (90-9,939 chars, mean ~1,497) and 323 real test
queries with human relevance judgments (qrels). No synthetic/repeated text.

This measures two different things:
  1. Latency/throughput on real, varied-length documents (not identical
     repeated strings).
  2. Retrieval quality (Recall@10, MRR@10) against the dataset's own human
     relevance judgments - so "are these results good" has an actual answer,
     not just a speed number.

Run:
    pip install -r requirements.txt datasets
    python benchmark_real_data.py
"""
import json
import statistics
import time
from collections import defaultdict

from datasets import load_dataset
from qdrant_client import QdrantClient, models

MODEL_NAME = "BAAI/bge-small-en-v1.5"
TOP_K = 10

print("Loading BeIR/nfcorpus from Hugging Face...")
corpus = load_dataset("BeIR/nfcorpus", "corpus", split="corpus")
queries = load_dataset("BeIR/nfcorpus", "queries", split="queries")
qrels = load_dataset("BeIR/nfcorpus-qrels", split="test")

query_text_by_id = {q["_id"]: q["text"] for q in queries}
relevant_by_query = defaultdict(set)
for row in qrels:
    relevant_by_query[row["query-id"]].add(row["corpus-id"])

test_query_ids = list(relevant_by_query.keys())
corpus_texts = [(row["_id"], f"{row['title']} {row['text']}".strip()) for row in corpus]

doc_lens = [len(t) for _, t in corpus_texts]
results = {
    "dataset": "BeIR/nfcorpus",
    "corpus_size": len(corpus_texts),
    "test_query_count": len(test_query_ids),
    "corpus_char_len": {
        "min": min(doc_lens), "max": max(doc_lens),
        "mean": statistics.mean(doc_lens),
    },
}

# 1. Embed + index the real corpus, in batches, measuring wall-clock throughput
client = QdrantClient(":memory:")
vector_size = client.get_embedding_size(MODEL_NAME)
client.create_collection(
    collection_name="nfcorpus",
    vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
)

BATCH = 64
t0 = time.perf_counter()
for i in range(0, len(corpus_texts), BATCH):
    chunk = corpus_texts[i:i + BATCH]
    client.upsert(
        collection_name="nfcorpus",
        points=[
            models.PointStruct(
                id=idx,
                vector=models.Document(text=text, model=MODEL_NAME),
                payload={"doc_id": doc_id},
            )
            for idx, (doc_id, text) in enumerate(chunk, start=i)
        ],
    )
elapsed = time.perf_counter() - t0
results["corpus_index_seconds"] = elapsed
results["corpus_index_docs_per_sec"] = len(corpus_texts) / elapsed

id_to_doc_id = {i: doc_id for i, (doc_id, _) in enumerate(corpus_texts)}

# 2. Real per-query latency (323 distinct real queries, not one repeated string)
#    + retrieval quality against the dataset's own human relevance judgments.
query_latencies_ms = []
recall_hits = []
reciprocal_ranks = []

for qid in test_query_ids:
    qtext = query_text_by_id[qid]
    t0 = time.perf_counter()
    response = client.query_points(
        collection_name="nfcorpus",
        query=models.Document(text=qtext, model=MODEL_NAME),
        limit=TOP_K,
    )
    query_latencies_ms.append((time.perf_counter() - t0) * 1000)

    retrieved_doc_ids = [id_to_doc_id[p.id] for p in response.points]
    relevant = relevant_by_query[qid]
    hit_ranks = [rank for rank, d in enumerate(retrieved_doc_ids, start=1) if d in relevant]
    recall_hits.append(1 if hit_ranks else 0)
    reciprocal_ranks.append(1.0 / hit_ranks[0] if hit_ranks else 0.0)

query_latencies_ms.sort()
n = len(query_latencies_ms)
results["real_query_latency_ms"] = {
    "mean": statistics.mean(query_latencies_ms),
    "p50": query_latencies_ms[n // 2],
    "p95": query_latencies_ms[int(n * 0.95)],
    "min": query_latencies_ms[0],
    "max": query_latencies_ms[-1],
}
results["retrieval_quality"] = {
    "hit_rate_at_10": statistics.mean(recall_hits),
    "mrr_at_10": statistics.mean(reciprocal_ranks),
    "note": (
        "hit_rate@10 = fraction of queries with >=1 known-relevant document "
        "in the top 10 (a lenient hit-rate, NOT the standard Recall@10 which "
        "averages the fraction of ALL relevant docs retrieved per query). "
        "mrr_at_10 = mean reciprocal rank of the first relevant hit (0 if "
        "none in top 10). Judged against BeIR/nfcorpus-qrels (test split, "
        "sparse human judgments - not every relevant doc is necessarily "
        "labeled)."
    ),
}

with open("results_real_data.json", "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
