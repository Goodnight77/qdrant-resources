"""Pattern C counterpart to benchmark_real_data.py - same corpus, same real
queries, same model, but embedded *server-side* by Qdrant Cloud's built-in
inference instead of client-side FastEmbed. Text goes over the wire to your
Qdrant Cloud cluster and gets embedded there.

This is the number this whole guide's "avoid the external round trip"
argument needs a counterpoint for: Pattern C removes the local embedding
step but reintroduces a network hop - just to your own Qdrant Cloud cluster
instead of a third-party embedding API. Compare
results_qdrant_cloud_inference.json's real_query_latency_ms against
results_real_data.json's (Pattern A, local FastEmbed, no network hop) and
results_real_data_gpu.json's (local GPU, no network hop) to see that cost.

Requires a Qdrant Cloud cluster with built-in inference enabled and
BAAI/bge-small-en-v1.5 in its supported model list - check the Cloud console
/ https://qdrant.tech/documentation/cloud/inference/ (the list changes over
time). Creates a scratch collection ("nfcorpus-cloud-bench") on your cluster
and deletes it only after a full successful run.

Resumable: if the collection already has points from a prior interrupted
run (network drop, killed process, power loss), this picks up indexing from
that point count instead of re-embedding from scratch - re-running the same
command after any crash is always the right move, never delete the
collection by hand first.

Setup:
    export QDRANT_URL="https://<cluster>.cloud.qdrant.io"
    export QDRANT_API_KEY="..."
    pip install -r requirements.txt datasets
    python benchmark_qdrant_cloud_inference.py

Cost note: server-side inference is billed by your Qdrant Cloud plan; this
run embeds ~3,633 docs + 323 queries once.
"""
import json
import os
import statistics
import time
from collections import defaultdict

from datasets import load_dataset
from qdrant_client import QdrantClient, models

MODEL_NAME = "BAAI/bge-small-en-v1.5"
TOP_K = 10
COLLECTION = "nfcorpus-cloud-bench"
MAX_RETRIES = 5


def with_retry(fn, label):
    """Retries transient network errors (SSL drops, read timeouts) on a
    long-running loop of sequential HTTPS calls - without this, one hiccup
    over ~15-20 minutes of indexing kills the whole run."""
    for attempt in range(MAX_RETRIES):
        try:
            return fn()
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raise
            wait = 2 ** attempt
            print(f"{label} failed ({e}), retry {attempt + 1}/{MAX_RETRIES - 1} in {wait}s", flush=True)
            time.sleep(wait)

url = os.environ["QDRANT_URL"]
api_key = os.environ["QDRANT_API_KEY"]

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

results = {
    "runtime": "Qdrant Cloud built-in inference (server-side embed)",
    "dataset": "BeIR/nfcorpus",
    "model": MODEL_NAME,
    "corpus_size": len(corpus_texts),
    "test_query_count": len(test_query_ids),
}

client = QdrantClient(url=url, api_key=api_key, timeout=120)

BATCH = 16
if client.collection_exists(COLLECTION):
    # Resume: points are inserted in order with id == corpus index, in
    # whole batches (each upsert either fully lands or gets retried) - so
    # the current point count is always a safe batch-aligned resume point.
    start_index = (client.get_collection(COLLECTION).points_count // BATCH) * BATCH
    print(f"resuming from a prior run: {start_index}/{len(corpus_texts)} already indexed", flush=True)
else:
    vector_size = client.get_embedding_size(MODEL_NAME)
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
    )
    start_index = 0

try:
    # 1. Embed + index the real corpus server-side, in batches, measuring
    #    wall-clock throughput including the network round trip. Only the
    #    batches actually sent *this* process are timed - on a resumed run,
    #    corpus_index_docs_per_sec reflects the remaining tail, not the
    #    full corpus (resumed_from_docs in the output says how much that is).
    t0 = time.perf_counter()
    for i in range(start_index, len(corpus_texts), BATCH):
        chunk = corpus_texts[i:i + BATCH]
        with_retry(
            lambda chunk=chunk, i=i: client.upsert(
                collection_name=COLLECTION,
                points=[
                    models.PointStruct(
                        id=idx,
                        vector=models.Document(text=text, model=MODEL_NAME),
                        payload={"doc_id": doc_id},
                    )
                    for idx, (doc_id, text) in enumerate(chunk, start=i)
                ],
            ),
            label=f"upsert batch at {i}",
        )
        if i % (BATCH * 20) == 0:
            print(f"indexed {i + len(chunk)}/{len(corpus_texts)}", flush=True)
    elapsed = time.perf_counter() - t0
    docs_timed = len(corpus_texts) - start_index
    results["resumed_from_docs"] = start_index
    results["corpus_index_seconds"] = elapsed
    results["corpus_index_docs_per_sec"] = docs_timed / elapsed if elapsed > 0 else None

    id_to_doc_id = {i: doc_id for i, (doc_id, _) in enumerate(corpus_texts)}

    # 2. Real per-query latency (embed + search + network, end to end) +
    #    retrieval quality, same methodology as benchmark_real_data.py.
    query_latencies_ms = []
    recall_hits = []
    reciprocal_ranks = []
    for qi, qid in enumerate(test_query_ids):
        qtext = query_text_by_id[qid]
        t0 = time.perf_counter()
        response = with_retry(
            lambda qtext=qtext: client.query_points(
                collection_name=COLLECTION,
                query=models.Document(text=qtext, model=MODEL_NAME),
                limit=TOP_K,
            ),
            label=f"query {qid}",
        )
        query_latencies_ms.append((time.perf_counter() - t0) * 1000)
        if qi % 50 == 0:
            print(f"queried {qi + 1}/{len(test_query_ids)}", flush=True)

        retrieved_doc_ids = [id_to_doc_id[p.id] for p in response.points]
        relevant = relevant_by_query[qid]
        hit_ranks = [rank for rank, d in enumerate(retrieved_doc_ids, start=1) if d in relevant]
        recall_hits.append(1 if hit_ranks else 0)
        reciprocal_ranks.append(1.0 / hit_ranks[0] if hit_ranks else 0.0)

    # Only reached on a full successful run - safe to clean up now. On any
    # exception above, this is skipped and the collection is left in place
    # so the next run can resume instead of re-embedding from scratch.
    client.delete_collection(COLLECTION)

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
            "Same methodology as results_real_data.json: hit_rate@10 = "
            "fraction of queries with >=1 known-relevant doc in the top 10; "
            "mrr_at_10 = mean reciprocal rank of the first relevant hit."
        ),
    }
except Exception:
    print("crashed - collection left in place, rerun this same command to resume", flush=True)
    raise

with open("results_qdrant_cloud_inference.json", "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
