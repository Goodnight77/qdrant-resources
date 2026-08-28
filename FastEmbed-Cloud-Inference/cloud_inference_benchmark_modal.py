"""Runs benchmark_qdrant_cloud_inference.py's Pattern C benchmark (Qdrant
Cloud built-in inference) from a Modal container instead of your local
machine.

Why: the local run kept dying on a residential connection - power outages,
SSL drops, DNS blips over the ~15-20 minutes indexing takes. Modal's
container sits in a stable cloud network, so this survives all of that, and
runs fully detached like gpu_benchmark_modal.py (keeps going even if your
laptop is off).

Methodology note: this changes what corpus_index_docs_per_sec and
real_query_latency_ms actually measure - it's now Modal's datacenter to your
Qdrant Cloud cluster, not your own machine to Qdrant Cloud. That's arguably
more representative of a real deployment anyway (a production app server
usually calls Qdrant Cloud from another datacenter, not a laptop on
residential wifi) - just don't read this as "your app's" exact latency
without re-checking from wherever you actually deploy.

No GPU needed - this is I/O-bound (HTTP calls to Qdrant Cloud, embedding
happens server-side there), so it's a plain CPU container, cheap.

Setup:
    pip install modal
    modal token set --token-id <id> --token-secret <secret>
    export QDRANT_URL="https://<cluster>.cloud.qdrant.io"
    export QDRANT_API_KEY="..."
    modal run --detach cloud_inference_benchmark_modal.py::main
    # prints a call id and returns immediately - runs on Modal independent
    # of this process.
    modal run cloud_inference_benchmark_modal.py::fetch --call-id <id>
    # blocks (up to 1800s) until done, then writes
    # results_qdrant_cloud_inference.json

Resumable: same as the local script - if interrupted, rerun and it picks up
from the collection's current point count instead of re-embedding.
"""
import json
import os

import modal

app = modal.App("qdrant-cloud-inference-benchmark")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "datasets", "qdrant-client", "fastembed", "huggingface_hub"
)

MODEL_NAME = "BAAI/bge-small-en-v1.5"
TOP_K = 10
COLLECTION = "nfcorpus-cloud-bench"
MAX_RETRIES = 5
BATCH = 16


@app.function(
    image=image,
    timeout=1800,
    secrets=[modal.Secret.from_dict({
        "QDRANT_URL": os.environ["QDRANT_URL"],
        "QDRANT_API_KEY": os.environ["QDRANT_API_KEY"],
    })],
)
def run_benchmark() -> dict:
    import statistics
    import time
    from collections import defaultdict

    from datasets import load_dataset
    from qdrant_client import QdrantClient, models

    def with_retry(fn, label):
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
        "runtime": "Qdrant Cloud built-in inference (server-side embed, client on Modal)",
        "dataset": "BeIR/nfcorpus",
        "model": MODEL_NAME,
        "corpus_size": len(corpus_texts),
        "test_query_count": len(test_query_ids),
    }

    client = QdrantClient(url=url, api_key=api_key, timeout=120)

    if client.collection_exists(COLLECTION):
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

        # Only reached on full success - collection left in place on any
        # exception above so a rerun can resume.
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
                "fraction of queries with >=1 known-relevant doc in the top "
                "10; mrr_at_10 = mean reciprocal rank of the first relevant "
                "hit."
            ),
        }
    except Exception:
        print("crashed - collection left in place, rerun to resume", flush=True)
        raise

    return results


@app.local_entrypoint()
def main():
    call = run_benchmark.spawn()
    print(f"Spawned detached run. Function call id: {call.object_id}")
    print("This keeps running on Modal's infrastructure independent of this process.")
    print(f"Fetch results once it's done with:")
    print(f"  modal run cloud_inference_benchmark_modal.py::fetch --call-id {call.object_id}")


@app.local_entrypoint()
def fetch(call_id: str):
    call = modal.FunctionCall.from_id(call_id)
    results = call.get(timeout=1800)
    print(json.dumps(results, indent=2))
    with open("results_qdrant_cloud_inference.json", "w") as f:
        json.dump(results, f, indent=2)
