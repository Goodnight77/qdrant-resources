"""GPU counterpart to benchmark_real_data.py - same corpus, same model, same
document lengths, run on a Modal T4 GPU instead of a local CPU core.

NOTE ON METHODOLOGY: this uses `sentence-transformers` + `torch` on CUDA,
not FastEmbed's ONNX runtime. FastEmbed's own GPU path (`fastembed-gpu` +
onnxruntime-gpu) needs CUDA/cuDNN versions matched to the exact onnxruntime
build, which is fragile to pin inside a container without hands-on
iteration. sentence-transformers on stock CUDA-enabled torch wheels is the
well-trodden path for running this exact model on GPU, and it's the same
public model weights (`BAAI/bge-small-en-v1.5`), so this is still a fair
CPU-runtime-vs-GPU-runtime comparison for the same architecture - just be
aware the runtime differs, not only the hardware, when you read the numbers
side by side with benchmark_real_data.py's.

Latency/throughput above measures embedding compute only (no Qdrant round
trip in the timed section) - that isolates the actual bottleneck the CPU run
identified (embedding compute), since local Qdrant search overhead over
~3.6K vectors was negligible next to embedding time in the CPU run.

Retrieval quality (hit-rate@10, MRR@10) is computed separately, off the
timed critical path: the GPU-encoded corpus/query vectors are indexed into
an in-memory Qdrant and searched, same methodology as
benchmark_real_data.py, so hit_rate_at_10/mrr_at_10 here are directly
comparable to results_real_data.json's.

Runs detached: launching it hands the job to Modal's infrastructure and
returns immediately, so it keeps running even if your terminal, laptop, or
SSH session disconnects. Fetch the result later with the printed call id.

Setup (on a machine/session where gRPC isn't blocked by an egress proxy):
    pip install modal
    modal token set --token-id <id> --token-secret <secret>   # or env vars
    modal run gpu_benchmark_modal.py::main
    # prints a call id immediately and returns - the job keeps running
    # on Modal even after this command exits.
    modal run gpu_benchmark_modal.py::fetch --call-id <the printed id>
    # run this once the job has had time to finish; if it's still running,
    # this call blocks (up to 900s) waiting for it, then writes
    # results_real_data_gpu.json.

Cost note: allocates a T4 GPU for the duration of the run (a few minutes
including model download) - expect well under $1 on Modal's per-second T4
pricing, but confirm current pricing before running if that matters to you.
"""

import json
import statistics
import time

import modal

app = modal.App("fastembed-gpu-benchmark")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch", "sentence-transformers", "datasets", "huggingface_hub", "qdrant-client"
)

MODEL_NAME = "BAAI/bge-small-en-v1.5"
TOP_K = 10


@app.function(image=image, gpu="T4", timeout=900)
def run_benchmark() -> dict:
    from collections import defaultdict

    from datasets import load_dataset
    from qdrant_client import QdrantClient, models
    from sentence_transformers import SentenceTransformer

    t0 = time.perf_counter()
    model = SentenceTransformer(MODEL_NAME, device="cuda")
    model_load_seconds = time.perf_counter() - t0

    corpus = load_dataset("BeIR/nfcorpus", "corpus", split="corpus")
    queries = load_dataset("BeIR/nfcorpus", "queries", split="queries")
    qrels = load_dataset("BeIR/nfcorpus-qrels", split="test")

    query_text_by_id = {q["_id"]: q["text"] for q in queries}
    relevant_by_query = defaultdict(set)
    for row in qrels:
        relevant_by_query[row["query-id"]].add(row["corpus-id"])
    test_query_ids = list(relevant_by_query.keys())

    corpus_ids = [row["_id"] for row in corpus]
    corpus_texts = [f"{row['title']} {row['text']}".strip() for row in corpus]

    # warm-up (first CUDA call pays kernel/context init cost)
    model.encode(corpus_texts[:8], batch_size=8)

    # 1. Full real corpus, batched - the number that mirrors
    #    corpus_index_docs_per_sec in results_real_data.json
    t0 = time.perf_counter()
    corpus_vectors = model.encode(corpus_texts, batch_size=64, show_progress_bar=False)
    corpus_elapsed = time.perf_counter() - t0

    # 2. Real per-query latency, one at a time (323 distinct real queries),
    #    mirrors real_query_latency_ms in results_real_data.json. Embedding
    #    time only - the retrieval-quality pass below reuses these vectors
    #    but isn't part of the timed latency.
    query_latencies_ms = []
    query_vectors = {}
    for qid in test_query_ids:
        text = query_text_by_id[qid]
        t0 = time.perf_counter()
        vec = model.encode([text])[0]
        query_latencies_ms.append((time.perf_counter() - t0) * 1000)
        query_vectors[qid] = vec

    query_latencies_ms.sort()
    n = len(query_latencies_ms)

    # 3. Retrieval quality: index the already-computed GPU embeddings into
    #    an in-memory Qdrant and search with the already-computed query
    #    vectors - same hit-rate@10 / MRR@10 methodology as
    #    benchmark_real_data.py, so it's directly comparable to
    #    results_real_data.json. Not timed - off the critical path above.
    qclient = QdrantClient(":memory:")
    qclient.create_collection(
        collection_name="nfcorpus",
        vectors_config=models.VectorParams(
            size=corpus_vectors.shape[1], distance=models.Distance.COSINE
        ),
    )
    qclient.upload_points(
        collection_name="nfcorpus",
        points=[
            models.PointStruct(id=i, vector=vec.tolist(), payload={"doc_id": doc_id})
            for i, (doc_id, vec) in enumerate(zip(corpus_ids, corpus_vectors))
        ],
    )

    recall_hits = []
    reciprocal_ranks = []
    for qid in test_query_ids:
        response = qclient.query_points(
            collection_name="nfcorpus",
            query=query_vectors[qid].tolist(),
            limit=TOP_K,
        )
        retrieved_doc_ids = [corpus_ids[p.id] for p in response.points]
        relevant = relevant_by_query[qid]
        hit_ranks = [
            rank for rank, d in enumerate(retrieved_doc_ids, start=1) if d in relevant
        ]
        recall_hits.append(1 if hit_ranks else 0)
        reciprocal_ranks.append(1.0 / hit_ranks[0] if hit_ranks else 0.0)

    return {
        "runtime": "sentence-transformers (torch/CUDA), T4 GPU",
        "model": MODEL_NAME,
        "corpus_size": len(corpus_texts),
        "test_query_count": n,
        "model_load_seconds": model_load_seconds,
        "corpus_index_seconds": corpus_elapsed,
        "corpus_index_docs_per_sec": len(corpus_texts) / corpus_elapsed,
        "real_query_latency_ms": {
            "mean": statistics.mean(query_latencies_ms),
            "p50": query_latencies_ms[n // 2],
            "p95": query_latencies_ms[int(n * 0.95)],
            "min": query_latencies_ms[0],
            "max": query_latencies_ms[-1],
        },
        "retrieval_quality": {
            "hit_rate_at_10": statistics.mean(recall_hits),
            "mrr_at_10": statistics.mean(reciprocal_ranks),
            "note": (
                "Same methodology as results_real_data.json: hit_rate@10 = "
                "fraction of queries with >=1 known-relevant doc in the top "
                "10; mrr_at_10 = mean reciprocal rank of the first relevant "
                "hit. Computed from the same GPU-encoded vectors used above "
                "against an in-memory Qdrant index - not timed as part of "
                "the latency/throughput numbers."
            ),
        },
    }


@app.local_entrypoint()
def main():
    call = run_benchmark.spawn()
    print(f"Spawned detached run. Function call id: {call.object_id}")
    print("This keeps running on Modal's infrastructure independent of this process.")
    print("Fetch results once it's done with:")
    print(f"  modal run gpu_benchmark_modal.py::fetch --call-id {call.object_id}")
    with open("call_id.txt", "w") as f:
        f.write(call.object_id)


@app.local_entrypoint()
def fetch(call_id: str):
    call = modal.FunctionCall.from_id(call_id)
    results = call.get(timeout=900)
    print(json.dumps(results, indent=2))
    with open("results_real_data_gpu.json", "w") as f:
        json.dump(results, f, indent=2)
