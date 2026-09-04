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

The actual benchmark logic (resumable indexing, retries, retrieval quality)
lives in cloud_inference_lib.py, shared with cloud_inference_benchmark_modal.py
(the same benchmark run detached on Modal instead of your local machine).

Requires a Qdrant Cloud cluster with built-in inference enabled and
BAAI/bge-small-en-v1.5 in its supported model list - check the Cloud console
/ https://qdrant.tech/documentation/cloud/inference/ (the list changes over
time).

Setup:
    export QDRANT_URL="https://<cluster>.cloud.qdrant.io"
    export QDRANT_API_KEY="..."
    pip install datasets qdrant-client fastembed
    python benchmark_qdrant_cloud_inference.py

Cost note: server-side inference is billed by your Qdrant Cloud plan; this
run embeds ~3,633 docs + 323 queries once.
"""

import json
import os

from cloud_inference_lib import run_benchmark

url = os.environ["QDRANT_URL"]
api_key = os.environ["QDRANT_API_KEY"]

results = run_benchmark(
    url, api_key, "Qdrant Cloud built-in inference (server-side embed)"
)

with open("results_qdrant_cloud_inference.json", "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
