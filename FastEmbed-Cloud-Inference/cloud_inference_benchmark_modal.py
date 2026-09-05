"""Runs benchmark_qdrant_cloud_inference.py's Pattern C benchmark (Qdrant
Cloud built-in inference) from a Modal container instead of your local
machine.

Why: the local run kept dying on a residential connection - power outages,
SSL drops, DNS blips over the ~15-20 minutes indexing takes. Modal's
container sits in a stable cloud network, so this survives all of that, and
runs fully detached like gpu_benchmark_modal.py (keeps going even if your
laptop is off).

The benchmark logic itself lives in cloud_inference_lib.py, shared with
benchmark_qdrant_cloud_inference.py (the local runner) - this file is just
the Modal plumbing (image, secret, spawn/fetch).

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

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("datasets", "qdrant-client", "fastembed", "huggingface_hub")
    .add_local_python_source("cloud_inference_lib")
)


@app.function(
    image=image,
    timeout=1800,
    secrets=[
        modal.Secret.from_dict(
            {
                "QDRANT_URL": os.environ["QDRANT_URL"],
                "QDRANT_API_KEY": os.environ["QDRANT_API_KEY"],
            }
        )
    ],
)
def run_benchmark_modal() -> dict:
    from cloud_inference_lib import run_benchmark

    return run_benchmark(
        os.environ["QDRANT_URL"],
        os.environ["QDRANT_API_KEY"],
        "Qdrant Cloud built-in inference (server-side embed, client on Modal)",
    )


@app.local_entrypoint()
def main():
    call = run_benchmark_modal.spawn()
    print(f"Spawned detached run. Function call id: {call.object_id}")
    print("This keeps running on Modal's infrastructure independent of this process.")
    print("Fetch results once it's done with:")
    print(
        f"  modal run cloud_inference_benchmark_modal.py::fetch --call-id {call.object_id}"
    )


@app.local_entrypoint()
def fetch(call_id: str):
    call = modal.FunctionCall.from_id(call_id)
    results = call.get(timeout=1800)
    print(json.dumps(results, indent=2))
    with open("results_qdrant_cloud_inference.json", "w") as f:
        json.dump(results, f, indent=2)
