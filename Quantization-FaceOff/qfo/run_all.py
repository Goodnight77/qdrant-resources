"""Run the whole faceoff end to end with one command:

    uv run python -m qfo.run_all

Each step runs as `python -m qfo.<module>` in a fresh process (clean state, mirrors the
manual commands). Stops at the first failure.

Note: ingest + add_bq rebuild the collections from scratch (drop + re-upload).
"""

import subprocess
import sys

# (module, args, description)
STEPS = [
    ("qfo.data.dbpedia", [], "build + cache dataset"),
    ("qfo.pipeline.ingest", [], "create baseline/sq/pq + upload payload"),
    ("qfo.pipeline.add_bq", [], "create bq (binary) collection"),
    ("qfo.data.ground_truth", [], "exact-NN ground truth"),
    ("qfo.bench.eval", [], "recall + latency -> results/ + W&B"),
    ("qfo.bench.rescore_sweep", ["pq"], "PQ oversampling/rescore sweep"),
    ("qfo.bench.rescore_sweep", ["bq"], "BQ sweep"),
    ("qfo.bench.throughput", [], "QPS under concurrency"),
    ("qfo.viz.report", [], "dashboard chart -> assets/"),
    ("qfo.viz.rescore_chart", ["pq"], "PQ sweep chart"),
    ("qfo.viz.rescore_chart", ["bq"], "BQ sweep chart"),
]


def main():
    for i, (mod, args, desc) in enumerate(STEPS, 1):
        print(
            f"\n=== [{i}/{len(STEPS)}] {mod} {' '.join(args)} - {desc} ===", flush=True
        )
        subprocess.run([sys.executable, "-m", mod, *args], check=True)
    print("\nfaceoff complete. charts in assets/, metrics in results/ + W&B.")


if __name__ == "__main__":
    main()
