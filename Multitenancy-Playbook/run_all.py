"""Run all experiments and regenerate charts.

Experiments 1-2 require a Qdrant server (localhost:6333 by default).
Experiment 2 also requires cluster mode for custom shards. Experiment 3 uses local memory
unless QDRANT_URL is set; set it to the Docker endpoint to use the server.
"""

import json
from pathlib import Path

from mtp import config, qdrant_graph_isolation, qdrant_tiered
from mtp.sim import bm25_comparison, per_tenant_idf
from mtp.viz import charts


def main():
    print("=" * 70)
    print("Experiment 1: Qdrant indexed shared vs. tenant-optimized search")
    print("=" * 70)
    qdrant_graph_isolation.run(
        Path(config.results_path("exp1_qdrant_graph_isolation.json"))
    )

    print()
    print("=" * 70)
    print("Experiment 2: tiered multitenancy (whale + long tail)")
    print("=" * 70)
    qdrant_tiered.run(Path(config.results_path("exp2_qdrant_tiered.json")))

    print()
    print("=" * 70)
    print("Experiment 3: per-tenant IDF statistics")
    print("=" * 70)
    r3 = per_tenant_idf.run()
    with open(config.results_path("exp3_per_tenant_idf.json"), "w") as f:
        json.dump(r3, f, indent=2)

    print("Experiment 3 extension: raw-count vs. BM25 document weights")
    comparison = bm25_comparison.run()
    with open(config.results_path("exp3_bm25_comparison.json"), "w") as f:
        json.dump(comparison, f, indent=2, allow_nan=False)

    print()
    print("rendering charts ...")
    charts.make_all()


if __name__ == "__main__":
    main()
