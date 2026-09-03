"""Run every experiment and regenerate every chart.

Experiments 1 and 2 (mtp.sim.graph_isolation, mtp.sim.tiered) are pure
hnswlib simulations: no Qdrant server involved, nothing to start first.

Experiment 3 (mtp.sim.per_tenant_idf) runs real Qdrant sparse-vector/IDF
code. By default it uses qdrant-client's embedded local mode (no server
needed). Set QDRANT_URL (and QDRANT_API_KEY, if needed) before running this
to reproduce it against a real deployment instead, e.g. `docker compose up
-d` for a local server, or point at Qdrant Cloud.
"""

import json

from mtp import config
from mtp.sim import graph_isolation, per_tenant_idf, tiered
from mtp.viz import charts


def main():
    print("=" * 70)
    print("Experiment 1: payload partitioning vs. dedicated (is_tenant) graphs")
    print("=" * 70)
    r1 = graph_isolation.run()
    with open(config.results_path("exp1_graph_isolation.json"), "w") as f:
        json.dump(r1, f, indent=2)

    print()
    print("=" * 70)
    print("Experiment 2: tiered multitenancy (whale + long tail)")
    print("=" * 70)
    r2 = tiered.run()
    with open(config.results_path("exp2_tiered.json"), "w") as f:
        json.dump(r2, f, indent=2)

    print()
    print("=" * 70)
    print("Experiment 3: per-tenant IDF statistics")
    print("=" * 70)
    r3 = per_tenant_idf.run()
    with open(config.results_path("exp3_per_tenant_idf.json"), "w") as f:
        json.dump(r3, f, indent=2)

    print()
    print("rendering charts ...")
    charts.make_all()


if __name__ == "__main__":
    main()
