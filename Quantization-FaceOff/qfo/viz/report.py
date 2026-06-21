"""Build the faceoff dashboard (one 2x2 PNG) from results.json + computed memory.

Memory here is the RAM-resident quantized-vector size (exact math) — the thing
quantization actually shrinks. Measured per-config RAM is in Qdrant's dashboard memory panel.
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qfo.config import DATASET_N, N_QUERY, DIM, results_path

OUT = "assets"
LABEL = {"baseline": "f32", "sq": "SQ int8", "pq": "PQ x16", "bq": "BQ x32"}
COMPRESS = {"baseline": 1, "sq": 4, "pq": 16, "bq": 32}


def main():
    os.makedirs(OUT, exist_ok=True)
    with open(results_path("results.json")) as f:
        R = json.load(f)

    n_base = DATASET_N - N_QUERY
    mem_mb = {
        k: n_base * DIM * 4 / 1e6 / COMPRESS[k] for k in R
    }  # resident vector bytes
    names = list(R.keys())
    labels = [LABEL.get(n, n) for n in names]
    palette = ["#4C72B0", "#DD8452", "#55A868", "#8172B3"]
    colors = [palette[i % len(palette)] for i in range(len(names))]

    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(
        f"Qdrant Quantization Face-Off — dbpedia 1536-d, {n_base:,} vectors",
        fontsize=14,
        weight="bold",
    )

    # recall@10
    ax[0, 0].bar(labels, [R[n]["recall_at_k"] for n in names], color=colors)
    ax[0, 0].set_title("Recall@10 (higher = better)")
    ax[0, 0].set_ylim(0, 1.05)
    for i, n in enumerate(names):
        ax[0, 0].text(
            i, R[n]["recall_at_k"] + 0.01, f"{R[n]['recall_at_k']:.3f}", ha="center"
        )

    # latency percentiles
    x = range(len(names))
    w = 0.25
    for j, (p, c) in enumerate(
        [("p50", "#8172B3"), ("p95", "#CCB974"), ("p99", "#C44E52")]
    ):
        ax[0, 1].bar(
            [i + (j - 1) * w for i in x],
            [R[n][f"latency_{p}_ms"] for n in names],
            w,
            label=p,
            color=c,
        )
    ax[0, 1].set_title("Latency (ms, lower = better)")
    ax[0, 1].set_xticks(list(x))
    ax[0, 1].set_xticklabels(labels)
    ax[0, 1].legend()

    # memory (resident quantized vectors)
    ax[1, 0].bar(labels, [mem_mb[n] for n in names], color=colors)
    ax[1, 0].set_title("Vectors in RAM (MB, computed — lower = better)")
    ax[1, 0].set_ylim(
        0, max(mem_mb.values()) * 1.25
    )  # headroom so labels clear the title
    for i, n in enumerate(names):
        ax[1, 0].text(
            i,
            mem_mb[n] + max(mem_mb.values()) * 0.02,
            f"{mem_mb[n]:.0f} MB ({COMPRESS[n]}x)",
            ha="center",
        )

    # tradeoff: recall vs p50 latency, bubble size = memory
    for i, n in enumerate(names):
        ax[1, 1].scatter(
            R[n]["latency_p50_ms"],
            R[n]["recall_at_k"],
            s=mem_mb[n] * 2,
            color=colors[i],
            alpha=0.7,
            label=labels[i],
        )
        ax[1, 1].annotate(
            labels[i],
            (R[n]["latency_p50_ms"], R[n]["recall_at_k"]),
            textcoords="offset points",
            xytext=(8, 0),
        )
    ax[1, 1].set_title("Tradeoff: recall vs p50 latency (bubble = RAM)")
    ax[1, 1].set_xlabel("p50 latency (ms)")
    ax[1, 1].set_ylabel("recall@10")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    path = os.path.join(OUT, "faceoff.png")
    plt.savefig(path, dpi=120)
    print("wrote", path)
    for n in names:
        print(
            f"{LABEL.get(n, n):8} recall={R[n]['recall_at_k']:.3f}  p50={R[n]['latency_p50_ms']:.1f}ms  ram={mem_mb[n]:.0f}MB ({COMPRESS[n]}x)"
        )


if __name__ == "__main__":
    main()
