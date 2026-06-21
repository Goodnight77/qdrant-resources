"""Chart a rescore sweep: quantized recall climbing toward baseline as oversampling
rises, with p50 latency on the second axis.
Usage: python -m qfo.viz.rescore_chart [collection]   default: pq
Reads results/rescore_sweep_<coll>.json + results/results.json -> assets/rescore_<coll>.png
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qfo.config import results_path

COMPRESS = {"pq": "16x", "bq": "32x", "sq": "4x"}


def main():
    coll = sys.argv[1] if len(sys.argv) > 1 else "pq"

    with open(results_path(f"rescore_sweep_{coll}.json")) as f:
        rows = json.load(f)
    with open(results_path("results.json")) as f:
        base_recall = json.load(f)["baseline"]["recall_at_k"]

    os_pts = [r for r in rows if r["rescore"]]
    nr = next(r for r in rows if not r["rescore"])
    xs = [r["oversampling"] for r in os_pts]
    rec = [r["recall_at_k"] for r in os_pts]
    lat = [r["latency_p50_ms"] for r in os_pts]

    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    ax1.set_title(f"{coll.upper()} {COMPRESS.get(coll, '')}: rescoring buys back recall for a few ms", weight="bold")

    ax1.axhline(base_recall, ls="--", color="#4C72B0", label=f"baseline f32 recall ({base_recall:.3f})")
    ax1.plot(xs, rec, "-o", color="#55A868", lw=2, label=f"{coll.upper()} recall (rescored)")
    ax1.scatter([1], [nr["recall_at_k"]], color="#C44E52", marker="X", s=110, zorder=5,
                label=f"{coll.upper()} no rescore ({nr['recall_at_k']:.3f})")
    for x, r in zip(xs, rec):
        ax1.annotate(f"{r:.3f}", (x, r), textcoords="offset points", xytext=(0, 8), ha="center")
    ax1.set_xlabel("oversampling factor")
    ax1.set_ylabel("Recall@10")
    ax1.set_xticks(xs)
    ax1.set_ylim(min(nr["recall_at_k"], min(rec)) - 0.03, 1.01)

    ax2 = ax1.twinx()
    ax2.plot(xs, lat, "--s", color="#888", alpha=0.8, label="p50 latency")
    ax2.set_ylabel("p50 latency (ms)")

    l1, lab1 = ax1.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lab1 + lab2, loc="center right", fontsize=9)

    plt.tight_layout()
    os.makedirs("assets", exist_ok=True)
    out = f"assets/rescore_{coll}.png"
    plt.savefig(out, dpi=120)
    print("wrote", out)


if __name__ == "__main__":
    main()
