"""Renders assets/cloud-embedding-qdrant/cpu-vs-gpu-comparison.png from
results_real_data.json (CPU) and results_real_data_gpu.json (GPU).
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

HERE = Path(__file__).parent
ASSETS = HERE.parent / "assets" / "cloud-embedding-qdrant"

CPU = "#2a78d6"
GPU = "#eb6834"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

cpu = json.loads((HERE / "results_real_data.json").read_text())
gpu = json.loads((HERE / "results_real_data_gpu.json").read_text())

plt.rcParams.update({
    "font.family": "sans-serif",
    "text.color": INK,
    "axes.edgecolor": GRID,
    "axes.labelcolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
fig.suptitle(
    "BAAI/bge-small-en-v1.5 on real data (BeIR/nfcorpus, 3,633 docs, 323 queries)\nCPU (1 core) vs Modal T4 GPU",
    fontsize=11, color=INK, y=0.99,
)

# Panel 1: indexing throughput (log scale - ~104x gap)
labels = ["CPU\n(1 core)", "GPU\n(T4)"]
throughput = [cpu["corpus_index_docs_per_sec"], gpu["corpus_index_docs_per_sec"]]
bars1 = ax1.bar(labels, throughput, color=[CPU, GPU], width=0.55)
ax1.set_yscale("log")
ax1.set_ylabel("docs/sec indexed (log scale)")
ax1.set_title("Corpus indexing throughput", fontsize=10, color=INK)
ax1.yaxis.set_major_formatter(mticker.ScalarFormatter())
ax1.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
ax1.set_axisbelow(True)
for spine in ("top", "right"):
    ax1.spines[spine].set_visible(False)
for bar, val in zip(bars1, throughput):
    ax1.annotate(f"{val:.1f}", (bar.get_x() + bar.get_width() / 2, val),
                 textcoords="offset points", xytext=(0, 5), ha="center",
                 fontsize=9, color=INK)
speedup = throughput[1] / throughput[0]
ax1.annotate(f"{speedup:.0f}x", xy=(0.5, 0.92), xycoords="axes fraction",
             ha="center", fontsize=12, color=INK, fontweight="bold")

# Panel 2: real per-query latency, p50 vs p95
x = range(2)
width = 0.35
p50 = [cpu["real_query_latency_ms"]["p50"], gpu["real_query_latency_ms"]["p50"]]
p95 = [cpu["real_query_latency_ms"]["p95"], gpu["real_query_latency_ms"]["p95"]]
bars_p50 = ax2.bar([i - width / 2 for i in x], p50, width, label="p50", color=[CPU, GPU])
bars_p95 = ax2.bar([i + width / 2 for i in x], p95, width, label="p95", color=[CPU, GPU], alpha=0.55)
ax2.set_xticks(list(x))
ax2.set_xticklabels(labels)
ax2.set_ylabel("query latency, ms")
ax2.set_title("Real per-query latency (323 distinct queries)", fontsize=10, color=INK)
ax2.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
ax2.set_axisbelow(True)
for spine in ("top", "right"):
    ax2.spines[spine].set_visible(False)
for bars in (bars_p50, bars_p95):
    for bar in bars:
        h = bar.get_height()
        ax2.annotate(f"{h:.1f}", (bar.get_x() + bar.get_width() / 2, h),
                     textcoords="offset points", xytext=(0, 4), ha="center",
                     fontsize=8, color=INK)

from matplotlib.patches import Patch
ax2.legend(
    handles=[Patch(facecolor=MUTED, alpha=1.0, label="p50"), Patch(facecolor=MUTED, alpha=0.55, label="p95")],
    frameon=False, loc="upper right", fontsize=9,
)

fig.tight_layout(rect=[0, 0, 1, 0.9])
out = ASSETS / "cpu-vs-gpu-comparison.png"
fig.savefig(out, dpi=180, facecolor=SURFACE)
print(f"wrote {out}")
