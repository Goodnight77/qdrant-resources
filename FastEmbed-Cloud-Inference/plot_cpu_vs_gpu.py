"""Renders assets/cloud-embedding-qdrant/cpu-vs-gpu-comparison.png from the
three benchmark results in this folder: results_real_data.json (Pattern A,
local FastEmbed CPU), results_real_data_gpu.json (local GPU via Modal T4),
and results_qdrant_cloud_inference.json (Pattern C, Qdrant Cloud built-in
inference), if that last one exists.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch

HERE = Path(__file__).parent
ASSETS = HERE.parent / "assets" / "cloud-embedding-qdrant"

CPU = "#2a78d6"
GPU = "#eb6834"
CLOUD = "#1baf7a"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

cpu = json.loads((HERE / "results_real_data.json").read_text())
gpu = json.loads((HERE / "results_real_data_gpu.json").read_text())
cloud_path = HERE / "results_qdrant_cloud_inference.json"
cloud = json.loads(cloud_path.read_text()) if cloud_path.exists() else None

runs = [("CPU\n(1 core)", cpu, CPU), ("GPU\n(T4)", gpu, GPU)]
if cloud:
    runs.append(("Cloud\n(built-in)", cloud, CLOUD))
labels = [r[0] for r in runs]
colors = [r[2] for r in runs]
n_runs = len(runs)

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

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.2))
fig.suptitle(
    "BAAI/bge-small-en-v1.5 on real data (BeIR/nfcorpus, 3,633 docs, 323 queries)",
    fontsize=11, color=INK, y=0.99,
)
run_names = {"CPU\n(1 core)": "CPU (1 core)", "GPU\n(T4)": "GPU (T4)", "Cloud\n(built-in)": "Qdrant Cloud (built-in inference)"}
fig.legend(
    handles=[Patch(facecolor=c, label=run_names[lbl]) for lbl, _, c in runs],
    loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=n_runs, frameon=False, fontsize=9.5,
)

# Panel 1: indexing throughput (log scale - CPU vs GPU is ~104x)
throughput = [r[1]["corpus_index_docs_per_sec"] for r in runs]
bars1 = ax1.bar(labels, throughput, color=colors, width=0.55, zorder=3)
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
ax1.annotate(f"GPU {speedup:.0f}x CPU", xy=(0.83, 0.55), xycoords="axes fraction",
             ha="center", fontsize=11, color=INK, fontweight="bold")

# Panel 2: real per-query latency, p50 (solid) vs p95 (faded) per runtime -
# same hue per runtime as panel 1, shade is the only p50/p95 signal, called
# out with a direct caption instead of a same-color legend swatch.
x = range(n_runs)
width = 0.35
p50 = [r[1]["real_query_latency_ms"]["p50"] for r in runs]
p95 = [r[1]["real_query_latency_ms"]["p95"] for r in runs]
bars_p50 = ax2.bar([i - width / 2 for i in x], p50, width, color=colors, zorder=3)
bars_p95 = ax2.bar([i + width / 2 for i in x], p95, width, color=colors, alpha=0.5, zorder=3)
ax2.set_xticks(list(x))
ax2.set_xticklabels(labels)
ax2.set_ylabel("query latency, ms")
ax2.set_title("Real per-query latency (323 distinct queries)\nsolid = p50 · faded = p95",
              fontsize=10, color=INK)
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

fig.tight_layout(rect=[0, 0, 1, 0.92])
out = ASSETS / "cpu-vs-gpu-comparison.png"
fig.savefig(out, dpi=180, facecolor=SURFACE)
print(f"wrote {out}")

# Panel 3 (separate figure): retrieval quality parity check - same model,
# does hardware/runtime change ranking quality? Only drawn once all runs
# that computed it are present (GPU's retrieval_quality is a later addition;
# older results_real_data_gpu.json files won't have it).
quality_runs = [(lbl, r, c) for lbl, r, c in runs if "retrieval_quality" in r]
if len(quality_runs) >= 2:
    fig2, ax3 = plt.subplots(figsize=(5.5, 4.2))
    qlabels = [q[0] for q in quality_runs]
    qcolors = [q[2] for q in quality_runs]
    hit_rate = [q[1]["retrieval_quality"]["hit_rate_at_10"] * 100 for q in quality_runs]
    mrr = [q[1]["retrieval_quality"]["mrr_at_10"] * 100 for q in quality_runs]
    xq = range(len(quality_runs))
    w = 0.35
    b1 = ax3.bar([i - w / 2 for i in xq], hit_rate, w, color=qcolors, zorder=3)
    b2 = ax3.bar([i + w / 2 for i in xq], mrr, w, color=qcolors, alpha=0.5, zorder=3)
    ax3.set_xticks(list(xq))
    ax3.set_xticklabels(qlabels)
    ax3.set_ylabel("% (hit-rate@10, MRR@10 x100)")
    ax3.set_title("Retrieval quality: same across hardware/runtime\nsolid = hit-rate@10 · faded = MRR@10 x100",
                  fontsize=10, color=INK)
    fig2.legend(
        handles=[Patch(facecolor=c, label=run_names[lbl]) for lbl, _, c in quality_runs],
        loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=len(quality_runs), frameon=False, fontsize=9,
    )
    ax3.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax3.set_axisbelow(True)
    for spine in ("top", "right"):
        ax3.spines[spine].set_visible(False)
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax3.annotate(f"{h:.1f}", (bar.get_x() + bar.get_width() / 2, h),
                         textcoords="offset points", xytext=(0, 4), ha="center",
                         fontsize=8, color=INK)
    fig2.tight_layout(rect=[0, 0, 1, 0.92])
    out2 = ASSETS / "retrieval-quality-comparison.png"
    fig2.savefig(out2, dpi=180, facecolor=SURFACE)
    print(f"wrote {out2}")
