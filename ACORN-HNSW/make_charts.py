import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("results.json") as f:
    R = json.load(f)
rows, sweep = R["rows"], R["sweep"]

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "figure.dpi": 150,
    }
)
BLUE, GREEN, RED = "#2563eb", "#16a34a", "#dc2626"

sel = [r["selectivity"] * 100 for r in rows]
labels = [r["label"] for r in rows]

# ---- Chart 1: recall vs selectivity ----
fig, ax = plt.subplots(figsize=(8.2, 5))
ax.plot(
    sel,
    [r["vanilla_recall"] for r in rows],
    "o-",
    color=BLUE,
    lw=2.4,
    ms=7,
    label="Vanilla filtered HNSW",
)
ax.plot(
    sel,
    [r["acorn_recall"] for r in rows],
    "s-",
    color=GREEN,
    lw=2.4,
    ms=7,
    label="ACORN",
)
ax.plot(
    sel,
    [r["vanilla_recall_p10"] for r in rows],
    "o--",
    color=BLUE,
    lw=1.2,
    ms=4,
    alpha=0.45,
    label="Vanilla p10 (worst queries)",
)
ax.plot(
    sel,
    [r["acorn_recall_p10"] for r in rows],
    "s--",
    color=GREEN,
    lw=1.2,
    ms=4,
    alpha=0.45,
    label="ACORN p10 (worst queries)",
)
ax.set_xscale("log")
ax.set_xticks(sel)
ax.set_xticklabels(labels)
ax.set_xlabel("Filter selectivity (% of points passing the filter)")
ax.set_ylabel("Recall@10")
ax.set_ylim(0, 1.05)
ax.set_title(
    "Recall@10 across filter selectivity — 200k vectors, dim 96, hnsw_ef=64",
    fontsize=12.5,
)
ax.legend(loc="lower right", frameon=False)
fig.tight_layout()
fig.savefig("assets/recall-vs-selectivity.png", bbox_inches="tight")

# ---- Chart 2: latency vs selectivity ----
fig, ax = plt.subplots(figsize=(8.2, 5))
ax.plot(
    sel,
    [r["vanilla_ms"] for r in rows],
    "o-",
    color=BLUE,
    lw=2.4,
    ms=7,
    label="Vanilla filtered HNSW (median)",
)
ax.plot(
    sel,
    [r["acorn_ms"] for r in rows],
    "s-",
    color=GREEN,
    lw=2.4,
    ms=7,
    label="ACORN (median)",
)
ax.fill_between(
    sel,
    [r["vanilla_ms"] for r in rows],
    [r["vanilla_ms_p95"] for r in rows],
    color=BLUE,
    alpha=0.12,
)
ax.fill_between(
    sel,
    [r["acorn_ms"] for r in rows],
    [r["acorn_ms_p95"] for r in rows],
    color=GREEN,
    alpha=0.12,
)
ax.plot(
    sel,
    [r["default_ms"] for r in rows],
    "^--",
    color="#f59e0b",
    lw=2,
    ms=7,
    label="ACORN default (max_selectivity=0.4)",
)
ax.annotate(
    "auto fallback to vanilla\nabove the 40% threshold",
    xy=(50, rows[5]["default_ms"]),
    xytext=(18, 14),
    fontsize=10,
    color="#b45309",
    arrowprops={"arrowstyle": "->", "color": "#b45309"},
)
ax.set_xscale("log")
ax.set_xticks(sel)
ax.set_xticklabels(labels)
ax.set_xlabel("Filter selectivity (% of points passing the filter)")
ax.set_ylabel("Query latency (ms)")
ax.set_title(
    "Latency across filter selectivity (shaded = median to p95)", fontsize=12.5
)
ax.legend(loc="upper right", frameon=False)
fig.tight_layout()
fig.savefig("assets/latency-vs-selectivity.png", bbox_inches="tight")

# ---- Chart 3: recall/latency tradeoff at ~1% selectivity (ef sweep) ----
fig, ax = plt.subplots(figsize=(8.2, 5))
ax.plot(
    [s["vanilla_ms"] for s in sweep],
    [s["vanilla_recall"] for s in sweep],
    "o-",
    color=BLUE,
    lw=2.4,
    ms=7,
    label="Vanilla filtered HNSW",
)
ax.plot(
    [s["acorn_ms"] for s in sweep],
    [s["acorn_recall"] for s in sweep],
    "s-",
    color=GREEN,
    lw=2.4,
    ms=7,
    label="ACORN",
)
for i, s in enumerate(sweep):
    ax.annotate(
        f'ef={s["ef"]}',
        (s["vanilla_ms"], s["vanilla_recall"]),
        textcoords="offset points",
        xytext=(6, -12),
        fontsize=9,
        color=BLUE,
    )
    acorn_dy = (
        -14 if i == 0 else 6
    )  # ef=16 and ef=32 markers sit too close for both labels above
    ax.annotate(
        f'ef={s["ef"]}',
        (s["acorn_ms"], s["acorn_recall"]),
        textcoords="offset points",
        xytext=(6, acorn_dy),
        fontsize=9,
        color=GREEN,
    )
ax.set_xlabel("Median latency (ms)")
ax.set_ylabel("Recall@10")
ax.set_title(
    "The tradeoff curve at ~1% selectivity: recall vs latency while sweeping hnsw_ef",
    fontsize=12,
)
ax.legend(loc="lower right", frameon=False)
fig.tight_layout()
fig.savefig("assets/tradeoff-1pct.png", bbox_inches="tight")

print("charts saved")
