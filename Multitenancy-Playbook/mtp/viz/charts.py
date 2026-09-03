import json

import matplotlib.pyplot as plt
import numpy as np

from mtp import config
from mtp.viz.style import DEDICATED_COLOR, INK_MUTED, SHARED_COLOR, apply_style, style_axes


def _load(name):
    with open(config.results_path(name)) as f:
        return json.load(f)


def chart_exp1_latency():
    data = _load("exp1_graph_isolation.json")
    rows = data["rows"]
    x = [r["n_tenants"] for r in rows]
    shared = [r["shared_latency_ms"] for r in rows]
    dedicated = [r["dedicated_latency_ms"] for r in rows]

    apply_style()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x, shared, marker="o", color=SHARED_COLOR, linewidth=2, label="one shared graph + tenant filter")
    ax.plot(x, dedicated, marker="o", color=DEDICATED_COLOR, linewidth=2, label="dedicated per-tenant graph")
    ax.set_xscale("log")
    ax.set_yscale("log")
    vectors_label = f"{data['n_total']:,} vectors total" if "n_total" in data else "real embeddings"
    ax.set_xlabel(f"tenants sharing the collection ({vectors_label})")
    ax.set_ylabel("mean query latency (ms, log scale)")
    ax.set_title("Payload partitioning vs. dedicated tenant graphs")
    style_axes(ax)
    ax.legend(loc="upper left")
    last = rows[-1]
    ax.annotate(
        f"{last['shared_latency_ms']/last['dedicated_latency_ms']:.0f}x slower\nat {last['selectivity_pct']:.2f}% selectivity",
        xy=(x[-1], shared[-1]), xytext=(-150, -60), textcoords="offset points",
        fontsize=9, color=INK_MUTED, ha="center",
    )
    fig.tight_layout()
    fig.savefig(config.asset_path("exp1_latency.png"), dpi=160)
    plt.close(fig)


def chart_exp1_recall():
    data = _load("exp1_graph_isolation.json")
    rows = data["rows"]
    x = [r["n_tenants"] for r in rows]
    shared = [r["shared_recall_at_10"] for r in rows]
    dedicated = [r["dedicated_recall_at_10"] for r in rows]

    apply_style()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    width = 0.35
    idx = np.arange(len(x))
    ax.bar(idx - width / 2, shared, width, color=SHARED_COLOR, label="shared graph + filter")
    ax.bar(idx + width / 2, dedicated, width, color=DEDICATED_COLOR, label="dedicated graph")
    ax.set_xticks(idx)
    ax.set_xticklabels([str(v) for v in x])
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("tenants sharing the collection")
    ax.set_ylabel("recall@10")
    ax.set_title("Recall holds either way: the cost is latency, not accuracy")
    style_axes(ax)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(config.asset_path("exp1_recall.png"), dpi=160)
    plt.close(fig)


def chart_exp2_tiered():
    data = _load("exp2_tiered.json")
    minnow = data["minnow_summary"]
    whale = data["whale_summary"]

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))

    for ax, group, title in [(axes[0], minnow, "small tenants (long tail)"), (axes[1], whale, "the whale tenant")]:
        vals = [group["naive_latency_ms"], group["tiered_latency_ms"]]
        colors = [SHARED_COLOR, DEDICATED_COLOR]
        bars = ax.bar(["naive\n(one shared graph)", "tiered\n(dedicated + fallback)"], vals, color=colors, width=0.55)
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.2f} ms", xy=(b.get_x() + b.get_width() / 2, v), xytext=(0, 4),
                        textcoords="offset points", ha="center", fontsize=10, color="#0b0b0b")
        ax.set_title(title)
        ax.set_ylabel("mean query latency (ms)")
        style_axes(ax)

    fig.suptitle(
        f"Whale is {data['whale_share_pct']:.0f}% of the collection: tiering is a big win for the long tail",
        fontsize=11, y=1.03,
    )
    fig.tight_layout()
    fig.savefig(config.asset_path("exp2_tiered_latency.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def chart_exp3_real_flagship():
    data = _load("exp3_per_tenant_idf.json")["real_flagship"]
    categories = list(data["per_category"].keys())
    global_scores = [data["per_category"][c]["global_ndcg_mean"] for c in categories]
    scoped_scores = [data["per_category"][c]["scoped_ndcg_mean"] for c in categories]

    apply_style()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    idx = np.arange(len(categories))
    width = 0.35
    ax.bar(idx - width / 2, global_scores, width, color=SHARED_COLOR, label="global IDF (default)")
    ax.bar(idx + width / 2, scoped_scores, width, color=DEDICATED_COLOR, label="per-tenant IDF (idf.corpus)")
    ax.set_xticks(idx)
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("NDCG@10")
    ax.set_title("Real AG News: per-tenant IDF vs. global IDF, by category")
    style_axes(ax)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(config.asset_path("exp3_real_flagship.png"), dpi=160)
    plt.close(fig)


def chart_exp3_real_sweep():
    rows = _load("exp3_per_tenant_idf.json")["real_category_sweep"]
    x = [r["n_tenants"] for r in rows]
    global_scores = [r["global_ndcg_mean"] for r in rows]
    scoped_scores = [r["scoped_ndcg_mean"] for r in rows]

    apply_style()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x, global_scores, marker="o", color=SHARED_COLOR, linewidth=2, label="global IDF (default)")
    ax.plot(x, scoped_scores, marker="o", color=DEDICATED_COLOR, linewidth=2, label="per-tenant IDF (idf.corpus)")
    ax.set_ylim(0, 1.08)
    ax.set_xticks(x)
    ax.set_xlabel("real AG News categories sharing the collection")
    ax.set_ylabel("mean NDCG@10")
    ax.set_title("Real data: the IDF gap doesn't need scale to show up")
    style_axes(ax)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(config.asset_path("exp3_real_sweep.png"), dpi=160)
    plt.close(fig)


def chart_exp3_synthetic_worst_case():
    data = _load("exp3_per_tenant_idf.json")["synthetic_flagship"]
    tenants = list(data["per_tenant"].keys())
    global_scores = [data["per_tenant"][t]["global_ndcg_mean"] for t in tenants]
    scoped_scores = [data["per_tenant"][t]["scoped_ndcg_mean"] for t in tenants]

    apply_style()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    idx = np.arange(len(tenants))
    width = 0.35
    ax.bar(idx - width / 2, global_scores, width, color=SHARED_COLOR, label="global IDF (default)")
    ax.bar(idx + width / 2, scoped_scores, width, color=DEDICATED_COLOR, label="per-tenant IDF (idf.corpus)")
    ax.set_xticks(idx)
    ax.set_xticklabels(tenants)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("NDCG@10")
    ax.set_title("Constructed worst case: the mechanism at its most visible")
    style_axes(ax)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(config.asset_path("exp3_synthetic_worst_case.png"), dpi=160)
    plt.close(fig)


def make_all():
    chart_exp1_latency()
    chart_exp1_recall()
    chart_exp2_tiered()
    chart_exp3_real_flagship()
    chart_exp3_real_sweep()
    chart_exp3_synthetic_worst_case()
    print("charts written to", config.ASSETS_DIR)


if __name__ == "__main__":
    make_all()
