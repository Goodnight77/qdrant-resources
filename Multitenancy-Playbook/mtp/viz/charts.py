import json

import matplotlib.pyplot as plt
import numpy as np

from mtp import config
from mtp.viz.style import (
    DEDICATED_COLOR,
    INK_MUTED,
    SHARED_COLOR,
    apply_style,
    style_axes,
)


def _load(name):
    with open(config.results_path(name)) as f:
        return json.load(f)


def chart_exp1_latency():
    data = _load("exp1_qdrant_graph_isolation.json")
    if not data["complete"]:
        raise ValueError(
            "Finish the Qdrant experiment 1 run before rendering its charts"
        )
    rows = data["rows"]
    x = [r["n_tenants"] for r in rows]
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, scope, title in zip(
        axes,
        ["server", "client"],
        ["Server-reported processing", "Client-observed REST round trip"],
    ):
        for layout, color, label in [
            ("shared", SHARED_COLOR, "indexed shared"),
            ("tenant", DEDICATED_COLOR, "tenant-optimized"),
        ]:
            for statistic, linestyle, marker in [
                ("median", "-", "o"),
                ("p95", "--", "^"),
            ]:
                values = [r["summary"][layout][f"{scope}_{statistic}_ms"] for r in rows]
                ax.plot(
                    x,
                    values,
                    color=color,
                    linestyle=linestyle,
                    marker=marker,
                    label=f"{label}: {statistic}",
                )
        ax.set_xscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{r['n_tenants']}{'' if r['quality_comparable'] else '*'}" for r in rows]
        )
        ax.set_xlabel("tenants sharing the collection")
        ax.set_ylabel("query latency (ms)")
        if scope == "server":
            ax.set_yscale("log")
            ax.set_ylabel("query latency (ms, log scale)")
        else:
            ax.set_ylim(bottom=0)
        ax.set_title(title)
        ax.legend(fontsize=8)
        style_axes(ax)
    server_version = data["environment"]["server"]["version"]
    fig.suptitle(
        f"Qdrant {server_version}: {data['n_total']:,} vectors, {data['repeats']} warm query passes"
    )
    if any(not r["quality_comparable"] for r in rows):
        fig.text(
            0.5,
            0.01,
            "* Recall target missed: latency is not an equal-quality speed comparison.",
            ha="center",
            fontsize=9,
        )
        fig.tight_layout(rect=(0, 0.06, 1, 1))
    else:
        fig.tight_layout()
    fig.savefig(config.asset_path("exp1_latency.png"), dpi=160)
    plt.close(fig)


def chart_exp1_recall():
    data = _load("exp1_qdrant_graph_isolation.json")
    if not data["complete"]:
        raise ValueError(
            "Finish the Qdrant experiment 1 run before rendering its charts"
        )
    rows = data["rows"]
    apply_style()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    idx = np.arange(len(rows))
    for offset, layout, color, label in [
        (-0.18, "shared", SHARED_COLOR, "indexed shared"),
        (0.18, "tenant", DEDICATED_COLOR, "tenant-optimized"),
    ]:
        values = [r["summary"][layout]["recall_at_10"] for r in rows]
        ax.bar(idx + offset, values, width=0.36, color=color, label=label)
    ax.axhline(
        data["recall_target"],
        color=INK_MUTED,
        linestyle="--",
        linewidth=1,
        label=f"validation target: {data['recall_target']:.2f}",
    )
    ax.set_xticks(idx)
    ax.set_xticklabels([str(r["n_tenants"]) for r in rows])
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("tenants sharing the collection")
    ax.set_ylabel("recall@10 on evaluation queries")
    ax.set_title("Qdrant recall vs. exact cosine ground truth")
    style_axes(ax)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(config.asset_path("exp1_recall.png"), dpi=120)
    plt.close(fig)


def chart_exp2_tiered():
    data = _load("exp2_qdrant_tiered.json")
    if not data["complete"]:
        raise ValueError(
            "Finish the Qdrant experiment 2 run before rendering its chart"
        )
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7))
    groups = [data["groups"][key] for key in ["minnows", "whale"]]
    positions = np.arange(2)
    for ax, scope, title in zip(
        axes,
        ["server", "client"],
        ["Server-reported processing", "Client-observed REST round trip"],
    ):
        for offset, layout, color, label in [
            (-0.19, "shared", SHARED_COLOR, "one shared shard"),
            (0.19, "tiered", DEDICATED_COLOR, "dedicated + fallback"),
        ]:
            medians = [g["summary"][layout][f"{scope}_median_ms"] for g in groups]
            p95s = [g["summary"][layout][f"{scope}_p95_ms"] for g in groups]
            bars = ax.bar(
                positions + offset, medians, width=0.34, color=color, label=label
            )
            ax.vlines(positions + offset, medians, p95s, color=color, linewidth=2)
            ax.scatter(positions + offset, p95s, marker="_", s=150, color=color)
            for bar, value in zip(bars, medians):
                ax.annotate(
                    f"{value:.3f}" if scope == "server" else f"{value:.2f}",
                    (bar.get_x() + bar.get_width() / 2, value),
                    (0, 4),
                    textcoords="offset points",
                    ha="center",
                    fontsize=9,
                )
        ax.set_xticks(positions)
        ax.set_xticklabels(["small tenants", "whale tenant"])
        ax.set_ylabel("query latency (ms)")
        ax.set_ylim(
            0,
            max(
                g["summary"][layout][f"{scope}_p95_ms"]
                for g in groups
                for layout in ["shared", "tiered"]
            )
            * 1.32,
        )
        ax.set_title(title)
        ax.legend(fontsize=8, loc="upper left")
        style_axes(ax)
    fig.suptitle(
        f"Qdrant {data['environment']['server']['version']}: shared vs. tiered shards on one node"
    )
    footnote = "Bars: median; whisker caps: p95 (not confidence intervals). Three warm query passes."
    if any(not group["quality_comparable"] for group in groups):
        footnote += " Recall target missed; see result JSON."
    fig.text(0.5, 0.01, footnote, ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(config.asset_path("exp2_tiered_latency.png"), dpi=160)
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
    ax.bar(
        idx - width / 2,
        global_scores,
        width,
        color=SHARED_COLOR,
        label="global IDF (default)",
    )
    ax.bar(
        idx + width / 2,
        scoped_scores,
        width,
        color=DEDICATED_COLOR,
        label="per-tenant IDF (idf.corpus)",
    )
    ax.set_xticks(idx)
    ax.set_xticklabels(["sci-tech" if c == "scitech" else c for c in categories])
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
    ax.plot(
        x,
        global_scores,
        marker="o",
        color=SHARED_COLOR,
        linewidth=2,
        label="global IDF (default)",
    )
    ax.plot(
        x,
        scoped_scores,
        marker="o",
        color=DEDICATED_COLOR,
        linewidth=2,
        label="per-tenant IDF (idf.corpus)",
    )
    ax.set_ylim(0, 1.08)
    ax.set_xticks(x)
    ax.set_xlabel("real AG News categories sharing the collection")
    ax.set_ylabel("mean NDCG@10")
    ax.set_title("AG News scenarios: categories and query pairs vary")
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
    ax.bar(
        idx - width / 2,
        global_scores,
        width,
        color=SHARED_COLOR,
        label="global IDF (default)",
    )
    ax.bar(
        idx + width / 2,
        scoped_scores,
        width,
        color=DEDICATED_COLOR,
        label="per-tenant IDF (idf.corpus)",
    )
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


def chart_exp3_bm25_comparison():
    data = _load("exp3_bm25_comparison.json")["variants"]
    categories = list(data["raw_tf"]["per_category"])
    labels = ["sci-tech" if c == "scitech" else c for c in categories] + ["overall"]
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    x = np.arange(len(labels))
    for ax, mode, title in zip(
        axes, ["global", "scoped"], ["Global IDF", "Per-tenant IDF"]
    ):
        field = f"{mode}_ndcg_mean"
        for offset, variant, color, label in [
            (-0.18, "raw_tf", SHARED_COLOR, "raw counts"),
            (0.18, "bm25", DEDICATED_COLOR, "BM25 weights"),
        ]:
            values = [data[variant]["per_category"][c][field] for c in categories] + [
                data[variant][field]
            ]
            ax.bar(x + offset, values, width=0.36, color=color, label=label)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20)
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
        ax.legend(loc="lower left")
        style_axes(ax)
    axes[0].set_ylabel("NDCG@10")
    fig.suptitle("Same AG News documents and queries: raw counts vs. BM25 weights")
    fig.tight_layout()
    fig.savefig(config.asset_path("exp3_bm25_comparison.png"), dpi=160)
    plt.close(fig)


def make_all():
    chart_exp1_latency()
    chart_exp1_recall()
    chart_exp2_tiered()
    chart_exp3_real_flagship()
    chart_exp3_real_sweep()
    chart_exp3_synthetic_worst_case()
    chart_exp3_bm25_comparison()
    print("charts written to", config.ASSETS_DIR)


if __name__ == "__main__":
    make_all()
