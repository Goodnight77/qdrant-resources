"""Shared matplotlib styling: a small, fixed categorical palette (validated for
colorblind-safe adjacent contrast) instead of matplotlib's default cycle."""

import matplotlib.pyplot as plt

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
VIOLET = "#4a3aa7"
RED = "#e34948"

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
SURFACE = "#fcfcfb"

# fixed roles used consistently across every chart in this project
SHARED_COLOR = BLUE       # unpartitioned / naive / global-IDF ("before")
DEDICATED_COLOR = ORANGE  # is_tenant / dedicated shard / per-tenant IDF ("after")
WHALE_COLOR = VIOLET


def apply_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRIDLINE,
        "axes.labelcolor": INK_SECONDARY,
        "text.color": INK,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "grid.color": GRIDLINE,
        "font.size": 11,
        "font.family": "sans-serif",
        "axes.grid": True,
        "grid.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlecolor": INK,
        "axes.titleweight": "bold",
        "legend.frameon": False,
    })


def style_axes(ax):
    ax.grid(True, axis="y", linewidth=0.7, color=GRIDLINE)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRIDLINE)
