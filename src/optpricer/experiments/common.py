"""Shared helpers for experiments: figure style, saving figures and the CSVs behind them."""

from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from cycler import cycler  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from ..paths import results_dir  # noqa: E402

# Categorical order (fixed, never cycled): blue, orange, aqua, yellow, magenta, green, violet, red
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
SEQ_BLUE = LinearSegmentedColormap.from_list("seq_blue", ["#eaf2fc", "#2a78d6", "#0d3a73"])
DIVERGING = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#d9d8d4", "#eb6834"])


def setup_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": MUTED, "axes.labelcolor": TEXT, "text.color": TEXT,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.7, "axes.spines.top": False, "axes.spines.right": False,
        "axes.prop_cycle": cycler(color=PALETTE), "lines.linewidth": 2.0,
        "lines.markersize": 5, "font.size": 9.5, "axes.titlesize": 10.5, "legend.frameon": False,
        "figure.dpi": 110,
    })


setup_style()


def fig_dir() -> Path:
    return results_dir("figures")


def table_dir() -> Path:
    return results_dir("tables")


def save_fig(fig, name: str, footnote: str) -> Path:
    """Save ``results/figures/<name>.png`` with a footnote (data label, settings, assumptions)."""
    fig.text(0.01, 0.005, footnote, fontsize=7, color=MUTED, ha="left", va="bottom", wrap=True)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    path = fig_dir() / f"{name}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def save_table(df: pd.DataFrame, name: str) -> Path:
    path = table_dir() / f"{name}.csv"
    df.to_csv(path, index=False)
    return path


@contextmanager
def timed(log: dict, key: str):
    t0 = time.perf_counter()
    yield
    log[key] = round(time.perf_counter() - t0, 3)
