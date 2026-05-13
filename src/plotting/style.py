"""Publication-ready matplotlib defaults. NO seaborn, by project policy.

Use:
    from src.plotting import apply_style, save_figure
    apply_style()
    fig, ax = plt.subplots(...)
    ...
    save_figure(fig, "results/tokenization_audit/tp_bar")
    # writes tp_bar.png and tp_bar.pdf
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


def apply_style() -> None:
    mpl.rcParams.update({
        "figure.figsize": (6.0, 4.0),
        "figure.dpi": 130,
        "savefig.dpi": 220,
        "savefig.bbox": "tight",
        "font.family": "DejaVu Sans",  # ships with matplotlib; renders CJK basics
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#dddddd",
        "grid.linestyle": "-",
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 1.6,
        "patch.linewidth": 0.6,
        "xtick.direction": "out",
        "ytick.direction": "out",
    })


def save_figure(fig: plt.Figure, base_path: str | Path) -> tuple[Path, Path]:
    """Save the same figure as PNG and PDF; return both paths."""
    base = Path(base_path)
    base.parent.mkdir(parents=True, exist_ok=True)
    png = base.with_suffix(".png")
    pdf = base.with_suffix(".pdf")
    fig.savefig(png)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf
