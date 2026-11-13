"""Data figures of the manuscript (EPS, fonts embedded)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common import FIGURES, ROOT, summarize  # noqa: E402
from results import beta, has_source_only_imbalance, imbalance  # noqa: E402

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.grid": True,
    "grid.color": "#dddddd",
    "ps.fonttype": 42,
    "pdf.fonttype": 42,
})
COLORS = {"PLDCA": "#4878d0", "Source-only": "#8c8c8c", "PLDCA+AP": "#ee854a",
          "SDAP (batch-level)": "#956cb4", "SDAP": "#6acc64"}
FRACTIONS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]


def _save(fig, name: str) -> None:
    path = FIGURES / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="eps", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT)}")


def figure_imbalance() -> dict:
    """Office-31 final accuracy versus the retained fraction of non-dominant classes."""
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    x = np.arange(len(FRACTIONS))
    curves = {}
    methods = (["PLDCA"] + (["Source-only"] if has_source_only_imbalance() else [])
               + ["PLDCA+AP", "SDAP (batch-level)", "SDAP"])
    for method in methods:
        stats = [summarize(imbalance(method, f))["Office-31"] for f in FRACTIONS]
        curves[method] = (np.array([v[0] for v in stats]), np.array([v[1] for v in stats]))
    # Means only: the spread over tasks (8.6 to 24.8 p.p.) hides the curves; the std is kept in the
    # returned summary. Distinct markers and line styles keep the curves apart in grayscale.
    style = {"PLDCA": ("s", "-"), "Source-only": ("x", "--"), "PLDCA+AP": ("^", "-"),
             "SDAP (batch-level)": ("v", "-."), "SDAP": ("o", "-")}
    for method, (mean, std) in curves.items():
        marker, ls = style[method]
        ax.plot(x, mean, marker=marker, ms=4.5, lw=1.5, ls=ls, color=COLORS[method], label=method)
    ax.set_xticks(x, [f"{f:.1f}" for f in FRACTIONS])
    ax.set_xlabel("Retained fraction $r$ of the non-dominant classes")
    ax.set_ylabel("Final accuracy (%)")
    ax.set_ylim(40, 95)
    ax.legend(loc="lower left", frameon=True)
    _save(fig, "imbalance_office31.eps")
    return {k: {"mean": v[0].tolist(), "std": v[1].tolist()} for k, v in curves.items()}


BETAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def figure_beta() -> dict:
    stats = [summarize(beta(b))["Office-31"] for b in BETAS]
    mean = np.array([v[0] for v in stats])
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    ax.plot(BETAS, mean, marker="o", ms=4, color=COLORS["SDAP"])
    i = int(np.argmax(mean))
    ax.annotate(f"{mean[i]:.2f}%", (BETAS[i], mean[i]), textcoords="offset points", xytext=(6, -12))
    ax.set_ylim(89, 93)
    ax.set_xticks(BETAS)
    ax.set_xlabel(r"Confidence threshold $\beta$")
    ax.set_ylabel("Final accuracy (%)")
    _save(fig, "beta_office31.eps")
    return {"beta": BETAS, "mean": mean.tolist(), "std": [v[1] for v in stats]}
