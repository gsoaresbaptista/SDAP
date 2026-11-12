"""Load every method once and compute the per-task values used by tables, statistics and figures."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from common import (DATA, DATASETS, TASKS, by_task, check_runs, final_accuracy, load, other_metrics,
                    pseudo_label_curve, task_means)

# Display names. "Source-only" is the SDAP ablation with every adaptation module disabled:
# the projector on frozen DINOv2 features is trained with the source cross-entropy only.
METHODS = {
    "ProCA": ("proca_resnet50.json", None),
    "PLDCA": ("pldca_resnet50.json", None),
    "Source-only": ("sdap_ablation.json", frozenset({"cl", "il", "triplet", "dist"})),
    "PLDCA+AP": ("pldca_ap.json", None),
    "SDAP (old PL)": ("sdap_old_pl.json", None),
    "SDAP": ("sdap.json", None),
}
BACKBONE = {
    "ProCA": "ResNet-50", "PLDCA": "ResNet-50",
    "Source-only": "DINOv2", "PLDCA+AP": "DINOv2", "SDAP (old PL)": "DINOv2", "SDAP": "DINOv2",
}
ABLATION_MODULES = ["il", "cl", "triplet", "dist"]


def ablation_flags(suffix: str) -> frozenset:
    """Ablation keys look like '1.0_cl_triplet': the listed modules are DISABLED."""
    parts = suffix.split("_")
    assert parts[0] == "1.0", suffix
    return frozenset(p for p in parts[1:] if p)


@lru_cache(maxsize=None)
def raw(name: str) -> dict:
    return load(name)


def runs(method: str, metric):
    """{dataset: {task: [value per run]}} for one method."""
    fname, disabled = METHODS[method]
    data = raw(fname)
    if disabled is None:
        return by_task(data, metric)
    return by_task(data, metric, keep=lambda i: ablation_flags(i["suffix"]) == disabled)


@lru_cache(maxsize=None)
def accuracy(method: str) -> dict:
    """{dataset: {task: [A per run]}}."""
    r = runs(method, final_accuracy)
    check_runs(r, label=method)
    return r


@lru_cache(maxsize=None)
def accuracy_task_means(method: str) -> dict:
    return task_means(accuracy(method))


def ablation_accuracy(disabled: frozenset) -> dict:
    """Final accuracy per task for one ablation configuration; full SDAP comes from sdap.json."""
    if not disabled:
        return accuracy("SDAP")
    r = by_task(raw("sdap_ablation.json"), final_accuracy,
                keep=lambda i: ablation_flags(i["suffix"]) == disabled)
    check_runs(r, label=f"ablation -{sorted(disabled)}")
    return r


@lru_cache(maxsize=None)
def other(method: str) -> dict:
    """{dataset: {task: [dict(S1, Sf, H, F) per run]}}."""
    return runs(method, other_metrics)


@lru_cache(maxsize=None)
def pseudo_curves(method: str) -> dict:
    """{dataset: {task: [curve per run]}}, curve = pseudo-label accuracy per increment."""
    return runs(method, pseudo_label_curve)


def has_source_only_imbalance() -> bool:
    return (DATA / "source_only_imbalance.json").is_file()


def imbalance(method: str, fraction: float) -> dict:
    """Final accuracy per task under the imbalance protocol at one retained fraction.

    PLDCA (ResNet-50) was run under the imbalance protocol only for r < 1.0. With r = 1.0 the
    protocol discards no image, so that point repeats the balanced runs (pldca_resnet50.json),
    produced by the same code, source weights and seeds.
    Source-only (the SDAP ablation with every module disabled) was run under the imbalance
    protocol only for r < 1.0 as well; its r = 1.0 point repeats the balanced ablation runs.
    """
    if method in ("PLDCA", "Source-only") and fraction == 1.0:
        return accuracy(method)
    fname = {"PLDCA": "pldca_resnet50_imbalance.json", "PLDCA+AP": "pldca_ap_imbalance.json",
             "SDAP": "sdap_imbalance.json", "Source-only": "source_only_imbalance.json"}[method]
    tag = {"PLDCA": "pldca", "PLDCA+AP": "pldca+ap", "SDAP": "sdap",
           "Source-only": "cl_triplet_dist_il"}[method]
    want = f"{fraction:.1f}_{tag}"
    r = by_task(raw(fname), final_accuracy, keep=lambda i: i["suffix"] == want)
    check_runs(r, label=f"imbalance {method} {fraction}")
    return r


def beta(value: float) -> dict:
    want = f"1.0__{value:.1f}"
    r = by_task(raw("sdap_beta.json"), final_accuracy, keep=lambda i: i["suffix"] == want)
    check_runs(r, label=f"beta {value}")
    return r


def paired(method_a: str, method_b: str, dataset: str) -> tuple[np.ndarray, np.ndarray]:
    """Task means of two methods aligned on the task order of `dataset`."""
    a = accuracy_task_means(method_a)[dataset]
    b = accuracy_task_means(method_b)[dataset]
    tasks = TASKS[dataset]
    return np.array([a[t] for t in tasks]), np.array([b[t] for t in tasks])


__all__ = ["METHODS", "BACKBONE", "DATASETS", "ABLATION_MODULES", "accuracy", "accuracy_task_means", "has_source_only_imbalance",
           "ablation_accuracy", "other", "pseudo_curves", "imbalance", "beta", "paired"]
