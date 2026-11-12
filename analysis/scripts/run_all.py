"""Regenerate every table, statistic and figure, and dump the numbers to analysis/output/summary.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from common import DATASETS, OUTPUT, ROOT, summarize  # noqa: E402
from figures import figure_beta, figure_imbalance  # noqa: E402
from results import METHODS, accuracy  # noqa: E402
from stats import run_tests, table_stats  # noqa: E402
from tables import (ablation_summary, imbalance_summary, other_summary, pseudo_summary,  # noqa: E402
                    table_ablation, table_imbalance, table_literature, table_other, table_per_task,
                    table_pseudo, table_reimplementation, table_same_backbone, table_trajectory, trajectory)

N_CLASSES = {"Office-Home": 65, "Office-31": 31, "ImageNet-Caltech": 84}


def projector_parameters(n_classes: int, d_in: int = 1536, width: int = 1024) -> int:
    """Trainable parameters of codes/RAPTOR/models/projector.py ('small_2'):
    two blocks Linear -> BatchNorm1d -> ReLU, then a linear classifier."""
    linear1 = d_in * width + width
    bn1 = 2 * width
    linear2 = width * width + width
    bn2 = 2 * width
    head = width * n_classes + n_classes
    return linear1 + bn1 + linear2 + bn2 + head


def main() -> None:
    summary = {m: summarize(accuracy(m)) for m in METHODS}
    table_literature(summary)
    table_same_backbone(summary)
    osum = other_summary()
    table_other(osum)
    psum = pseudo_summary()
    table_pseudo(psum)
    traj = trajectory()
    table_trajectory(traj)
    isum = imbalance_summary()
    table_imbalance(isum)
    asum = ablation_summary()
    table_ablation(asum)
    reimpl = table_reimplementation()
    table_per_task(summary)
    tests = run_tests()
    table_stats(tests)
    fig_imb = figure_imbalance()
    fig_beta = figure_beta()

    dump = {
        "aggregation": "mean and sample std (ddof=1) over task means; each task mean averages 5 runs",
        "final_accuracy": {m: {ds: summary[m][ds][:2] for ds in DATASETS} for m in summary},
        "gain_sdap_over": {m: {ds: summary["SDAP"][ds][0] - summary[m][ds][0] for ds in DATASETS}
                           for m in summary if m != "SDAP"},
        "other_metrics": {m: {k: {ds: v[ds][:2] for ds in DATASETS} for k, v in d.items()} for m, d in osum.items()},
        "pseudo_label_accuracy": {m: {ds: v[ds][:2] for ds in DATASETS} for m, v in psum.items()},
        "pseudo_label_trajectory": {m: {ds: list(map(float, v[ds])) for ds in DATASETS} for m, v in traj.items()},
        "imbalance": {m: {str(f): {ds: v[ds][:2] for ds in DATASETS} for f, v in d.items()} for m, d in isum.items()},
        # Imbalance accuracy minus balanced Source-only (Source-only was not run under imbalance;
        # it has no target loss, only target batch-norm statistics).
        "imbalance_minus_source_only": {m: {str(f): {ds: v[ds][0] - summary["Source-only"][ds][0]
                                                     for ds in DATASETS} for f, v in d.items()}
                                        for m, d in isum.items()},
        "ablation": {"-".join(sorted(k)) or "full": {ds: v[ds][:2] for ds in DATASETS} for k, v in asum.items()},
        "reimplementation_mean_diff": reimpl,
        "statistics": tests,
        "figure_imbalance_office31": fig_imb,
        "figure_beta_office31": fig_beta,
        "projector_trainable_parameters": {ds: projector_parameters(n) for ds, n in N_CLASSES.items()},
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "summary.json"
    path.write_text(json.dumps(dump, indent=2, default=float) + "\n")
    print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
