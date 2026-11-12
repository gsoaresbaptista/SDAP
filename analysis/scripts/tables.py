"""Write every LaTeX table of the manuscript to paper/tables/."""

from __future__ import annotations

import csv
from itertools import combinations

import numpy as np

from common import (DATA, DATASETS, TABLES, TASKS, bold_pm, pm, summarize, task_means, task_tex, write)
from results import (ABLATION_MODULES, BACKBONE, ablation_accuracy, accuracy, accuracy_task_means,
                     has_source_only_imbalance, imbalance, other, pseudo_curves)

SHORT = {"Office-Home": "Office-Home", "Office-31": "Office-31", "ImageNet-Caltech": "ImageNet-Caltech"}

# Final accuracy (%) reported in the literature, all with a ResNet-50 backbone.
# Every row except HMA is copied from ProCA (Lin et al., ECCV 2022, Tables 1-2); HMA comes from
# PLDCA (Wei et al., TIP 2024, Tables I-II). Both verified against the PDFs on 2026-11-08.
# "--" = not reported.
LITERATURE = {
    "Source-only": (60.9, 77.5, 71.5),
    "DANN": (40.5, 74.8, 45.1),
    "HMA": (42.9, 77.4, None),
    "PADA": (37.1, 57.4, 41.6),
    "ETN": (33.4, 70.6, 2.3),
    "BA$^3$US": (44.6, 78.3, 52.9),
    "CIDA": (47.5, 71.6, 59.2),
}


def _best(values: dict, lower_is_better: bool = False) -> dict:
    """{dataset: best mean among rows}."""
    out = {}
    for ds in DATASETS:
        col = [v[ds][0] for v in values.values()]
        out[ds] = min(col) if lower_is_better else max(col)
    return out


def _cell(mean, std, best, digits=1):
    return bold_pm(mean, std, digits) if round(mean, digits) == round(best, digits) else pm(mean, std, digits)


# ---------------------------------------------------------------- Table: literature (ResNet-50)

def table_literature(summary: dict) -> None:
    rows = []
    for name, vals in LITERATURE.items():
        cells = " & ".join("--" if v is None else f"${v:.1f}$" for v in vals)
        rows.append(f"{name} & ResNet-50 & {cells} \\\\")
    rows.append("\\midrule")
    ours = {m: summary[m] for m in ["ProCA", "PLDCA", "SDAP"]}
    best = _best(ours)
    for m in ["ProCA", "PLDCA"]:
        cells = " & ".join(_cell(*summary[m][ds][:2], best[ds]) for ds in DATASETS)
        rows.append(f"{m}$^{{\\dagger}}$ & ResNet-50 & {cells} \\\\")
    rows.append("\\midrule")
    cells = " & ".join(_cell(*summary["SDAP"][ds][:2], best[ds]) for ds in DATASETS)
    rows.append(f"SDAP (ours) & DINOv2 ViT-g/14 & {cells} \\\\")
    body = "\n".join(rows)
    write(TABLES / "literature.tex", f"""\\begin{{tabular}}{{llccc}}
\\toprule
Method & Backbone & Office-Home & Office-31 & ImageNet-Caltech \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}""")


# ---------------------------------------------------------------- Table: same backbone

SAME_BACKBONE = ["Source-only", "PLDCA+AP", "SDAP (batch-level)", "SDAP"]


def table_same_backbone(summary: dict) -> None:
    vals = {m: summary[m] for m in SAME_BACKBONE}
    best = _best(vals)
    rows = []
    for m in SAME_BACKBONE:
        cells = " & ".join(_cell(*summary[m][ds][:2], best[ds]) for ds in DATASETS)
        rows.append(f"{m} & {cells} \\\\")
    write(TABLES / "same_backbone.tex", f"""\\begin{{tabular}}{{lccc}}
\\toprule
Method & Office-Home & Office-31 & ImageNet-Caltech \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}""")


# ---------------------------------------------------------------- Table: S1, Sf, H, F

OTHER_METHODS = ["ProCA", "PLDCA", "PLDCA+AP", "SDAP"]
METRIC_LABEL = {"S1": "$S_1$ ($\\uparrow$)", "Sf": "$S_f$ ($\\uparrow$)",
                "H": "$H$ ($\\uparrow$)", "F": "$F$ ($\\downarrow$)"}


def other_summary() -> dict:
    """{method: {metric: {dataset: (mean, std, n)}}}."""
    out = {}
    for m in OTHER_METHODS:
        r = other(m)
        out[m] = {}
        for metric in METRIC_LABEL:
            per_task = {ds: {t: [run[metric] for run in v] for t, v in tasks.items()} for ds, tasks in r.items()}
            out[m][metric] = summarize(per_task)
    return out


def table_other(osum: dict) -> None:
    rows = []
    for i, m in enumerate(OTHER_METHODS):
        for j, metric in enumerate(METRIC_LABEL):
            best = _best({mm: osum[mm][metric] for mm in OTHER_METHODS}, lower_is_better=(metric == "F"))
            cells = " & ".join(_cell(*osum[m][metric][ds][:2], best[ds]) for ds in DATASETS)
            name = f"\\multirow{{4}}{{*}}{{{m}}}" if j == 0 else ""
            rows.append(f"{name} & {METRIC_LABEL[metric]} & {cells} \\\\")
        if i < len(OTHER_METHODS) - 1:
            rows.append("\\midrule")
    write(TABLES / "other_metrics.tex", f"""\\begin{{tabular}}{{llccc}}
\\toprule
Method & Metric & Office-Home & Office-31 & ImageNet-Caltech \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}""")


# ---------------------------------------------------------------- Table: pseudo-label accuracy

PL_METHODS = ["ProCA", "PLDCA", "PLDCA+AP", "SDAP (batch-level)", "SDAP"]


def pseudo_summary() -> dict:
    """Mean pseudo-label accuracy over increments; {method: {dataset: (mean, std, n)}}."""
    out = {}
    for m in PL_METHODS:
        r = pseudo_curves(m)
        per_task = {ds: {t: [float(np.mean(c)) for c in v] for t, v in tasks.items()} for ds, tasks in r.items()}
        out[m] = summarize(per_task)
    return out


def table_pseudo(psum: dict) -> None:
    best = _best(psum)
    rows = []
    for m in PL_METHODS:
        cells = " & ".join(_cell(*psum[m][ds][:2], best[ds]) for ds in DATASETS)
        rows.append(f"{m} & {BACKBONE[m]} & {cells} \\\\")
    write(TABLES / "pseudo_labels.tex", f"""\\begin{{tabular}}{{llccc}}
\\toprule
Method & Backbone & Office-Home & Office-31 & ImageNet-Caltech \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}""")


def trajectory() -> dict:
    """{method: {dataset: [mean over tasks of the per-increment pseudo-label accuracy]}}."""
    out = {}
    for m in ["PLDCA", "PLDCA+AP", "SDAP (batch-level)", "SDAP"]:
        means = task_means(pseudo_curves(m))
        out[m] = {ds: np.mean([means[ds][t] for t in TASKS[ds]], axis=0) for ds in DATASETS}
    return out


def table_trajectory(traj: dict) -> None:
    blocks = []
    for ds in DATASETS:
        k = len(traj["SDAP"][ds])
        head = " & ".join(str(i + 1) for i in range(k))
        lines = [f"\\multicolumn{{{k + 1}}}{{l}}{{\\textit{{{ds}}}}} \\\\",
                 f"Method & {head} \\\\", "\\midrule"]
        for m, per_ds in traj.items():
            lines.append(f"{m} & " + " & ".join(f"${v:.1f}$" for v in per_ds[ds]) + " \\\\")
        blocks.append("\n".join(lines))
    body = "\n\\midrule\n".join(blocks)
    write(TABLES / "pseudo_label_trajectory.tex", f"""\\begin{{tabular}}{{lcccccccc}}
\\toprule
{body}
\\bottomrule
\\end{{tabular}}""")


# ---------------------------------------------------------------- Table: imbalance

FRACTIONS = [1.0, 0.5, 0.3, 0.2, 0.1]


def imbalance_methods() -> list[str]:
    return (["Source-only"] if has_source_only_imbalance() else []) + ["PLDCA+AP", "SDAP"]


def imbalance_summary() -> dict:
    return {m: {f: summarize(imbalance(m, f)) for f in FRACTIONS} for m in imbalance_methods()}


def table_imbalance(isum: dict) -> None:
    rows = []
    for i, ds in enumerate(DATASETS):
        methods = list(isum)
        for j, m in enumerate(methods):
            vals = [isum[m][f][ds][0] for f in FRACTIONS]
            drop = vals[-1] - vals[0]
            name = f"\\multirow{{{len(methods)}}}{{*}}{{{ds}}}" if j == 0 else ""
            rows.append(f"{name} & {m} & " + " & ".join(f"${v:.1f}$" for v in vals) + f" & ${drop:+.1f}$ \\\\")
        if i < len(DATASETS) - 1:
            rows.append("\\midrule")
    head = " & ".join(f"{f:.1f}" for f in FRACTIONS)
    write(TABLES / "imbalance.tex", f"""\\begin{{tabular}}{{llcccccc}}
\\toprule
 & & \\multicolumn{{{len(FRACTIONS)}}}{{c}}{{Retained fraction $r$}} & \\\\
\\cmidrule(lr){{3-{len(FRACTIONS) + 2}}}
Dataset & Method & {head} & $\\Delta_{{1.0 \\rightarrow 0.1}}$ \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}""")


# ---------------------------------------------------------------- Tables: ablation

ALL_CONFIGS = [frozenset(c) for n in range(len(ABLATION_MODULES), -1, -1)
               for c in combinations(ABLATION_MODULES, n)]
# cumulative order used in the main text: disabled sets
CUMULATIVE = [
    frozenset({"il", "cl", "triplet", "dist"}),
    frozenset({"cl", "triplet", "dist"}),
    frozenset({"triplet", "dist"}),
    frozenset({"dist"}),
    frozenset(),
]


def ablation_summary() -> dict:
    return {cfg: summarize(ablation_accuracy(cfg)) for cfg in ALL_CONFIGS}


def _marks(disabled: frozenset) -> str:
    return " & ".join("$\\times$" if mod in disabled else "\\checkmark" for mod in ABLATION_MODULES)


def table_ablation(asum: dict) -> None:
    full = asum[frozenset()]
    rows = []
    for cfg in CUMULATIVE:
        cells = " & ".join(_cell(*asum[cfg][ds][:2], full[ds][0]) for ds in DATASETS)
        rows.append(f"{_marks(cfg)} & {cells} \\\\")
    write(TABLES / "ablation.tex", f"""\\begin{{tabular}}{{cccc ccc}}
\\toprule
IL & CL & $\\mathcal{{L}}_{{triplet}}$ & $\\mathcal{{L}}_{{dist}}$ & Office-Home & Office-31 & ImageNet-Caltech \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}""")

    best = {ds: max(asum[c][ds][0] for c in ALL_CONFIGS) for ds in DATASETS}
    rows = []
    for cfg in ALL_CONFIGS:
        cells = " & ".join(_cell(*asum[cfg][ds][:2], best[ds], 2) for ds in DATASETS)
        rows.append(f"{_marks(cfg)} & {cells} \\\\")
    write(TABLES / "ablation_full.tex", f"""\\begin{{tabular}}{{cccc ccc}}
\\toprule
IL & CL & $\\mathcal{{L}}_{{triplet}}$ & $\\mathcal{{L}}_{{dist}}$ & Office-Home & Office-31 & ImageNet-Caltech \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}""")


# ---------------------------------------------------------------- Tables: re-implementation check

def original_reported() -> dict:
    out = {}
    with open(DATA / "original_reported_accuracy.csv") as fh:
        for row in csv.DictReader(line for line in fh if not line.startswith("#")):
            out[(row["dataset"], row["task"])] = (float(row["proca_original"]), float(row["pldca_original"]))
    return out


def table_reimplementation() -> dict:
    orig = original_reported()
    diffs = {}
    for ds, label in [("Office-Home", "office_home"), ("Office-31", "office31")]:
        tasks = TASKS[ds]
        pro = accuracy_task_means("ProCA")[ds]
        pld = accuracy_task_means("PLDCA")[ds]
        d_pro = [pro[t] - orig[(ds, t)][0] for t in tasks]
        d_pld = [pld[t] - orig[(ds, t)][1] for t in tasks]
        diffs[ds] = {"ProCA": float(np.mean(d_pro)), "PLDCA": float(np.mean(d_pld))}
        head = " & ".join(task_tex(t) for t in tasks)
        if ds == "Office-Home":  # long: tasks as rows
            rows = [f"{task_tex(t)} & ${a:+.1f}$ & ${b:+.1f}$ \\\\" for t, a, b in zip(tasks, d_pro, d_pld)]
            rows += ["\\midrule", f"Average & ${np.mean(d_pro):+.1f}$ & ${np.mean(d_pld):+.1f}$ \\\\"]
            text = f"""\\begin{{tabular}}{{lcc}}
\\toprule
Task & ProCA & PLDCA \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}"""
        else:
            rows = [f"{m} & " + " & ".join(f"${v:+.1f}$" for v in d) + f" & ${np.mean(d):+.1f}$ \\\\"
                    for m, d in [("ProCA", d_pro), ("PLDCA", d_pld)]]
            text = f"""\\begin{{tabular}}{{l{'c' * (len(tasks) + 1)}}}
\\toprule
Method & {head} & Average \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}"""
        write(TABLES / f"reimplementation_{label}.tex", text)
    return diffs


# ---------------------------------------------------------------- Appendix: per-task accuracy

PERTASK_METHODS = ["ProCA", "PLDCA", "Source-only", "PLDCA+AP", "SDAP (batch-level)", "SDAP"]


def table_per_task(summary: dict) -> None:
    for ds, label in [("Office-Home", "office_home"), ("Office-31", "office31"),
                      ("ImageNet-Caltech", "imagenet_caltech")]:
        runs = {m: accuracy(m)[ds] for m in PERTASK_METHODS}
        rows = []
        for t in TASKS[ds]:
            means = {m: float(np.mean(runs[m][t])) for m in PERTASK_METHODS}
            best = max(means.values())
            cells = []
            for m in PERTASK_METHODS:
                v = np.asarray(runs[m][t])
                cells.append(_cell(v.mean(), v.std(ddof=1), best, 2))
            rows.append(f"{task_tex(t)} & " + " & ".join(cells) + " \\\\")
        best = max(summary[m][ds][0] for m in PERTASK_METHODS)
        avg = " & ".join(_cell(*summary[m][ds][:2], best, 2) for m in PERTASK_METHODS)
        rows += ["\\midrule", f"Average & {avg} \\\\"]
        head = " & ".join(PERTASK_METHODS)
        write(TABLES / f"per_task_{label}.tex", f"""\\begin{{tabular}}{{l{'c' * len(PERTASK_METHODS)}}}
\\toprule
Task & {head} \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}""")
