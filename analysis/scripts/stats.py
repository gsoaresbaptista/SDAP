"""Non-parametric tests over task-level means (scipy / statsmodels only).

Unit of analysis: one task (source -> target pair); each value is the mean of five runs.
- Friedman test across all compared methods (scipy.stats.friedmanchisquare).
- Two-sided exact Wilcoxon signed-rank test for SDAP against each other method
  (scipy.stats.wilcoxon, method="exact"), Holm-corrected within each dataset
  (statsmodels.stats.multitest.multipletests).
ImageNet-Caltech has only two tasks and is reported descriptively.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests

from common import TABLES, TASKS, write
from results import accuracy_task_means, paired

COMPARED = ["ProCA", "PLDCA", "Source-only", "PLDCA+AP", "SDAP (old PL)"]
TESTED_DATASETS = ["Office-Home", "Office-31"]


def run_tests() -> dict:
    out = {}
    for ds in TESTED_DATASETS:
        tasks = TASKS[ds]
        matrix = [[accuracy_task_means(m)[ds][t] for t in tasks] for m in COMPARED + ["SDAP"]]
        fr = friedmanchisquare(*matrix)
        comps = []
        for m in COMPARED:
            sdap, other = paired("SDAP", m, ds)
            diff = sdap - other
            w = wilcoxon(sdap, other, alternative="two-sided", method="exact")
            comps.append({
                "method": m,
                "mean_diff": float(diff.mean()),
                "wins": int((diff > 0).sum()), "ties": int((diff == 0).sum()), "losses": int((diff < 0).sum()),
                "statistic": float(w.statistic), "p": float(w.pvalue),
            })
        holm = multipletests([c["p"] for c in comps], method="holm")[1]
        for c, p in zip(comps, holm):
            c["p_holm"] = float(p)
        out[ds] = {"n_tasks": len(tasks), "friedman_stat": float(fr.statistic),
                   "friedman_p": float(fr.pvalue), "comparisons": comps}
    # descriptive only
    ic = {}
    for m in COMPARED:
        sdap, other = paired("SDAP", m, "ImageNet-Caltech")
        d = sdap - other
        ic[m] = {"mean_diff": float(d.mean()), "wins": int((d > 0).sum()), "losses": int((d < 0).sum())}
    out["ImageNet-Caltech"] = {"n_tasks": 2, "descriptive": ic}
    return out


def _p(p: float) -> str:
    return "$<0.001$" if p < 0.001 else f"${p:.3f}$"


def _p_eq(p: float) -> str:
    return "$p < 0.001$" if p < 0.001 else f"$p = {p:.3f}$"


def table_stats(res: dict) -> None:
    rows = []
    for m in COMPARED:
        cells = []
        for ds in TESTED_DATASETS:
            c = next(c for c in res[ds]["comparisons"] if c["method"] == m)
            cells += [f"${c['mean_diff']:+.1f}$", f"{c['wins']}/{c['ties']}/{c['losses']}", _p(c["p"]), _p(c["p_holm"])]
        rows.append(f"SDAP vs. {m} & " + " & ".join(cells) + " \\\\")
    fr = " & ".join(f"\\multicolumn{{4}}{{c}}{{Friedman $\\chi^2_{{{len(COMPARED)}}} = {res[ds]['friedman_stat']:.1f}$, {_p_eq(res[ds]['friedman_p'])}}}"
                    for ds in TESTED_DATASETS)
    write(TABLES / "statistics.tex", f"""\\begin{{tabular}}{{l cccc cccc}}
\\toprule
 & \\multicolumn{{4}}{{c}}{{Office-Home ($n = {res['Office-Home']['n_tasks']}$ tasks)}} & \\multicolumn{{4}}{{c}}{{Office-31 ($n = {res['Office-31']['n_tasks']}$ tasks)}} \\\\
\\cmidrule(lr){{2-5}} \\cmidrule(lr){{6-9}}
Comparison & $\\Delta$ & W/T/L & $p$ & $p_{{\\text{{Holm}}}}$ & $\\Delta$ & W/T/L & $p$ & $p_{{\\text{{Holm}}}}$ \\\\
\\midrule
{chr(10).join(rows)}
\\midrule
 & {fr} \\\\
\\bottomrule
\\end{{tabular}}""")
