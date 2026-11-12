"""Shared loading and aggregation helpers for the SDAP analysis.

Aggregation rule used everywhere in the paper:
  1. average the five runs of each task (source -> target pair);
  2. report the mean and the sample standard deviation (ddof=1) of the task means
     within each dataset.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "analysis" / "data"
TABLES = ROOT / "analysis" / "output" / "tables"
FIGURES = ROOT / "analysis" / "output" / "figures"
OUTPUT = ROOT / "analysis" / "output"

DATASETS = ["Office-Home", "Office-31", "ImageNet-Caltech"]

DOMAIN_ABBR = {
    "amazon": "A", "dslr": "D", "webcam": "W",
    "art": "Ar", "clipart": "Cl", "product": "Pr", "real-world": "Rw",
    "imagenet": "I", "caltech": "C",
}
DATASET_OF = {
    "A": "Office-31", "D": "Office-31", "W": "Office-31",
    "Ar": "Office-Home", "Cl": "Office-Home", "Pr": "Office-Home", "Rw": "Office-Home",
    "I": "ImageNet-Caltech", "C": "ImageNet-Caltech",
}
TASKS = {
    "Office-Home": [f"{s}->{t}" for s in ["Ar", "Cl", "Pr", "Rw"] for t in ["Ar", "Cl", "Pr", "Rw"] if s != t],
    "Office-31": [f"{s}->{t}" for s in ["A", "D", "W"] for t in ["A", "D", "W"] if s != t],
    "ImageNet-Caltech": ["C->I", "I->C"],
}

_DOMAIN_RE = r"(amazon|dslr|webcam|art|clipart|product|real[-_]worl?d|imagenet|caltech)"
_KEY_RE = re.compile(rf"^(?:(?P<pre>[0-9.]+)_)?{_DOMAIN_RE}_{_DOMAIN_RE}_(?P<run>\d+)(?:_(?P<rest>.*))?$", re.I)


def _norm(domain: str) -> str:
    d = domain.lower().replace("_", "-")
    return "real-world" if d.startswith("real-wor") else d  # one data file spells it "real-word"


def parse_key(key: str) -> dict:
    """Split a run key into task, dataset, run and the remaining suffix."""
    m = _KEY_RE.match(key)
    if not m:
        raise ValueError(f"unrecognized run key: {key}")
    src = DOMAIN_ABBR[_norm(m.group(2))]
    tgt = DOMAIN_ABBR[_norm(m.group(3))]
    return {
        "task": f"{src}->{tgt}",
        "dataset": DATASET_OF[src],
        "run": int(m.group("run")),
        "prefix": m.group("pre"),
        "suffix": m.group("rest") or "",
    }


def load(name: str) -> dict:
    with open(DATA / name) as fh:
        return json.load(fh)


def steps(entry: dict, field: str) -> list[dict]:
    d = entry[field]
    return [d[k] for k in sorted(d, key=int)]


# ---------------------------------------------------------------- run metrics

def final_accuracy(entry: dict) -> float:
    """A: accuracy on all classes seen so far, after the last increment (%)."""
    return 100 * steps(entry, "all_data_until_increment_steps")[-1]["acc"]


def other_metrics(entry: dict) -> dict:
    """S1, Sf, H and F (%) for one run.

    a_{j,j} = each_step_eval[j]: accuracy on increment j right after learning it.
    a_{K,j} = re_eval[j]: accuracy on increment j with the final model.
    F = mean_{j<K} (a_{j,j} - a_{K,j}); the intermediate a_{l,j} (j<l<K) were not logged.
    """
    just_learned = [100 * s["acc"] for s in steps(entry, "each_step_eval")]
    final = [100 * s["acc"] for s in steps(entry, "re_eval")]
    s1, sf = final[0], final[-1]
    h = 0.0 if s1 + sf == 0 else 2 * s1 * sf / (s1 + sf)
    f = float(np.mean([just_learned[j] - final[j] for j in range(len(final) - 1)]))
    return {"S1": s1, "Sf": sf, "H": h, "F": f}


def pseudo_label_curve(entry: dict) -> list[float]:
    """Pseudo-label accuracy (%) at the end of each increment."""
    return [100 * s["pseudo_label_acc"] for s in steps(entry, "all_data_until_increment_steps")]


# ---------------------------------------------------------------- aggregation

def by_task(data: dict, metric, keep=lambda info: True) -> dict:
    """Return {dataset: {task: [values over runs]}} for runs accepted by `keep`."""
    out: dict = {}
    for key, entry in data.items():
        info = parse_key(key)
        if not keep(info):
            continue
        out.setdefault(info["dataset"], {}).setdefault(info["task"], []).append(metric(entry))
    return out


def task_means(per_task: dict) -> dict:
    """{dataset: {task: mean over runs}} (works for scalars and vectors)."""
    return {ds: {t: np.mean(np.asarray(v, dtype=float), axis=0) for t, v in tasks.items()}
            for ds, tasks in per_task.items()}


def summarize(per_task: dict) -> dict:
    """{dataset: (mean, std, n_tasks)} over task means (std with ddof=1)."""
    out = {}
    for ds, tasks in task_means(per_task).items():
        vals = np.array([tasks[t] for t in sorted(tasks)], dtype=float)
        out[ds] = (float(vals.mean(axis=0)) if vals.ndim == 1 else vals.mean(axis=0),
                   float(vals.std(axis=0, ddof=1)) if vals.ndim == 1 else vals.std(axis=0, ddof=1),
                   len(vals))
    return out


def check_runs(per_task: dict, expected: int = 5, label: str = "") -> None:
    for ds, tasks in per_task.items():
        for t, v in tasks.items():
            if len(v) != expected:
                print(f"WARNING {label}: {ds} {t} has {len(v)} runs (expected {expected})")
        missing = set(TASKS[ds]) - set(tasks)
        if missing:
            print(f"WARNING {label}: {ds} missing tasks {sorted(missing)}")


# ---------------------------------------------------------------- formatting

def num(x: float, digits: int = 1) -> str:
    """Fixed-point number without a negative zero."""
    s = f"{x:.{digits}f}"
    return s[1:] if s.startswith("-") and float(s) == 0 else s


def pm(mean: float, std: float, digits: int = 1) -> str:
    return f"${num(mean, digits)} \\pm {num(std, digits)}$"


def bold_pm(mean: float, std: float, digits: int = 1) -> str:
    return f"$\\mathbf{{{num(mean, digits)} \\pm {num(std, digits)}}}$"


def task_tex(task: str) -> str:
    s, t = task.split("->")
    return f"{s}$\\rightarrow${t}"


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "% Generated by analysis/scripts; do not edit by hand. Run `make analysis`.\n"
    path.write_text(header + text.rstrip() + "\n")
    print(f"wrote {path.relative_to(ROOT)}")
