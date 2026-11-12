#!/usr/bin/env python3
"""Progress and ETA of the PLDCA (ResNet-50) imbalance run.

Standard library only; run it with the system Python while the experiment is running:

    python3 scripts/progress.py             # one report
    python3 scripts/progress.py --watch 60  # refresh every 60 s

How the estimate works
----------------------
Each task (source -> target, run, r) is described by the amount of work it does, computed from
the real number of images per class in the dataset and the imbalance protocol of
IncrementalDataset.load_data (first class of each increment keeps all images, the others keep
max(1, int(n * r))), mirroring PLDCA.increment:

  x_train : images processed by training steps. The loop zips the source loader (all source
            images) with the imbalanced target loader, so steps = min(source, target) batches
            per epoch, times the number of epochs, summed over increments.
  x_val   : images evaluated: validation on all target images of the seen classes after every
            epoch and once at the end, plus the final re-evaluation.
  x_pass  : inference passes over the imbalanced target training set (shared-class detection and
            pseudo-labels at the start, pseudo-labels every epoch after warm-up, detection every
            `step_update` epochs, final evaluation and pseudo-label accuracy).
  1       : fixed overhead per task (model loading, DataLoader start-up).

time(task) = b0 + b1*x_train + b2*x_val + b3*x_pass. The coefficients start from a prior (relative
weights below) scaled by the tasks measured so far, and are re-fitted on every call by ridge
regression on the measured times in results/run.log, so the estimate improves as tasks finish
(a new target domain, or a smaller r, adds information). The ETA band uses the relative error of
the fitted model on the finished tasks.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Mirrors DATASETS / run_pldca_imbalance in src/experiment/train_proca_pldca_runner.py.
DATASETS = {
    "office-31": {
        "folder": "office-31",
        "config": "data/office_31/config.json",
        "domains": ["Amazon", "DSLR", "Webcam"],
        "num_classes": 31,
        "epochs": 15,
    },
}
DEFAULT_FRACTIONS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
DEFAULT_RUNS = 5
BATCH = 24
WARM_UP = 6
STEP_UPDATE = 3
INC = 10
DEFAULT_DATA_ROOT = "datasets"

# Prior relative cost (seconds per unit, up to a common scale fitted on the data).
PRIOR_REL = {"overhead": 20.0, "train": 1.0, "val": 0.3, "pass": 0.5}
RIDGE = 2.0  # weight of the prior, in "equivalent observations"

LINE = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) (START|END) (\S+)(.*)$")
KEY = re.compile(r"^([^_]+)_([^_]+)_(\d+)_(\d+\.\d)_pldca$")


# ----------------------------------------------------------------------------- dataset sizes
def class_counts(data_root: Path, ds: dict) -> dict[str, list[int]]:
    idx_to_class = {v: k for k, v in json.load(open(ROOT / ds["config"])).items()}
    counts = {}
    for dom in ds["domains"]:
        d = data_root / ds["folder"] / dom
        counts[dom] = [len(os.listdir(d / idx_to_class[c])) for c in range(ds["num_classes"])]
    return counts


def features(counts, ds, source, target, r):
    E = ds["epochs"]
    n_inc = ds["num_classes"] // INC
    src_batches = math.ceil(sum(counts[source]) / BATCH)
    tgt = counts[target]
    x_train = x_val = x_pass = 0.0
    passes = 2 + (E - WARM_UP) + len([e for e in range(WARM_UP, E) if e % STEP_UPDATE == 0]) + 2
    for i in range(n_inc):
        cls = range(i * INC, (i + 1) * INC)
        n_tr = sum(tgt[c] if (j == 0 or r >= 1.0) else max(1, int(tgt[c] * r))
                   for j, c in enumerate(cls))
        n_val = sum(tgt[: (i + 1) * INC])
        x_train += E * BATCH * min(src_batches, math.ceil(n_tr / BATCH))
        x_val += (E + 1) * n_val + sum(tgt[c] for c in cls)  # + final re-evaluation
        x_pass += passes * n_tr
    return [1.0, x_train, x_val, x_pass]


# ----------------------------------------------------------------------------- plan and log
def running_cmdline(pid_file: Path):
    try:
        pid = int(pid_file.read_text().strip())
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except (OSError, ValueError):
        return None, None
    return pid, [a.decode() for a in raw.split(b"\0") if a]


def arg_values(argv, name):
    if name not in argv:
        return None
    out = []
    for a in argv[argv.index(name) + 1:]:
        if a.startswith("--"):
            break
        out.append(a)
    return out


def build_plan(datasets, fractions, runs):
    plan = []
    for name in datasets:
        doms = DATASETS[name]["domains"]
        for r in fractions:
            for run in range(1, runs + 1):
                for s in doms:
                    for t in doms:
                        if s != t:
                            plan.append((name, f"{s}_{t}_{run}_{r:.1f}_pldca", s, t, r))
    return plan


def read_log(path: Path):
    """Return {key: seconds} for finished tasks and (key, start_datetime) of the open one."""
    durations, open_task, first_start = {}, None, None
    if not path.is_file():
        return durations, None, None
    for line in path.read_text(errors="replace").splitlines():
        m = LINE.match(line)
        if not m:
            continue
        ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
        kind, key, rest = m.group(2), m.group(3), m.group(4)
        if kind == "START":
            open_task = (key, ts)
            first_start = first_start or ts
        else:
            e = re.search(r"elapsed=([\d.]+)s", rest)
            if e:
                durations[key] = float(e.group(1))
            elif open_task and open_task[0] == key:
                durations[key] = (ts - open_task[1]).total_seconds()
            if open_task and open_task[0] == key:
                open_task = None
    return durations, open_task, first_start


# ----------------------------------------------------------------------------- model fit
def solve(A, b):
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        p = max(range(c, n), key=lambda k: abs(M[k][c]))
        M[c], M[p] = M[p], M[c]
        if abs(M[c][c]) < 1e-12:
            raise ZeroDivisionError
        for k in range(n):
            if k != c:
                f = M[k][c] / M[c][c]
                M[k] = [x - f * y for x, y in zip(M[k], M[c])]
    return [M[i][n] / M[i][i] for i in range(n)]


def fit(obs, scale):
    """obs: list of (x, t). Returns coefficients b (seconds per unit) and relative RMS error."""
    rel = [PRIOR_REL["overhead"], PRIOR_REL["train"], PRIOR_REL["val"], PRIOR_REL["pass"]]
    if not obs:
        return None, None
    # prior: relative weights scaled so that they reproduce the total measured time
    k = sum(t for _, t in obs) / sum(sum(r * xi for r, xi in zip(rel, x)) for x, _ in obs)
    b0 = [r * k for r in rel]
    # ridge in normalised units z_j = b_j * scale_j (seconds contributed on a typical task)
    z0 = [b * s for b, s in zip(b0, scale)]
    lam = RIDGE * (sum(t for _, t in obs) / len(obs)) ** 2 / max(sum(z * z for z in z0), 1e-9)
    n = 4
    A = [[lam if i == j else 0.0 for j in range(n)] for i in range(n)]
    rhs = [lam * z for z in z0]
    for x, t in obs:
        u = [xi / s for xi, s in zip(x, scale)]
        for i in range(n):
            rhs[i] += u[i] * t
            for j in range(n):
                A[i][j] += u[i] * u[j]
    try:
        z = solve(A, rhs)
    except ZeroDivisionError:
        z = z0
    z = [max(v, 0.0) for v in z]
    b = [v / s for v, s in zip(z, scale)]
    pred = lambda x: sum(bi * xi for bi, xi in zip(b, x))
    # keep the total right after clipping
    tot_p = sum(pred(x) for x, _ in obs)
    if tot_p > 0:
        f = sum(t for _, t in obs) / tot_p
        b = [v * f for v in b]
    errs = [(t / max(pred(x), 1e-9)) - 1 for x, t in obs]
    rms = math.sqrt(sum(e * e for e in errs) / len(errs))
    return b, rms


# ----------------------------------------------------------------------------- report
def hms(sec):
    sec = max(0, int(sec))
    d, rem = divmod(sec, 86400)
    h, rem = divmod(rem, 3600)
    m, _ = divmod(rem, 60)
    return (f"{d}d " if d else "") + f"{h:02d}h{m:02d}m"


def bar(frac, width=40):
    n = int(round(frac * width))
    return "[" + "#" * n + "." * (width - n) + "]"


def report(args):
    results_path = ROOT / args.results
    pid, argv = running_cmdline(ROOT / args.pid_file)
    alive = argv is not None
    argv = argv or []

    datasets = args.datasets or arg_values(argv, "--datasets") or ["office-31"]
    fr = args.fractions or arg_values(argv, "--fractions")
    fractions = [float(f) for f in fr] if fr else DEFAULT_FRACTIONS
    rv = args.runs or (arg_values(argv, "--runs") or [None])[0]
    runs = int(rv) if rv else DEFAULT_RUNS
    data_root = Path(args.data_root or os.environ.get("CIUDA_DATA_ROOT", DEFAULT_DATA_ROOT))

    unknown = [d for d in datasets if d not in DATASETS]
    if unknown:
        sys.exit(f"Datasets not configured in this script: {unknown}")
    counts = {d: class_counts(data_root, DATASETS[d]) for d in datasets}

    plan = build_plan(datasets, fractions, runs)
    feats = {key: features(counts[d], DATASETS[d], s, t, r) for d, key, s, t, r in plan}

    done = set(json.load(open(results_path))) if results_path.is_file() else set()
    durations, open_task, first_start = read_log(ROOT / args.log)

    # observations: any finished task in the log whose features we can compute
    obs = []
    for key, sec in durations.items():
        m = KEY.match(key)
        if not m:
            continue
        x = feats.get(key)
        if x is None:
            d = datasets[0]
            x = features(counts[d], DATASETS[d], m.group(1), m.group(2), float(m.group(4)))
        obs.append((x, sec))

    scale = [1.0] + [sum(f[j] for f in feats.values()) / len(feats) for j in (1, 2, 3)]
    b, rms = fit(obs, scale)
    if b is None:
        print("Nenhuma tarefa terminada no log ainda; sem base para o ETA.")
        return
    pred = {key: sum(bi * xi for bi, xi in zip(b, x)) for key, x in feats.items()}
    band = max(rms, 0.10 if len(obs) < 5 else 0.05)

    now = datetime.now()
    running_key, running_elapsed, running_left = None, 0.0, 0.0
    if open_task and alive and open_task[0] not in done:
        running_key = open_task[0]
        running_elapsed = (now - open_task[1]).total_seconds()
        p = pred.get(running_key, 0.0)
        running_left = p - running_elapsed if running_elapsed < p else 0.1 * running_elapsed

    pending = [k for _, k, *_ in plan if k not in done]
    left = sum(pred[k] for k in pending if k != running_key) + running_left
    work_total = sum(pred.values())
    work_done = sum(pred[k] for _, k, *_ in plan if k in done)
    if running_key in pred:
        work_done += min(running_elapsed, pred[running_key])

    n_done = sum(1 for _, k, *_ in plan if k in done)
    print(f"PLDCA ResNet-50 imbalance  ({now:%Y-%m-%d %H:%M:%S})")
    print(f"processo: {'rodando, PID ' + str(pid) if alive else 'NAO esta rodando'}")
    print(f"plano: {', '.join(datasets)}; r = {' '.join(f'{f:g}' for f in fractions)}; "
          f"{runs} execuções; {len(plan)} tarefas")
    print()
    print(f"tarefas   {bar(n_done / len(plan))} {100 * n_done / len(plan):5.1f}%  "
          f"({n_done}/{len(plan)})")
    print(f"trabalho  {bar(work_done / work_total)} {100 * work_done / work_total:5.1f}%  "
          f"(ponderado pelo custo estimado de cada tarefa)")
    if running_key:
        print(f"em curso: {running_key}  {hms(running_elapsed)} de ~{hms(pred[running_key])} previstos")
    if first_start:
        print(f"decorrido desde o primeiro START do log: {hms((now - first_start).total_seconds())}")
    lo, hi = left * (1 - band), left * (1 + band)
    print()
    print(f"ETA: {hms(left)}  (faixa {hms(lo)} a {hms(hi)})")
    print(f"término previsto: {now + timedelta(seconds=left):%a %d/%m %H:%M}  "
          f"(entre {now + timedelta(seconds=lo):%d/%m %H:%M} e {now + timedelta(seconds=hi):%d/%m %H:%M})")

    # completion time of each fraction, following the order of the plan
    print()
    print("por fração (na ordem em que o runner as executa):")
    print("   r    feitas   término previsto   acc média final")
    acc = {}
    if results_path.is_file():
        res = json.load(open(results_path))
        for k, v in res.items():
            try:
                acc[k] = v["all_data_until_increment_steps"][str(max(map(int, v["all_data_until_increment_steps"])))]["acc"]
            except (KeyError, ValueError, TypeError):
                pass
    cum = 0.0
    for r in fractions:
        keys = [k for _, k, _, _, rr in plan if rr == r]
        for k in keys:
            if k in done:
                continue
            cum += running_left if k == running_key else pred[k]
        nd = sum(1 for k in keys if k in done)
        when = "pronta" if nd == len(keys) else f"{now + timedelta(seconds=cum):%d/%m %H:%M}"
        accs = [acc[k] for k in keys if k in acc]
        a = f"{100 * sum(accs) / len(accs):5.1f}% (n={len(accs)})" if accs else "-"
        print(f"  {r:.1f}   {nd:3d}/{len(keys):<3d}  {when:>16}   {a}")

    # how good the model is, per target domain
    print()
    print(f"modelo de tempo ajustado em {len(obs)} tarefas; erro relativo RMS {100 * rms:.1f}%")
    by_t = {}
    for key, sec in durations.items():
        m = KEY.match(key)
        if m and key in pred:
            by_t.setdefault(m.group(2), []).append((sec, pred[key]))
    for d in datasets:
        for t in DATASETS[d]["domains"]:
            meas = by_t.get(t, [])
            ptask = [pred[k] for _, k, s, tt, _ in plan if tt == t]
            mean_p = sum(ptask) / len(ptask)
            if meas:
                print(f"  alvo {t:<7} medido {hms(sum(a for a, _ in meas) / len(meas))} por tarefa "
                      f"(n={len(meas)}), previsto {hms(sum(p for _, p in meas) / len(meas))}")
            else:
                print(f"  alvo {t:<7} sem medida ainda; previsto {hms(mean_p)} por tarefa (extrapolado)")
    if args.verbose:
        names = ["overhead (s)", "s/img treino", "s/img val", "s/img passe"]
        print("  coeficientes: " + ", ".join(f"{n}={v:.3g}" for n, v in zip(names, b)))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--watch", type=float, default=0, help="refresh every N seconds")
    p.add_argument("--results", default="results/pldca_resnet50_imbalance.json")
    p.add_argument("--log", default="results/run.log")
    p.add_argument("--pid-file", default="results/run.pid")
    p.add_argument("--data-root", default=None, help="default: $CIUDA_DATA_ROOT or ./datasets")
    p.add_argument("--datasets", nargs="+", default=None, help="default: read from the running process")
    p.add_argument("--fractions", nargs="+", default=None, help="default: read from the running process")
    p.add_argument("--runs", default=None, help="default: read from the running process")
    p.add_argument("-v", "--verbose", action="store_true", help="print the fitted coefficients")
    args = p.parse_args()
    if not args.watch:
        report(args)
        return
    try:
        while True:
            print("\033[2J\033[H", end="")
            report(args)
            time.sleep(args.watch)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
