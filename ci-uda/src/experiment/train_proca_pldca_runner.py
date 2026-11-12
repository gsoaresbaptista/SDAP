import json
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path

import torch

from datasets import IncrementalDataset
from experiment.manager import setup_dataloader
from incremental_old import load_incremental_loader
from models.base_model import PrototypeResNet
from training.common.metrics_utils import get_final_eval_metrics
from training.train_pldca import PLDCA
from training.train_proca import ProCA
from training.train_source import SourceTrainer
from utils.settings import dataset_path, set_seed

DATASETS = {
    "office-31": {
        "path": str(dataset_path("office-31")),
        "config": "data/office_31/config.json",
        "domains": ["Amazon", "DSLR", "Webcam"],
        "num_classes": 31,
        "epochs": 10,
        # PLDCA reference (analysis/data/pldca_resnet50.json) was run with 15 epochs
        # per increment, as recorded in resultados/asd/output_lcad55.txt.
        "pldca_epochs": 15,
    },
    "office-home": {
        "path": str(dataset_path("office-home-pattern")),
        "config": "data/office_home/config.json",
        "domains": ["product", "real_world", "art", "clipart"],
        "num_classes": 65,
        "epochs": 30,
        "pldca_epochs": 30,
    },
    "caltech-imagenet": {
        "path": str(dataset_path("caltech_imagenet")),
        "config": "data/caltech_imagenet/config.json",
        "domains": ["caltech", "imagenet"],
        "num_classes": 80,
        "epochs": 15,
        "pldca_epochs": 15,
    },
}


def run_proca_pldca(method="proca", runs=5):
    datasets = DATASETS

    results_file = f"all_metrics_{method}_runs.json"
    results = json.load(open(results_file)) if Path(results_file).is_file() else {}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    lr = 1e-3
    batch_size = 24
    THRESHOLD = 0.15
    BETA = 0.6
    WARM_UP = 6
    STEP_UPDATE = 3

    for dataset_name, cfg in datasets.items():
        print(f"\n=== Dataset: {dataset_name} ===")

        for run_id in range(1, runs + 1):
            for source in cfg["domains"]:
                for target in cfg["domains"]:
                    if source == target:
                        continue

                    key = f"{source}_{target}_{dataset_name}_{run_id}"
                    if key in results:
                        print(f"[Skip] {key}")
                        continue

                    print(f"\n[{run_id}] {method.upper()} {source} → {target}")

                    pretrained_path = Path(f"weights/{source}_source.pth")
                    pretrained_model = PrototypeResNet(cfg["num_classes"], pretrained_path).to(device)
                    s_loader = setup_dataloader(Path(cfg["path"]) / source, Path(cfg["config"]), range(cfg["num_classes"]), batch_size, False)

                    # inicializa método
                    if method == "proca":
                        trainer = ProCA(pretrained_model, s_loader, lr, THRESHOLD, device)
                    else:
                        trainer = PLDCA(pretrained_model, s_loader, lr, THRESHOLD, BETA, WARM_UP, STEP_UPDATE, device)

                    set_seed(run_id * 123)

                    # incremental learning loop
                    for i in range(cfg["num_classes"] // 10):
                        train_ids = range(i * 10, (i + 1) * 10)
                        val_ids = range(0, (i + 1) * 10)
                        t_loader = setup_dataloader(Path(cfg["path"]) / target, Path(cfg["config"]), train_ids, batch_size, False, True, True, method == "pldca")
                        val_loader = setup_dataloader(Path(cfg["path"]) / target, Path(cfg["config"]), val_ids, batch_size, False, False)
                        # PLDCA uses pldca_epochs (15 on Office-31, as in the reference runs)
                        epochs = cfg["pldca_epochs"] if method == "pldca" else cfg["epochs"]
                        trainer.increment(t_loader, val_loader, epochs)

                    # re-eval
                    experiment = {
                        "dataset": {
                            "path": cfg["path"],
                            "config": cfg["config"],
                            "source": source,
                            "target": target,
                            "n_classes": cfg["num_classes"],
                        },
                        "train": {"batch": batch_size, "class_increment": 10},
                        "eval": {"batch": batch_size},
                    }

                    trainer.re_eval = get_final_eval_metrics(trainer.model, experiment, device, cfg["num_classes"], 10)

                    experiment = {
                        "dataset": {
                            "path": cfg["path"],
                            "config": cfg["config"],
                            "source": source,
                            "target": target,
                            "n_classes": cfg["num_classes"],
                        },
                        "train": {"batch": batch_size, "class_increment": 10},
                        "eval": {"batch": batch_size},
                    }

                    trainer.re_eval = get_final_eval_metrics(trainer.model, experiment, device, cfg["num_classes"], 10)
                    print(f"Reavaliação final concluída ({source} → {target})")

                    results[key] = trainer.export_metrics()

                    with open(results_file, "w") as f:
                        json.dump(results, f, indent=4)

IMBALANCE_FRACTIONS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
SMOKE_FRACTIONS = [1.0, 0.1]


def log_progress(log_file, message):
    """Append a timestamped progress line to the run log (flushed to disk immediately)."""
    path = Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {message}\n")
        f.flush()
        os.fsync(f.fileno())


def save_results_atomic(results_file, results):
    """Write the results JSON atomically (temp file in the same directory, then os.replace)."""
    path = Path(results_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(results, f, indent=4)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def per_class_counts(cfg, target, class_start, class_qtd, imbalance_factor,
                     max_images_per_class=None):
    """Number of target training images per class for one increment (uses the real dataset code)."""
    dataset = IncrementalDataset(
        Path(cfg["path"]) / target,
        Path(cfg["config"]),
        range(class_start, class_start + class_qtd),
        imbalance_factor=imbalance_factor,
        max_images_per_class=max_images_per_class,
    )
    counts = {}
    for label in dataset.labels:
        counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items()))


def log_imbalance_counts(cfg, target, class_start, class_qtd, fractions, log_file,
                         max_images_per_class=None):
    for factor in fractions:
        counts = per_class_counts(cfg, target, class_start, class_qtd, factor,
                                  max_images_per_class=max_images_per_class)
        line = (f"IMBALANCE target={target} increment={class_start // class_qtd} "
                f"classes={class_start}-{class_start + class_qtd - 1} r={factor:.1f} "
                f"total={sum(counts.values())} counts={json.dumps(counts)}")
        print(line)
        log_progress(log_file, line)


def run_pldca_imbalance(dataset_names=("office-31",), fractions=IMBALANCE_FRACTIONS, runs=5,
                        results_file="results/pldca_resnet50_imbalance.json",
                        log_file="results/run.log", epochs=None, images_per_class=None,
                        source_domains=None, target_domains=None,
                        weights_pattern="weights/{source}_source.pth"):
    """PLDCA (ResNet-50) under the imbalance protocol used for SDAP and PLDCA+AP.

    In every increment, the first class keeps all its target training images and each other
    class keeps a fraction `r` of them (IncrementalDataset, imbalance_factor). Evaluation uses all
    target images of the classes seen so far. Output keys follow the other imbalance files:
    "<source>_<target>_<run>_<r>_pldca" -> {all_data_until_increment_steps, each_step_eval, re_eval}.

    Results are saved atomically after every completed task; the JSON is loaded on start, so an
    interrupted run resumes by skipping keys that are already complete. A task interrupted in the
    middle is not saved and is re-run from its first increment.
    """
    results = json.load(open(results_file)) if Path(results_file).is_file() else {}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    lr = 1e-3
    batch_size = 24
    THRESHOLD = 0.15
    BETA = 0.6
    WARM_UP = 6
    STEP_UPDATE = 3

    for dataset_name in dataset_names:
        cfg = DATASETS[dataset_name]
        sources = source_domains if source_domains is not None else cfg["domains"]
        targets = target_domains if target_domains is not None else cfg["domains"]
        num_epochs = cfg["pldca_epochs"] if epochs is None else epochs
        n_increments = cfg["num_classes"] // 10
        print(f"\n=== Dataset: {dataset_name} ===")

        for imbalance in fractions:
            for run_id in range(1, runs + 1):
                for source in sources:
                    for target in targets:
                        if source == target:
                            continue

                        key = f"{source}_{target}_{run_id}_{imbalance:.1f}_pldca"
                        if key in results:
                            print(f"[Skip] {key}")
                            continue

                        print(f"\n[{run_id}, r={imbalance:.1f}] PLDCA {source} -> {target}")
                        started = time.time()
                        log_progress(log_file, f"START {key}")

                        pretrained_path = Path(weights_pattern.format(source=source))
                        if not pretrained_path.is_file():
                            raise FileNotFoundError(
                                f"Missing source weights: {pretrained_path}. "
                                f"Train them with scripts/train_sources.py first."
                            )
                        pretrained_model = PrototypeResNet(cfg["num_classes"], pretrained_path).to(device)
                        s_loader = setup_dataloader(Path(cfg["path"]) / source, Path(cfg["config"]),
                                                    range(cfg["num_classes"]), batch_size, False,
                                                    max_images_per_class=images_per_class)
                        trainer = PLDCA(pretrained_model, s_loader, lr, THRESHOLD, BETA, WARM_UP,
                                        STEP_UPDATE, device)

                        set_seed(run_id * 123)

                        for i in range(n_increments):
                            train_ids = range(i * 10, (i + 1) * 10)
                            val_ids = range(0, (i + 1) * 10)
                            # same arguments as run_proca_pldca, plus the imbalance of the training increment
                            t_loader = setup_dataloader(Path(cfg["path"]) / target, Path(cfg["config"]), train_ids,
                                                        batch_size, False, True, True, True,
                                                        imbalance_factor=imbalance,
                                                        max_images_per_class=images_per_class)
                            val_loader = setup_dataloader(Path(cfg["path"]) / target, Path(cfg["config"]), val_ids,
                                                          batch_size, False, False,
                                                          max_images_per_class=images_per_class)
                            trainer.increment(t_loader, val_loader, num_epochs)

                        experiment = {
                            "dataset": {
                                "path": cfg["path"],
                                "config": cfg["config"],
                                "source": source,
                                "target": target,
                                "n_classes": cfg["num_classes"],
                            },
                            "train": {"batch": batch_size, "class_increment": 10},
                            "eval": {"batch": batch_size},
                        }
                        trainer.re_eval = get_final_eval_metrics(trainer.model, experiment, device,
                                                                 cfg["num_classes"], 10)

                        results[key] = trainer.export_metrics()
                        save_results_atomic(results_file, results)

                        # export_metrics keeps integer step keys in memory (JSON turns them into strings)
                        final_acc = results[key]["all_data_until_increment_steps"][n_increments - 1]["acc"]
                        elapsed = time.time() - started
                        print(f"[Done] {key} final_acc={final_acc:.6f} elapsed={elapsed:.1f}s")
                        log_progress(log_file, f"END {key} acc={final_acc:.6f} elapsed={elapsed:.1f}s")


def train_smoke_source(source="Amazon", num_classes=31, images_per_class=3, epochs=1):
    """Tiny source model for smoke tests only (weights/smoke_<domain>_source.pth)."""
    weight_path = Path(f"weights/smoke_{source}_source.pth")
    if weight_path.exists():
        print(f"[Smoke] source model already trained: {weight_path}")
        return weight_path

    data_root = Path(dataset_path("office-31"))
    config_path = Path("data/office_31/config.json")
    batch_size = 32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader = setup_dataloader(data_root / source, config_path, range(num_classes), batch_size,
                                    True, max_images_per_class=images_per_class)
    val_loader = setup_dataloader(data_root / source, config_path, range(num_classes), batch_size,
                                  False, False, max_images_per_class=images_per_class)
    set_seed(987)
    trainer = SourceTrainer(num_classes, 1e-3, 5e-4, device, 0.1)
    trainer.train(train_loader, val_loader, epochs, 50)
    weight_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.model.save_model(weight_path)
    print(f"[Smoke] saved source model: {weight_path}")
    return weight_path


def run_smoke(fractions=SMOKE_FRACTIONS):
    """End-to-end smoke test on a tiny Office-31 subset, isolated from the real results."""
    log_file = "results/smoke_run.log"
    results_file = "results/smoke_pldca_resnet50_imbalance.json"
    # 12 images/class keeps the run tiny while still exercising the reply buffer (needs >= 10).
    images_per_class = 12
    print("[Smoke] tiny Amazon source model (1 epoch, 12 images/class)")
    train_smoke_source(images_per_class=images_per_class)
    log_imbalance_counts(DATASETS["office-31"], "DSLR", 10, 10, fractions, log_file,
                         max_images_per_class=images_per_class)
    run_pldca_imbalance(
        dataset_names=("office-31",),
        fractions=fractions,
        runs=1,
        results_file=results_file,
        log_file=log_file,
        epochs=1,
        images_per_class=images_per_class,
        source_domains=("Amazon",),
        target_domains=("DSLR",),
        weights_pattern="weights/smoke_{source}_source.pth",
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ProCA / PLDCA (ResNet-50) runs for CI-UDA")
    parser.add_argument("--imbalance", action="store_true",
                        help="run PLDCA under the imbalance protocol (writes results/pldca_resnet50_imbalance.json)")
    parser.add_argument("--smoke", action="store_true",
                        help="tiny end-to-end smoke test, writes results/smoke_pldca_resnet50_imbalance.json")
    parser.add_argument("--check-imbalance", action="store_true",
                        help="only print/log per-class target image counts for the given fractions")
    parser.add_argument("--datasets", nargs="+", default=["office-31"],
                        help="datasets for --imbalance (office-31, office-home, caltech-imagenet)")
    parser.add_argument("--fractions", nargs="+", type=float, default=None,
                        help=f"imbalance fractions (default: {IMBALANCE_FRACTIONS}; smoke: {SMOKE_FRACTIONS})")
    parser.add_argument("--runs", type=int, default=None,
                        help="number of runs (default: 5; smoke: 1)")
    args = parser.parse_args()

    if args.check_imbalance:
        log_imbalance_counts(DATASETS["office-31"], "DSLR", 10, 10,
                             args.fractions or SMOKE_FRACTIONS, "results/run.log")
    elif args.smoke:
        run_smoke(args.fractions or SMOKE_FRACTIONS)
    elif args.imbalance:
        run_pldca_imbalance(args.datasets,
                            fractions=args.fractions or IMBALANCE_FRACTIONS,
                            runs=args.runs or 5)
    else:
        # balanced runs used in the paper (all_metrics_proca_runs.json, all_metrics_pldca_runs.json)
        run_proca_pldca("proca", runs=args.runs or 5)
        run_proca_pldca("pldca", runs=args.runs or 5)
