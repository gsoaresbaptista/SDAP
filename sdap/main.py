import argparse
import json
import pathlib
import itertools
import copy
from pathlib import Path

import torch
from models.projector import Projector
from process.train import train
from utils.setup import load_configs_yaml
from models.feature_matrix import FEATURE_SIZE
from process.train_pldca import PLDCA
from data.incremental import load_incremental_loader


def run_table(imbalance_gen: bool = False, disable: bool = False):
    DATASETS = [
        "office-31",
        "office-home",
        "caltech-imagenet",
    ]
    EPOCHS = {"office-31": 10, "office-home": 30, "caltech-imagenet": 15}
    running = True

    imbalance_values = [1.0]
    disabled = []

    if imbalance_gen:
        imbalance_values.extend([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        imbalance_values = list(reversed(imbalance_values))

    if disable:
        disable_mech = ['cl', 'triplet', 'dist', 'il']
        for i in range(1, len(disable_mech) + 1):
            for combo in itertools.combinations(disable_mech, i):
                disabled.append(list(combo))

    while running:
        try:
            for backbone in FEATURE_SIZE.keys():
                FILE_NAME = f"all_metrics_{backbone}_disabled.json"

                if pathlib.Path(FILE_NAME).is_file():
                    with open(FILE_NAME, "r") as old_result:
                        results = json.loads(old_result.read())
                else:
                    results = {}

                for imbalance in imbalance_values:
                    for dataset_name in DATASETS:
                        experiment = load_configs_yaml(
                            dataset={"name": dataset_name, "source": "", "target": ""}
                        )
                        experiment['backbone'] = backbone
                        experiment["train"]["imbalance_factor"] = imbalance
                        experiment["train"]["epochs"] = EPOCHS[dataset_name]
                        DOMAINS = experiment["dataset"]["domains"]

                        for disable_i in disabled or ['']:
                            print(f"Disabled: {disable_i}")

                            for i in range(5):
                                for source in DOMAINS:
                                    for target in DOMAINS:
                                        if (source == target) or (
                                            f"{source}_{target}_{i+1}_{imbalance}_{"_".join(disable_i)}" in results
                                        ):
                                            continue

                                        experiment["dataset"]["source"] = source
                                        experiment["dataset"]["target"] = target

                                        print(f"[{i+1}, {imbalance}] {source} -> {target}")
                                        features = FEATURE_SIZE[experiment['backbone']]
                                        model = Projector(
                                            experiment["dataset"]["n_classes"], "small_2", features
                                        )
                                        steps, single_steps, final_metrics, label_distillation = train(
                                            model, experiment, 0.1, disable_i
                                        )
                                        results[f"{source}_{target}_{i+1}_{imbalance}_{"_".join(disable_i)}"] = {
                                            "all_data_until_increment_steps": steps,
                                            "each_step_eval": single_steps,
                                            "re_eval": final_metrics,
                                            # "label_distillation": label_distillation,
                                        }
                                        with open(FILE_NAME, "w") as json_file:
                                            json.dump(results, json_file, indent=4, sort_keys=True)
        except KeyboardInterrupt:
            exit(0)
        except Exception as error:
            print(error)
            continue
        else:
            running = False
            break


def main():
    experiment = load_configs_yaml()
    features = FEATURE_SIZE[experiment['backbone']]
    model = Projector(experiment["dataset"]["n_classes"], "small_2", features)
    steps, single_step, final_metrics, label_distillation = train(model, experiment, 0.1, [])
    print(steps, single_step, final_metrics)


def run_pldca_with_projector(experiment, imbalance_gen=False):
    datasets = {
        "office-31": {
            "path": "datasets/office-31",
            "config": "data/office_31/config.json",
            "domains": ["Amazon", "DSLR", "Webcam"],
            "num_classes": 31,
            "epochs": 10,
        },
        "office-home": {
            "path": "datasets/office-home-pattern",
            "config": "data/office_home/config.json",
            "domains": ["product", "real_world", "art", "clipart"],
            "num_classes": 65,
            "epochs": 30,
        },
        "caltech-imagenet": {
            "path": "datasets/caltech_imagenet",
            "config": "data/caltech_imagenet/config.json",
            "domains": ["caltech", "imagenet"],
            "num_classes": 84,
            "epochs": 15,
        },
    }

    # == NOVO: lista de imbalance para rodar igual ao run_table ==
    imbalance_values = [1.0]
    if imbalance_gen:
        imbalance_values = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0][::-1]

    lr = 1e-3
    THRESHOLD = 0.15
    BETA = 0.6
    WARM_UP = 6
    STEP_UPDATE = 3

    results_file = f"pldca_with_projector_imbalance.json"
    results = json.load(open(results_file)) if Path(results_file).is_file() else {}

    for dataset_name, cfg in datasets.items():
        print(f"\n=== Dataset: {dataset_name} ===")

        for imbalance in imbalance_values:
            print(f"== Running with imbalance={imbalance} ==")

            experiment = load_configs_yaml(
                dataset={"name": dataset_name, "source": "", "target": ""}
            )
            experiment['backbone'] = 'dinov2_vitg14'
            experiment["train"]["imbalance_factor"] = imbalance
            experiment["train"]["epochs"] = cfg['epochs']

            for run_id in range(1, 6):
                for source in cfg["domains"]:

                    features = FEATURE_SIZE[experiment['backbone']]
                    experiment["dataset"]["source"] = source
                    experiment["dataset"]["target"] = source

                    source_loader, *_ = load_incremental_loader(
                        experiment, 0, cfg['num_classes'], 'cuda', False
                    )

                    source_model = Projector(
                        experiment["dataset"]["n_classes"], "small_2", features
                    ).to('cuda')

                    optim = torch.optim.SGD(source_model.parameters(), lr, 0.9, weight_decay=1e-6)

                    # pre-train
                    for _ in range(100):
                        for inputs, targets, *_ in source_loader:
                            inputs, targets = inputs.to('cuda'), targets.to('cuda')
                            _, outputs = source_model(inputs)
                            loss = torch.nn.CrossEntropyLoss()(outputs, targets)
                            loss.backward()
                            optim.step()
                            optim.zero_grad()

                    for target in cfg["domains"]:
                        if source == target:
                            continue

                        key = f"{source}_{target}_{dataset_name}_{run_id}_{imbalance}"
                        if key in results:
                            print(f"[Skip] {key}")
                            continue

                        experiment["dataset"]["target"] = target

                        pldca = PLDCA(
                            copy.deepcopy(source_model),
                            cfg['num_classes'],
                            source_loader,
                            lr,
                            THRESHOLD,
                            BETA,
                            WARM_UP,
                            STEP_UPDATE,
                            'cuda'
                        )

                        for i in range(cfg['num_classes'] // 10):
                            _, t_loader, val_loader = load_incremental_loader(
                                experiment, i * 10, 10, 'cuda', False
                            )
                            pldca.increment(t_loader, val_loader, cfg['epochs'])

                        results[key] = pldca.export_metrics()

                        with open(results_file, "w") as f:
                            json.dump(results, f, indent=4)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="RAPTOR")
    parser.add_argument("--table", action="store_true", default=False)
    parser.add_argument("--imbalance", action="store_true", default=False)
    parser.add_argument("--disable", action="store_true", default=False)
    parser.add_argument("--pldca", action="store_true", default=False)
    args = parser.parse_args()

    if args.table:
        run_table(args.imbalance, args.disable)
    elif args.pldca:
        experiment = load_configs_yaml()
        run_pldca_with_projector(experiment)
    else:
        main()
