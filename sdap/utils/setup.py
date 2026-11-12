import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def set_seed(seed=42):
    torch.manual_seed(seed)
    random.seed(0)
    np.random.seed(0)
    torch.use_deterministic_algorithms(True)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def load_configs_yaml(dataset=None):
    with open("experiment.yml", "r") as conf_file:
        experiment = yaml.safe_load(conf_file)

    if dataset is not None:
        experiment["dataset"] = dataset

    # Update Dataset information based on the name provided in experiment.yml
    with open("datasets.yml", "r") as conf_file:
        datasets = yaml.safe_load(conf_file)

    # Loads the information regarding the dataset
    dataset = experiment["dataset"]
    datasets[dataset["name"]]["source"] = dataset["source"]
    datasets[dataset["name"]]["target"] = dataset["target"]
    datasets[dataset["name"]]["name"] = dataset["name"]
    dataset = datasets[dataset["name"]]

    if "remap" in dataset:  # If the remap was informed, loads the JSON file
        with open(dataset["remap"], "r") as remap_file:
            dataset["remap"] = json.load(remap_file)
    else:  # Otherwise, open all folders in alphabetical order
        dataset_path = Path(dataset["path"]) / dataset["source"]
        sorted_classes = sorted(folder.name for folder in dataset_path.glob("*/"))
        dataset["remap"] = {name: i for i, name in enumerate(sorted_classes)}

    experiment["dataset"] = dataset
    experiment["train"]["learning_rate"] = float(experiment["train"]["learning_rate"])

    return experiment
