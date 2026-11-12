import os
import sys

sys.path.insert(0, "./")

from pathlib import Path

from torch.utils.data import DataLoader

from data.dataset import ImageDataset
from data.transform import get_transforms
from models.feature_matrix import FeatureMatrix
from utils.setup import load_configs_yaml
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
import copy
from data.incremental import load_incremental_loader
from torch.nn import functional as F


def get_data(path, remap, curr_classes):
    images, labels = [], []

    if len(remap) == 0:
        remap = {label: i for i, label in enumerate(sorted(os.listdir(path)))}
        curr_classes = remap.keys()

    for name in curr_classes:
        curr_images = list((path / name).glob("*"))
        labels.extend(len(curr_images) * [remap[name]])
        images.extend(curr_images)
    return images, labels


def gen():
    experiment = load_configs_yaml()

    # load settings
    DATASET = experiment["dataset"]["name"]
    DOMAINS = experiment["dataset"]["domains"]
    remap = experiment["dataset"]["remap"]
    path = Path(experiment["dataset"]["path"])
    batch = experiment["eval"]["batch"]
    n_classes = experiment["dataset"]["n_classes"]

    train_transform, test_transform, aug_transform = get_transforms(experiment)
    model = FeatureMatrix(n_classes, experiment["train"]["device"], experiment['backbone'])

    transforms = {
        "train": train_transform,
        "test": test_transform,
        "aug": aug_transform,
    }
    posfix =  '_resnet' if experiment['backbone'] == 'resnet' else ''

    for domain in DOMAINS:
        # load domain dataset
        domain_path = path / domain
        images, labels = get_data(domain_path, remap, remap.keys())

        for name, transform in transforms.items():
            dataset = ImageDataset(images, labels, transform)
            loader = DataLoader(dataset, 32, False)
            model = FeatureMatrix(n_classes, experiment["train"]["device"], experiment['backbone'])
            model._freeze()
            model.create_and_save_feature_matrix(
                loader, f"features/{DATASET}/{domain}_{name}{posfix}.bin", True
            )

if __name__ == "__main__":
    gen()
