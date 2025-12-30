from pathlib import Path
import torch

import numpy as np
from torch.utils.data import DataLoader

from data.dataset import AugmentedDataset, FeatureDataset
from models.feature_matrix import FeatureMatrix
from scipy.spatial.distance import cosine


def load_augmented_loader(
    experiment, class_start, class_qtd=10, device="cuda", drop_last=True
):
    posfix =  '_resnet' if experiment['backbone'] == 'resnet' else ''
    train_indexes = range(class_start, class_start + class_qtd)
    train_batch = experiment["train"]["batch"]

    source_path = Path(experiment["dataset"]["features"])
    source, target = experiment["dataset"]["source"], experiment["dataset"]["target"]
    source_train = FeatureMatrix.load_features(source_path / f"{source}_train{posfix}.bin")
    features = source_train["features"]
    mean, std = features.mean(axis=0), features.std(axis=0)

    aug_1_data = FeatureMatrix.load_features(source_path / f"{target}_aug_1{posfix}.bin")
    aug_mask = np.isin(aug_1_data["labels"], train_indexes)
    features_aug_1 = (aug_1_data["features"][aug_mask] - mean) / std

    aug_2_data = FeatureMatrix.load_features(source_path / f"{target}_aug_2{posfix}.bin")
    features_aug_2 = (aug_2_data["features"][aug_mask] - mean) / std

    dataset = AugmentedDataset(features_aug_1, features_aug_2, device)
    return DataLoader(
        dataset,
        train_batch,
        True,
        drop_last=drop_last,
    )


def load_source_aug_loader(experiment, device="cuda"):
    source = experiment["dataset"]["source"]
    source_path = Path(experiment["dataset"]["features"])
    posfix =  '_resnet' if experiment['backbone'] == 'resnet' else ''
    source_train = FeatureMatrix.load_features(source_path / f"{source}_aug{posfix}.bin")
    features = source_train["features"]
    train_batch = experiment["train"]["batch"]

    # normalize features
    mean, std = features.mean(axis=0), features.std(axis=0)
    features = (features - mean) / std
    labels = source_train["labels"]
    s_dataset = FeatureDataset(features, labels, device)

    return DataLoader(
        s_dataset,
        train_batch,
        True,
        drop_last=False,
    )


def load_incremental_loader(
    experiment, class_start, class_qtd=10, device="cuda", drop_last=True, model=None
):
    source, target = experiment["dataset"]["source"], experiment["dataset"]["target"]
    source_path = Path(experiment["dataset"]["features"])
    target_path = Path(experiment["dataset"]["features"])
    train_batch, eval_batch = experiment["train"]["batch"], experiment["eval"]["batch"]
    imbalance_factor = experiment["train"]["imbalance_factor"]
    posfix =  '_resnet' if experiment['backbone'] == 'resnet' else ''

    # Calculates class intervals
    if experiment["dataset"]["name"] == "caltech-imagenet" and target == "imagenet":
        lists = [
            experiment["dataset"]["increments"][i + 1]
            for i in range((class_start // class_qtd) + 1)
        ]
        train_indexes = experiment["dataset"]["increments"][
            (class_start // class_qtd) + 1
        ]
        eval_indexes = [x for list in lists for x in list]
    else:
        train_indexes = range(class_start, class_start + class_qtd)
        eval_indexes = range(0, class_start + class_qtd)

    # source loader
    source_train = FeatureMatrix.load_features(source_path / f"{source}_train{posfix}.bin")
    features = source_train["features"]

    # normalize features
    mean, std = features.mean(axis=0), features.std(axis=0)
    features = (features - mean) / std

    labels = source_train["labels"]
    s_dataset = FeatureDataset(features, labels, device)

    source_loader = DataLoader(
        s_dataset,
        train_batch,
        True,
        drop_last=drop_last,
    )

    # target loader
    target_train = FeatureMatrix.load_features(target_path / f"{target}_test{posfix}.bin")

    # Determinar as classes do target
    target_train = FeatureMatrix.load_features(target_path / f"{target}_test{posfix}.bin")
    target_mask = np.isin(target_train["labels"], train_indexes)

    features = target_train["features"][target_mask]
    features = (features - mean) / std
    labels = target_train["labels"][target_mask]

    aug_train = FeatureMatrix.load_features(target_path / f"{target}_test{posfix}.bin")
    aug_feas = aug_train["features"][target_mask]
    aug_feas = (features - mean) / std

    if imbalance_factor < 1.0:
        indices_final = []
        for i, cls in enumerate(np.unique(labels)):
            indices = np.where(labels == cls)[0]
            n_keep = len(indices) * imbalance_factor
            n_keep = max(int(n_keep), 1)
            if cls == class_start:
                indices_final.extend(indices)
            else:
                chosen = indices[:n_keep]
                indices_final.extend(chosen)
        indices_final = np.array(indices_final)
        features = features[indices_final]
        labels = labels[indices_final]
        aug_feas = aug_feas[indices_final]

    t_dataset = FeatureDataset(features, labels, device, aug_feas)

    target_loader = DataLoader(
        t_dataset,
        train_batch,
        True,
        drop_last=drop_last,
    )

    # eval loader (previous and current target data)
    eval_mask = np.isin(target_train["labels"], eval_indexes)
    features = target_train["features"][eval_mask]
    features = (features - mean) / std
    labels = target_train["labels"][eval_mask]
    eval_dataset = FeatureDataset(features, labels, device)

    eval_loader = DataLoader(
        eval_dataset,
        eval_batch,
        False,
        drop_last=False,
    )

    return source_loader, target_loader, eval_loader
