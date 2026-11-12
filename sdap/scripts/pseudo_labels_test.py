import sys

import torch

sys.path.insert(0, "./")

from data.incremental import load_incremental_loader
from models.dinov2_projector import DINOv2Projector
from process.prototype import compute_forward, compute_labels
from utils.setup import load_configs_yaml


def compute_pseudo_label_accuracy(pseudo_labels, true_labels):
    correct = (pseudo_labels == true_labels).sum().item()
    total = len(true_labels)
    accuracy = correct / total
    return accuracy


if __name__ == "__main__":
    # TODO: ARRUMAR
    experiment = load_configs_yaml()
    model = DINOv2Projector(experiment["dataset"]["n_classes"], "base", "huge")
    model.load_weights(
        "weights/study_prototype/rw_to_prod_wo_con.pth", experiment["train"]["device"]
    )
    source_loader, target_loader, _ = load_incremental_loader(experiment, 0, 10)
    source_feas, source_labels, _ = compute_forward(model, source_loader)
    target_feas, target_labels, target_indexes = compute_forward(model, target_loader)
    pseudo_labels = compute_labels(
        model, source_feas, source_labels, target_feas, target_indexes
    )

    target_labels_re = torch.empty_like(target_labels)
    target_labels_re[target_indexes] = target_labels

    print(
        "Pseudo Labels Acc.:",
        compute_pseudo_label_accuracy(pseudo_labels, target_labels_re),
    )
