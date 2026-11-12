import sys

sys.path.insert(0, "./")

from data.incremental import load_incremental_loader
from models.dinov2_projector import DINOv2Projector
from process.prototype import compute_class_centroids, compute_forward
from utils.preview import plot_source_target
from utils.setup import load_configs_yaml

if __name__ == "__main__":
    # TODO: arrumar
    experiment = load_configs_yaml()
    model = DINOv2Projector(experiment["dataset"]["n_classes"], "base", "huge")
    model.load_weights(
        "weights/study_prototype/rw_to_prod_mmd_prot.pth", experiment["train"]["device"]
    )
    source_loader, target_loader, _ = load_incremental_loader(experiment, 0, 10)
    source_centroids = compute_class_centroids(model, source_loader).to("cpu")
    target_feas, target_labels, _ = compute_forward(model, target_loader)
    plot_source_target(
        source_centroids,
        range(65),
        target_feas,
        target_labels,
        "Source to Target (Real World -> Product)",
        "weights/study_prototype/source_target_mmd_prot.png",
    )
