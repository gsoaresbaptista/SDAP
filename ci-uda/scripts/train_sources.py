import argparse
import os
from pathlib import Path
import torch
from experiment.manager import setup_dataloader
from models.base_model import PrototypeResNet
from utils.settings import dataset_path, set_seed
from training.train_source import SourceTrainer

def main(dataset_names=None):
    datasets = {
        "office-home": {
            "path": dataset_path("office-home-pattern"),
            "config": Path("data/office_home/config.json"),
            "domains": ["product", "real_world", "art", "clipart"],
            "num_classes": 65
        },
        "office-31": {
            "path": dataset_path("office-31"),
            "config": Path("data/office_31/config.json"),
            "domains": ["Amazon", "DSLR", "Webcam"],
            "num_classes": 31
        },
        "caltech-imagenet": {
            "path": dataset_path("caltech_imagenet"),
            "config": Path("data/caltech_imagenet/config.json"),
            "domains": ["caltech", "imagenet"],
            "num_classes": 80
        }
    }

    if dataset_names:
        datasets = {name: cfg for name, cfg in datasets.items() if name in dataset_names}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_epochs = 100
    weight_decay = 5e-4
    batch_size = 32
    smooth = 0.1
    step = 50
    lr = 1e-3

    for dataset_name, cfg in datasets.items():
        for source in cfg["domains"]:
            weight_path = Path(f"weights/{source}_source.pth")
            if weight_path.exists():
                print(f"Skipping {source} ({dataset_name}) — already trained.")
                continue

            print(f"Training source for {source} ({dataset_name})...")
            all_ids = range(cfg["num_classes"])
            train_loader = setup_dataloader(cfg["path"] / source, cfg["config"], all_ids, batch_size, True)
            val_loader = setup_dataloader(cfg["path"] / source, cfg["config"], all_ids, batch_size, False, False)

            set_seed(987)
            trainer = SourceTrainer(cfg["num_classes"], lr, weight_decay, device, smooth)
            trainer.train(train_loader, val_loader, num_epochs, step)
            trainer.model.save_model(weight_path)
            print(f"Saved model: {weight_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the source models used by ProCA/PLDCA")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="restrict to datasets (office-31, office-home, caltech-imagenet); default: all")
    args = parser.parse_args()
    main(args.datasets)
