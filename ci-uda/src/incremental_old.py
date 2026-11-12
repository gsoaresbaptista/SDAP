from pathlib import Path
from torch.utils.data import DataLoader
from datasets import IncrementalDataset
from torchvision import transforms

def load_incremental_loader(experiment, class_start, class_qtd=10, device="cuda", drop_last=True, model=None):
    data_dir = Path(experiment["dataset"]["path"])
    config_path = experiment["dataset"]["config"]
    batch_size = experiment["train"]["batch"]
    augment = experiment["train"].get("augmentation", True)

    # define classes
    train_ids = range(class_start, class_start + class_qtd)
    eval_ids = range(0, class_start + class_qtd)

    # transformações padrão (iguais às do setup_dataloader)
    base_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
    ])

    # loaders
    source_path = data_dir / experiment["dataset"]["source"]
    target_path = data_dir / experiment["dataset"]["target"]

    source_dataset = IncrementalDataset(
        source_path,
        config_path,
        class_ids=range(experiment["dataset"]["n_classes"]),
        transform=base_transform,
        shuffle=True,
    )

    target_dataset = IncrementalDataset(
        target_path,
        config_path,
        class_ids=train_ids,
        transform=base_transform,
        shuffle=True,
        return_id=True,
        augmentation=augment,
    )

    eval_dataset = IncrementalDataset(
        target_path,
        config_path,
        class_ids=eval_ids,
        transform=base_transform,
        shuffle=False,
    )

    source_loader = DataLoader(source_dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last)
    target_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    return source_loader, target_loader, eval_loader
