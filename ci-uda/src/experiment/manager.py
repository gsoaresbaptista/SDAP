import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from datasets import IncrementalDataset
from utils.settings import seed_worker

transform_train = v2.Compose(
    [
        v2.Resize((256, 256)),
        v2.RandomCrop((224, 224)),
        v2.RandomHorizontalFlip(),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)

transform_test = v2.Compose(
    [
        v2.Resize((224, 224)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)


def setup_dataloader(
    data_dir,
    config_path,
    class_ids,
    batch_size,
    is_train=True,
    shuffle=True,
    return_id=False,
    augmentation=False,
    imbalance_factor=1.0,
    max_images_per_class=None,
):
    transform = transform_train if is_train else transform_test

    dataset = IncrementalDataset(
        data_dir=data_dir,
        config_path=config_path,
        class_ids=class_ids,
        transform=transform,
        shuffle=shuffle,
        return_id=return_id,
        augmentation=augmentation,
        imbalance_factor=imbalance_factor,
        max_images_per_class=max_images_per_class,
    )

    g = torch.Generator()
    g.manual_seed(0)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        worker_init_fn=seed_worker,
        generator=g,
    )

    return dataloader
