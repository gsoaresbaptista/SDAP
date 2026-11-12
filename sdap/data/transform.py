import torch
from torchvision.transforms import v2


def get_transforms(experiment):
    img_size = experiment["dataset"]["image_size"]

    train_transform = v2.Compose(
        [
            v2.Resize(img_size + 32),
            v2.RandomResizedCrop(size=img_size),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandAugment(3, 6),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    test_transform = v2.Compose(
        [
            v2.Resize((img_size, img_size)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    aug_transform = v2.Compose(
        [
            v2.RandomResizedCrop(size=img_size),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandAugment(2, 10),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return train_transform, test_transform, aug_transform
