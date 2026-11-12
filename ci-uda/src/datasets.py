import hashlib
import json
import os

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import RandAugment, v2


class IncrementalDataset(Dataset):
    def __init__(
        self,
        data_dir,
        config_path,
        class_ids=None,
        transform=None,
        shuffle=False,
        return_id=False,
        augmentation=False,
        imbalance_factor=1.0,
        max_images_per_class=None,
    ):
        self.data_dir = data_dir
        self.transform = transform
        self.class_to_idx = self.load_class_ids(config_path)
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.imbalance_factor = imbalance_factor
        self.max_images_per_class = max_images_per_class
        self.image_paths, self.labels = self.load_data(class_ids)
        self.return_id = return_id
        self.aug = augmentation

        self.strong = RandAugment(2, 10)
        self.weak = v2.Compose(
            [
                v2.Resize((256, 256)),
                v2.RandomCrop((224, 224)),
                v2.RandomHorizontalFlip(),
            ]
        )

        if shuffle:
            self.deterministic_shuffle()

    def load_class_ids(self, config_path):
        with open(config_path, 'r') as f:
            class_to_idx = json.load(f)
        return class_to_idx

    def load_data(self, class_ids):
        image_paths, labels = [], []

        for i, class_id in enumerate(class_ids):
            class_name = self.idx_to_class.get(class_id)
            class_dir = os.path.join(self.data_dir, class_name)
            imgs = [os.path.join(class_dir, x) for x in os.listdir(class_dir)]

            if self.max_images_per_class is not None:
                imgs = imgs[:self.max_images_per_class]

            if self.imbalance_factor < 1.0:
                if i == 0:
                    selected = imgs
                else:
                    n = max(1, int(len(imgs) * self.imbalance_factor))
                    selected = imgs[:n]
            else:
                selected = imgs

            image_paths.extend(selected)
            labels.extend([class_id] * len(selected))

        return image_paths, labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image_transformed = self.transform(image)

        if self.return_id:
            if self.aug:
                return (
                    image_transformed,
                    label,
                    idx,
                    self.transform(self.strong(image)),
                    self.transform(self.weak(image)),
                )
            return image_transformed, label, idx

        if self.aug:
            return image_transformed, label, self.transform(self.strong(image))
        return image_transformed, label

    def deterministic_shuffle(self):
        combined = list(zip(self.image_paths, self.labels))
        combined.sort(key=lambda x: hashlib.md5(os.path.basename(x[0]).encode()).hexdigest())
        self.image_paths, self.labels = zip(*combined)
