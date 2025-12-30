import torch
from torch.utils.data import Dataset
from torchvision.io import read_image


class ImageDataset(Dataset):
    def __init__(self, images, labels, transform):
        super().__init__()
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = read_image(self.images[index])
        label = self.labels[index]

        if self.transform:
            image = self.transform(image)

        return image, label, index


class FeatureDataset(Dataset):
    def __init__(self, features, labels, device="cpu", aug_feas=None):
        self.features = torch.tensor(features, device=device)
        self.labels = torch.tensor(labels, device=device)
        self.aug_feas = aug_feas

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]

        if self.aug_feas is not None:
            return feature, label, idx, self.aug_feas[idx]

        return feature, label, idx


class AugmentedDataset(Dataset):
    def __init__(self, features_1, features_2, device="cpu"):
        self.features_1 = torch.tensor(features_1, device=device)
        self.features_2 = torch.tensor(features_2, device=device)

    def __len__(self):
        return len(self.features_1)

    def __getitem__(self, idx):
        feature_1 = self.features_1[idx]
        feature_2 = self.features_2[idx]
        return feature_1, feature_2
