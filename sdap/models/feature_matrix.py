import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.base_model import PrototypeResNet

FEATURE_SIZE = {
    "dinov2_vitg14": 1536,
    # "dinov2_vitl14": 1024,
    # "dinov2_vitb14": 768,
    # "dinov2_vits14": 384,
    # "convnext_tiny": 1000,
    # "convnext_small": 1000,
    # "convnext_base": 1000,
    # "convnext_large": 1000,
    # "resnet_finetuned": 1536,
    # "resnet50": 2048,
}

class FeatureMatrix:
    def __init__(self, n_classes, device="cpu", backbone='dinov2'):
        self.device = device
        self.backbone_name = backbone

        if backbone == 'dinov2_vitg14':
            self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitg14_reg")
        elif backbone == 'resnet':
            self.backbone = PrototypeResNet(n_classes)
        elif backbone == 'dinov2_vits14':
            self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg")

    def _freeze(self):
        self.backbone.requires_grad_(False)
        self.backbone.eval()
        self.backbone.to(self.device)

    def create_and_save_feature_matrix(
        self, dataloader: DataLoader, save_path: str, verbose=False
    ):
        num_samples = len(dataloader.dataset)

        if self.backbone_name == 'resnet':
            feature_dim = 768
        else:
            feature_dim = self.backbone.norm.normalized_shape[0]

        feature_matrix = np.zeros((num_samples, feature_dim), dtype=np.float32)
        labels = np.zeros(num_samples, dtype=np.int64)

        data_iterator = tqdm(dataloader, desc="Processing batches", disable=not verbose)

        with torch.no_grad():
            for images, targets, indices in data_iterator:
                images = images.to(self.device)
                features = self.backbone(images)
                feature_matrix[indices.cpu().numpy()] = features.cpu().numpy()
                labels[indices.cpu().numpy()] = targets.cpu().numpy()

        data = {"features": feature_matrix, "labels": labels}
        with open(save_path, "wb") as f:
            pickle.dump(data, f)

    def delete_backbone(self):
        del self.backbone
        torch.cuda.empty_cache()

    @staticmethod
    def load_features(file_path: str):
        with open(file_path, "rb") as f:
            return pickle.load(f)
