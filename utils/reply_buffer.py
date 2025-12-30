from collections import defaultdict
from itertools import cycle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class ReplyBuffer:
    def __init__(self, imgs_per_class: int = 10):
        self.exemplar_set = defaultdict(list)
        self.soft_pred = defaultdict(list)
        self.labels = defaultdict(list)
        self.m = imgs_per_class

    def remove_label(self, label):
        if label in self.exemplar_set:
            del self.exemplar_set[label]
        if label in self.soft_pred:
            del self.soft_pred[label]
        if label in self.labels:
            del self.labels[label]

    @torch.no_grad()
    def compute_class_mean(self, model, images):
        model.eval()
        feas, output = model(images.cuda())
        output, feas = output.cpu(), feas.cpu()

        feature_extractor_output = F.normalize(feas).numpy()
        class_mean = np.mean(feature_extractor_output, axis=0)
        output = nn.Softmax(dim=1)(output)

        return class_mean, feature_extractor_output, output

    def construct_exemplar_set(self, model, images, label):
        class_mean, feature_extractor_output, buffer_output = self.compute_class_mean(
            model, images
        )
        exemplar, soft_predar = [], []
        now_class_mean = np.zeros((1, model.head.in_features))

        for i in range(self.m):
            diff = class_mean - (now_class_mean + feature_extractor_output) / (i + 1)
            index = np.argmin(np.linalg.norm(diff, axis=1))
            now_class_mean += feature_extractor_output[index]
            exemplar.append(images[index])
            soft_predar.append(buffer_output[index])

        labels = [label for _ in range(self.m)]

        if len(exemplar) and len(soft_predar) and len(labels):
            self.exemplar_set[label] = exemplar
            self.soft_pred[label] = soft_predar
            self.labels[label] = labels

    def get_dataloader(self, batch_size: int = 24):
        if len(self.exemplar_set) == 0:
            return None

        valid_exemplars = [exemplar for exemplar in self.exemplar_set.values()]
        valid_labels = [label for label in self.labels.values()]
        valid_soft_preds = [pred for pred in self.soft_pred.values()]

        exemplars = torch.tensor(np.concatenate(valid_exemplars), dtype=torch.float32)
        labels = torch.tensor(np.concatenate(valid_labels))
        soft_preds = torch.cat([torch.stack(pred) for pred in valid_soft_preds])

        dataset = TensorDataset(exemplars, labels, soft_preds)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        return cycle(dataloader)
