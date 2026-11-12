import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from tqdm import tqdm

from src.experiment.manager import setup_dataloader
from src.models.base_model import PrototypeResNet


def compute_confusion_matrix(true_cls, pred_cls, num_cls):
    TP, FP, FN, TN = 0, 0, 0, 0
    for trues, preds in zip(true_cls, pred_cls):
        TP += len([i for i in preds if i in trues])
        FP += len([i for i in preds if i not in trues])
        FN += len([i for i in range(num_cls) if (i in trues) and (i not in preds)])
        TN += len([i for i in range(num_cls) if (i not in preds) and (i not in trues)])
    return np.array([[TP, FP], [FN, TN]])


@torch.no_grad()
def detect_shared_classes(num_classes, dataloader, threshold, device):
    shared_classes = torch.zeros(num_classes, device=device)

    for inputs, *_ in dataloader:
        inputs = inputs.to(device)
        outputs, _ = pretrained_model(inputs)
        probs = F.softmax(outputs, dim=1)
        shared_classes += probs.sum(dim=0)

    diff = shared_classes.max() - shared_classes.min()
    shared_classes = (shared_classes - shared_classes.min()) / diff

    return (
        shared_classes.cpu(),
        (shared_classes >= threshold).nonzero(as_tuple=True)[0].tolist(),
    )


true_classes = [
    list(range(0, 10)),
    list(range(10, 20)),
    list(range(20, 30)),
    list(range(30, 40)),
    list(range(40, 50)),
    list(range(50, 60)),
]

weights_dir = 'weights/'
model_files = [f for f in os.listdir(weights_dir) if f.endswith('.pth')]
results = []

SOURCE = 'art'
data_dir = Path('datasets/office-home-pattern')
conf_dir = Path('data/office_home/config.json')
data_path = data_dir / SOURCE
THRESHOLD = 0.15
DEVICE = 'cuda'
NUM_CLS = 65

models_results = defaultdict(list)

for model_file in tqdm(model_files):
    model_path = os.path.join(weights_dir, model_file)
    pretrained_model = PrototypeResNet(NUM_CLS, model_path).to(DEVICE)
    s_loader = (data_dir / SOURCE, conf_dir, range(NUM_CLS), 24, False)

    for i in range(NUM_CLS // 10):
        train_ids = range(i * 10, (i + 1) * 10)
        t_loader = setup_dataloader(data_path, conf_dir, train_ids, 24, False, True, True)
        _, increment_classes = detect_shared_classes(NUM_CLS, t_loader, THRESHOLD, DEVICE)
        models_results[model_path].append(increment_classes)

    pred_classes = models_results[model_path]
    conf_matrix = compute_confusion_matrix(true_classes, pred_classes, NUM_CLS)

    # Calcular pontuação do modelo: TP + TN - FP - FN
    score = conf_matrix[0, 0] + conf_matrix[1, 1] - conf_matrix[0, 1] - conf_matrix[1, 0]
    results.append((model_file, score, conf_matrix))

results.sort(key=lambda x: x[1], reverse=True)
best_model = results[0]

results_df = pd.DataFrame(results, columns=['Model File', 'Score', 'Confusion Matrix'])
results_df.to_csv(f'{THRESHOLD}_model_performance_results.csv', index=False)
