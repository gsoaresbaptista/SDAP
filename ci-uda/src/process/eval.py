import numpy as np
import torch
from sklearn.metrics import f1_score

@torch.no_grad()
def eval(model, test_loader, device):
    model.eval()
    all_preds, all_label = [], []

    for batch in test_loader:
        x, y = batch[0].to(device), batch[1]
        logits, _ = model(x)
        preds = torch.argmax(logits, dim=-1)

        if len(all_preds) == 0:
            all_preds.append(preds.cpu())
            all_label.append(y.cpu())
        else:
            all_preds[0] = np.append(all_preds[0], preds.cpu(), axis=0)
            all_label[0] = np.append(all_label[0], y.cpu(), axis=0)

    all_preds, all_label = all_preds[0], all_label[0]
    accuracy = (all_preds == all_label).sum() / all_label.shape[0]

    return accuracy.item(), f1_score(all_label, all_preds, average="macro")
