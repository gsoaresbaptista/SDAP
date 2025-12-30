import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import hmean

from process.knn import (
    reciprocal_knn,
    remove_outliers_cosine,
    remove_outliers_euclidean,
)
from utils.data import get_data


@torch.no_grad()
def generate_pseudo_labels(model, device, target_loader, increment_classes):
    model.eval()
    all_feas, all_outputs, all_indices = [], [], []

    for inputs, _, idx, _ in target_loader:
        inputs = inputs.to(device)
        feas, outputs = model(inputs)
        all_feas.append(feas.cpu())
        all_outputs.append(outputs.cpu())
        all_indices.append(idx.float())

    all_fea = torch.cat(all_feas, dim=0)
    all_output = F.softmax(torch.cat(all_outputs, dim=0), dim=1)
    all_idx = torch.cat(all_indices, dim=0)

    all_fea = F.normalize(
        torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1), p=2, dim=1
    )

    confi_class_idx = torch.tensor(increment_classes, device=all_fea.device)
    initc = F.normalize(all_output.T @ all_fea, p=2, dim=1)[confi_class_idx]
    similarities = torch.mm(all_fea, initc.t())
    prediction_c1 = confi_class_idx[similarities.argmax(dim=1)]

    all_idxs = all_idx.int().tolist()
    preds = prediction_c1.tolist()
    confs = all_output.max(dim=1)[0] > 0.6

    return dict(zip(all_idxs, preds)), dict(zip(all_idxs, confs))


def build_labels_tensor(labels, device, indices):
    labels = [labels[idx.item()] for idx in indices]
    return torch.tensor(labels, device=device)


@torch.no_grad()
def old_detect_shared_classes(model, device, dataloader, num_classes, threshold=0.1):
    model.eval()
    shared_classes = torch.zeros(num_classes, device=device)

    for inputs, *_ in dataloader:
        inputs = inputs.to(device)
        _, outputs = model(inputs)
        probs = F.softmax(outputs, dim=1)
        shared_classes += probs.sum(dim=0)

    diff = shared_classes.max() - shared_classes.min()
    shared_classes = (shared_classes - shared_classes.min()) / diff

    return (
        (
            torch.arange(0, num_classes, device=device)[shared_classes >= threshold]
        ).tolist(),
        (shared_classes[shared_classes >= threshold]),
    )

def calcular_similaridades_medias(pseudo_labels, max_similarities):
    similaridades_por_label = {}
    for label, sim in zip(pseudo_labels, max_similarities):
        if label not in similaridades_por_label:
            similaridades_por_label[label] = []
        similaridades_por_label[label].append(sim)
    medias_por_label = {}
    for label in similaridades_por_label:
        medias_por_label[label] = np.mean(similaridades_por_label[label])
    labels_ordenados = sorted(medias_por_label.keys())
    mu = [medias_por_label[label] for label in labels_ordenados]
    return mu

def assign_pseudo_labels_with_similarity(source_features, source_labels, target_features):
    similarity_matrix = cosine_similarity(target_features, source_features)
    max_similarity_indices = np.argmax(similarity_matrix, axis=1)
    max_similarities = similarity_matrix[np.arange(len(target_features)), max_similarity_indices]
    pseudo_labels = source_labels[max_similarity_indices]
    return pseudo_labels, max_similarities

def compute_centroids(s_feas, s_labels):
    unique_labels = s_labels.unique()
    centroids = torch.stack([s_feas[s_labels == label].mean(dim=0) for label in unique_labels])
    return centroids, unique_labels

# def detect_shared_classes(model, source_loader, target_loader, classes=None, beta=0.3):
#     s_feas, s_labels, *_ = get_data(model, source_loader)
#     t_feas, *_, all_idxs = get_data(model, target_loader)

#     # s_feas, s_labels = remove_outliers_cosine(s_feas, s_labels, 0.75)
#     # s_feas, s_labels = compute_centroids(s_feas, s_labels)
#     s_feas, s_labels = s_feas.cpu().numpy(), s_labels.cpu().numpy()
#     t_feas, all_idxs = t_feas.cpu().numpy(), all_idxs.cpu().numpy()

#     # if classes is not None:
#     #     mask = np.isin(s_labels, classes)
#     #     s_feas = s_feas[mask]
#     #     s_labels = s_labels[mask]

#     # preds, confs = assign_pseudo_labels_with_similarity(s_feas, s_labels, t_feas)
#     preds, preds_w, _, _, confs = reciprocal_knn(s_feas, s_labels, t_feas, classes, 0)

#     confs_mask = np.where(confs > beta, True, False)
#     mu = calcular_similaridades_medias(preds[confs_mask], confs[confs_mask])

#     if classes is not None:
#         mask = ~np.isin(preds, classes)
#         preds[mask] = -1
#         confs[mask] = 0

#     return (
#         dict(zip(all_idxs, preds_w.tolist())),
#         dict(zip(all_idxs, preds_w.tolist())),
#         dict(zip(all_idxs, confs_mask.tolist())),
#         np.unique(preds[confs_mask]).tolist(),
#         mu,
#     )

def detect_shared_classes(model, source_loader, target_loader, classes=None, beta=0.3):
    s_feas, s_labels, *_ = get_data(model, source_loader)
    t_feas, *_, all_idxs = get_data(model, target_loader)

    # s_feas, s_labels = remove_outliers_cosine(s_feas, s_labels, 0.95)
    s_feas, s_labels = compute_centroids(s_feas, s_labels)
    s_feas, s_labels = s_feas.cpu().numpy(), s_labels.cpu().numpy()
    t_feas, all_idxs = t_feas.cpu().numpy(), all_idxs.cpu().numpy()

    if classes is not None:
        mask = np.isin(s_labels, classes)
        s_feas = s_feas[mask]
        s_labels = s_labels[mask]

    preds, preds_without_conf, uniques, qtds, mu = reciprocal_knn(
        s_feas, s_labels, t_feas, classes, beta
    )

    confs = (preds != -1).tolist()
    all_idxs = all_idxs.tolist()

    # mask = qtds > 1
    # uniques = uniques[mask]
    # qtds = qtds[mask]
    # preds[~np.isin(preds, uniques)] = -1
    # preds_without_conf[~np.isin(preds_without_conf, uniques)] = -1
    mu = [mu[key] for key in sorted(uniques)]

    return (
        dict(zip(all_idxs, preds)),
        dict(zip(all_idxs, preds_without_conf)),
        dict(zip(all_idxs, confs)),
        uniques.tolist(),
        mu,
    )
