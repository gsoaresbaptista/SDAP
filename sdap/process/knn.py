import numpy as np
import torch
from scipy.spatial.distance import cdist
from scipy.stats import mode
from collections import defaultdict

from utils.data import get_data


def remove_outliers_euclidean(features, labels, percentile=0.75):
    unique_labels = torch.unique(labels)
    keep_indices = []

    for lbl in unique_labels.tolist():
        idx = torch.where(labels == lbl)[0]
        X = features[idx]
        centroid = torch.mean(X, axis=0)
        distances = (X - centroid).norm(dim=1)
        threshold = torch.quantile(distances, percentile)
        keep = idx[distances <= threshold]
        keep_indices.append(keep)

    keep_indices = torch.concatenate(keep_indices)
    keep_indices, _ = torch.sort(keep_indices)
    filtered_features = features[keep_indices]
    filtered_labels = labels[keep_indices]
    return filtered_features, filtered_labels


def remove_outliers_cosine(features, labels, percentile=0.75):
    unique_labels = torch.unique(labels)
    keep_indices = []

    for lbl in unique_labels.tolist():
        idx = torch.where(labels == lbl)[0]
        X = features[idx]
        centroid = torch.mean(X, axis=0)
        centroid_norm = centroid / torch.norm(centroid)
        X_norm = X / torch.norm(X, dim=1, keepdim=True)
        cosine_distances = 1 - torch.matmul(X_norm, centroid_norm)
        threshold = torch.quantile(cosine_distances, percentile)
        keep = idx[cosine_distances <= threshold]
        keep_indices.append(keep)

    keep_indices = torch.concatenate(keep_indices)
    keep_indices, _ = torch.sort(keep_indices)
    filtered_features = features[keep_indices]
    filtered_labels = labels[keep_indices]
    return filtered_features, filtered_labels


def find_best_k(source_loader, aug_loader, model, beta=2, k_percentage=0.9):
    s_feas, s_labels, *_ = get_data(model, source_loader)
    aug_feas, aug_labels, *_ = get_data(model, aug_loader)

    # s_feas, s_labels = remove_outliers_euclidean(s_feas, s_labels)

    curr_k = 3
    best_k = 3
    best_acc = 0
    optimal_backtrack = False

    while True:
        full_pseudo_labels, _, _ = reciprocal_knn(
            s_feas, s_labels, aug_feas, curr_k, beta
        )
        full_pseudo_labels = torch.tensor(full_pseudo_labels, device=aug_labels.device)

        correct = (full_pseudo_labels == aug_labels).sum().item()
        total = aug_labels.shape[0]
        current_acc = correct / total
        print(f"k={curr_k} | Acc: {current_acc:.2%}")

        if current_acc > best_acc:
            if optimal_backtrack and (current_acc > k_percentage):
                best_k = curr_k
                best_acc = current_acc
                break
            if current_acc > k_percentage:
                optimal_backtrack = True
                curr_k = (curr_k // 2) + 1
                continue
            if not optimal_backtrack:
                best_k = curr_k
                best_acc = current_acc
                curr_k = 2 * curr_k
            else:
                curr_k += 1
        else:
            break

    print(f"Best k found: {best_k} with acc. {best_acc:.2%}")
    return best_k


@torch.no_grad()
def generate_knn_pseudo_labels(model, source, target, classes, k, beta):
    model.eval()
    s_feas, s_labels, *_ = get_data(model, source)
    t_feas, *_, t_index = get_data(model, target)
    full_pseudo_labels, _, _ = reciprocal_knn(
        s_feas, s_labels, t_feas, k, beta, classes
    )
    confs = torch.where(full_pseudo_labels != -1, True, False).tolist()
    t_index = t_index.tolist()
    full_pseudo_labels.tolist()
    return dict(zip(t_index, full_pseudo_labels)), dict(zip(t_index, confs))


def reciprocal_knn(s_features, s_labels, t_features, classes=None, beta=0.5):
    source_features = s_features
    source_labels = s_labels
    target_features = t_features
    # distances = (cdist(target_features, source_features, metric="cosine") + 1) / 2
    distances = cdist(target_features, source_features, metric="cosine")
    pseudo_labels = []
    confidences = []
    confidences_per_cls = defaultdict(list)
    for i in range(len(target_features)):
        dists = distances[i]
        unique_classes = np.unique(source_labels)
        best_conf = -np.inf
        best_class = None
        min_dist = 0
        for cls in unique_classes:
            cls_mask = source_labels == cls
            if not np.any(cls_mask):
                continue
            d_cls = dists[cls_mask]
            min_d_cls = d_cls.min()
            other_mask = source_labels != cls
            if np.any(other_mask):
                d_other = dists[other_mask]
                min_d_other = d_other.min()
            else:
                min_d_other = np.inf
            conf = 1 - (min_d_cls / (min_d_other + 1e-8))
            if conf > best_conf:
                best_conf = conf
                best_class = cls
                min_dist = min_d_cls
        pseudo_labels.append(best_class)
        confidences.append(best_conf)
        confidences_per_cls[best_class].append(best_conf)
    pseudo_labels = np.array(pseudo_labels)
    confidences = np.array(confidences)
    final_labels = np.where(confidences >= beta, pseudo_labels, -1)
    unique_confident = np.unique(final_labels[final_labels != -1])
    counts = np.array([np.sum(final_labels == lbl) for lbl in unique_confident])
    unique_confident = np.sort(unique_confident)

    normalized = {}
    for key in unique_confident:
        normalized[key.item()] = np.mean(confidences_per_cls[key])
    # print(normalized)

    return final_labels, pseudo_labels, unique_confident, counts, normalized
