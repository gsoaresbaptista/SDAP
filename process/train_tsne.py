import copy
import os
from itertools import chain, cycle
from statistics import mean
<<<<<<< HEAD
from utils.data import get_data
=======
import seaborn as sns

>>>>>>> 593ba19 (WIP: progresso local)
import numpy as np
import torch
from torch.nn import functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from data.incremental import load_incremental_loader, load_source_aug_loader
from process.eval import eval
from process.knn import find_best_k
from process.labels import (
    build_labels_tensor,
    detect_shared_classes,
    generate_pseudo_labels,
)
from utils.losses import contrastive_alignment_loss, distillation_loss
from utils.reply_buffer import ReplyBuffer
<<<<<<< HEAD
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
=======
from utils.data import get_data
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


sns.set_style("whitegrid")
sns.set_context("talk", font_scale=1.4)
fig, axs = plt.subplots(2, 3, figsize=(8 * 3, 8 * 2), sharex=True, sharey=True)
palette = sns.color_palette("muted")
axs[0, 0].set_ylabel("SDAP Without Triplet-Loss Alignment", fontsize=26, fontweight='bold', labelpad=30)
axs[1, 0].set_ylabel("SDAP", fontsize=26, fontweight='bold', labelpad=30)
# fig.text(0.06, 0.3, "SDAP", fontsize=18, fontweight='bold', rotation=90, va='center')
increments = ["Increment 1", "Increment 2", "Increment 3"]
for i in range(3):
    # axs[0, i].set_title(increments[i], fontsize=26)
    axs[1, i].set_xlabel(increments[i], fontsize=26, fontweight='bold', labelpad=30)
plt.tight_layout(rect=[0, 0, 1, 1])
plt.savefig(f'figure.pdf')
axs[0, 0].grid(True, linestyle='--', alpha=0.6); axs[0, 0].set_facecolor("#f8f8f8")
axs[0, 1].grid(True, linestyle='--', alpha=0.6); axs[0, 1].set_facecolor("#f8f8f8")
axs[0, 2].grid(True, linestyle='--', alpha=0.6); axs[0, 2].set_facecolor("#f8f8f8")
axs[1, 0].grid(True, linestyle='--', alpha=0.6); axs[1, 0].set_facecolor("#f8f8f8")
axs[1, 1].grid(True, linestyle='--', alpha=0.6); axs[1, 1].set_facecolor("#f8f8f8")
axs[1, 0].grid(True, linestyle='--', alpha=0.6); axs[1, 0].set_facecolor("#f8f8f8")
axs[1, 2].grid(True, linestyle='--', alpha=0.6); axs[1, 2].set_facecolor("#f8f8f8")
# exit(0)
>>>>>>> 593ba19 (WIP: progresso local)

def process_features(features_np):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features_np)
    tsne = TSNE(
        n_components=2,
        perplexity=100,
        learning_rate='auto',
        init='random',
        random_state=42
    )
    return tsne.fit_transform(scaled)

def get_final_eval_metrics(model, experiment, device, n_classes, class_increment):
    model.eval()
    results = {}

    for increment_id in range(n_classes // class_increment):
        _, first_increment, _ = load_incremental_loader(
            experiment, increment_id * class_increment, class_increment, device, False
        )
        acc_increment, f1_increment = eval(model, first_increment, device)
        results[increment_id] = {"acc": acc_increment, "f1": f1_increment}

    return results


def get_classes_imgs(loader, labels, increment_classes):
    class_imgs_dict = {class_idx: [] for class_idx in increment_classes}

    for data in loader:
        inputs, *_, sample_idx, _ = data
        for label in increment_classes:
            class_mask = torch.tensor(
                [labels[int(idx.item())] == label for idx in sample_idx]
            )
            if class_mask.any():
                class_inputs = inputs[class_mask].cpu()
                class_imgs_dict[label].append(class_inputs)

    for label in class_imgs_dict:
        if len(class_imgs_dict[label]) > 0:
            class_imgs_dict[label] = torch.cat(class_imgs_dict[label], dim=0)

    return class_imgs_dict


def get_pos_neg(prototypes, num_classes, device, q_label: int):
    feature_pos = prototypes[q_label].unsqueeze(0)
    neg_idx = torch.tensor([i for i in range(num_classes) if i != q_label])
    negative_pairs = prototypes[neg_idx]
    return torch.cat((feature_pos, negative_pairs), dim=0).to(device)


def class_level_label_distillation(mu_history, uk, buffer, increment_classes):
    to_remove = []
    for i, uki in enumerate(uk):
        label = increment_classes[i]
        if label not in mu_history:
            mu_history[label] = uki
        elif uki >= mu_history[label]:
            mu_history[label] = uki
            buffer.remove_label(label)
        elif uki < mu_history[label]:
            to_remove.append(label)

    for label in to_remove:
        increment_classes.remove(label)

    return increment_classes


def compute_triplet_loss_from_source(
    target_features, target_labels, source_features, source_labels, margin=0.2
):
    target_norm = F.normalize(target_features, p=2, dim=1)
    source_norm = F.normalize(source_features, p=2, dim=1)
    sims = torch.mm(target_norm, source_norm.t())

    pos_mask = target_labels.unsqueeze(1) == source_labels.unsqueeze(0)
    neg_mask = target_labels.unsqueeze(1) != source_labels.unsqueeze(0)

    fill_val = -1.0
    neg_sims = torch.where(neg_mask, sims, torch.tensor(fill_val, device=sims.device))

    neg_max, _ = neg_sims.max(dim=1)
    pos_probs = pos_mask.float()
    pos_probs[pos_probs.sum(dim=1) == 0] = 1e-6  # Evita soma zero
    pos_indices = torch.multinomial(pos_probs, 1).squeeze(1)
    pos_random = sims.gather(1, pos_indices.unsqueeze(1)).squeeze(1)

    loss = F.softplus(neg_max - pos_random + margin)
    return loss.mean()

<<<<<<< HEAD
=======

>>>>>>> 593ba19 (WIP: progresso local)
def train(model, experiment, triplet):
    # experiment settings
    n_classes = experiment["dataset"]["n_classes"]
    class_increment = experiment["train"]["class_increment"]
    device = experiment["train"]["device"]
    epochs = experiment["train"]["epochs"]
    lr = experiment["train"]["learning_rate"]

    buffer_loader = None
    buffer = ReplyBuffer()
    mu_history = {}
    increment_id = 0
    WARM_UP = 3
    source_centroids_mem = torch.zeros(
        (n_classes, model.head.in_features), device=device
    )
    added_classes = []
    incremental_stats = []

    pretrained = copy.deepcopy(model)
    pretrained.to(device)
    pretrained.train()
    source_loader, *_ = load_incremental_loader(experiment, 0, n_classes, device, False)
    optimizer = SGD(pretrained.parameters(), 1e-3, 0.9, weight_decay=1e-6)
    scheduler = ReduceLROnPlateau(optimizer, "min", 0.1, 3)
    for epoch in range(100):
        loss, total = 0, 0
        for s_im, s_labels, *_ in source_loader:
            s_features, s_outs = pretrained(s_im)
            ce_loss = F.cross_entropy(s_outs, s_labels)
            ce_loss.backward()
            loss += ce_loss.item()
            total += 1
            optimizer.step()
            optimizer.zero_grad()
        scheduler.step(loss / total)
    pretrained.eval()

<<<<<<< HEAD
=======
    
>>>>>>> 593ba19 (WIP: progresso local)
    for increment_id in range(n_classes // class_increment):
        print(f"Increment {increment_id + 1}")
        _, shared_loader, _ = load_incremental_loader(
            experiment,
            increment_id * class_increment,
            class_increment,
            device,
            False,
        )
        _, labels, confs, increment_classes, uk = detect_shared_classes(
            pretrained, source_loader, shared_loader, None, 0.1
        )
<<<<<<< HEAD
        increment_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        uk = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
=======
>>>>>>> 593ba19 (WIP: progresso local)
        increment_classes = class_level_label_distillation(
            mu_history, uk, buffer, increment_classes
        )
        print("Classes:", increment_classes)
        source_loader, target_loader, eval_loader = load_incremental_loader(
            experiment, increment_id * class_increment, class_increment, device, False
        )
        buffer_loader = buffer.get_dataloader(target_loader.batch_size)

        model.to(device)
        model.train()

        optimizer = SGD(model.parameters(), lr, 0.9, weight_decay=1e-6)
        scheduler = ReduceLROnPlateau(optimizer, "min", 0.1, 3)
        incremental_stats.append([])

        for epoch in range(epochs):
            losses = []
            for s_batch, t_batch in zip(source_loader, cycle(target_loader)):
                # compute source cross-entropy
                s_im, s_labels = s_batch[0], s_batch[1]
                s_features, s_outs = model(s_im)
                ce_loss = F.cross_entropy(s_outs, s_labels)

                # CE on target data
                ce_loss_target = torch.tensor(0, device=device, dtype=torch.float32)

                if epoch >= WARM_UP:
                    t_im, t_indexes = t_batch[0], t_batch[2]
                    indices_list = t_indexes.tolist()
                    confidence_mask = torch.tensor(
                        [confs.get(idx, False) for idx in indices_list]
                    )
                    confident_indices = t_indexes[confidence_mask]
                    confidence_mask = confidence_mask.to(device)
                    t_features, t_outs = model(t_im)

                # Domain-Level Contrastive Alignment
                with torch.no_grad():
                    for label in range(n_classes):
                        mask = s_labels == label
                        if mask.sum() > 1:
                            if label in added_classes:
                                source_centroids_mem[label] = (
                                    0.99 * source_centroids_mem[label]
                                    + 0.01 * s_features[mask].mean(dim=0)
                                )
                            else:
                                added_classes.append(label)
                                source_centroids_mem[label] = s_features[mask].mean(
                                    dim=0
                                )

                # Loss IDC
                loss_idc = torch.tensor(0, device=device, dtype=torch.float32)
<<<<<<< HEAD
                loss_ic = torch.tensor(0, device=device, dtype=torch.float32)

                if triplet:
                    if epoch >= WARM_UP and confident_indices.shape[0]:
                        pseudo_labels = build_labels_tensor(
                            labels, device, confident_indices
                        )
                        loss_idc = compute_triplet_loss_from_source(
                            t_features[confidence_mask],
                            pseudo_labels,
                            source_centroids_mem,
                            torch.arange(0, n_classes, device=device),
                            0.2,
                        )

                contras_loss = loss_idc + loss_ic
=======

                if (epoch >= WARM_UP) and (confident_indices.shape[0]) and triplet:
                    pseudo_labels = build_labels_tensor(
                        labels, device, confident_indices
                    )
                    loss_idc = compute_triplet_loss_from_source(
                        t_features[confidence_mask],
                        pseudo_labels,
                        source_centroids_mem,
                        torch.arange(0, n_classes, device=device),
                        0.2,
                    )

>>>>>>> 593ba19 (WIP: progresso local)
                distill_loss = torch.tensor(0, device=device, dtype=torch.float32)

                if buffer_loader:
                    re_org_img, _, re_org_sp = next(buffer_loader)
                    re_org_img = re_org_img.cuda()
                    re_org_sp = re_org_sp.cuda()
                    _, reply_ouputs = model(re_org_img)
                    distill_loss = distillation_loss(re_org_sp, reply_ouputs)

                # model optimization
<<<<<<< HEAD
                loss = ce_loss + ce_loss_target + contras_loss + distill_loss
=======
                loss = ce_loss + ce_loss_target + loss_idc + distill_loss
>>>>>>> 593ba19 (WIP: progresso local)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                losses.append(loss.item())

            accuracy, f1 = eval(model, eval_loader, device)
            scheduler.step(losses[-1])
            print(
                f"Epoch {epoch + 1} | Loss: {mean(losses):.4f} | Acc.: {accuracy:.4f} | F1: {f1:.4f}"
            )

            if epoch >= WARM_UP:
                model.eval()
                _, shared_loader, _ = load_incremental_loader(
                    experiment,
                    increment_id * class_increment,
                    class_increment,
                    device,
                    False,
                )
                _, labels, confs, tmp_classes, tmp_uk = detect_shared_classes(
                    model,
                    source_loader,
                    shared_loader,
                    increment_classes,
                    beta=0.1,
                    # beta=beta_val,
                )
                to_update = [lbs for lbs in increment_classes if lbs not in tmp_classes]
                increment_classes = [lbs for lbs in increment_classes if lbs in tmp_classes]
                uk = [tmp_uk[i] for i, lbs in enumerate(tmp_classes) if lbs in increment_classes]

                for i, label in enumerate(increment_classes):
                    mu_history[label] = uk[i]
                for removed in to_update:
<<<<<<< HEAD
                    del mu_history[removed]
                print("Classes:", increment_classes)
            model.train()

        break

    features, labels, *_ = get_data(model, shared_loader)
    features = process_features(features.cpu().numpy())
    labels = labels.cpu().numpy()
    return features, labels


def sdap_plot_tsne(model, experiment):
    without_feas, without_labels = train(model, experiment, False)
    with_feas, with_labels = train(model, experiment, True)
    n_classes = experiment["dataset"]["n_classes"]

    sns.set_style("whitegrid")
    sns.set_context("talk", font_scale=1.4)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    # palette = sns.color_palette("Set3", n_colors=n_classes)
    palette = sns.color_palette("muted")

    sns.scatterplot(
        x=without_feas[:, 0],
        y=without_feas[:, 1],
        hue=without_labels,
        palette=palette,
        ax=ax1,
        legend=False,
        alpha=0.8,
        s=40,
        edgecolor="none"
    )
    # ax1.set_title('Dataset 1 t-SNE Visualization', fontsize=14, pad=20)
    ax1.set_xlabel('(a) SDAP w/o Triplet-Loss Alignment.', fontsize=32)
    # ax1.set_ylabel('t-SNE Component 2', fontsize=12)

    # Plot second dataset
    sns.scatterplot(
        x=with_feas[:, 0],
        y=with_feas[:, 1],
        hue=with_labels,
        palette=palette,
        ax=ax2,
        legend=False,
        alpha=0.8,
        s=40,
        edgecolor="none"
    )
    # ax2.set_title('Dataset 2 t-SNE Visualization', fontsize=14, pad=20)
    ax2.set_xlabel('(b) SDAP.', fontsize=32)
    # ax2.set_ylabel('t-SNE Component 2', fontsize=12)

    # Add common styling
    for ax in [ax1, ax2]:
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_facecolor("#f8f8f8")
        ax.tick_params(axis='both', which='major', labelsize=26)

    plt.tight_layout()
    plt.savefig('tsne_explain.png')
    plt.savefig('tsne_explain.eps')
    plt.savefig('tsne_explain.pdf')
=======
                    if removed in mu_history:
                        del mu_history[removed]
                print("Classes:", increment_classes)
            model.train()

        imgs = get_classes_imgs(target_loader, labels, increment_classes)
        for i, label in enumerate(increment_classes):
            mu_history[label] = uk[i]
            if len(imgs[label]) >= 10:
                buffer.construct_exemplar_set(model, imgs[label], label)

        if isinstance(accuracy, torch.Tensor):
            accuracy = accuracy.tolist()

        _, first_increment, _ = load_incremental_loader(
            experiment, increment_id * class_increment, class_increment, device, False
        )
        acc_increment, f1_increment = eval(model, first_increment, device)

        features, labels, *_ = get_data(model, shared_loader)
        features = process_features(features.cpu().numpy())
        labels = labels.cpu().numpy()

        sns.scatterplot(
            x=features[:, 0],
            y=features[:, 1],
            hue=labels,
            palette=palette,
            ax=axs[1 if triplet else 0][increment_id],
            legend=False,
            alpha=0.8,
            s=40,
            edgecolor="none"
        )
        plt.tight_layout(rect=[0, 0, 1, 0.9])
        plt.savefig(f'figure.pdf')

def sdap_plot_tsne(model, experiment):
    from models.projector import Projector
    model = Projector(experiment["dataset"]["n_classes"], "small_2")
    experiment["train"]['batch'] = 32
    experiment["eval"]['batch'] = 32
    train(model, experiment, True)
    experiment["eval"]['batch'] = 512
    experiment["train"]['batch'] = 512
    model = Projector(experiment["dataset"]["n_classes"], "small_2")
    train(model, experiment, False)

# def sdap_plot_tsne(model, experiment):
#     without_feas, without_labels = train(model, experiment, False)
#     with_feas, with_labels = train(model, experiment, True)
#     n_classes = experiment["dataset"]["n_classes"]

#     sns.set_style("whitegrid")
#     sns.set_context("talk", font_scale=1.4)
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
#     # palette = sns.color_palette("Set3", n_colors=n_classes)
#     palette = sns.color_palette("muted")

#     sns.scatterplot(
#         x=without_feas[:, 0],
#         y=without_feas[:, 1],
#         hue=without_labels,
#         palette=palette,
#         ax=ax1,
#         legend=False,
#         alpha=0.8,
#         s=40,
#         edgecolor="none"
#     )
#     # ax1.set_title('Dataset 1 t-SNE Visualization', fontsize=14, pad=20)
#     ax1.set_xlabel('(a) SDAP w/o Triplet-Loss Alignment.', fontsize=32)
#     # ax1.set_ylabel('t-SNE Component 2', fontsize=12)

#     # Plot second dataset
#     sns.scatterplot(
#         x=with_feas[:, 0],
#         y=with_feas[:, 1],
#         hue=with_labels,
#         palette=palette,
#         ax=ax2,
#         legend=False,
#         alpha=0.8,
#         s=40,
#         edgecolor="none"
#     )
#     # ax2.set_title('Dataset 2 t-SNE Visualization', fontsize=14, pad=20)
#     ax2.set_xlabel('(b) SDAP.', fontsize=32)
#     # ax2.set_ylabel('t-SNE Component 2', fontsize=12)

#     # Add common styling
#     for ax in [ax1, ax2]:
#         ax.grid(True, linestyle='--', alpha=0.6)
#         ax.set_facecolor("#f8f8f8")
#         ax.tick_params(axis='both', which='major', labelsize=26)

#     plt.tight_layout()
#     plt.savefig('tsne_explain.png')
#     plt.savefig('tsne_explain.eps')
#     plt.savefig('tsne_explain.pdf')
>>>>>>> 593ba19 (WIP: progresso local)
