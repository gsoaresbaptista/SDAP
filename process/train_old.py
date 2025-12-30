import copy
import os
from itertools import chain, cycle
from statistics import mean
from pathlib import Path
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
    old_detect_shared_classes,
    generate_pseudo_labels,
)
from utils.losses import contrastive_alignment_loss, distillation_loss
from utils.reply_buffer import ReplyBuffer


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
    for i, uki in enumerate(uk.tolist()):
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
    sims = torch.mm(target_norm, source_norm.t())  # (N_target, N_source)

    # Create masks: positive if labels match, negative if they differ.
    pos_mask = target_labels.unsqueeze(1) == source_labels.unsqueeze(0)
    neg_mask = target_labels.unsqueeze(1) != source_labels.unsqueeze(0)

    # Use a fill value so that non-matching similarities are ignored (-1 is a low cosine similarity)
    fill_val = -1.0
    pos_sims = torch.where(pos_mask, sims, torch.tensor(fill_val, device=sims.device))
    neg_sims = torch.where(neg_mask, sims, torch.tensor(fill_val, device=sims.device))

    # For each target sample, take the highest cosine similarity among positive and negative source samples
    pos_max, _ = pos_sims.max(dim=1)
    neg_max, _ = neg_sims.max(dim=1)

    # Apply softplus-based triplet loss: smoother penalty than hinge loss
    loss = F.softplus(neg_max - pos_max + margin)
    return loss.mean()


def train(model, experiment):
    # experiment settings
    n_classes = experiment["dataset"]["n_classes"]
    class_increment = experiment["train"]["class_increment"]
    device = experiment["train"]["device"]
    epochs = experiment["train"]["epochs"]
    source = experiment["dataset"]["source"]
    target = experiment["dataset"]["target"]
    lr = experiment["train"]["learning_rate"]
    step_level = {}
    single_step_level = {}

    buffer_loader = None
    buffer = ReplyBuffer()
    mu_history = {}
    increment_id = 0
    WARM_UP = 0
    source_centroids_mem = torch.zeros(
        (n_classes, model.resnet.fc.in_features), device=device
    )
    added_classes = []
    incremental_stats = []

    if os.path.exists(f"{source}_{target}.pth"):
        pretrained = torch.load(f"{source}_{target}.pth")
    else:
        pretrained = copy.deepcopy(model)
        pretrained.to(device)
        pretrained.train()
        source_loader, *_ = load_incremental_loader(experiment, 0, n_classes, device, False)
        optimizer = SGD(pretrained.parameters(), 1e-3, 0.9, weight_decay=1e-6)
        for epoch in range(100):
            for s_im, s_labels, *_ in source_loader:
                s_im = s_im.to(device)
                s_labels = s_labels.to(device)
                s_features, s_outs = pretrained(s_im)
                ce_loss = F.cross_entropy(s_outs, s_labels)
                ce_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            accuracy, f1 = eval(pretrained, source_loader, device)
            print(accuracy)
        pretrained.eval()
        torch.save(pretrained, f"{source}_{target}.pth")

    pretrained.to(device)
    pretrained.train()

    for increment_id in range(n_classes // class_increment):
        print(f"Increment {increment_id + 1}")
        _, shared_loader, _ = load_incremental_loader(
            experiment,
            increment_id * class_increment,
            class_increment,
            device,
            False,
        )
        increment_classes, uk = old_detect_shared_classes(
            pretrained,
            device,
            shared_loader,
            n_classes,
        )
        increment_classes = class_level_label_distillation(
            mu_history, uk, buffer, increment_classes
        )
        print("Classes:", increment_classes)
        labels, confs = generate_pseudo_labels(pretrained, device, shared_loader, increment_classes)
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
                s_im = s_im.to(device)
                s_labels = s_labels.to(device)
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
                    t_im = t_im.to(device)
                    t_features, t_outs = model(t_im)

                    if confident_indices.shape[0]:
                        conf_target_labels = build_labels_tensor(
                            labels, device, confident_indices
                        )
                        ce_loss_target = F.cross_entropy(
                            t_outs[confidence_mask], conf_target_labels
                        )

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
                loss_ic = torch.tensor(0, device=device, dtype=torch.float32)

                if epoch >= WARM_UP and confident_indices.shape[0]:
                    for label in increment_classes:
                        mask = [
                            i
                            for i, idx in enumerate(indices_list)
                            if (labels[idx] == label and confs[idx])
                        ]
                        if len(mask) and label in added_classes:
                            fea = t_features[mask].mean(dim=0)
                            pos_neg_pair = torch.vstack(
                                [
                                    source_centroids_mem[label],
                                    t_features[
                                        [
                                            i
                                            for i, idx in enumerate(indices_list)
                                            if (labels[idx] != label and confs[idx])
                                        ]
                                    ],
                                    source_centroids_mem[
                                        [
                                            b_label
                                            for b_label in added_classes
                                            if b_label != label
                                        ]
                                    ],
                                ]
                            )
                            loss_idc += contrastive_alignment_loss(fea, pos_neg_pair, 5, 0.3)
                    loss_idc /= len(increment_classes)

                # loss_ic
                if epoch >= WARM_UP:
                    aug_images = t_batch[3]
                    unconf_id = [i for i, idx in enumerate(indices_list) if (not confs[idx])]
                    aug_feas, _ = model(aug_images.to(device))

                if epoch >= WARM_UP and len(unconf_id):
                    for idx in unconf_id:
                        fea = t_features[idx]
                        pos_neg_pair = torch.vstack(
                            [aug_feas[idx], aug_feas[[i for i in unconf_id if i != idx]]]
                        )
                        loss_ic += contrastive_alignment_loss(fea, pos_neg_pair, 5, 0.3)
                    loss_ic /= len(unconf_id)

                contras_loss = loss_idc + loss_ic
                distill_loss = torch.tensor(0, device=device, dtype=torch.float32)

                if buffer_loader:
                    re_org_img, _, re_org_sp = next(buffer_loader)
                    re_org_img = re_org_img.cuda()
                    re_org_sp = re_org_sp.cuda()
                    _, reply_ouputs = model(re_org_img)
                    distill_loss = distillation_loss(re_org_sp, reply_ouputs)

                # model optimization
                loss = ce_loss + ce_loss_target + contras_loss + distill_loss
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

            if epoch >= 0:
                model.eval()
                _, shared_loader, _ = load_incremental_loader(
                    experiment,
                    increment_id * class_increment,
                    class_increment,
                    device,
                    False,
                )
                tmp_classes, tmp_uk = old_detect_shared_classes(
                    model,
                    device,
                    shared_loader,
                    n_classes,
                )
                indices = [
                    i
                    for i in range(len(tmp_classes))
                    if tmp_classes[i] in increment_classes
                ]
                to_remove = [cls for cls in increment_classes if cls not in tmp_classes]
                increment_classes = [tmp_classes[i] for i in indices]
                uk = [tmp_uk[i] for i in indices]
                for i, label in enumerate(increment_classes):
                    mu_history[label] = uk[i]
                for label in to_remove:
                    mu_history[label] = 0
                print("Classes:", increment_classes)
                labels, confs = generate_pseudo_labels(model, device, shared_loader, increment_classes)
            model.train()

        if increment_id != ((n_classes // class_increment) - 1):
            imgs = get_classes_imgs(target_loader, labels, increment_classes)
            for i, label in enumerate(increment_classes):
                if len(imgs[label]) >= 1:
                    mu_history[label] = uk[i]
                    buffer.construct_exemplar_set(model, imgs[label], label)

        if isinstance(accuracy, torch.Tensor):
            accuracy = accuracy.tolist()
        step_level[increment_id] = {"acc": accuracy, "f1": f1}

        _, first_increment, _ = load_incremental_loader(
            experiment, increment_id * class_increment, class_increment, device, False
        )
        acc_increment, f1_increment = eval(model, first_increment, device)
        single_step_level[increment_id] = {"acc": acc_increment, "f1": f1_increment}

    return (
        step_level,
        single_step_level,
        get_final_eval_metrics(model, experiment, device, n_classes, class_increment),
    )
