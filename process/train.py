import copy
from itertools import cycle
from statistics import mean

import torch
from torch.nn import functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from data.incremental import load_incremental_loader
from process.eval import eval
from process.labels import (
    build_labels_tensor,
    detect_shared_classes,
)
from utils.losses import distillation_loss
from utils.reply_buffer import ReplyBuffer

def pseudo_label_evaluate(pseudo_labels, target_loader):
    corrects, total = 0, 0
    for _, labels, indices, *_ in target_loader:
        for i, id in enumerate(indices.tolist()):
            if pseudo_labels[id] == labels[i]:
                corrects += 1
            total += 1
    return corrects / total


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


def class_level_label_distillation(mu_history, uk, buffer, increment_classes, inc_id, distillation_metrics):
    to_remove = []

    distillation_metrics['relearning'].append(0)
    distillation_metrics['wrong_relearning'].append(0)
    distillation_metrics['removing'].append(0)
    distillation_metrics['wrong_class'].append(0)
    distillation_metrics['wrong_removing'].append(0)

    for i, uki in enumerate(uk):
        label = increment_classes[i]
        if label not in mu_history:
            mu_history[label] = uki
            if not (inc_id*10 <= label < (inc_id+1)*10):
                distillation_metrics['wrong_class'][-1] += 1
        elif uki >= mu_history[label]:
            mu_history[label] = uki
            buffer.remove_label(label)
            distillation_metrics['relearning'][-1] += 1
            if not (inc_id*10 <= label < (inc_id+1)*10):
                distillation_metrics['wrong_relearning'][-1] += 1
        elif uki < mu_history[label]:
            to_remove.append(label)
            distillation_metrics['removing'][-1] += 1
            if inc_id*10 <= label < (inc_id+1)*10:
                distillation_metrics['wrong_removing'][-1] += 1

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

def train(model, experiment, beta_val, disable):

    distillation_metrics = {
        'relearning': [],
        'removing': [],
        'wrong_class': [],
        'wrong_relearning': [],
        'wrong_removing': [],
    }

    # experiment settings
    n_classes = experiment["dataset"]["n_classes"]
    class_increment = experiment["train"]["class_increment"]
    device = experiment["train"]["device"]
    epochs = experiment["train"]["epochs"]
    lr = experiment["train"]["learning_rate"]
    step_level = {}
    single_step_level = {}

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
            pretrained, source_loader, shared_loader, None, beta_val if 'cl' not in disable else 0
        )
        if 'cl' not in disable:
            increment_classes = class_level_label_distillation(
                mu_history, uk, buffer, increment_classes, increment_id, distillation_metrics
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

                if (epoch >= WARM_UP) and (confident_indices.shape[0]) and ('triplet' not in disable):
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

                distill_loss = torch.tensor(0, device=device, dtype=torch.float32)

                if buffer_loader and ('dist' not in disable):
                    re_org_img, _, re_org_sp = next(buffer_loader)
                    re_org_img = re_org_img.cuda()
                    re_org_sp = re_org_sp.cuda()
                    _, reply_ouputs = model(re_org_img)
                    distill_loss = distillation_loss(re_org_sp, reply_ouputs)

                # model optimization

                if 'triplet' in disable:
                    loss_idc = torch.tensor(0, device=device, dtype=torch.float32)
                if 'dist' in disable:
                    distill_loss = torch.tensor(0, device=device, dtype=torch.float32)

                loss = ce_loss + ce_loss_target + loss_idc + distill_loss
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

            if epoch >= WARM_UP and 'il' not in disable:
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
                    beta=beta_val,
                )
                to_update = [lbs for lbs in increment_classes if lbs not in tmp_classes]
                increment_classes = [lbs for lbs in increment_classes if lbs in tmp_classes]
                uk = [tmp_uk[i] for i, lbs in enumerate(tmp_classes) if lbs in increment_classes]

                for i, label in enumerate(increment_classes):
                    mu_history[label] = uk[i]
                for removed in to_update:
                    if removed in mu_history:
                        del mu_history[removed]
                print("Classes:", increment_classes)
            model.train()


        imgs = get_classes_imgs(target_loader, labels, increment_classes)
        for i, label in enumerate(increment_classes):
            mu_history[label] = uk[i]
            if len(imgs[label]) >= 1:
                buffer.construct_exemplar_set(model, imgs[label], label)

        if isinstance(accuracy, torch.Tensor):
            accuracy = accuracy.tolist()

        pseudo_label_acc = pseudo_label_evaluate(labels, target_loader)
        step_level[increment_id] = {"acc": accuracy, "f1": f1, "pseudo_label_acc": pseudo_label_acc}

        _, first_increment, _ = load_incremental_loader(
            experiment, increment_id * class_increment, class_increment, device, False
        )
        acc_increment, f1_increment = eval(model, first_increment, device)
        single_step_level[increment_id] = {"acc": acc_increment, "f1": f1_increment}

    return (
        step_level,
        single_step_level,
        get_final_eval_metrics(model, experiment, device, n_classes, class_increment),
        distillation_metrics
    )
