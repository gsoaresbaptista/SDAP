from pathlib import Path
from collections import defaultdict
from itertools import cycle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score


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
        class_mean, feature_extractor_output, buffer_output = self.compute_class_mean(model, images)
        exemplar, soft_predar = [], []
        now_class_mean = np.zeros((1, 1024))

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


def contrastive_alignment_loss(
    anchor: torch.Tensor, pos_neg_pair: torch.Tensor, gamma: float, margin: float = 0
) -> torch.Tensor:
    result = F.cosine_similarity(anchor.unsqueeze(0), pos_neg_pair, dim=-1)
    max_result = torch.max(result / gamma)
    numerator = torch.exp((result[0] - margin) / gamma - max_result)
    denominator = numerator + torch.sum(torch.exp(result[1:] / gamma - max_result)) + 1e-8
    loss = -torch.log(numerator / denominator)
    return loss


def distillation_loss(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    q = F.softmax(q, dim=1)
    q = torch.log(q)
    soft_pred_loss = torch.sum(-1 * p * q, dim=1)
    return torch.mean(soft_pred_loss)


@torch.no_grad()
def eval(model, test_loader, device):
    model.eval()
    all_preds, all_label = [], []

    for batch in test_loader:
        x, y = batch[0].to(device), batch[1]
        _, logits = model(x)
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


@torch.no_grad()
def pseudo_label_evaluate(pseudo_labels, target_loader):
    corrects, total = 0, 0
    for _, labels, indices, *_ in target_loader:
        for i, idx in enumerate(indices.tolist()):
            if idx in pseudo_labels and pseudo_labels[idx] == labels[i].item():
                corrects += 1
            total += 1
    return corrects / total if total > 0 else 0.0


@torch.no_grad()
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


class PLDCA:
    def __init__(
        self, model, n_classes, source_loader, lr, threshold, beta, warm_up, step_update, device
    ):
        self.device = device
        self.threshold = threshold
        self.source_loader = source_loader

        self.model = model
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr, 0.9, weight_decay=1e-6)
        self.num_classes = n_classes

        self.buffer = ReplyBuffer()
        self.mu_history = {}
        self.increment_id = 0
        self.beta = beta
        self.source_centroids_mem = torch.zeros((self.num_classes, 1024), device=device)

        self.warm_up = warm_up
        self.step_update = step_update
        self.added_classes = []

        self.step_level = {}
        self.single_step_level = {}
        self.re_eval = {}


    @torch.no_grad()
    def generate_pseudo_labels(self, target_loader, with_target=False):
        self.model.eval()
        all_feas, all_outputs, all_indices = [], [], []

        for inputs, _, idx, *_ in target_loader:
            inputs = inputs.to(self.device)
            feas, outputs = self.model(inputs)
            all_feas.append(feas.cpu())
            all_outputs.append(outputs.cpu())
            all_indices.append(idx.float())

        all_fea = torch.cat(all_feas, dim=0)
        all_output = F.softmax(torch.cat(all_outputs, dim=0), dim=1)
        all_idx = torch.cat(all_indices, dim=0)

        all_fea = F.normalize(torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1), p=2, dim=1)

        confi_class_idx = torch.tensor(self.increment_classes, device=all_fea.device)
        initc = F.normalize(all_output.T @ all_fea, p=2, dim=1)[confi_class_idx]
        similarities = F.cosine_similarity(all_fea.unsqueeze(1), initc.unsqueeze(0), dim=2)
        prediction_c1 = confi_class_idx[similarities.argmax(dim=1)]

        all_idxs = all_idx.int().tolist()
        preds = prediction_c1.tolist()
        confs = all_output.max(dim=1)[0] > self.beta

        return dict(zip(all_idxs, preds)), dict(zip(all_idxs, confs))

    @torch.no_grad()
    def detect_shared_classes(self, dataloader, with_target=False):
        self.model.eval()
        shared_classes = torch.zeros(self.num_classes, device=self.device)

        for inputs, *_ in dataloader:
            inputs = inputs.to(self.device)
            _, outputs = self.model(inputs)
            probs = F.softmax(outputs, dim=1)
            shared_classes += probs.sum(dim=0)

        diff = shared_classes.max() - shared_classes.min()
        shared_classes = (shared_classes - shared_classes.min()) / diff

        return (
            shared_classes.cpu(),
            (shared_classes >= self.threshold).nonzero(as_tuple=True)[0].tolist(),
        )

    @torch.no_grad()
    def validate(self, loader):
        self.model.eval()
        correct, total = 0, 0

        for images, labels, _ in loader:
            images, labels = images.to(self.device), labels.to(self.device)
            _, out_t = self.model(images)
            pred = F.softmax(out_t, dim=1).argmax(dim=1)
            total += images.size(0)
            correct += (pred == labels).sum()

        return correct / total

    def build_labels_tensor(self, indices):
        labels = [self.labels[idx.item()] for idx in indices]
        return torch.tensor(labels, device=self.device)

    def get_classes_imgs(self, loader):
        class_imgs_dict = {class_idx: [] for class_idx in self.increment_classes}

        for data in loader:
            inputs, *_, sample_idx, _ = data
            for label in self.increment_classes:
                class_mask = torch.tensor(
                    [self.labels[int(idx.item())] == label for idx in sample_idx]
                )
                if class_mask.any():
                    class_inputs = inputs[class_mask].cpu()
                    class_imgs_dict[label].append(class_inputs)

        for label in class_imgs_dict:
            if len(class_imgs_dict[label]) > 0:
                class_imgs_dict[label] = torch.cat(class_imgs_dict[label], dim=0)

        return class_imgs_dict

    def get_pos_neg(self, q_label: int):
        feature_pos = self.prototypes[q_label].unsqueeze(0)
        neg_idx = torch.tensor([i for i in range(self.num_classes) if i != q_label])
        random_neg_idx = torch.randint(0, len(neg_idx), (10,))
        negative_pairs = self.prototypes[neg_idx[random_neg_idx]]
        return torch.cat((feature_pos, negative_pairs), dim=0).to(self.device)

    def class_level_label_distillation(self):
        to_remove = []
        for label in self.increment_classes:
            if label not in self.mu_history:
                self.mu_history[label] = self.uk[label]
            elif self.uk[label] >= self.mu_history[label]:
                self.mu_history[label] = self.uk[label]
                self.buffer.remove_label(label)
            elif self.uk[label] < self.mu_history[label]:
                to_remove.append(label)

        for label in to_remove:
            self.increment_classes.remove(label)

    def increment(self, target_loader, val_loader, num_epochs=30):
        # === Mostrar classes reais do incremento atual ===
        class_start = self.increment_id * 10
        class_end = class_start + 10
        true_increment_classes = list(range(class_start, class_end))
        print(f"[Incremento {self.increment_id}] Classes reais esperadas: {true_increment_classes}")

        self.uk, self.increment_classes = self.detect_shared_classes(target_loader)
        self.labels, self.confs = self.generate_pseudo_labels(target_loader)
        self.class_level_label_distillation()
        print(f"[Incremento {self.increment_id}] Classes detectadas: {self.increment_classes}")

        buffer_loader = self.buffer.get_dataloader(target_loader.batch_size)

        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0

            # Iterate over both source and target data
            for (source_inputs, source_labels, _), (target_inputs, _, indices, aug_images) in zip(
                self.source_loader, target_loader
            ):
                self.optimizer.zero_grad()
                torch.cuda.empty_cache()

                source_inputs = source_inputs.to(self.device)
                source_labels = source_labels.to(self.device)
                target_inputs = target_inputs.to(self.device)

                indices_list = indices.tolist()
                confidence_mask = torch.tensor([self.confs.get(idx, False) for idx in indices_list])
                confident_indices = indices[confidence_mask]
                confidence_mask = confidence_mask.to(self.device)

                # CE on source data
                source_feas, source_outputs = self.model(source_inputs)
                loss_ce = F.cross_entropy(source_outputs, source_labels)

                # CE on target data
                target_feas, target_outputs = self.model(target_inputs)
                if confident_indices.shape[0]:
                    conf_target_labels = self.build_labels_tensor(confident_indices)
                    loss_ce += F.cross_entropy(target_outputs[confidence_mask], conf_target_labels)

                # Domain-Level Contrastive Alignment
                batch_labels = []
                with torch.no_grad():
                    for label in range(self.num_classes):
                        mask = source_labels == label
                        if mask.sum() > 1:
                            batch_labels.append(label)
                            if label in self.added_classes:
                                self.source_centroids_mem[label] = 0.99 * self.source_centroids_mem[
                                    label
                                ] + 0.01 * source_feas[mask].mean(dim=0)
                            else:
                                self.added_classes.append(label)
                                self.source_centroids_mem[label] = source_feas[mask].mean(dim=0)

                # Loss IDC
                loss_idc = torch.tensor(0.0).cuda()
                loss_ic = torch.tensor(0.0).cuda()

                if confident_indices.shape[0]:
                    for label in self.increment_classes:
                        mask = [
                            i
                            for i, idx in enumerate(indices_list)
                            if (self.labels[idx] == label and self.confs[idx])
                        ]
                        if len(mask) and label in self.added_classes:
                            fea = target_feas[mask].mean(dim=0)
                            pos_neg_pair = torch.vstack(
                                [
                                    self.source_centroids_mem[label],
                                    target_feas[
                                        [
                                            i
                                            for i, idx in enumerate(indices_list)
                                            if (self.labels[idx] != label and self.confs[idx])
                                        ]
                                    ],
                                    self.source_centroids_mem[
                                        [
                                            b_label
                                            for b_label in self.added_classes
                                            if b_label != label
                                        ]
                                    ],
                                ]
                            )
                            loss_idc += contrastive_alignment_loss(fea, pos_neg_pair, 5, 0.3)
                    loss_idc /= len(self.increment_classes)

                # loss_ic
                unconf_id = [i for i, idx in enumerate(indices_list) if (not self.confs[idx])]
                aug_feas, _ = self.model(aug_images.to(self.device))

                if len(unconf_id):
                    for idx in unconf_id:
                        fea = target_feas[idx]
                        pos_neg_pair = torch.vstack(
                            [aug_feas[idx], aug_feas[[i for i in unconf_id if i != idx]]]
                        )
                        loss_ic += contrastive_alignment_loss(fea, pos_neg_pair, 5, 0.3)
                    loss_ic /= len(unconf_id)

                contras_loss = loss_idc + loss_ic
                distill_loss = torch.tensor(0.0).cuda()

                if buffer_loader:
                    re_org_img, _, re_org_sp = next(buffer_loader)
                    re_org_img = re_org_img.cuda()
                    re_org_sp = re_org_sp.cuda()
                    _, reply_ouputs = self.model(re_org_img)
                    distill_loss = distillation_loss(re_org_sp, reply_ouputs)

                # Backpropagation and optimizer step
                loss = loss_ce + contras_loss + distill_loss
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            self.last_acc = self.validate(val_loader)

            print(
                f'Epoch [{epoch + 1:2d}/{num_epochs:2d}], '
                f'Loss: {running_loss / len(self.source_loader):.6f}, '
                f'Accuracy: {self.last_acc:.6f}'
            )

            if epoch >= self.warm_up:
                if epoch % self.step_update == 0:
                    self.model.eval()
                    self.uk, _ = self.detect_shared_classes(target_loader, True)
                    new_classes = self.increment_classes.copy()

                    for label in self.increment_classes:
                        if self.uk[label] < 0.1:
                            new_classes.remove(label)
                            if label in self.added_classes:
                                self.added_classes.remove(label)

                    self.increment_classes = new_classes
                    print('Classes:', self.increment_classes)
                self.labels, self.confs = self.generate_pseudo_labels(target_loader, True)

        # === Cálculo e armazenamento das métricas do incremento ===
        accuracy, f1 = eval(self.model, val_loader, self.device)
        pseudo_label_acc = pseudo_label_evaluate(self.labels, target_loader)

        self.step_level[self.increment_id] = {
            "acc": accuracy if not isinstance(accuracy, torch.Tensor) else accuracy.item(),
            "f1": f1,
            "pseudo_label_acc": pseudo_label_acc,
        }

        # Avaliação isolada do incremento atual
        acc_inc, f1_inc = eval(self.model, target_loader, self.device)
        self.single_step_level[self.increment_id] = {
            "acc": acc_inc if not isinstance(acc_inc, torch.Tensor) else acc_inc.item(),
            "f1": f1_inc,
        }

        self.increment_id += 1

        imgs = self.get_classes_imgs(target_loader)
        for label in self.increment_classes:
            if len(imgs[label]) >= 10:
                self.mu_history[label] = self.uk[label]
                self.buffer.construct_exemplar_set(self.model, imgs[label], label)
    
    def export_metrics(self):
        return {
            "all_data_until_increment_steps": self.step_level,
            "each_step_eval": self.single_step_level,
            "re_eval": self.re_eval,
        }


# if __name__ == '__main__':
#     # Path configurations
#     data_dir = Path('datasets/office-home-pattern')
#     config_dir = Path('data/office_home/config.json')

#     # Experiment settings
#     num_classes = 65
#     num_epochs = 30
#     learning_rate = 0.001
#     SOURCE = 'art'
#     TARGET = 'clipart'
#     batch_size = 24
#     threshold = 0.1
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     BETA = 0.6
#     WARM_UP = 6
#     STEP_UPDATE = 3

#     # Paths
#     pretrained_path = Path(f'weights/{SOURCE}_source.pth')
#     data_path = data_dir / TARGET
#     results = {}

#     # Initialize the PLDCA class
#     pretrained_model = PrototypeResNet(num_classes, pretrained_path).to(device)
#     s_loader = setup_dataloader(data_dir / SOURCE, config_dir, range(65), batch_size, False)

#     set_seed(42)
#     pldca = PLDCA(
#         pretrained_model, s_loader, learning_rate, threshold, BETA, WARM_UP, STEP_UPDATE, device
#     )

#     # Perform incremental training for each increment
#     for i in range(num_classes // 10):
#         train_ids = range(i * 10, (i + 1) * 10)
#         val_ids = range(0, (i + 1) * 10)
#         t_loader = setup_dataloader(
#             data_path, config_dir, train_ids, batch_size, False, True, True, True
#         )
#         val_loader = setup_dataloader(data_path, config_dir, val_ids, batch_size, False, False)
#         pldca.increment(t_loader, val_loader, num_epochs)

#     print(pldca.last_acc)
