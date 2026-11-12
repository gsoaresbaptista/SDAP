from pathlib import Path

import torch
import torch.nn.functional as F

from experiment.manager import setup_dataloader
from models.base_model import PrototypeResNet
from training.common.losses import contrastive_alignment_loss, distillation_loss
from training.common.reply_buffer import ReplyBuffer
from utils.settings import set_seed
from incremental_old import load_incremental_loader
from src.process.eval import eval
from pprint import pprint


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


class ProCA:
    def __init__(self, pretrained_model, source_loader, lr, threshold, device):
        self.device = device
        self.threshold = threshold
        self.pretrained_model = pretrained_model
        self.pretrained_model.eval()
        self.source_loader = source_loader

        self.model = PrototypeResNet(num_classes=pretrained_model.resnet.fc.out_features).to(device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr, 0.9, weight_decay=1e-6)
        self.num_classes = pretrained_model.resnet.fc.out_features

        self.buffer = ReplyBuffer()
        self.mu_history = {}
        self.increment_id = 0

        self.compute_labeled_prototypes(source_loader)
        self.step_level = {}
        self.single_step_level = {}
        self.re_eval = {}

    @torch.no_grad()
    def compute_labeled_prototypes(self, dataloader):
        embedding_dim = self.model.resnet.fc.in_features

        self.prototypes = torch.zeros((self.num_classes, embedding_dim), device=self.device)
        counts = torch.zeros(self.num_classes, device=self.device)

        for inputs, labels in dataloader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            _, embeddings = self.pretrained_model(inputs)

            # Accumulate embeddings for each class using scatter_add
            labels_expanded = labels.view(-1, 1).expand(-1, embedding_dim)
            self.prototypes.scatter_add_(0, labels_expanded, embeddings)
            counts.scatter_add_(
                0,
                labels,
                torch.ones_like(labels, dtype=counts.dtype, device=self.device),
            )

        mask = counts > 0
        self.prototypes[mask] /= counts[mask].unsqueeze(1)

    @torch.no_grad()
    def generate_pseudo_labels(self, target_loader):
        self.pretrained_model.eval()
        all_feas, all_outputs, all_indices = [], [], []

        for inputs, _, idx, *_ in target_loader:
            inputs = inputs.to(self.device)
            outputs, feas = self.pretrained_model(inputs)
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
        prediction_c0 = confi_class_idx[(1 - similarities).argmin(dim=1)]

        aff = torch.zeros((prediction_c0.size(0), all_output.size(1)), device=all_fea.device)
        aff[torch.arange(prediction_c0.size(0)), prediction_c0] = 1
        initc = (aff.T @ all_fea)[confi_class_idx]

        similarities = F.cosine_similarity(all_fea.unsqueeze(1), initc.unsqueeze(0), dim=2)
        prediction_c1 = confi_class_idx[(1 - similarities).argmin(dim=1)]

        return dict(zip(all_idx.int().tolist(), prediction_c1.tolist()))

    @torch.no_grad()
    def detect_shared_classes(self, dataloader):
        shared_classes = torch.zeros(self.num_classes, device=self.device)

        for inputs, *_ in dataloader:
            inputs = inputs.to(self.device)
            outputs, _ = self.pretrained_model(inputs)
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

        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)
            out_t, _ = self.model(images)
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
            inputs, *_, sample_idx = data
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

    def increment(self, target_loader, val_loader, num_epochs=30):
        # === Mostrar classes reais do incremento atual ===
        class_start = self.increment_id * 10
        class_end = class_start + 10
        true_increment_classes = list(range(class_start, class_end))
        print(f"[Incremento {self.increment_id}] Classes reais esperadas: {true_increment_classes}")


        self.uk, self.increment_classes = self.detect_shared_classes(target_loader)
        self.labels = self.generate_pseudo_labels(target_loader)
        print(f"[Incremento {self.increment_id}] Classes detectadas: {self.increment_classes}")
        buffer_loader = self.buffer.get_dataloader(target_loader.batch_size)

        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0

            # Iterate over both source and target data
            for (source_inputs, source_labels), (target_inputs, _, indices) in zip(
                self.source_loader, target_loader
            ):
                self.optimizer.zero_grad()
                torch.cuda.empty_cache()

                source_inputs = source_inputs.to(self.device)
                source_labels = source_labels.to(self.device)
                target_inputs = target_inputs.to(self.device)

                # CE on source data
                source_outputs, _ = self.model(source_inputs)
                loss_ce = F.cross_entropy(source_outputs, source_labels)

                # CE on target data
                target_outputs, _ = self.model(target_inputs)
                loss_ce += F.cross_entropy(target_outputs, self.build_labels_tensor(indices))

                contras_loss = torch.tensor(0.0).cuda()
                distill_loss = torch.tensor(0.0).cuda()

                if buffer_loader:
                    re_org_img, re_org_label, re_org_sp = next(buffer_loader)
                    re_org_img = re_org_img.cuda()
                    re_org_label = re_org_label.cuda()
                    re_org_sp = re_org_sp.cuda()
                    reply_ouputs, reply_feas = self.model(re_org_img)

                    # Contrastive Alignment
                    for idx, fea in enumerate(reply_feas):
                        pos_neg_pair = self.get_pos_neg(self.labels[re_org_label[idx].item()])
                        contras_loss += contrastive_alignment_loss(fea, pos_neg_pair, 0.07, 0.3)
                    contras_loss /= len(reply_feas)

                    # Distillation
                    distill_loss = distillation_loss(re_org_sp, reply_ouputs)

                # Backpropagation and optimizer step
                loss = loss_ce + 0.1 * contras_loss + distill_loss
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            self.last_acc = self.validate(val_loader)

            print(
                f'Epoch [{epoch + 1:2d}/{num_epochs:2d}], '
                f'Loss: {running_loss / len(self.source_loader):.6f}, '
                f'Accuracy: {self.last_acc:.6f}'
            )

            if epoch >= 3:
                imgs = self.get_classes_imgs(target_loader)
                for label in self.increment_classes:
                    if len(imgs[label]) > 0:
                        if label not in self.mu_history or self.uk[label] >= self.mu_history[label]:
                            self.mu_history[label] = self.uk[label]
                            self.buffer.construct_exemplar_set(self.model, imgs[label], label)
                buffer_loader = self.buffer.get_dataloader(target_loader.batch_size)

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
        print("------- Temporary Metrics after Increment -------")
        pprint(self.export_metrics()) # AQUI
        print("-------------------------------------------------")

    def export_metrics(self):
        return {
            "all_data_until_increment_steps": self.step_level,
            "each_step_eval": self.single_step_level,
            "re_eval": self.re_eval,
        }


if __name__ == '__main__':
    # Path configurations
    data_dir = Path('datasets/office-home')
    config_dir = Path('data/office_home/config.json')

    # Experiment settings
    num_classes = 65
    num_epochs = 30
    learning_rate = 0.001
    SOURCE = 'art'
    TARGET = 'real_world'
    batch_size = 24
    threshold = 0.15
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Paths
    pretrained_path = Path(f'weights/{SOURCE}_source.pth')
    data_path = data_dir / TARGET

    # Initialize the ProCA class
    pretrained_model = PrototypeResNet(num_classes, pretrained_path).to(device)
    s_loader = setup_dataloader(data_dir / SOURCE, config_dir, range(65), batch_size, False)
    proca = ProCA(pretrained_model, s_loader, learning_rate, threshold, device)
    set_seed(987)

    # Perform incremental training for each increment
    for i in range(num_classes // 10):
        train_ids = range(i * 10, (i + 1) * 10)
        val_ids = range(0, (i + 1) * 10)
        t_loader = setup_dataloader(data_path, config_dir, train_ids, batch_size, False, True, True)
        val_loader = setup_dataloader(data_path, config_dir, val_ids, batch_size, False, False)
        proca.increment(t_loader, val_loader)
