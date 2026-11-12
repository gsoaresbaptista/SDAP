import copy
from pathlib import Path

import torch
import torch.nn.functional as F

from experiment.manager import setup_dataloader
from models.base_model import PrototypeResNet
from utils.settings import set_seed


class SourceTrainer:
    def __init__(self, num_classes, lr, weight_decay, device, smooth):
        self.device = device
        self.model = PrototypeResNet(num_classes=num_classes).to(device)
        self.num_classes = num_classes
        self.label_smoothing = smooth

        param_group = []
        for k, v in self.model.resnet.named_parameters():
            if k[:2] == 'fc':
                param_group += [{'params': v, 'lr': lr * 10}]
            else:
                param_group += [{'params': v, 'lr': lr}]

        self.optimizer = torch.optim.SGD(param_group, lr, 0.9, weight_decay=weight_decay)

    @torch.no_grad()
    def validate(self, loader):
        self.model.eval()
        corrects, total = 0, 0

        for inputs, labels in loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            outputs, _ = self.model(inputs)
            outputs = F.softmax(outputs, dim=1).argmax(dim=1)
            corrects += (outputs == labels).sum().item()
            total += inputs.shape[0]

        return corrects / total

    def train(self, train_loader, val_loader, num_epochs, step):
        self.best_acc = 0.0
        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=step, gamma=0.1)

        for epoch in range(num_epochs):
            self.model.train()
            running_loss, acc = 0.0, 0.0

            for inputs, labels in train_loader:
                self.optimizer.zero_grad()
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs, _ = self.model(inputs)
                loss_ce = F.cross_entropy(outputs, labels, label_smoothing=self.label_smoothing)

                loss = loss_ce
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            scheduler.step()
            acc = self.validate(val_loader)
            if acc > self.best_acc:
                self.best_acc = acc
                self.best_model = copy.deepcopy(self.model).cpu()
            print(
                f'Epoch [{epoch + 1:2d}/{num_epochs:2d}], '
                f'Train Loss: {running_loss / len(train_loader):.6f}, '
                f'Accuracy: {acc:.6f}'
            )

        self.model = self.best_model
        self.model = self.model.to(self.device)
        del self.best_model


if __name__ == '__main__':
    # Path configurations
    # data_dir = Path('datasets/office-home')
    # data_dir = Path('datasets/office-31')
    data_dir = Path('datasets/caltech_imagenet')
    # config_dir = Path('data/office_home/config.json')
    # config_dir = Path('data/office_31/config.json')
    config_dir = Path('data/caltech_imagenet/config.json')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Experiment settings
    # num_classes = 65 # office-home
    # num_classes = 31 # office-31
    num_classes = 80 # caltech-imagenet
    num_epochs = 100
    WEIGHT_DECAY = 5e-4
    BATCH_SIZE = 32
    SMOOTH = 0.1
    STEP = 50
    LR = 1e-3
    # DOMAINS = ['product', 'real_world', 'art', 'clipart'] # office-home
    # DOMAINS = ['Amazon', 'DSLR', 'Webcam'] # office-31
    DOMAINS = ['caltech', 'imagenet'] # caltech-imagenet

    for source in DOMAINS:
        all_ids = range(num_classes)
        train_loader = setup_dataloader(data_dir / source, config_dir, all_ids, BATCH_SIZE, True)
        val_loader = setup_dataloader(
            data_dir / source, config_dir, all_ids, BATCH_SIZE, False, False
        )
        set_seed(987)
        trainer = SourceTrainer(num_classes, LR, WEIGHT_DECAY, device, SMOOTH)
        trainer.train(train_loader, val_loader, num_epochs, STEP)
        trainer.model.save_model(f'weights/{source}_source.pth')
