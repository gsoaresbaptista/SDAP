import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50


class PrototypeResNet(nn.Module):
    def __init__(self, num_classes, pretrained_weights_path=None):
        """
        Initializes the PrototypeResNet model with dual output for classification and prototype embeddings.

        Args:
            num_classes (int): The number of output classes for classification.
            pretrained_weights_path (str, optional): Path to the pretrained model weights file.
                                                    If None, loads default ImageNet weights.
        """
        super(PrototypeResNet, self).__init__()
        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
        self.num_classes = num_classes

        if pretrained_weights_path:
            self.load_pretrained_weights(pretrained_weights_path)

    def load_pretrained_weights(self, weights_path):
        checkpoint = torch.load(weights_path, weights_only=True)
        self.load_state_dict(checkpoint['model_state_dict'])

    def save_model(self, save_path):
        """
        Saves the model state to the specified file.

        Args:
            save_path (str): Path where the model state should be saved.
        """
        torch.save({'model_state_dict': self.state_dict()}, save_path)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        embedding = self.resnet.avgpool(x)
        embedding = torch.flatten(embedding, 1)

        return embedding
