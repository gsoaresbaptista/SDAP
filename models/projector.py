import torch
from torch import nn

__maximum_layer_node = {
    "huge": 8192,
    "large": 4096,
    "base": 2048,
    "small_2": 1024,
    "small": 256,
}


class ConditionalBatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features)
        self.eps = eps

    def forward(self, x):
        if x.size(0) > 1:
            return self.bn(x)
        else:
            with torch.no_grad():
                weight = self.bn.weight
                bias = self.bn.bias
                running_mean = self.bn.running_mean
                running_var = self.bn.running_var
                x_normalized = (x - running_mean) / torch.sqrt(running_var + self.eps)
                return x_normalized * weight + bias


def get_bottleneck(bottleneck_size: str, input_feature: int) -> nn.Module:
    previous_layer = input_feature
    next_layer = __maximum_layer_node[bottleneck_size]
    layers = []

    for _ in range(2):
        layers.append(nn.Linear(previous_layer, next_layer))
        layers.append(ConditionalBatchNorm1d(next_layer))
        layers.append(nn.ReLU(inplace=True))
        previous_layer = next_layer

    return nn.Sequential(*layers)


class Projector(nn.Module):
    def __init__(self, num_classes, bottleneck_size="base", features=1536):
        super().__init__()
        self.projector = get_bottleneck(bottleneck_size, input_feature=features)
        self.head = nn.Linear(self.projector[-3].out_features, num_classes)

    def forward(self, x):
        features = self.projector(x)
        output = self.head(features)
        return features, output

    def save_weights(self, filepath):
        torch.save(self.state_dict(), filepath)

    def load_weights(self, filepath, device="cpu"):
        self.load_state_dict(
            torch.load(filepath, map_location=device, weights_only=True)
        )
