"""
MLP Block - Multi-Layer Perceptron
Reference: FuxiCTR fuxictr/pytorch/layers/blocks/mlp_block.py
"""
import torch
from torch import nn


class MLP(nn.Module):
    """Configurable MLP block with optional batch norm and dropout."""

    def __init__(self,
                 input_dim,
                 hidden_units=(256, 128),
                 activations="ReLU",
                 dropout=0.0,
                 batch_norm=False,
                 output_dim=None,
                 output_activation=None):
        super().__init__()
        layers = []
        if not isinstance(dropout, (list, tuple)):
            dropout = [dropout] * len(hidden_units)
        dims = [input_dim] + list(hidden_units)
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if batch_norm:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
            if activations == "ReLU":
                layers.append(nn.ReLU())
            elif activations == "PReLU":
                layers.append(nn.PReLU())
            elif activations == "Dice":
                layers.append(Dice(dims[i + 1]))
            if dropout[i] > 0:
                layers.append(nn.Dropout(p=dropout[i]))
        if output_dim is not None:
            layers.append(nn.Linear(dims[-1], output_dim))
        if output_activation == "sigmoid":
            layers.append(nn.Sigmoid())
        elif output_activation == "tanh":
            layers.append(nn.Tanh())
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class Dice(nn.Module):
    """Dice activation used in DIN."""

    def __init__(self, num_features, eps=1e-9):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features, eps=eps)
        self.alpha = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        p = torch.sigmoid(self.bn(x))
        return p * x + (1 - p) * self.alpha * x
