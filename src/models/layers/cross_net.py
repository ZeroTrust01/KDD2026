"""
CrossNet V2 - Deep & Cross Network V2
Reference: FuxiCTR fuxictr/pytorch/layers/interactions/cross_net.py
Paper: DCN V2 (Google, WWW 2021)
"""
import torch
from torch import nn


class CrossNetV2(nn.Module):
    """DCN V2 cross network: X_i = X_i + X_0 * W_i(X_i) + b_i"""

    def __init__(self, input_dim, num_layers=3):
        super().__init__()
        self.num_layers = num_layers
        self.cross_layers = nn.ModuleList(
            [nn.Linear(input_dim, input_dim) for _ in range(num_layers)]
        )

    def forward(self, x0):
        xi = x0  # [B, D]
        for i in range(self.num_layers):
            xi = xi + x0 * self.cross_layers[i](xi)
        return xi
