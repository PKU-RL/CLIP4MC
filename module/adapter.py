from __future__ import annotations
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

from typing import *
import torch
import torch.nn as nn


class AdapterHead(nn.Module):
    def __init__(self,
                 adapter_layers: int,
                 feature_dim) -> None:
        super().__init__()

        self.adapter_layers = adapter_layers
        self.residual_weight = None
        if adapter_layers == 0:
            self.adapter = nn.Identity()
        else:
            self.adapter = nn.Sequential(
                *([nn.Linear(feature_dim, feature_dim), nn.ReLU()] * (adapter_layers - 1)),
                nn.Linear(feature_dim, feature_dim))
            self.residual_weight = nn.Parameter(torch.tensor(4.0))
        self.layers = 1

    def get_layer(self, layer: int):
        assert layer == 0
        return self.adapter, self.residual_weight

    def forward(self, features):
        if self.residual_weight is None:
            return self.adapter(features)
        else:
            res = torch.sigmoid(self.residual_weight)
            return res * features + (1.0 - res) * self.adapter(features)


def build_adapter(config_name='video_adapter_config'):
    import yaml
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)[config_name]
    adapter = AdapterHead(adapter_layers=config['adapter_layers'],
                          feature_dim=config['feature_dim'])
    return adapter
