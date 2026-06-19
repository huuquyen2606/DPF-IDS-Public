"""Feed-forward neural network used by the DPF-IDS notebooks."""

from __future__ import annotations

import torch
import torch.nn as nn

class FFNN(nn.Module):
    def __init__(self, in_features: int, num_classes: int, dropout_rate: float = 0.1):
        super().__init__()

        self.in_features = in_features
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        self.layer1 = nn.Linear(in_features, 64)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=dropout_rate)

        self.layer2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=dropout_rate)

        self.layer3 = nn.Linear(32, 16)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(p=dropout_rate)

        self.output_layer = nn.Linear(16, num_classes)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)

        x = self.layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.layer2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.layer3(x)
        x = self.relu3(x)
        x = self.dropout3(x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(x)
        logits = self.output_layer(features)
        return features,logits
