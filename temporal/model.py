"""1D-CNN that classifies the play phase of a window of per-second event probabilities."""

from __future__ import annotations

import torch
from torch import nn

from common.schema import CLASSES


class PhaseNet(nn.Module):
    """``(N, C, T)`` -> ``(N, n_phases)`` logits; max+mean pooling over time."""

    def __init__(self, in_ch: int = len(CLASSES), n_phases: int = len(CLASSES), hidden: int = 32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, hidden, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden * 2, 3, padding=1),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden * 4, n_phases)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        return self.head(torch.cat([h.mean(dim=2), h.amax(dim=2)], dim=1))


class WindowMajor(nn.Module):
    """``(N, T, C)`` in, as the service API and ONNX export use."""

    def __init__(self, net: PhaseNet):
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.transpose(1, 2))
