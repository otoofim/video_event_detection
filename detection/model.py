"""Per-step event classifier over precomputed clip features (used for LoL-V2T)."""

from __future__ import annotations

import torch
from torch import nn

from common.lolv2t import CLASSES


class _Block(nn.Module):
    def __init__(self, ch: int, dilation: int, dropout: float):
        super().__init__()
        self.conv = nn.Conv1d(ch, ch, 3, padding=dilation, dilation=dilation)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop(torch.relu(self.conv(x)))


class FeatureEventNet(nn.Module):
    """``(N, D, T)`` features -> ``(N, C, T)`` per-step logits.

    A 1x1 projection followed by dilated residual convolutions (receptive field about 29 steps).
    Feature mean/std are stored as buffers so the exported model normalises its own input.
    """

    def __init__(
        self,
        in_dim: int,
        n_classes: int = len(CLASSES),
        hidden: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.register_buffer("mean", torch.zeros(1, in_dim, 1))
        self.register_buffer("std", torch.ones(1, in_dim, 1))
        self.proj = nn.Sequential(nn.Dropout(dropout), nn.Conv1d(in_dim, hidden, 1), nn.ReLU())
        self.blocks = nn.Sequential(*[_Block(hidden, d, dropout) for d in (1, 2, 4, 8)])
        self.out = nn.Conv1d(hidden, n_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.blocks(self.proj((x - self.mean) / self.std)))


class StepMajor(nn.Module):
    """``(N, T, D)`` in, ``(N, T, C)`` out: the layout the service API and ONNX export use."""

    def __init__(self, net: FeatureEventNet):
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.transpose(1, 2)).transpose(1, 2)
