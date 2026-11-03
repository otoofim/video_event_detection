"""1D-CNN autoencoder over windows of event features; reconstruction error = irregularity score."""
from __future__ import annotations

import torch
from torch import nn

from temporal.dataset import N_FEATURES


class ConvAE(nn.Module):
    """``(N, F, T)`` -> reconstruction ``(N, F, T)``; ``T`` must be divisible by 4."""

    def __init__(self, in_ch: int = N_FEATURES, latent: int = 8):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv1d(in_ch, 16, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv1d(16, 32, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv1d(32, latent, 3, padding=1),
        )
        self.dec = nn.Sequential(
            nn.Conv1d(latent, 32, 3, padding=1), nn.ReLU(),
            nn.ConvTranspose1d(32, 16, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose1d(16, in_ch, 4, stride=2, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dec(self.enc(x))


class WindowScorer(nn.Module):
    """Wraps the autoencoder to output one score per window (what gets exported to ONNX)."""

    def __init__(self, ae: ConvAE):
        super().__init__()
        self.ae = ae

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ((self.ae(x) - x) ** 2).mean(dim=(1, 2))
