"""Frame-level event classifier: a small CNN that runs comfortably on CPU."""
from __future__ import annotations

import torch
from torch import nn

from common.schema import CLASSES


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class EventNet(nn.Module):
    """Input ``(N, 3, H, W)`` -> logits ``(N, len(CLASSES))``.

    The pooled grid keeps coarse spatial layout: banners are told apart by *where* they are as well
    as by colour, so a global average pool would throw away exactly the signal that matters.
    """

    def __init__(self, num_classes: int = len(CLASSES), width: int = 16):
        super().__init__()
        self.features = nn.Sequential(
            _block(3, width), _block(width, width * 2), _block(width * 2, width * 4),
            _block(width * 4, width * 4),
        )
        self.pool = nn.AdaptiveAvgPool2d((3, 5))
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.2), nn.Linear(width * 4 * 15, 64), nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(self.features(x)))
