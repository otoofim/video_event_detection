"""TSN ResNet-50 feature networks, loaded from the mmaction2 checkpoints the dataset used.

Two backends build the same network:

- ``mmaction2``: mmaction2 0.x's own ``ResNet`` backbone (same code as the dataset). Needs the
  ``extract`` extra plus ``mmcv-full``; see the README.
- ``torch``: torchvision's ResNet-50 with the checkpoint's keys renamed. Lighter, but only trust it
  after it passes ``python -m preprocess.verify`` against the shipped features.

Either way the feature is the backbone output, global-average-pooled to 2048 numbers per input.
"""

from __future__ import annotations

import re
from pathlib import Path

import torch
from torch import nn

_RULES = [
    (r"^conv1\.conv\.", "conv1."),
    (r"^conv1\.bn\.", "bn1."),
    (r"^(layer\d+\.\d+)\.conv(\d)\.conv\.", r"\1.conv\2."),
    (r"^(layer\d+\.\d+)\.conv(\d)\.bn\.", r"\1.bn\2."),
    (r"^(layer\d+\.\d+)\.downsample\.conv\.", r"\1.downsample.0."),
    (r"^(layer\d+\.\d+)\.downsample\.bn\.", r"\1.downsample.1."),
]


def mmaction_to_torchvision(state_dict: dict) -> dict:
    """Rename an mmaction2 ``Recognizer2D`` state dict's backbone keys to torchvision's ResNet."""
    out = {}
    for k, v in state_dict.items():
        if not k.startswith("backbone."):
            continue  # cls_head.* is the ActivityNet classifier, not used for features
        k = k[len("backbone.") :]
        for pat, rep in _RULES:
            k = re.sub(pat, rep, k)
        out[k] = v
    return out


class _Pooled(nn.Module):
    """Backbone feature map -> global average pool -> ``(N, 2048)``."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x).mean(dim=(2, 3))


def build_resnet50(in_channels: int = 3) -> nn.Module:
    """torchvision ResNet-50 whose ``forward`` returns pooled 2048-d features."""
    from torchvision.models import resnet50

    m = resnet50(weights=None)
    if in_channels != 3:
        m.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    m.fc = nn.Identity()
    return m


def build_mmaction_resnet50(in_channels: int = 3) -> nn.Module:
    try:
        from mmaction.models.backbones import ResNet
    except ImportError as exc:
        raise ImportError(
            "backend 'mmaction2' needs mmaction2 0.x and mmcv-full (see README); "
            "or use --backend torch"
        ) from exc
    return _Pooled(ResNet(depth=50, pretrained=None, in_channels=in_channels, norm_eval=False))


def resolve_checkpoint(path_or_url: str) -> Path:
    """A local path, or download a URL once into torch's hub cache."""
    if not path_or_url.startswith(("http://", "https://")):
        return Path(path_or_url)
    d = Path(torch.hub.get_dir()) / "checkpoints"
    d.mkdir(parents=True, exist_ok=True)
    dest = d / path_or_url.rsplit("/", 1)[-1]
    if not dest.exists():
        torch.hub.download_url_to_file(path_or_url, str(dest))
    return dest


def load_state_dict(path: Path, trust: bool = False) -> dict:
    """Checkpoint tensors only (``weights_only``); ``trust=True`` allows full unpickling."""
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=not trust)
    except Exception as exc:  # pickle.UnpicklingError for objects in the checkpoint's meta
        if trust:
            raise
        raise RuntimeError(
            f"{path} holds non-tensor objects; pass --trust-checkpoint if you trust its source"
        ) from exc
    return ckpt.get("state_dict", ckpt)


def load_network(
    checkpoint: str, in_channels: int, backend: str = "mmaction2", trust: bool = False
) -> nn.Module:
    sd = load_state_dict(resolve_checkpoint(checkpoint), trust)
    if backend == "torch":
        net = build_resnet50(in_channels)
        net.load_state_dict(mmaction_to_torchvision(sd), strict=True)
    elif backend == "mmaction2":
        net = build_mmaction_resnet50(in_channels)
        bb = {k[len("backbone.") :]: v for k, v in sd.items() if k.startswith("backbone.")}
        net.backbone.load_state_dict(bb, strict=True)
    else:
        raise ValueError(f"unknown backend {backend!r}")
    return net.eval()
