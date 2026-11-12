"""Per-step event detectors over clip features (N, D, T).

VariationalUNet is the detector; FeatureEventNet (dilated convolutions) and the U-Net
with variational=False are the ablations. build_model picks one by name.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from common.lolv2t import CLASSES

IGNORE = -100  # label of padded steps


class _Block(nn.Module):
    def __init__(self, ch: int, dilation: int, dropout: float):
        super().__init__()
        self.conv = nn.Conv1d(ch, ch, 3, padding=dilation, dilation=dilation)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop(torch.relu(self.conv(x)))


class FeatureEventNet(nn.Module):
    """``(N, D, T)`` features -> ``(N, C, T)`` per-step logits.

    A 1x1 projection followed by dilated residual convolutions; the receptive field is
    ``2 * sum(dilations) + 1`` steps (29 for the default). Feature mean/std are stored as buffers
    so the exported model normalises its own input.
    """

    def __init__(
        self,
        in_dim: int,
        n_classes: int = len(CLASSES),
        hidden: int = 256,
        dropout: float = 0.3,
        in_dropout: float | None = None,
        noise_std: float = 0.0,
        dilations: tuple[int, ...] = (1, 2, 4, 8),
    ):
        super().__init__()
        self.noise_std = noise_std
        self.register_buffer("mean", torch.zeros(1, in_dim, 1))
        self.register_buffer("std", torch.ones(1, in_dim, 1))
        in_drop = dropout if in_dropout is None else in_dropout
        self.proj = nn.Sequential(nn.Dropout(in_drop), nn.Conv1d(in_dim, hidden, 1), nn.ReLU())
        self.blocks = nn.Sequential(*[_Block(hidden, d, dropout) for d in dilations])
        self.out = nn.Conv1d(hidden, n_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = (x - self.mean) / self.std
        if self.training and self.noise_std > 0:
            h = h + torch.randn_like(h) * self.noise_std
        return self.out(self.blocks(self.proj(h)))


class GRUEventNet(nn.Module):
    """``(N, D, T)`` features -> ``(N, C, T)`` logits through a bidirectional GRU over the clip.

    Every step sees the whole clip. Padded steps (all-zero raw features) are packed out so the
    backward direction does not start from padding.
    """

    def __init__(
        self,
        in_dim: int,
        n_classes: int = len(CLASSES),
        hidden: int = 64,
        layers: int = 2,
        dropout: float = 0.3,
        in_dropout: float | None = None,
        noise_std: float = 0.0,
    ):
        super().__init__()
        self.noise_std = noise_std
        self.register_buffer("mean", torch.zeros(1, in_dim, 1))
        self.register_buffer("std", torch.ones(1, in_dim, 1))
        in_drop = dropout if in_dropout is None else in_dropout
        self.proj = nn.Sequential(nn.Dropout(in_drop), nn.Conv1d(in_dim, hidden, 1), nn.ReLU())
        self.gru = nn.GRU(
            hidden,
            hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.out = nn.Conv1d(2 * hidden, n_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lengths = (x.abs().sum(1) > 0).sum(1).clamp(min=1).cpu()
        h = (x - self.mean) / self.std
        if self.training and self.noise_std > 0:
            h = h + torch.randn_like(h) * self.noise_std
        h = self.proj(h).transpose(1, 2)
        packed = pack_padded_sequence(h, lengths, batch_first=True, enforce_sorted=False)
        out, _ = self.gru(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=h.shape[1])
        return self.out(self.drop(out).transpose(1, 2))


class TransformerEventNet(nn.Module):
    """``(N, D, T)`` features -> ``(N, C, T)`` logits through a pre-norm transformer encoder.

    A 1x1 projection to ``d_model``, sinusoidal positions, ``layers`` self-attention blocks over the
    whole clip, and a per-step classifier. Padded steps (all-zero raw features) are masked out of
    the attention keys.
    """

    def __init__(
        self,
        in_dim: int,
        n_classes: int = len(CLASSES),
        d_model: int = 64,
        layers: int = 2,
        heads: int = 4,
        ff: int = 128,
        dropout: float = 0.2,
        in_dropout: float | None = None,
        noise_std: float = 0.0,
        max_len: int = 4096,
        checkpoint: bool = False,
    ):
        super().__init__()
        self.noise_std = noise_std
        self.checkpoint = checkpoint
        self.register_buffer("mean", torch.zeros(1, in_dim, 1))
        self.register_buffer("std", torch.ones(1, in_dim, 1))
        in_drop = dropout if in_dropout is None else in_dropout
        self.proj = nn.Sequential(nn.Dropout(in_drop), nn.Conv1d(in_dim, d_model, 1))
        pos = torch.arange(max_len)[:, None]
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)[:, : d_model // 2]
        self.register_buffer("pe", pe, persistent=False)
        block = nn.TransformerEncoderLayer(
            d_model, heads, ff, dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Conv1d(d_model, n_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = x.abs().sum(1) == 0
        h = (x - self.mean) / self.std
        if self.training and self.noise_std > 0:
            h = h + torch.randn_like(h) * self.noise_std
        h = self.proj(h).transpose(1, 2) + self.pe[: x.shape[2]]
        mask = pad if bool(pad.any()) else None
        for layer in self.encoder.layers:
            if self.checkpoint and self.training:
                h = torch.utils.checkpoint.checkpoint(
                    layer, h, src_key_padding_mask=mask, use_reentrant=False
                )
            else:
                h = layer(h, src_key_padding_mask=mask)
        return self.out(self.norm(h).transpose(1, 2))


# --- Variational U-Net ------------------------------------------------------------------------


class _ConvBlock(nn.Module):
    """Two 3-step convolutions with a residual connection (no batch norm: batches are padded)."""

    def __init__(self, ch: int, dropout: float):
        super().__init__()
        self.c1 = nn.Conv1d(ch, ch, 3, padding=1)
        self.c2 = nn.Conv1d(ch, ch, 3, padding=1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x + self.c2(self.drop(torch.relu(self.c1(x)))))


class VariationalUNet(nn.Module):
    """1D U-Net over time with a variational bottleneck: ``(N, D, T)`` -> per-step logits.

    - Encoder: ``levels`` stride-2 convolutions (channels double each level), skips to the decoder.
    - Bottleneck: ``mu`` and ``log_var`` of a ``latent``-dim Gaussian per latent step; ``z`` is
      sampled in training, ``z = mu`` in eval unless ``noise`` is given.
    - Classification head: decoder from ``z`` *with* skips -> ``(N, C, T)`` logits.
    - Reconstruction head: decoder from ``z`` *without* skips -> the normalised input. Since it
      cannot copy the input through a skip, the latent has to carry information.

    Any length works: stride-2 convolutions round up (``ceil(T / 2**levels)`` latent steps, the
    same as zero-padding T to a multiple of ``2**levels``) and decoder outputs are cropped back.
    ``variational=False`` gives the plain U-Net ablation (no sampling, no reconstruction, no KL).
    """

    def __init__(
        self,
        in_dim: int,
        n_classes: int = len(CLASSES),
        hidden: int = 64,
        latent: int = 32,
        levels: int = 2,
        dropout: float = 0.1,
        variational: bool = True,
        in_dropout: float | None = None,
        noise_std: float = 0.0,
        blocks: int = 1,
    ):
        super().__init__()
        self.levels, self.latent, self.variational = levels, latent, variational
        self.noise_std = noise_std
        self.factor = 2**levels
        self.register_buffer("mean", torch.zeros(1, in_dim, 1))
        self.register_buffer("std", torch.ones(1, in_dim, 1))
        chs = [hidden * 2**i for i in range(levels + 1)]
        in_drop = dropout if in_dropout is None else in_dropout
        self.inp = nn.Sequential(nn.Dropout(in_drop), nn.Conv1d(in_dim, hidden, 1), nn.ReLU())

        def stack(ch: int) -> nn.Module:
            if blocks == 1:
                return _ConvBlock(ch, dropout)
            return nn.Sequential(*[_ConvBlock(ch, dropout) for _ in range(blocks)])

        self.enc = nn.ModuleList(stack(chs[i]) for i in range(levels))
        self.down = nn.ModuleList(
            nn.Conv1d(chs[i], chs[i + 1], 3, stride=2, padding=1) for i in range(levels)
        )
        self.mid = stack(chs[-1])
        self.up = nn.ModuleList(
            nn.ConvTranspose1d(chs[i + 1], chs[i], 2, stride=2) for i in reversed(range(levels))
        )
        self.dec = nn.ModuleList(
            nn.Sequential(nn.Conv1d(2 * chs[i], chs[i], 1), nn.ReLU(), stack(chs[i]))
            for i in reversed(range(levels))
        )
        self.out = nn.Conv1d(hidden, n_classes, 1)
        if variational:
            self.to_mu = nn.Conv1d(chs[-1], latent, 1)
            self.to_logvar = nn.Conv1d(chs[-1], latent, 1)
            self.from_z = nn.Conv1d(latent, chs[-1], 1)
            self.rec_up = nn.ModuleList(
                nn.ConvTranspose1d(chs[i + 1], chs[i], 2, stride=2) for i in reversed(range(levels))
            )
            self.rec_blocks = nn.ModuleList(
                _ConvBlock(chs[i], dropout) for i in reversed(range(levels))
            )
            self.rec_out = nn.Conv1d(hidden, in_dim, 1)

    def _decode_recon(self, z: torch.Tensor, lengths: list) -> torch.Tensor:
        h = torch.relu(self.from_z(z))
        for up, blk, n in zip(self.rec_up, self.rec_blocks, lengths, strict=True):
            h = blk(torch.relu(up(h))[..., :n])
        return self.rec_out(h)

    def reconstruct(self, z: torch.Tensor, t: int) -> torch.Tensor:
        """Reconstruction head alone: latent ``(N, Z, Tz)`` -> normalised features ``(N, D, t)``."""
        lengths = [t]
        for _ in range(self.levels - 1):
            lengths.append(-(-lengths[-1] // 2))
        return self._decode_recon(z, lengths[::-1])

    def forward(self, x: torch.Tensor, noise: torch.Tensor | None = None) -> dict:
        target = (x - self.mean) / self.std
        noisy = target
        if self.training and self.noise_std > 0:
            noisy = target + torch.randn_like(target) * self.noise_std
        h = self.inp(noisy)
        skips = []
        for blk, down in zip(self.enc, self.down, strict=True):
            h = blk(h)
            skips.append(h)
            h = torch.relu(down(h))
        h = self.mid(h)
        mu = logvar = recon = None
        if self.variational:
            mu, logvar = self.to_mu(h), self.to_logvar(h).clamp(-10.0, 10.0)
            if noise is None:
                noise = torch.randn_like(mu) if self.training else torch.zeros_like(mu)
            z = mu + noise * torch.exp(0.5 * logvar)
            recon = self._decode_recon(z, [s.shape[-1] for s in reversed(skips)])
            h = torch.relu(self.from_z(z))
        for up, dec, skip in zip(self.up, self.dec, reversed(skips), strict=True):
            h = torch.relu(up(h))[..., : skip.shape[-1]]
            h = dec(torch.cat([h, skip], dim=1))
        return {
            "logits": self.out(h),
            "recon": recon,
            "target": target,
            "mu": mu,
            "logvar": logvar,
            "factor": self.factor,
        }


def latent_mask(mask: torch.Tensor, factor: int) -> torch.Tensor:
    """``(N, T)`` valid-step mask -> ``(N, ceil(T / factor))``: a latent step counts if any of
    its ``factor`` input steps is valid."""
    pad = (-mask.shape[1]) % factor
    m = F.pad(mask.float(), (0, pad))
    return m.view(mask.shape[0], -1, factor).amax(dim=2) > 0


def vae_loss(
    out: dict, y: torch.Tensor, weight: torch.Tensor | None, lam: float, beta: float
) -> tuple[torch.Tensor, dict[str, float]]:
    """Masked class-weighted CE + ``lam`` * reconstruction MSE + ``beta`` * KL.

    ``y`` holds ``IGNORE`` at padded steps; padding is left out of all three terms. MSE is averaged
    over valid steps and feature dims; KL is summed over latent dims and averaged over valid
    latent steps.
    """
    ce = F.cross_entropy(out["logits"], y, weight=weight, ignore_index=IGNORE)
    parts = {"ce": float(ce.detach())}
    if out["mu"] is None:
        return ce, parts
    mask = y != IGNORE
    m = mask.unsqueeze(1).float()
    n_feat = out["target"].shape[1]
    recon = ((out["recon"] - out["target"]) ** 2 * m).sum() / (m.sum() * n_feat).clamp(min=1)
    lm = latent_mask(mask, out["factor"]).unsqueeze(1).float()
    mu, lv = out["mu"], out["logvar"]
    kl = (-0.5 * (1 + lv - mu**2 - lv.exp()) * lm).sum() / lm.sum().clamp(min=1)
    parts.update(recon=float(recon.detach()), kl=float(kl.detach()))
    return ce + lam * recon + beta * kl, parts


def beta_at(step: int, warmup_steps: int, beta_max: float) -> float:
    """KL weight ramped linearly from 0 to ``beta_max`` over ``warmup_steps``, then held."""
    if warmup_steps <= 0:
        return beta_max
    return beta_max * min(1.0, step / warmup_steps)


@torch.no_grad()
def mc_probs(
    model: nn.Module, x: torch.Tensor, n_samples: int = 16
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean and std of per-step class probabilities over ``n_samples`` draws of ``z``.

    The std is the model's predictive uncertainty. Deterministic models give a std of zero.
    """
    model.eval()
    if not getattr(model, "variational", False):
        p = torch.softmax(_logits(model, x), dim=1)
        return p, torch.zeros_like(p)
    mu = model(x)["mu"]
    ps = torch.stack(
        [
            torch.softmax(model(x, noise=torch.randn_like(mu))["logits"], dim=1)
            for _ in range(n_samples)
        ]
    )
    return ps.mean(0), ps.std(0, unbiased=False)


def _logits(model: nn.Module, x: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
    out = model(x) if noise is None else model(x, noise=noise)
    return out["logits"] if isinstance(out, dict) else out


ARCHS = {"vunet", "unet", "dilated", "bigru", "transformer"}


def build_model(arch: str, in_dim: int, n_classes: int = len(CLASSES), **kw) -> nn.Module:
    """``vunet`` (the detector), ``unet`` (ablation: no variational parts), ``dilated``
    (ablation: the earlier ``FeatureEventNet``), ``bigru`` (bidirectional GRU over the clip)."""
    if arch == "dilated":
        keys = ("hidden", "dropout", "in_dropout", "noise_std")
        opts = {k: kw[k] for k in keys if k in kw}
        if "dilations" in kw:
            opts["dilations"] = tuple(kw["dilations"])
        return FeatureEventNet(in_dim, n_classes, **opts)
    if arch == "transformer":
        keys = (
            "d_model",
            "layers",
            "heads",
            "ff",
            "dropout",
            "in_dropout",
            "noise_std",
            "checkpoint",
        )
        return TransformerEventNet(in_dim, n_classes, **{k: kw[k] for k in keys if k in kw})
    if arch == "bigru":
        keys = ("hidden", "layers", "dropout", "in_dropout", "noise_std")
        return GRUEventNet(in_dim, n_classes, **{k: kw[k] for k in keys if k in kw})
    if arch in ("vunet", "unet"):
        keys = ("hidden", "latent", "levels", "dropout", "in_dropout", "noise_std", "blocks")
        return VariationalUNet(
            in_dim, n_classes, variational=arch == "vunet", **{k: kw[k] for k in keys if k in kw}
        )
    raise ValueError(f"unknown arch {arch!r}; expected one of {sorted(ARCHS)}")


class StepMajor(nn.Module):
    """``(N, T, D)`` in, ``(N, T, C)`` logits out: the layout of the service API and ONNX export.

    For the variational model a second input ``noise`` ``(N, ceil(T / 2**levels), latent)`` sets
    ``z = mu + noise * sigma``: zeros give the deterministic prediction, random draws give samples.
    """

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        n = None if noise is None else noise.transpose(1, 2)
        return _logits(self.net, x.transpose(1, 2), n).transpose(1, 2)
