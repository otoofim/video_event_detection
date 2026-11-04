"""Train and evaluate the event detector on LoL-V2T clip features (weak caption labels).

    python -m data.scripts.download_lolv2t --parts annotations train val test
    python -m detection.train --out models/detector.pt                       # variational U-Net
    python -m detection.train --arch unet --out models/detector_unet.pt      # ablation
    python -m detection.train --arch dilated --out models/detector_dilated.pt  # ablation

Metrics are against events derived from commentary keywords, not human event annotations.
Loss: class-weighted CE (masked) + ``lam`` * reconstruction MSE + ``beta`` * KL, with ``beta``
ramped linearly from 0 over the first ``beta_warmup_epochs``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset

from common.lolv2t import (
    CLASSES,
    available_clips,
    load_annotations,
    load_clip,
    step_labels,
    step_times,
    weak_events,
)
from detection.events import EventAggregator
from detection.metrics import confusion_matrix, event_prf, per_class_prf
from detection.model import IGNORE, _logits, beta_at, build_model, mc_probs, vae_loss


class ClipDataset(Dataset):
    """Loads features lazily: the train features alone are ~11 GB and do not fit in RAM."""

    def __init__(self, root: Path, split: str, clip_ids: list[str], ann: dict, cfg: dict):
        self.root, self.split, self.ids, self.ann, self.cfg = root, split, clip_ids, ann, cfg

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int):
        cid = self.ids[i]
        x = load_clip(self.root, self.split, cid, self.cfg["use_flow"])
        a = self.ann[cid]
        y = step_labels(weak_events(a), len(x), a["duration"], self.cfg["label_span_s"])
        return torch.from_numpy(x), torch.from_numpy(y)


def collate(batch):
    t = max(len(x) for x, _ in batch)
    xs = torch.zeros(len(batch), t, batch[0][0].shape[1])
    ys = torch.full((len(batch), t), IGNORE, dtype=torch.long)
    for k, (x, y) in enumerate(batch):
        xs[k, : len(x)], ys[k, : len(y)] = x, y
    return xs.transpose(1, 2), ys  # (N, D, T), (N, T)


def pick(root: Path, split: str, cfg: dict) -> tuple[list[str], dict]:
    ann = load_annotations(root, split)
    ids = available_clips(root, split, ann, cfg["use_flow"])
    if cfg["max_clips"]:
        ids = ids[: cfg["max_clips"]]
    if not ids:
        raise SystemExit(
            f"no feature files found for split '{split}'; run data.scripts.download_lolv2t"
        )
    return ids, ann


def compute_loss(
    model: nn.Module, x: torch.Tensor, y: torch.Tensor, weight, lam: float, beta: float
) -> tuple[torch.Tensor, dict[str, float]]:
    """Training loss for any arch: the full VAE loss, or masked CE for the ablations."""
    out = model(x)
    if isinstance(out, dict):
        return vae_loss(out, y, weight, lam, beta)
    ce = nn.functional.cross_entropy(out, y, weight=weight, ignore_index=IGNORE)
    return ce, {"ce": float(ce.detach())}


def evaluate(model: nn.Module, ds: ClipDataset, cfg: dict) -> dict:
    """Step and event metrics from the deterministic prediction (``z = mu``).

    For the variational model it also reports the mean sampled std of the predicted class on
    correct versus wrong steps: useful uncertainty is higher where the model is wrong.
    """
    model.eval()
    cm = np.zeros((len(CLASSES),) * 2, dtype=np.int64)
    pooled: dict[str, dict[str, float]] = {}
    unc = {"correct": [], "wrong": []}
    n_mc = cfg.get("mc_samples", 0) if getattr(model, "variational", False) else 0
    with torch.inference_mode():
        for i, cid in enumerate(ds.ids):
            x, y = ds[i]
            probs = torch.softmax(_logits(model, x.T[None]), dim=1)[0].T.numpy()  # (T, C)
            pred = probs.argmax(1)
            cm += confusion_matrix(y.numpy(), pred, len(CLASSES))
            if n_mc:
                _, std = mc_probs(model, x.T[None], n_mc)
                s = std[0].T.numpy()[np.arange(len(pred)), pred]
                ok = pred == y.numpy()
                unc["correct"] += s[ok].tolist()
                unc["wrong"] += s[~ok].tolist()
            agg = EventAggregator(**cfg["aggregator"], classes=CLASSES)
            events = [
                e
                for t, p in zip(step_times(len(x), ds.ann[cid]["duration"]), probs, strict=True)
                for e in agg.update(float(t), p)
            ]
            gt = weak_events(ds.ann[cid])
            for etype, m in event_prf(events, gt, cfg["tolerance_s"]).items():
                p = pooled.setdefault(etype, {"tp": 0, "n_pred": 0, "n_gt": 0})
                for k in p:
                    p[k] += m[k]
    for p in pooled.values():
        pr, rc = p["tp"] / max(p["n_pred"], 1), p["tp"] / max(p["n_gt"], 1)
        p.update(precision=pr, recall=rc, f1=2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    report = {
        "confusion_matrix": cm.tolist(),
        "classes": CLASSES,
        "step": per_class_prf(cm, CLASSES),
        "event": pooled,
        "n_clips": len(ds),
    }
    if n_mc:
        report["uncertainty"] = {
            "mc_samples": n_mc,
            **{f"mean_std_{k}": float(np.mean(v)) if v else None for k, v in unc.items()},
        }
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("detection/configs/default.yaml"))
    ap.add_argument("--out", type=Path, default=Path("models/detector.pt"))
    ap.add_argument("--report", type=Path, default=Path("detection/results.json"))
    ap.add_argument("--arch", choices=["vunet", "unet", "dilated"], help="override config arch")
    a = ap.parse_args()
    cfg = yaml.safe_load(a.config.read_text())
    if a.arch:
        cfg["arch"] = a.arch
    root = Path(cfg["root"])
    torch.manual_seed(cfg["seed"])

    splits = {s: pick(root, s, cfg) for s in ("train", "val", "test")}
    ds = {s: ClipDataset(root, s, ids, ann, cfg) for s, (ids, ann) in splits.items()}
    print({s: len(d) for s, d in ds.items()}, "clips")

    # feature statistics and class weights from a sample of training clips
    sample = [ds["train"][i] for i in range(min(cfg["norm_clips"], len(ds["train"])))]
    feats = torch.cat([x for x, _ in sample])
    labels = torch.cat([y for _, y in sample])
    counts = torch.bincount(labels, minlength=len(CLASSES)).clamp(min=1).float()
    print("sample step labels:", dict(zip(CLASSES, counts.int().tolist(), strict=True)))
    kwargs = dict(cfg["model"][cfg["arch"]])
    model = build_model(cfg["arch"], feats.shape[1], **kwargs)
    model.mean.copy_(feats.mean(0).view(1, -1, 1))
    model.std.copy_(feats.std(0).clamp(min=1e-3).view(1, -1, 1))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    loader = DataLoader(
        ds["train"],
        cfg["batch_size"],
        shuffle=True,
        collate_fn=collate,
        num_workers=cfg["num_workers"],
    )
    weight = (counts.sum() / (len(counts) * counts)).sqrt()
    warmup = cfg["beta_warmup_epochs"] * len(loader)
    best, step = -1.0, 0
    a.out.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(cfg["epochs"]):
        model.train()
        sums: dict[str, float] = {}
        n = 0
        for x, y in loader:
            opt.zero_grad()
            beta = beta_at(step, warmup, cfg["beta"])
            loss, parts = compute_loss(model, x, y, weight, cfg["lam"], beta)
            loss.backward()
            opt.step()
            step += 1
            n += len(x)
            for k, v in parts.items():
                sums[k] = sums.get(k, 0.0) + v * len(x)
        val = evaluate(model, ds["val"], cfg)
        f1 = float(np.mean([m["f1"] for m in val["event"].values()])) if val["event"] else 0.0
        terms = "  ".join(f"{k} {v / n:.4f}" for k, v in sums.items())
        print(f"epoch {epoch + 1}/{cfg['epochs']}  {terms}  beta {beta:.3f}  val event F1 {f1:.3f}")
        if f1 > best:  # keep the best validation checkpoint
            best = f1
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "arch": cfg["arch"],
                    "model_kwargs": kwargs,
                    "in_dim": int(feats.shape[1]),
                    "classes": CLASSES,
                    "use_flow": cfg["use_flow"],
                    "aggregator": cfg["aggregator"],
                },
                a.out,
            )
    ckpt = torch.load(a.out, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["state_dict"])
    report = {
        "label_source": "weak: commentary keywords",
        "arch": cfg["arch"],
        "val": evaluate(model, ds["val"], cfg),
        "test": evaluate(model, ds["test"], cfg),
    }
    a.report.write_text(json.dumps(report, indent=1))
    print(json.dumps(report["test"]["event"], indent=1))


if __name__ == "__main__":
    main()
