"""Train and evaluate the feature-based event detector on LoL-V2T (weak labels).

    python -m data.scripts.download_lolv2t --parts annotations train val test
    python -m detection.train_lolv2t --out models/lolv2t.pt

Metrics are against events derived from commentary keywords, not human event annotations.
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
    LOL_CLASSES,
    available_clips,
    load_annotations,
    load_clip,
    step_labels,
    step_times,
    weak_events,
)
from detection.events import EventAggregator
from detection.metrics import confusion_matrix, event_prf, per_class_prf
from detection.seq_model import FeatureEventNet

IGNORE = -100


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


def evaluate(model: FeatureEventNet, ds: ClipDataset, cfg: dict) -> dict:
    model.eval()
    cm = np.zeros((len(LOL_CLASSES),) * 2, dtype=np.int64)
    pooled: dict[str, dict[str, float]] = {}
    with torch.inference_mode():
        for i, cid in enumerate(ds.ids):
            x, y = ds[i]
            probs = torch.softmax(model(x.T[None]), dim=1)[0].T.numpy()  # (T, C)
            cm += confusion_matrix(y.numpy(), probs.argmax(1), len(LOL_CLASSES))
            agg = EventAggregator(**cfg["aggregator"], classes=LOL_CLASSES, hud=False)
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
    return {
        "confusion_matrix": cm.tolist(),
        "classes": LOL_CLASSES,
        "step": per_class_prf(cm, LOL_CLASSES),
        "event": pooled,
        "n_clips": len(ds),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("detection/configs/lolv2t.yaml"))
    ap.add_argument("--out", type=Path, default=Path("models/lolv2t.pt"))
    ap.add_argument("--report", type=Path, default=Path("detection/results_lolv2t.json"))
    a = ap.parse_args()
    cfg = yaml.safe_load(a.config.read_text())
    root = Path(cfg["root"])
    torch.manual_seed(cfg["seed"])

    splits = {s: pick(root, s, cfg) for s in ("train", "val", "test")}
    ds = {s: ClipDataset(root, s, ids, ann, cfg) for s, (ids, ann) in splits.items()}
    print({s: len(d) for s, d in ds.items()}, "clips")

    # feature statistics and class weights from a sample of training clips
    sample = [ds["train"][i] for i in range(min(cfg["norm_clips"], len(ds["train"])))]
    feats = torch.cat([x for x, _ in sample])
    labels = torch.cat([y for _, y in sample])
    counts = torch.bincount(labels, minlength=len(LOL_CLASSES)).clamp(min=1).float()
    print("sample step labels:", dict(zip(LOL_CLASSES, counts.int().tolist(), strict=True)))
    model = FeatureEventNet(feats.shape[1], hidden=cfg["hidden"], dropout=cfg["dropout"])
    model.mean.copy_(feats.mean(0).view(1, -1, 1))
    model.std.copy_(feats.std(0).clamp(min=1e-3).view(1, -1, 1))
    loss_fn = nn.CrossEntropyLoss(
        weight=(counts.sum() / (len(counts) * counts)).sqrt(), ignore_index=IGNORE
    )
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    loader = DataLoader(
        ds["train"], cfg["batch_size"], shuffle=True, collate_fn=collate, num_workers=2
    )
    best = -1.0
    a.out.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(cfg["epochs"]):
        model.train()
        total, n = 0.0, 0
        for x, y in loader:
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            total, n = total + loss.item() * len(x), n + len(x)
        val = evaluate(model, ds["val"], cfg)
        f1 = float(np.mean([m["f1"] for m in val["event"].values()])) if val["event"] else 0.0
        print(
            f"epoch {epoch + 1}/{cfg['epochs']}  loss {total / n:.4f}  val event macro-F1 {f1:.3f}"
        )
        if f1 > best:  # keep the best validation checkpoint
            best = f1
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "in_dim": int(feats.shape[1]),
                    "hidden": cfg["hidden"],
                    "dropout": cfg["dropout"],
                    "classes": LOL_CLASSES,
                    "use_flow": cfg["use_flow"],
                    "aggregator": cfg["aggregator"],
                },
                a.out,
            )
    ckpt = torch.load(a.out, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    report = {
        "label_source": "weak: commentary keywords",
        "val": evaluate(model, ds["val"], cfg),
        "test": evaluate(model, ds["test"], cfg),
    }
    a.report.write_text(json.dumps(report, indent=1))
    print(json.dumps(report["test"]["event"], indent=1))


if __name__ == "__main__":
    main()
