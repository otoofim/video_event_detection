"""Train the frame classifier and evaluate it (frame- and event-level).

    python -m detection.train --config detection/configs/default.yaml --out models/detector.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn

from common.schema import CLASSES, Annotation
from common.sim import frame_labels
from common.video import iter_frames, preprocess
from detection.infer import Detector, detect_video
from detection.metrics import confusion_matrix, event_prf, per_class_prf
from detection.model import EventNet


def load_split(lo: int, hi: int, size, keep_bg: float, rng: np.random.Generator,
               raw: Path, ann_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for i in range(lo, hi):
        ann = Annotation.model_validate_json((ann_dir / f"match_{i:04d}.json").read_text())
        frames = [f for _, _, f in iter_frames(raw / ann.video)]
        labels = frame_labels(ann.events, len(frames), ann.fps)
        for f, y in zip(frames, labels, strict=True):
            if y == 0 and rng.random() > keep_bg:
                continue
            xs.append(preprocess(f, size))
            ys.append(y)
    return np.stack(xs), np.array(ys, dtype=np.int64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("detection/configs/default.yaml"))
    ap.add_argument("--out", type=Path, default=Path("models/detector.pt"))
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--annotations", type=Path, default=Path("data/annotations"))
    ap.add_argument("--report", type=Path, default=Path("detection/results.json"))
    a = ap.parse_args()
    cfg = yaml.safe_load(a.config.read_text())
    torch.manual_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    size = tuple(cfg["input_size"])

    x, y = load_split(*cfg["train_matches"], size, cfg["background_keep"], rng, a.raw, a.annotations)
    print(f"train frames: {len(y)}  per class: {np.bincount(y, minlength=len(CLASSES)).tolist()}")
    model = EventNet(width=cfg["width"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg["lr"], total_steps=cfg["epochs"] * int(np.ceil(len(y) / cfg["batch_size"])))
    counts = np.bincount(y, minlength=len(CLASSES)).clip(min=1)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(len(y) / (len(counts) * counts), dtype=torch.float32))
    xt, yt = torch.from_numpy(x), torch.from_numpy(y)
    for epoch in range(cfg["epochs"]):
        model.train()
        perm = torch.randperm(len(yt))
        total = 0.0
        for k in range(0, len(perm), cfg["batch_size"]):
            idx = perm[k:k + cfg["batch_size"]]
            opt.zero_grad()
            loss = loss_fn(model(xt[idx]), yt[idx])
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(idx)
        print(f"epoch {epoch + 1}/{cfg['epochs']}  loss {total / len(perm):.4f}")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "width": cfg["width"], "size": list(size),
                "classes": CLASSES}, a.out)

    # Evaluation on held-out matches: full videos through the real inference path, one pass.
    det = Detector(a.out)
    all_true, all_pred = [], []
    pooled: dict[str, dict[str, float]] = {}
    for i in range(*cfg["val_matches"]):
        ann = Annotation.model_validate_json((a.annotations / f"match_{i:04d}.json").read_text())
        events, preds = detect_video(a.raw / ann.video, det, **cfg["aggregator"])
        all_true.append(frame_labels(ann.events, len(preds), ann.fps))
        all_pred.append(preds)
        # events are matched within a match, then counts are pooled across matches
        for etype, m in event_prf(events, ann.events, cfg["tolerance_s"]).items():
            p = pooled.setdefault(etype, {"tp": 0, "n_pred": 0, "n_gt": 0})
            for k in p:
                p[k] += m[k]
    for p in pooled.values():
        prec, rec = p["tp"] / max(p["n_pred"], 1), p["tp"] / max(p["n_gt"], 1)
        p.update(precision=prec, recall=rec, f1=2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    cm = confusion_matrix(np.concatenate(all_true), np.concatenate(all_pred))
    report = {"confusion_matrix": cm.tolist(), "classes": CLASSES, "frame": per_class_prf(cm),
              "event": pooled}
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(report, indent=1))
    print(json.dumps(pooled, indent=1))


if __name__ == "__main__":
    main()
