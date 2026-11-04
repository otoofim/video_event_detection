"""Train the temporal phase classifier on detector output and evaluate it on the test split.

    python -m temporal.train --detector models/detector.pt --out models/temporal.pt

Input: per-second class probabilities from the detector (video features -> events). Label: the
play phase of the window taken from the captions (kill > objective > teamfight > background).
The model is trained on detector outputs of the val split, because the detector has fitted the
train split and its outputs there are cleaner than they will be on new clips.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn

from common.lolv2t import (
    available_clips,
    load_annotations,
    load_clip,
    step_labels,
    step_times,
    weak_events,
)
from common.schema import CLASSES
from detection.infer import Detector
from detection.metrics import confusion_matrix, per_class_prf
from temporal.dataset import bin_probs, make_windows, window_phases
from temporal.model import PhaseNet


def build(det: Detector, root: Path, split: str, cfg: dict):
    ann = load_annotations(root, split)
    ids = available_clips(root, split, ann, det.use_flow)
    if cfg["max_clips"]:
        ids = ids[: cfg["max_clips"]]
    if not ids:
        raise SystemExit(f"no feature files for split '{split}'; run data.scripts.download_lolv2t")
    xs, ys, clip_of = [], [], []
    for k, cid in enumerate(ids):
        a = ann[cid]
        feats = load_clip(root, split, cid, det.use_flow)
        probs = det.probs(feats)
        times = step_times(len(feats), a["duration"])
        w, starts = make_windows(
            bin_probs(probs, times, a["duration"], cfg["bin_s"]), cfg["window"], cfg["stride"]
        )
        if len(w) == 0:
            continue
        labels = step_labels(weak_events(a), len(feats), a["duration"])
        xs.append(w)
        ys.append(window_phases(labels, times, starts, cfg["window"], cfg["bin_s"]))
        clip_of += [k] * len(w)
    return np.concatenate(xs), np.concatenate(ys), np.array(clip_of)


def predict(model: PhaseNet, x: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.inference_mode():
        return model(torch.from_numpy(x).transpose(1, 2)).argmax(1).numpy()


def report(y: np.ndarray, pred: np.ndarray) -> dict:
    cm = confusion_matrix(y, pred, len(CLASSES))
    prf = per_class_prf(cm)
    return {
        "accuracy": float(np.trace(cm) / max(cm.sum(), 1)),
        "macro_f1": float(np.mean([v["f1"] for v in prf.values()])),
        "per_class": prf,
        "confusion_matrix": cm.tolist(),
        "classes": CLASSES,
        "n_windows": int(len(y)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("temporal/configs/default.yaml"))
    ap.add_argument("--detector", type=Path, default=Path("models/detector.pt"))
    ap.add_argument("--out", type=Path, default=Path("models/temporal.pt"))
    ap.add_argument("--report", type=Path, default=Path("temporal/results.json"))
    a = ap.parse_args()
    cfg = yaml.safe_load(a.config.read_text())
    root = Path(cfg["root"])
    torch.manual_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    det = Detector(a.detector)

    x, y, clip_of = build(det, root, "val", cfg)
    held = rng.random(clip_of.max() + 1) < cfg["holdout_frac"]
    tr, ho = ~held[clip_of], held[clip_of]
    print(
        f"windows: train {int(tr.sum())}, holdout {int(ho.sum())}; phases {np.bincount(y).tolist()}"
    )

    counts = np.bincount(y[tr], minlength=len(CLASSES)).clip(min=1)
    weight = torch.tensor(np.sqrt(counts.sum() / (len(counts) * counts)), dtype=torch.float32)
    model = PhaseNet(hidden=cfg["hidden"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    xt = torch.from_numpy(x[tr]).transpose(1, 2).contiguous()
    yt = torch.from_numpy(y[tr])
    best, best_state = -1.0, None
    for _ in range(cfg["epochs"]):
        model.train()
        perm = torch.randperm(len(yt))
        for k in range(0, len(perm), cfg["batch_size"]):
            idx = perm[k : k + cfg["batch_size"]]
            opt.zero_grad()
            loss_fn(model(xt[idx]), yt[idx]).backward()
            opt.step()
        f1 = report(y[ho], predict(model, x[ho]))["macro_f1"]
        if f1 > best:
            best, best_state = f1, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)

    xs, ys, _ = build(det, root, "test", cfg)
    baseline = xs.mean(axis=1)[:, 1:].argmax(1) + 1  # strongest mean event probability ...
    baseline[xs.mean(axis=1)[:, 1:].max(1) < 0.2] = 0  # ... or background if all are weak
    out = {
        "label_source": "weak: commentary keywords",
        "model": report(ys, predict(model, xs)),
        "baseline_argmax_mean_prob": report(ys, baseline),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "hidden": cfg["hidden"],
            "window": cfg["window"],
            "stride": cfg["stride"],
            "bin_s": cfg["bin_s"],
            "classes": CLASSES,
        },
        a.out,
    )
    a.report.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "label_source"}, indent=1)[:2000])


if __name__ == "__main__":
    main()
