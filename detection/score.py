"""Score a saved checkpoint on the val and test splits, in the format ``detection.train`` writes.

    python -m detection.score --ckpt models/detector.pt --report detection/results.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from common.device import torch_device
from detection.model import build_model
from detection.train import ClipDataset, evaluate, pick


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=Path("detection/configs/default.yaml"))
    a = ap.parse_args()
    cfg = yaml.safe_load(a.config.read_text())
    ckpt = torch.load(a.ckpt, map_location="cpu", weights_only=True)
    cfg.update(use_flow=ckpt["use_flow"], aggregator=ckpt["aggregator"], arch=ckpt["arch"])
    device = torch_device(cfg.get("device", "auto"))
    model = build_model(ckpt["arch"], ckpt["in_dim"], **ckpt["model_kwargs"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    root = Path(cfg["root"])
    report = {"label_source": "weak: commentary keywords", "arch": ckpt["arch"]}
    for split in ("val", "test"):
        ids, ann = pick(root, split, cfg)
        report[split] = evaluate(model, ClipDataset(root, split, ids, ann, cfg), cfg)
        print(split, {k: round(v["f1"], 3) for k, v in report[split]["event"].items()}, flush=True)
    a.report.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
