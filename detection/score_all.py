"""Score every checkpoint listed in checkpoints/manifest.json on train, val and test.

    python -m detection.score_all --out visualisation/results/scores.json

For each model ``min_conf`` is picked on val, then loss, step and event metrics, AUROC/AP and the
step confusion matrix are computed on all three splits. Each model is scored against labels built
with its own ``label_offset_s``; the features of a split are put on the device once and reused.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import yaml

from common.device import torch_device
from detection.gpu_data import EvalSet, GpuSplit
from detection.model import build_model
from detection.weight_search import evaluate, rank_metrics

SPLITS = ("val", "test", "train")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=Path("checkpoints/manifest.json"))
    ap.add_argument("--out", type=Path, default=Path("visualisation/results/scores.json"))
    ap.add_argument("--config", type=Path, default=Path("detection/configs/default.yaml"))
    ap.add_argument("--conf-grid", type=float, nargs="+", default=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    ap.add_argument("--only", nargs="+", help="score just these checkpoint names")
    a = ap.parse_args()

    cfg = yaml.safe_load(a.config.read_text())
    device, root, tol = (
        torch_device(cfg.get("device", "auto")),
        Path(cfg["root"]),
        cfg["tolerance_s"],
    )
    manifest = [m for m in json.loads(a.manifest.read_text()) if not a.only or m["name"] in a.only]
    present = [m for m in manifest if Path(m["file"]).exists()]
    for m in manifest:
        if m not in present:
            print(f"skipping {m['name']}: {m['file']} is not in this checkout", flush=True)
    manifest = present
    # with --only, keep the scores of the other checkpoints and add or replace these
    results = json.loads(a.out.read_text()) if a.only and a.out.exists() else {}
    models = {}
    for m in manifest:
        ck = torch.load(m["file"], map_location="cpu", weights_only=True)
        net = build_model(ck["arch"], ck["in_dim"], **ck["model_kwargs"])
        net.load_state_dict(ck["state_dict"])
        models[m["name"]] = (net.to(device).eval(), ck["aggregator"], float(m["label_offset_s"]))
        results[m["name"]] = {"label_offset_s": float(m["label_offset_s"])}
    offsets = sorted({off for _, _, off in models.values()})
    t0 = time.time()
    for split in SPLITS:
        base = None
        for off in offsets:
            gs = GpuSplit(root, split, {**cfg, "label_offset_s": off}, device)
            if base is None:
                base = EvalSet(gs, range(len(gs)))
                es = base
            else:
                es = base.with_labels(gs)
            for name, (net, agg, model_off) in models.items():
                if model_off != off:
                    continue
                if split == "val":
                    f1 = {
                        c: evaluate(net, es, {**agg, "min_conf": c}, tol)["mean_event_f1"]
                        for c in a.conf_grid
                    }
                    results[name]["min_conf"] = max(f1, key=f1.get)
                    results[name]["val_f1_by_conf"] = f1
                use = {**agg, "min_conf": results[name]["min_conf"]}
                r = evaluate(net, es, use, tol)
                r["rank"] = rank_metrics(net, es)
                results[name][split] = r
            print(f"{split} offset {off:+g}: done ({(time.time() - t0) / 60:.1f} min)", flush=True)
        del base
        torch.cuda.empty_cache()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(results, separators=(",", ":")))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
