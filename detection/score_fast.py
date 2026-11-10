"""Score checkpoints on train, val and test: loss, accuracy, precision, recall, F1, confusion matrix.

    python -m detection.score_fast models/reg4_c.pt models/detector_unet.pt --tune-conf

``--tune-conf`` picks ``min_conf`` on val from a grid, then applies it to every split. Without it
the checkpoint's stored aggregator is used. Step metrics use the argmax class; ``step_at_conf``
also applies ``min_conf``. Event metrics are the +-3 s matched events. The train split is the full
training set, scored in eval mode (no dropout or noise).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from common.device import torch_device
from detection.gpu_data import EvalSet, GpuSplit
from detection.model import build_model
from detection.weight_search import evaluate, rank_metrics

SPLITS = ["train", "val", "test"]


def show_matrix(step: dict) -> str:
    names = step["classes"]
    lines = ["    true \\ pred " + "".join(f"{n[:9]:>11s}" for n in names)]
    for name, row in zip(names, step["confusion_matrix"], strict=True):
        lines.append(f"    {name[:12]:12s}" + "".join(f"{v:11d}" for v in row))
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path("detection/results_scores.json"))
    ap.add_argument("--config", type=Path, default=Path("detection/configs/default.yaml"))
    ap.add_argument("--tune-conf", action="store_true")
    ap.add_argument("--conf-grid", type=float, nargs="+", default=[0.2, 0.3, 0.4, 0.5])
    ap.add_argument("--splits", nargs="+", default=SPLITS, choices=SPLITS)
    ap.add_argument("--matrices", action="store_true", help="print confusion matrices")
    a = ap.parse_args()

    cfg = yaml.safe_load(a.config.read_text())
    device = torch_device(cfg.get("device", "auto"))
    root, tol = Path(cfg["root"]), cfg["tolerance_s"]

    def load_set(split: str) -> EvalSet:
        gs = GpuSplit(root, split, cfg, device)
        return EvalSet(gs, range(len(gs)))

    val_set = load_set("val")
    models, results = {}, {}
    for path in a.ckpts:
        ck = torch.load(path, map_location="cpu", weights_only=True)
        model = build_model(ck["arch"], ck["in_dim"], **ck["model_kwargs"])
        model.load_state_dict(ck["state_dict"])
        model.to(device)
        agg = dict(ck["aggregator"])
        if a.tune_conf:
            f1 = {
                c: evaluate(model, val_set, {**agg, "min_conf": c}, tol)["mean_event_f1"]
                for c in a.conf_grid
            }
            agg["min_conf"] = max(f1, key=f1.get)
        models[path.stem] = (model, agg)
        results[path.stem] = {"aggregator": agg}

    for split in (s for s in SPLITS if s in a.splits):
        es = val_set if split == "val" else load_set(split)
        print(f"\n===== {split} ({es.n} clips) =====", flush=True)
        for name, (model, agg) in models.items():
            r = results[name][split] = evaluate(model, es, agg, tol)
            r["rank"] = rank_metrics(model, es)
            st = r["step"]
            print(
                f"{name:14s} conf {agg['min_conf']:.1f}  loss {r['ce']:.3f}  acc {st['accuracy']:.3f}"
                f"  step P {st['macro_precision']:.3f} R {st['macro_recall']:.3f}"
                f" F1 {st['macro_f1']:.3f}  event P {r['mean_event_precision']:.3f}"
                f" R {r['mean_event_recall']:.3f} F1 {r['mean_event_f1']:.3f}"
                f"  AUROC {r['rank']['mean_auroc']:.3f} AP lift "
                f"{r['rank']['mean_ap'] / r['rank']['mean_base_rate']:.2f}x",
                flush=True,
            )
            if a.matrices:
                print(show_matrix(st))
        if split != "val":
            del es
            torch.cuda.empty_cache()
    a.out.write_text(json.dumps(results, indent=1))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
