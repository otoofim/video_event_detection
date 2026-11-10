"""Train the detector from GPU-resident cached features, watching train/val divergence.

    python -m detection.gpu_data --splits train val          # once
    python -m detection.train_gpu --config detection/configs/unet_reg1.yaml --run-name reg1

Every epoch the model is scored in eval mode on a fixed set of training clips and on all val clips
with the same metrics (CE, event F1), so the two curves are comparable. Training stops when val CE
the train-val gap in CE or event F1 has exceeded its limit for ``--patience`` epochs in a row. Test is scored only with ``--test``.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.tensorboard import SummaryWriter

from common.device import torch_device
from common.lolv2t import CLASSES
from detection.gpu_data import EvalSet, GpuSplit
from detection.model import IGNORE, build_model, vae_loss
from detection.weight_search import evaluate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--arch", choices=["vunet", "unet", "dilated"])
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--report", type=Path)
    ap.add_argument("--logdir", type=Path, default=Path("runs"))
    ap.add_argument("--monitor-clips", type=int, default=500)
    ap.add_argument("--conf-grid", type=float, nargs="+", default=[0.2, 0.3, 0.4, 0.5])
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--max-ce-gap", type=float, default=0.10, help="val CE minus train CE")
    ap.add_argument("--max-f1-gap", type=float, default=0.10, help="train F1 minus val F1")
    ap.add_argument("--test", action="store_true", help="also score test (final config only)")
    a = ap.parse_args()

    cfg = yaml.safe_load(a.config.read_text())
    if a.arch:
        cfg["arch"] = a.arch
    out = a.out or Path(f"models/{a.run_name}.pt")
    report_path = a.report or Path(f"detection/results_{a.run_name}.json")
    device = torch_device(cfg.get("device", "auto"))
    torch.manual_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    root, agg, tol = Path(cfg["root"]), cfg["aggregator"], cfg["tolerance_s"]
    tb = SummaryWriter(a.logdir / a.run_name)
    tb.add_text("config", "    " + yaml.safe_dump(cfg).replace("\n", "\n    "))
    t0 = time.time()

    train = GpuSplit(root, "train", cfg, device, resident=True)
    val = GpuSplit(root, "val", cfg, device)
    train_mon = EvalSet(train, rng.permutation(len(train))[: a.monitor_clips])
    val_set = EvalSet(val, range(len(val)))
    print(
        f"data on device in {time.time() - t0:.0f}s: train {len(train)} val {len(val)} clips",
        flush=True,
    )

    n_norm = min(cfg["norm_clips"], len(train))
    feats = torch.cat([train.clip(i).float() for i in range(n_norm)])
    labels = torch.cat([train.y[i] for i in range(n_norm)])
    counts = torch.bincount(labels, minlength=len(CLASSES)).clamp(min=1).float()
    kwargs = dict(cfg["model"][cfg["arch"]])
    model = build_model(cfg["arch"], train.dim, **kwargs).to(device)
    model.mean.copy_(feats.mean(0).view(1, -1, 1))
    model.std.copy_(feats.std(0).clamp(min=1e-3).view(1, -1, 1))
    weight = (counts.sum() / (len(counts) * counts)).sqrt()
    del feats
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{cfg['arch']} {kwargs} {n_params:,} parameters", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    epochs, bs = cfg["epochs"], cfg["batch_size"]
    min_frac = cfg.get("lr_min_frac", 1.0)
    smooth = cfg.get("label_smoothing", 0.0)
    clip = cfg.get("grad_clip", 1.0)
    warmup = cfg["beta_warmup_epochs"] * math.ceil(len(train) / bs)

    best_f1, bad, step = -1.0, 0, 0
    history = []
    out.parent.mkdir(parents=True, exist_ok=True)
    stop_reason = "finished all epochs"
    for epoch in range(epochs):
        lr = cfg["lr"] * (
            min_frac + (1 - min_frac) * 0.5 * (1 + math.cos(math.pi * epoch / epochs))
        )
        for g in opt.param_groups:
            g["lr"] = lr
        model.train()
        perm = torch.randperm(len(train)).tolist()
        loss_sum, seen = 0.0, 0
        for k in range(0, len(perm), bs):
            x, y = train.batch(perm[k : k + bs])
            opt.zero_grad()
            res = model(x)
            if isinstance(res, dict) and res["mu"] is not None:
                beta = cfg["beta"] * min(1.0, step / warmup) if warmup else cfg["beta"]
                loss, _ = vae_loss(res, y, weight, cfg["lam"], beta)
            else:
                logits = res["logits"] if isinstance(res, dict) else res
                loss = F.cross_entropy(
                    logits, y, weight=weight, ignore_index=IGNORE, label_smoothing=smooth
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            step += 1
            loss_sum += float(loss.detach()) * len(x)
            seen += len(x)
        by_conf = {c: evaluate(model, val_set, {**agg, "min_conf": c}, tol) for c in a.conf_grid}
        conf = max(by_conf, key=lambda c: by_conf[c]["mean_event_f1"])
        va = by_conf[conf]
        tr = evaluate(model, train_mon, {**agg, "min_conf": conf}, tol)
        row = {
            "epoch": epoch + 1,
            "lr": lr,
            "train_loss": loss_sum / seen,
            "train_ce": tr["ce"],
            "min_conf": conf,
            "val_ce": va["ce"],
            "ce_gap": va["ce"] - tr["ce"],
            "train_f1": tr["mean_event_f1"],
            "val_f1": va["mean_event_f1"],
            "f1_gap": tr["mean_event_f1"] - va["mean_event_f1"],
        }
        history.append(row)
        for key, v in row.items():
            if key != "epoch":
                tb.add_scalar(f"monitor/{key}", v, epoch + 1)
        tb.flush()
        print(
            f"epoch {epoch + 1:3d}/{epochs}  loss {row['train_loss']:.3f}  "
            f"CE train {tr['ce']:.3f} val {va['ce']:.3f} gap {row['ce_gap']:+.3f}  "
            f"F1 train {tr['mean_event_f1']:.3f} val {va['mean_event_f1']:.3f} "
            f"gap {row['f1_gap']:+.3f}  conf {conf:.1f}  ({(time.time() - t0) / 60:.1f} min)",
            flush=True,
        )
        if va["mean_event_f1"] > best_f1:
            best_f1 = va["mean_event_f1"]
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "arch": cfg["arch"],
                    "model_kwargs": kwargs,
                    "in_dim": int(train.dim),
                    "classes": CLASSES,
                    "use_flow": cfg["use_flow"],
                    "aggregator": {**agg, "min_conf": conf},
                },
                out,
            )
        if row["ce_gap"] > a.max_ce_gap or row["f1_gap"] > a.max_f1_gap:
            bad += 1
            if bad >= a.patience:
                stop_reason = (
                    f"DIVERGENCE at epoch {epoch + 1}: CE gap {row['ce_gap']:+.3f} "
                    f"(limit {a.max_ce_gap}), F1 gap {row['f1_gap']:+.3f} (limit {a.max_f1_gap}) "
                    f"for {bad} epochs in a row"
                )
                break
        else:
            bad = 0
    print(stop_reason, flush=True)

    ckpt = torch.load(out, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["state_dict"])
    agg = ckpt["aggregator"]
    report = {
        "aggregator": agg,
        "run": a.run_name,
        "config": cfg,
        "stop_reason": stop_reason,
        "history": history,
        "best_val_f1": best_f1,
        "val": evaluate(model, val_set, agg, tol),
    }
    if a.test:
        test = GpuSplit(root, "test", cfg, device)
        report["test"] = evaluate(model, EvalSet(test, range(len(test))), agg, tol)
        print("test event F1:", round(report["test"]["mean_event_f1"], 4), flush=True)
    report_path.write_text(json.dumps(report, indent=1))
    print(f"best val F1 {best_f1:.4f}  checkpoint {out}  report {report_path}", flush=True)
    tb.close()


if __name__ == "__main__":
    main()
