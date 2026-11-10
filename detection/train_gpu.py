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
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter

from common.device import torch_device
from common.lolv2t import CLASSES
from detection.gpu_data import EvalSet, GpuSplit
from detection.model import IGNORE, build_model, vae_loss
from detection.weight_search import evaluate, rank_metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--arch", choices=["vunet", "unet", "dilated", "bigru"])
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--report", type=Path)
    ap.add_argument("--logdir", type=Path, default=Path("runs"))
    ap.add_argument("--monitor-clips", type=int, default=500)
    ap.add_argument("--conf-grid", type=float, nargs="+", default=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--max-ce-gap", type=float, default=0.10, help="val CE minus train CE")
    ap.add_argument("--max-f1-gap", type=float, default=0.10, help="train F1 minus val F1")
    ap.add_argument("--max-auroc-gap", type=float, default=0.10, help="train AUROC minus val AUROC")
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
    counts = torch.bincount(torch.cat(train.y), minlength=len(CLASSES)).clamp(min=1).float()
    kwargs = dict(cfg["model"][cfg["arch"]])
    model = build_model(cfg["arch"], train.dim, **kwargs).to(device)
    model.mean.copy_(feats.mean(0).view(1, -1, 1))
    model.std.copy_(feats.std(0).clamp(min=1e-3).view(1, -1, 1))
    power = cfg.get("class_weight_power", 0.5)
    weight = (counts.sum() / (len(counts) * counts)).pow(power)
    print(
        "train step share "
        + " ".join(
            f"{c} {n / counts.sum():.3f}" for c, n in zip(CLASSES, counts.tolist(), strict=True)
        )
        + f" | class weights (power {power}) "
        + " ".join(f"{c} {w:.2f}" for c, w in zip(CLASSES, weight.tolist(), strict=True)),
        flush=True,
    )
    del feats
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{cfg['arch']} {kwargs} {n_params:,} parameters", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    epochs, bs = cfg["epochs"], cfg["batch_size"]
    min_frac = cfg.get("lr_min_frac", 1.0)
    smooth = cfg.get("label_smoothing", 0.0)
    clip = cfg.get("grad_clip", 1.0)
    warmup = cfg["beta_warmup_epochs"] * math.ceil(len(train) / bs)

    select = cfg.get("select_metric", "event_f1")
    crop = cfg.get("crop")
    bg_keep = cfg.get("bg_keep", 1.0)
    ev_over = cfg.get("event_clip_oversample", 1.0)
    ev_pos = [np.flatnonzero(y.cpu().numpy()) for y in train.y]
    has_event = torch.tensor([len(p) > 0 for p in ev_pos])
    clip_p = 1.0 + (ev_over - 1.0) * has_event.float()
    print(
        f"sampling: crop {crop} bg_keep {bg_keep} event-clip oversample {ev_over} "
        f"({int(has_event.sum())}/{len(train)} clips have events)",
        flush=True,
    )

    def make_batch(idx: list[int]):
        xs, ys = [], []
        for i in idx:
            x, y = train.clip(i), train.y[i]
            if crop and len(y) > crop["min_len"]:
                n = int(rng.integers(crop["min_len"], crop["max_len"] + 1))
                if len(y) > n:
                    if len(ev_pos[i]) and rng.random() < crop["event_p"]:
                        e = int(ev_pos[i][rng.integers(len(ev_pos[i]))])
                        st = int(rng.integers(max(0, e - n + 1), min(e, len(y) - n) + 1))
                    else:
                        st = int(rng.integers(0, len(y) - n + 1))
                    x, y = x[st : st + n], y[st : st + n]
            xs.append(x)
            ys.append(y)
        x = pad_sequence(xs, batch_first=True).transpose(1, 2).float()
        y = pad_sequence(ys, batch_first=True, padding_value=IGNORE)
        if bg_keep == "balance":
            n = torch.bincount(y[y != IGNORE], minlength=len(CLASSES)).float()
            keep = min(1.0, max(0.05, float(n[1:].mean() / n[0]))) if n[0] > 0 else 1.0
            y = y.masked_fill((y == 0) & (torch.rand(y.shape, device=y.device) > keep), IGNORE)
        elif bg_keep < 1.0:
            y = y.masked_fill((y == 0) & (torch.rand(y.shape, device=y.device) > bg_keep), IGNORE)
        return x, y

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
        if ev_over != 1.0:
            perm = torch.multinomial(clip_p, len(train), replacement=True).tolist()
        else:
            perm = torch.randperm(len(train)).tolist()
        loss_sum, seen = 0.0, 0
        for k in range(0, len(perm), bs):
            x, y = make_batch(perm[k : k + bs])
            if not (y != IGNORE).any():
                continue
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
        rk_tr, rk_va = rank_metrics(model, train_mon), rank_metrics(model, val_set)
        score = {
            "step_macro_f1": va["step"]["macro_f1"],
            "auroc": rk_va["mean_auroc"],
        }.get(select, va["mean_event_f1"])
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
            "train_step_f1": tr["step"]["macro_f1"],
            "val_step_f1": va["step"]["macro_f1"],
            "train_step_recall": tr["step"]["macro_recall"],
            "val_step_recall": va["step"]["macro_recall"],
            "train_auroc": rk_tr["mean_auroc"],
            "val_auroc": rk_va["mean_auroc"],
            "auroc_gap": rk_tr["mean_auroc"] - rk_va["mean_auroc"],
            "val_ap_lift": rk_va["mean_ap"] / rk_va["mean_base_rate"],
            "train_step_precision": tr["step"]["macro_precision"],
            "val_step_precision": va["step"]["macro_precision"],
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
            f"gap {row['f1_gap']:+.3f} | step macro F1 train {row['train_step_f1']:.3f} "
            f"val {row['val_step_f1']:.3f} R val {row['val_step_recall']:.3f} "
            f"P val {row['val_step_precision']:.3f} | AUROC train {row['train_auroc']:.3f} "
            f"val {row['val_auroc']:.3f} gap {row['auroc_gap']:+.3f}  conf {conf:.1f}  "
            f"({(time.time() - t0) / 60:.1f} min)",
            flush=True,
        )
        if score > best_f1:
            best_f1 = score
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
        if (
            row["ce_gap"] > a.max_ce_gap
            or row["f1_gap"] > a.max_f1_gap
            or row["auroc_gap"] > a.max_auroc_gap
        ):
            bad += 1
            if bad >= a.patience:
                stop_reason = (
                    f"DIVERGENCE at epoch {epoch + 1}: CE gap {row['ce_gap']:+.3f} "
                    f"(limit {a.max_ce_gap}), F1 gap {row['f1_gap']:+.3f} (limit {a.max_f1_gap}), "
                    f"AUROC gap {row['auroc_gap']:+.3f} (limit {a.max_auroc_gap}) "
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
        "select_metric": select,
        "best_val_score": best_f1,
        "val": {**evaluate(model, val_set, agg, tol), "rank": rank_metrics(model, val_set)},
    }
    if a.test:
        test = GpuSplit(root, "test", cfg, device)
        test_set = EvalSet(test, range(len(test)))
        report["test"] = {
            **evaluate(model, test_set, agg, tol),
            "rank": rank_metrics(model, test_set),
        }
        print("test event F1:", round(report["test"]["mean_event_f1"], 4), flush=True)
    report_path.write_text(json.dumps(report, indent=1))
    print(f"best val {select} {best_f1:.4f}  checkpoint {out}  report {report_path}", flush=True)
    tb.close()


if __name__ == "__main__":
    main()
