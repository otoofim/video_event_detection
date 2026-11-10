"""Gradient-free weight editing ("weight-space programming") of a saved U-Net checkpoint.

The checkpoint is copied and its weights are changed directly, with no backprop and no optimizer:
a mutation is kept only if it raises event F1 on a fixed set of *training* clips. A second set of
training clips is scored but never used to accept anything, and val/test are scored once at the end.

    python -m detection.weight_search --ckpt models/detector_unet.pt --out models/detector_unet_wsp.pt
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.tensorboard import SummaryWriter

from common.device import torch_device
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
from detection.model import build_model

NC = len(CLASSES)


def fast_events(cls: np.ndarray, times: np.ndarray, min_run: int) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised ``EventAggregator``: class and start time of every run that reaches ``min_run``."""
    change = np.flatnonzero(np.concatenate(([True], cls[1:] != cls[:-1])))
    lens = np.diff(np.append(change, len(cls)))
    s = change[(cls[change] != 0) & (lens >= min_run)]
    return cls[s], np.round(times[s], 3)


def step_metrics(cm: np.ndarray) -> dict:
    """Accuracy, per-class and macro precision/recall/F1 from a confusion matrix (rows = truth)."""
    per, tot = {}, cm.sum()
    for i, name in enumerate(CLASSES):
        tp = cm[i, i]
        pr, rc = tp / max(cm[:, i].sum(), 1), tp / max(cm[i, :].sum(), 1)
        per[name] = {
            "precision": float(pr),
            "recall": float(rc),
            "f1": float(2 * pr * rc / (pr + rc)) if pr + rc else 0.0,
            "support": int(cm[i, :].sum()),
        }
    return {
        "accuracy": float(np.trace(cm) / max(tot, 1)),
        "macro_precision": float(np.mean([m["precision"] for m in per.values()])),
        "macro_recall": float(np.mean([m["recall"] for m in per.values()])),
        "macro_f1": float(np.mean([m["f1"] for m in per.values()])),
        "per_class": per,
        "confusion_matrix": cm.astype(int).tolist(),
        "classes": CLASSES,
    }


def match(pred_ts: np.ndarray, gt_ts: list[float], tol: float) -> int:
    """Same greedy one-to-one matching as ``detection.metrics.event_prf``."""
    used = [False] * len(gt_ts)
    tp = 0
    for pt in pred_ts:
        best, best_d = -1, tol + 1e-9
        for j, gt in enumerate(gt_ts):
            d = abs(pt - gt)
            if not used[j] and d <= best_d:
                best, best_d = j, d
        if best >= 0:
            used[best] = True
            tp += 1
    return tp


def self_check(min_run: int, min_conf: float) -> None:
    rng = np.random.default_rng(0)
    for _ in range(300):
        t = int(rng.integers(5, 200))
        probs = rng.dirichlet(np.full(NC, 0.3), size=t)
        times = np.arange(t) * 0.217
        agg = EventAggregator(min_run, min_conf, CLASSES)
        ref = [e for ts, p in zip(times, probs, strict=True) for e in agg.update(float(ts), p)]
        cls = probs.argmax(1)
        cls[probs.max(1) < min_conf] = 0
        c, ts = fast_events(cls, times, min_run)
        assert [(CLASSES[i], float(x)) for i, x in zip(c, ts, strict=True)] == [
            (e.event_type, e.timestamp) for e in ref
        ], "fast aggregator differs from EventAggregator"


class ClipSet:
    """Clips grouped by exact length (so no padding) and kept on the device."""

    def __init__(self, root: Path, split: str, ids: list[str], ann: dict, cfg: dict, device):
        def load(cid: str):
            x = load_clip(root, split, cid, cfg["use_flow"])
            a = ann[cid]
            y = step_labels(weak_events(a), len(x), a["duration"], cfg["label_span_s"])
            return cid, x, y

        with ThreadPoolExecutor(8) as ex:
            items = list(ex.map(load, ids))
        self.n = len(items)
        self.times, self.gt = {}, {}
        by_len: dict[int, list] = {}
        for cid, x, y in items:
            a = ann[cid]
            self.times[cid] = step_times(len(x), a["duration"])
            g = {k: [] for k in range(1, NC)}
            for e in weak_events(a):
                g[CLASSES.index(e.event_type)].append(e.timestamp)
            self.gt[cid] = {k: sorted(v) for k, v in g.items()}
            by_len.setdefault(len(x), []).append((cid, x, y))
        self.groups = []
        for members in by_len.values():
            x = torch.from_numpy(np.stack([m[1] for m in members])).transpose(1, 2).contiguous()
            y = torch.from_numpy(np.stack([m[2] for m in members]))
            self.groups.append(([m[0] for m in members], x.to(device), y.to(device)))


@torch.no_grad()
def evaluate(model, cs: ClipSet, agg: dict, tol: float) -> dict:
    model.eval()
    tp, npred, ngt = np.zeros(NC), np.zeros(NC), np.zeros(NC)
    ce, steps = 0.0, 0
    cm_raw = np.zeros((NC, NC), dtype=np.int64)
    cm_conf = np.zeros((NC, NC), dtype=np.int64)
    for ids, x, y in cs.groups:
        out = model(x.float())
        logits = out["logits"] if isinstance(out, dict) else out
        ce += float(F.cross_entropy(logits, y, reduction="sum"))
        steps += y.numel()
        conf, cls = torch.softmax(logits, 1).max(1)
        cm_raw += (
            torch.bincount((y * NC + cls).flatten(), minlength=NC * NC).view(NC, NC).cpu().numpy()
        )
        cls[conf < agg["min_conf"]] = 0
        cm_conf += (
            torch.bincount((y * NC + cls).flatten(), minlength=NC * NC).view(NC, NC).cpu().numpy()
        )
        cls = cls.cpu().numpy()
        for k, cid in enumerate(ids):
            c, ts = fast_events(cls[k], cs.times[cid], agg["min_run"])
            for t in range(1, NC):
                p = ts[c == t]
                g = cs.gt[cid][t]
                npred[t] += len(p)
                ngt[t] += len(g)
                if len(p) and len(g):
                    tp[t] += match(p, g, tol)
    event = {}
    for t in range(1, NC):
        pr, rc = tp[t] / max(npred[t], 1), tp[t] / max(ngt[t], 1)
        event[CLASSES[t]] = {
            "tp": int(tp[t]),
            "n_pred": int(npred[t]),
            "n_gt": int(ngt[t]),
            "precision": float(pr),
            "recall": float(rc),
            "f1": float(2 * pr * rc / (pr + rc)) if pr + rc else 0.0,
        }
    live = [m for m in event.values() if m["n_gt"] or m["n_pred"]]
    return {
        "mean_event_f1": float(np.mean([m["f1"] for m in live])) if live else 0.0,
        "mean_event_precision": float(np.mean([m["precision"] for m in live])) if live else 0.0,
        "mean_event_recall": float(np.mean([m["recall"] for m in live])) if live else 0.0,
        "ce": ce / steps,
        "event": event,
        "step": step_metrics(cm_raw),
        "step_at_conf": step_metrics(cm_conf),
        "min_conf": agg["min_conf"],
        "n_clips": cs.n,
    }


@torch.no_grad()
def rank_metrics(model, cs: ClipSet) -> dict:
    """One-vs-rest AUROC and average precision per event class, from the step probabilities."""
    model.eval()
    probs, ys = [], []
    for _, x, y in cs.groups:
        out = model(x.float())
        logits = out["logits"] if isinstance(out, dict) else out
        probs.append(torch.softmax(logits, 1).permute(0, 2, 1).reshape(-1, NC))
        ys.append(y.reshape(-1))
    p, y = torch.cat(probs), torch.cat(ys)
    res = {"auroc": {}, "ap": {}, "base_rate": {}}
    for k in range(1, NC):
        pos = (y == k).float()[torch.argsort(p[:, k], descending=True)]
        tp, fp = torch.cumsum(pos, 0), torch.cumsum(1 - pos, 0)
        n_pos, n_neg = pos.sum(), (1 - pos).sum()
        zero = torch.zeros(1, device=p.device)
        auc = torch.trapezoid(torch.cat([zero, tp / n_pos]), torch.cat([zero, fp / n_neg]))
        ap = (pos * tp / torch.arange(1, len(pos) + 1, device=p.device)).sum() / n_pos
        res["auroc"][CLASSES[k]] = float(auc)
        res["ap"][CLASSES[k]] = float(ap)
        res["base_rate"][CLASSES[k]] = float(n_pos / len(pos))
    res["mean_auroc"] = float(np.mean(list(res["auroc"].values())))
    res["mean_ap"] = float(np.mean(list(res["ap"].values())))
    res["mean_base_rate"] = float(np.mean(list(res["base_rate"].values())))
    return res


def better(new: dict, cur: dict) -> bool:
    if new["mean_event_f1"] > cur["mean_event_f1"] + 1e-9:
        return True
    return abs(new["mean_event_f1"] - cur["mean_event_f1"]) <= 1e-9 and new["ce"] < cur["ce"] - 1e-6


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=Path("models/detector_unet.pt"))
    ap.add_argument("--out", type=Path, default=Path("models/detector_unet_wsp.pt"))
    ap.add_argument("--report", type=Path, default=Path("detection/results_unet_wsp.json"))
    ap.add_argument("--config", type=Path, default=Path("detection/configs/default.yaml"))
    ap.add_argument("--logdir", type=Path, default=Path("runs"))
    ap.add_argument("--n-fit", type=int, default=1000)
    ap.add_argument("--n-holdout", type=int, default=500)
    ap.add_argument("--minutes-b", type=float, default=15)
    ap.add_argument("--minutes-c", type=float, default=45)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--final-splits", nargs="+", default=["val", "test"], choices=["val", "test"])
    a = ap.parse_args()

    cfg = yaml.safe_load(a.config.read_text())
    ckpt = torch.load(a.ckpt, map_location="cpu", weights_only=True)
    cfg.update(use_flow=ckpt["use_flow"])
    agg, tol = ckpt["aggregator"], cfg["tolerance_s"]
    self_check(agg["min_run"], agg["min_conf"])
    print("fast aggregator matches EventAggregator", flush=True)

    device = torch_device(cfg.get("device", "auto"))
    model = build_model(ckpt["arch"], ckpt["in_dim"], **ckpt["model_kwargs"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    root = Path(cfg["root"])
    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)
    tb = SummaryWriter(a.logdir / f"wsp-{ckpt['arch']}-{time.strftime('%Y%m%d-%H%M%S')}")
    t0 = time.time()

    def loadset(split: str, ids: list[str] | None = None, ann: dict | None = None) -> ClipSet:
        ann = ann or load_annotations(root, split)
        ids = ids if ids is not None else available_clips(root, split, ann, cfg["use_flow"])
        return ClipSet(root, split, ids, ann, cfg, device)

    # validate the fast evaluator against the stored val report before trusting it
    val = loadset("val")
    base_val = evaluate(model, val, agg, tol)
    stored = json.loads(Path("detection/results_unet.json").read_text())["val"]["event"]
    for k, m in stored.items():
        got = base_val["event"][k]
        assert (got["tp"], got["n_pred"], got["n_gt"]) == (m["tp"], m["n_pred"], m["n_gt"]), k
    print("fast evaluator reproduces the stored val report exactly", flush=True)
    del val
    torch.cuda.empty_cache()

    ann_tr = load_annotations(root, "train")
    ids = available_clips(root, "train", ann_tr, cfg["use_flow"])
    pick = rng.permutation(len(ids))
    fit_ids = [ids[i] for i in pick[: a.n_fit]]
    hold_ids = [ids[i] for i in pick[a.n_fit : a.n_fit + a.n_holdout]]
    fit, hold = loadset("train", fit_ids, ann_tr), loadset("train", hold_ids, ann_tr)
    print(f"loaded fit={fit.n} holdout={hold.n} train clips in {time.time() - t0:.0f}s", flush=True)

    original = {"fit": evaluate(model, fit, agg, tol), "holdout": evaluate(model, hold, agg, tol)}
    of, oh = original["fit"]["mean_event_f1"], original["holdout"]["mean_event_f1"]
    print(f"original  fit F1 {of:.4f}  holdout F1 {oh:.4f}", flush=True)

    params = [p for p in model.parameters()]
    scales = [max(float(p.detach().std()), 1e-2) for p in params]
    cur = original["fit"]
    n_eval = 0
    last_save = time.time()

    def log(stage: str, sigma: float | None = None) -> None:
        tb.add_scalar("fit/mean_event_f1", cur["mean_event_f1"], n_eval)
        tb.add_scalar("fit/ce", cur["ce"], n_eval)
        if sigma is not None:
            tb.add_scalar("search/sigma", sigma, n_eval)

    def hold_log(stage: str) -> None:
        h = evaluate(model, hold, agg, tol)
        tb.add_scalar("holdout/mean_event_f1", h["mean_event_f1"], n_eval)
        tb.add_scalar("holdout/ce", h["ce"], n_eval)
        print(
            f"[{stage}] evals {n_eval:5d}  fit F1 {cur['mean_event_f1']:.4f}  "
            f"holdout F1 {h['mean_event_f1']:.4f}  ({(time.time() - t0) / 60:.1f} min)",
            flush=True,
        )

    def save() -> None:
        torch.save({**ckpt, "state_dict": model.state_dict()}, a.out)

    # --- stage A: line search on the output biases -------------------------------------------
    tb.add_text("stage", "A: output-bias line search")
    bias = model.out.bias
    for _ in range(2):
        for c in range(1, NC):
            base = float(bias[c])
            best_v = base
            for d in np.linspace(-1.5, 1.5, 31):
                with torch.no_grad():
                    bias[c] = base + float(d)
                new = evaluate(model, fit, agg, tol)
                n_eval += 1
                if better(new, cur):
                    cur, best_v = new, base + float(d)
            with torch.no_grad():
                bias[c] = best_v
            log("A")
    hold_log("A")

    # --- stages B and C: block-wise (1+1) evolution strategy ----------------------------------
    def es(stage: str, idxs: list[int], sigma: float, minutes: float) -> None:
        nonlocal cur, n_eval, last_save
        tb.add_text("stage", stage)
        w = np.array([params[i].numel() ** 0.5 for i in idxs])
        w /= w.sum()
        end = time.time() + minutes * 60
        while time.time() < end:
            i = int(rng.choice(idxs, p=w))
            p = params[i]
            with torch.no_grad():
                backup = p.clone()
                p.add_(torch.randn_like(p) * (sigma * scales[i]))
            new = evaluate(model, fit, agg, tol)
            n_eval += 1
            if better(new, cur):
                cur = new
                sigma = min(sigma * 1.5, 0.5)
            else:
                with torch.no_grad():
                    p.copy_(backup)
                sigma = max(sigma * 1.5**-0.25, 1e-4)
            log(stage, sigma)
            if n_eval % 100 == 0:
                hold_log(stage)
            if time.time() - last_save > 300:
                save()
                last_save = time.time()

    names = [n for n, _ in model.named_parameters()]
    es("B: output layer", [names.index("out.weight"), names.index("out.bias")], 0.05, a.minutes_b)
    hold_log("B")
    es("C: all weights", list(range(len(params))), 0.02, a.minutes_c)
    hold_log("C")
    save()

    # --- final scoring ------------------------------------------------------------------------
    tb.add_text("stage", "final scoring")
    report = {
        "label_source": "weak: commentary keywords",
        "arch": ckpt["arch"],
        "n_evals": n_eval,
        "fit_clips": a.n_fit,
        "holdout_clips": a.n_holdout,
        "minutes": (time.time() - t0) / 60,
    }
    edited = {"fit": evaluate(model, fit, agg, tol), "holdout": evaluate(model, hold, agg, tol)}
    fit = hold = None  # free device memory before loading val/test
    torch.cuda.empty_cache()
    orig_state = ckpt["state_dict"]
    edited_state = {k: v.clone() for k, v in model.state_dict().items()}
    for split in a.final_splits:
        cs = loadset(split)
        model.load_state_dict(edited_state)
        edited[split] = evaluate(model, cs, agg, tol)
        model.load_state_dict(orig_state)
        original[split] = evaluate(model, cs, agg, tol)
        del cs
        torch.cuda.empty_cache()
    report["original"], report["edited"] = original, edited
    a.report.write_text(json.dumps(report, indent=1))
    for k, split in enumerate(["fit", "holdout", *a.final_splits]):
        o, e = original[split]["mean_event_f1"], edited[split]["mean_event_f1"]
        tb.add_scalar("final/original_f1", o, k)
        tb.add_scalar("final/edited_f1", e, k)
        print(f"{split:8s} original {o:.4f}  edited {e:.4f}", flush=True)
    tb.close()


if __name__ == "__main__":
    main()
