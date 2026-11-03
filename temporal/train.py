"""Train the window autoencoder on normal play and evaluate irregular-play flagging on held-out matches.

    python -m temporal.train --events-dir data/events        # on detector output (default pipeline)
    python -m temporal.train --use-gt                        # on ground-truth events
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from temporal.dataset import events_to_bins, load_match, make_windows, window_labels
from temporal.model import ConvAE


def build(lo: int, hi: int, cfg: dict, stride: int, ann_dir: Path, events_dir: Path | None):
    xs, ys, mids, starts = [], [], [], []
    for i in range(lo, hi):
        events, ann = load_match(i, ann_dir, events_dir)
        w, s = make_windows(events_to_bins(events, ann.duration), cfg["window"], stride)
        if len(w) == 0:
            continue
        xs.append(w)
        ys.append(window_labels(s, cfg["window"], ann.anomalies))
        mids += [i] * len(w)
        starts.append(s)
    return np.concatenate(xs), np.concatenate(ys), np.array(mids), np.concatenate(starts)


def scores_of(model: ConvAE, x: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.inference_mode():
        t = torch.from_numpy(x).transpose(1, 2)
        return ((model(t) - t) ** 2).mean(dim=(1, 2)).numpy()


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    """Rank-based AUROC (Mann-Whitney U), ties averaged."""
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    allv = np.concatenate([pos, neg])[order]
    ranks = np.empty(len(allv))
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2 + 1
        i = j + 1
    r_pos = ranks[np.argsort(order)][: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def kill_burst_baseline(x: np.ndarray) -> np.ndarray:
    """Rule baseline: most kills in any 8-second span of the window (blind to droughts)."""
    k = x[:, :, 0]
    c = np.cumsum(np.pad(k, ((0, 0), (1, 0))), axis=1)
    return (c[:, 8:] - c[:, :-8]).max(axis=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("temporal/configs/default.yaml"))
    ap.add_argument("--events-dir", type=Path, default=Path("data/events"))
    ap.add_argument("--use-gt", action="store_true", help="use ground-truth events instead")
    ap.add_argument("--annotations", type=Path, default=Path("data/annotations"))
    ap.add_argument("--out", type=Path, default=Path("models/temporal.pt"))
    ap.add_argument("--report", type=Path, default=Path("temporal/results.json"))
    a = ap.parse_args()
    cfg = yaml.safe_load(a.config.read_text())
    ev_dir = None if a.use_gt else a.events_dir
    torch.manual_seed(cfg["seed"])

    x, y, _, _ = build(*cfg["train_matches"], cfg, cfg["train_stride"], a.annotations, ev_dir)
    xn = torch.from_numpy(x[y == 0]).transpose(1, 2).contiguous()
    print(f"training on {len(xn)} normal windows ({int((y == 1).sum())} anomalous windows excluded)")
    model = ConvAE()
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    for epoch in range(cfg["epochs"]):
        model.train()
        perm = torch.randperm(len(xn))
        total = 0.0
        for k in range(0, len(perm), cfg["batch_size"]):
            b = xn[perm[k:k + cfg["batch_size"]]]
            opt.zero_grad()
            loss = ((model(b) - b) ** 2).mean()
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        if (epoch + 1) % 5 == 0:
            print(f"epoch {epoch + 1}/{cfg['epochs']}  mse {total / len(perm):.5f}")

    xc, yc, _, _ = build(*cfg["calib_matches"], cfg, cfg["stride"], a.annotations, ev_dir)
    threshold = float(np.quantile(scores_of(model, xc[yc == 0]), cfg["threshold_quantile"]))

    xt, yt, mt, st = build(*cfg["test_matches"], cfg, cfg["stride"], a.annotations, ev_dir)
    s = scores_of(model, xt)
    flag = s > threshold
    tp, fp, fn = int((flag & (yt == 1)).sum()), int((flag & (yt == 0)).sum()), int((~flag & (yt == 1)).sum())
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)

    # Interval-level recall: an annotated anomaly counts as found if any overlapping window is flagged.
    found, total = {}, {}
    for i in range(*cfg["test_matches"]):
        _, ann = load_match(i, a.annotations, ev_dir)
        for an in ann.anomalies:
            sel = (mt == i) & (st + cfg["window"] > an.start) & (st < an.end)
            total[an.kind] = total.get(an.kind, 0) + 1
            found[an.kind] = found.get(an.kind, 0) + int(flag[sel].any())
    report = {
        "events_source": "ground_truth" if a.use_gt else "detector_output",
        "threshold": threshold,
        "window_auroc": auroc(yt, s),
        "baseline_kill_burst_auroc": auroc(yt, kill_burst_baseline(xt)),
        "window_precision": prec, "window_recall": rec,
        "window_f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "anomaly_interval_recall": {k: found[k] / total[k] for k in total},
        "n_test_windows": len(yt), "n_anomalous_windows": int(yt.sum()),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "window": cfg["window"], "stride": cfg["stride"],
                "threshold": threshold}, a.out)
    a.report.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
