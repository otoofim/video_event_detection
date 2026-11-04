"""Acceptance check (manual; needs the dataset, the source video and the checkpoints).

Re-extract one LoL-V2T clip and compare with the shipped features, step by step:

    python -m preprocess.verify --clip-id 001_uMmV3EgD76U-039 --split val \\
        --video downloads/uMmV3EgD76U.mp4 --start 1234.5 --end 1258.1

``--start``/``--end`` (seconds) cut the clip out of the full broadcast; omit them if ``--video`` is
already the clip. The dataset does not ship scene timecodes, so the clip's position in the
broadcast must be found by hand (``preprocess.chunking.scene_chunks`` with the dataset's settings
should reproduce the cut). Target: cosine similarity above 0.99 at every step, for RGB and flow.
The report also tries small step offsets, which exposes an off-by-one in sampling.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common.lolv2t import load_features, split_dir
from preprocess.config import CHECKPOINTS
from preprocess.extract import Extractor, iter_frames
from preprocess.video import frame_count


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity of two ``(T, D)`` arrays (truncated to the shorter)."""
    t = min(len(a), len(b))
    a, b = a[:t], b[:t]
    return (a * b).sum(1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-12)


def compare(ours: np.ndarray, ref: np.ndarray, max_shift: int = 2) -> dict:
    """Per-step cosine at the best of a few step shifts, plus the summary at shift 0."""
    shifts = {}
    for s in range(-max_shift, max_shift + 1):
        a, b = (ours[s:], ref) if s >= 0 else (ours, ref[-s:])
        c = cosine(a, b)
        shifts[s] = float(c.mean()) if len(c) else float("nan")
    c0 = cosine(ours, ref)
    return {
        "steps_ours": len(ours),
        "steps_ref": len(ref),
        "mean": float(c0.mean()),
        "min": float(c0.min()),
        "frac_above_0.99": float((c0 > 0.99).mean()),
        "mean_by_shift": shifts,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data/lolv2t"))
    ap.add_argument("--split", choices=["train", "val", "test"], required=True)
    ap.add_argument("--clip-id", required=True)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--start", type=float, default=None, help="clip start in --video (s)")
    ap.add_argument("--end", type=float, default=None, help="clip end in --video (s)")
    ap.add_argument("--backend", choices=["mmaction2", "torch"], default="mmaction2")
    ap.add_argument("--trust-checkpoint", action="store_true")
    a = ap.parse_args()

    ex = Extractor.from_checkpoints(
        rgb=CHECKPOINTS["rgb"],
        flow=CHECKPOINTS["flow"],
        backend=a.backend,
        trust=a.trust_checkpoint,
    )
    n, fps = frame_count(a.video)
    s = int(round((a.start or 0.0) * fps))
    e = int(round(a.end * fps)) if a.end is not None else n
    frames = [
        f for i, f in enumerate(iter_frames(a.video, ex.cfg.width, ex.cfg.height)) if s <= i < e
    ]
    feats = ex.features(np.stack(frames), fps)["features"]
    d = split_dir(a.root, a.split)
    rgb_ref = load_features(d / f"{a.clip_id}_rgb.pkl")
    flow_ref = load_features(d / f"{a.clip_id}_flow.pkl")
    report = {
        "clip_id": a.clip_id,
        "fps": fps,
        "frames": len(frames),
        "rgb": compare(feats[:, :2048], rgb_ref),
        "flow": compare(feats[:, 2048:], flow_ref),
    }
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
