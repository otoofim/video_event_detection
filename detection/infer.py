"""Run the detector over a video (or directory of videos) and emit the event stream as JSON.

    python -m detection.infer --video data/raw/match_0000.mp4 --weights models/detector.pt
    python -m detection.infer --video-dir data/raw --weights models/detector.onnx --out-dir data/events
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from common.schema import Event
from common.video import iter_frames, preprocess
from detection.events import EventAggregator


class Detector:
    """Frame classifier over either a PyTorch checkpoint (.pt) or an ONNX file (.onnx)."""

    def __init__(self, weights: str | Path, size: tuple[int, int] | None = None):
        self.weights = str(weights)
        if self.weights.endswith(".onnx"):
            import onnxruntime as ort

            self._sess = ort.InferenceSession(self.weights, providers=["CPUExecutionProvider"])
            self._name = self._sess.get_inputs()[0].name
            self.size = tuple(size or (160, 96))
            self._model = None
        else:
            import torch

            from detection.model import EventNet

            ckpt = torch.load(self.weights, map_location="cpu")
            self.size = tuple(ckpt["size"])
            self._model = EventNet(width=ckpt["width"])
            self._model.load_state_dict(ckpt["state_dict"])
            self._model.eval()
            self._sess = None

    def probs(self, batch: np.ndarray) -> np.ndarray:
        """``(N, 3, H, W)`` float32 -> ``(N, n_classes)`` probabilities."""
        if self._model is not None:
            import torch

            with torch.inference_mode():
                logits = self._model(torch.from_numpy(batch)).numpy()
        else:
            logits = self._sess.run(None, {self._name: batch})[0]
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)


def detect_video(path: str | Path, det: Detector, min_run: int = 2, min_conf: float = 0.5,
                 batch_size: int = 32) -> tuple[list[Event], np.ndarray]:
    """Return the event stream and per-frame argmax predictions for a video."""
    agg = EventAggregator(min_run, min_conf)
    events: list[Event] = []
    preds: list[int] = []
    buf: list[tuple[float, np.ndarray]] = []

    def flush() -> None:
        if not buf:
            return
        p = det.probs(np.stack([preprocess(f, det.size) for _, f in buf]))
        for (ts, frame), pr in zip(buf, p, strict=True):
            preds.append(int(pr.argmax()))
            events.extend(agg.update(ts, pr, frame))
        buf.clear()

    for _, ts, frame in iter_frames(path):
        buf.append((ts, frame))
        if len(buf) >= batch_size:
            flush()
    flush()
    return events, np.array(preds, dtype=np.int64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path)
    ap.add_argument("--video-dir", type=Path)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--out", type=Path, help="event JSON for --video (default: stdout)")
    ap.add_argument("--out-dir", type=Path, default=Path("data/events"))
    ap.add_argument("--min-run", type=int, default=2)
    ap.add_argument("--min-conf", type=float, default=0.5)
    a = ap.parse_args()
    det = Detector(a.weights)
    videos = [a.video] if a.video else sorted(a.video_dir.glob("*.mp4"))
    for v in videos:
        t0 = time.perf_counter()
        events, _ = detect_video(v, det, a.min_run, a.min_conf)
        payload = json.dumps([e.model_dump() for e in events], indent=1)
        if a.video and not a.out:
            print(payload)
            continue
        out = a.out if a.video else a.out_dir / f"{v.stem}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(payload)
        print(f"{v.name}: {len(events)} events in {time.perf_counter() - t0:.1f}s -> {out}")


if __name__ == "__main__":
    main()
