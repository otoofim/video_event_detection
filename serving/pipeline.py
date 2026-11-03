"""End-to-end pipeline: frames -> detector -> events -> temporal model -> irregular-play flags.

Only needs onnxruntime + numpy + opencv (no torch), so the service image stays small.
"""
from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import numpy as np
import onnxruntime as ort

from common.schema import Anomaly, Event
from common.video import preprocess
from detection.events import EventAggregator
from temporal.dataset import StreamFeaturizer, events_to_bins, make_windows


def _session(path: Path) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1  # many small calls: avoid oversubscription across workers
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


class Models:
    """Shared, read-only model sessions."""

    def __init__(self, model_dir: str | Path):
        d = Path(model_dir)
        self.meta = json.loads((d / "meta.json").read_text())
        self.det = _session(d / "detector.onnx")
        self.tmp = _session(d / "temporal.onnx")
        self.size = tuple(self.meta["detector"]["size"])
        self.window = self.meta["temporal"]["window"]
        self.stride = self.meta["temporal"]["stride"]
        self.threshold = self.meta["temporal"]["threshold"]

    def detect_probs(self, batch: np.ndarray) -> np.ndarray:
        logits = self.det.run(None, {"frames": batch})[0]
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def score_windows(self, windows: np.ndarray) -> np.ndarray:
        """``(N, T, F)`` windows -> ``(N,)`` irregularity scores."""
        x = np.ascontiguousarray(windows.transpose(0, 2, 1), dtype=np.float32)
        return self.tmp.run(None, {"windows": x})[0]

    def preprocess(self, rgb: np.ndarray) -> np.ndarray:
        return preprocess(rgb, self.size)


class StreamSession:
    """Per-stream state: event aggregation, 1 s feature bins and a sliding temporal window."""

    def __init__(self, models: Models, fps: float = 10.0):
        self.m = models
        self.fps = fps
        self.agg = EventAggregator()
        self.feat = StreamFeaturizer(bin_s=1.0, lag=0.5)
        self.bins: deque[np.ndarray] = deque(maxlen=models.window)
        self.n_bins = 0
        self.frame_idx = 0

    @property
    def ts(self) -> float:
        return self.frame_idx / self.fps

    def on_probs(self, probs: np.ndarray, frame: np.ndarray) -> list[Event]:
        """Feed one frame's detector output; returns events that fired on this frame."""
        events = self.agg.update(self.ts, probs, frame)
        for e in events:
            self.feat.add_event(e)
        self.frame_idx += 1
        return events

    def due_windows(self) -> list[tuple[float, float, np.ndarray]]:
        """Windows that became complete since the last call: ``(start_s, end_s, (T, F))``."""
        out = []
        for row in self.feat.advance(self.ts):
            self.bins.append(row)
            self.n_bins += 1
            if self.n_bins >= self.m.window and (self.n_bins - self.m.window) % self.m.stride == 0:
                out.append((self.n_bins - self.m.window, float(self.n_bins), np.stack(self.bins)))
        return out

    def flags(self, windows: list[tuple[float, float, np.ndarray]], scores: np.ndarray) -> list[Anomaly]:
        return [Anomaly(start=s, end=e, score=float(sc))
                for (s, e, _), sc in zip(windows, scores, strict=True) if sc > self.m.threshold]


def analyze_events(models: Models, events: list[Event], duration: float) -> list[Anomaly]:
    """Batch path: score a finished event stream (used by POST /analyze)."""
    bins = events_to_bins(events, duration)
    w, starts = make_windows(bins, models.window, models.stride)
    if len(w) == 0:
        return []
    scores = models.score_windows(w)
    return [Anomaly(start=float(s), end=float(s + models.window), score=float(sc))
            for s, sc in zip(starts, scores, strict=True) if sc > models.threshold]

