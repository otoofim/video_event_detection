"""End-to-end pipeline: clip features -> detector -> events -> temporal model -> play phases.

Only needs onnxruntime + numpy (no torch), so the service image stays small.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort

from common.device import onnx_providers
from common.lolv2t import step_times
from common.schema import Event
from detection.events import events_from_probs
from temporal.dataset import bin_probs, make_windows


def _session(path: Path, device: str) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1  # many small calls: avoid oversubscription across workers
    return ort.InferenceSession(str(path), opts, providers=onnx_providers(device))


def _softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


class Models:
    """Shared, read-only model sessions."""

    def __init__(self, model_dir: str | Path, device: str | None = None):
        device = device or os.environ.get("DEVICE", "auto")
        d = Path(model_dir)
        self.meta = json.loads((d / "meta.json").read_text())
        self.det = _session(d / "detector.onnx", device)
        self.tmp = _session(d / "temporal.onnx", device)
        self.in_dim = self.meta["detector"]["in_dim"]
        self.classes = self.meta["detector"]["classes"]
        self.window = self.meta["temporal"]["window"]
        self.stride = self.meta["temporal"]["stride"]
        self.bin_s = self.meta["temporal"]["bin_s"]

    def step_probs(self, feats: np.ndarray) -> np.ndarray:
        """``(T, D)`` clip features -> ``(T, C)`` per-step class probabilities."""
        logits = self.det.run(None, {"features": feats[None].astype(np.float32)})[0][0]
        return _softmax(logits)

    def phase_probs(self, windows: np.ndarray) -> np.ndarray:
        """``(N, window, C)`` windows -> ``(N, n_phases)`` phase probabilities."""
        x = np.ascontiguousarray(windows, dtype=np.float32)
        return _softmax(self.tmp.run(None, {"windows": x})[0])

    def detect(self, feats: np.ndarray, duration: float) -> tuple[list[Event], np.ndarray]:
        probs = self.step_probs(feats)
        cfg = self.meta["detector"]["aggregator"]
        return events_from_probs(probs, step_times(len(feats), duration), **cfg), probs

    def windows(self, probs: np.ndarray, duration: float) -> tuple[np.ndarray, np.ndarray]:
        """Per-second windows of step probabilities for the temporal model."""
        bins = bin_probs(probs, step_times(len(probs), duration), duration, self.bin_s)
        return make_windows(bins, self.window, self.stride)

    def phases(self, scores: np.ndarray, starts: np.ndarray) -> list[dict]:
        """Format temporal-model output for the API."""
        return [
            {
                "start": float(s * self.bin_s),
                "end": float((s + self.window) * self.bin_s),
                "phase": self.classes[int(p.argmax())],
                "confidence": float(p.max()),
            }
            for s, p in zip(starts, scores, strict=True)
        ]
