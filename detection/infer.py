"""Run the detector over clip features and emit the event stream as JSON.

python -m detection.infer --split test --clip-id 001_uMmV3EgD76U-039 --weights models/detector.pt
python -m detection.infer --features features.npy --duration 23.6 --samples 16
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common.lolv2t import load_annotations, load_clip, step_times
from common.schema import Event
from detection.events import events_from_probs

_DEFAULT_AGG = {"min_run": 2, "min_conf": 0.5}


def _softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


class Detector:
    """Per-step classifier over a PyTorch checkpoint (.pt) or an exported ONNX file (.onnx)."""

    def __init__(self, weights: str | Path, seed: int = 0):
        self.weights = Path(weights)
        self._rng = np.random.default_rng(seed)
        if self.weights.suffix == ".onnx":
            import onnxruntime as ort

            self._sess = ort.InferenceSession(str(self.weights), providers=["CPUExecutionProvider"])
            self._model = None
            meta = json.loads((self.weights.parent / "meta.json").read_text())["detector"]
            self.variational, self.factor, self.latent = (
                meta["variational"],
                meta["factor"],
                meta["latent"],
            )
            self.use_flow, self.aggregator = meta["use_flow"], meta["aggregator"]
        else:
            from serving.onnx_export import load_detector

            self._model, ckpt = load_detector(self.weights)
            self._sess = None
            self.variational = bool(getattr(self._model, "variational", False))
            self.factor = getattr(self._model, "factor", 1)
            self.latent = getattr(self._model, "latent", 0)
            self.use_flow, self.aggregator = ckpt["use_flow"], ckpt.get("aggregator", _DEFAULT_AGG)

    def _logits(self, feats: np.ndarray, noise: np.ndarray | None) -> np.ndarray:
        """``(N, T, D)`` features (+ ``(N, Tz, Z)`` noise) -> ``(N, T, C)`` logits."""
        x = feats.astype(np.float32)
        if self._model is not None:
            import torch

            from detection.model import StepMajor

            with torch.inference_mode():
                n = None if noise is None else torch.from_numpy(noise)
                return StepMajor(self._model)(torch.from_numpy(x), n).numpy()
        feed = {"features": x}
        if self.variational:
            feed["noise"] = noise
        return self._sess.run(None, feed)[0]

    def _noise(self, n: int, t: int, random: bool) -> np.ndarray | None:
        if not self.variational:
            return None
        shape = (n, -(-t // self.factor), self.latent)
        return (self._rng.standard_normal(shape) if random else np.zeros(shape)).astype(np.float32)

    def probs(self, feats: np.ndarray) -> np.ndarray:
        """``(T, D)`` clip features -> ``(T, C)`` per-step probabilities (``z = mu``)."""
        return _softmax(self._logits(feats[None], self._noise(1, len(feats), False))[0])

    def sample(self, feats: np.ndarray, n_samples: int = 16) -> tuple[np.ndarray, np.ndarray]:
        """Mean and std of per-step probabilities over ``n_samples`` draws of ``z``."""
        if not self.variational:
            p = self.probs(feats)
            return p, np.zeros_like(p)
        x = np.repeat(feats[None], n_samples, axis=0)
        p = _softmax(self._logits(x, self._noise(n_samples, len(feats), True)))
        return p.mean(0), p.std(0)

    def detect(
        self, feats: np.ndarray, duration: float, n_samples: int = 1
    ) -> tuple[list[Event], np.ndarray]:
        """Events and per-step probabilities for one clip (with uncertainty if ``n_samples > 1``)."""
        times = step_times(len(feats), duration)
        if n_samples > 1 and self.variational:
            p, std = self.sample(feats, n_samples)
            return events_from_probs(p, times, **self.aggregator, std=std), p
        p = self.probs(feats)
        return events_from_probs(p, times, **self.aggregator), p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data/lolv2t"))
    ap.add_argument("--split", choices=["train", "val", "test"], default="test")
    ap.add_argument("--clip-id", help="a LoL-V2T clip (reads features and duration from --root)")
    ap.add_argument("--features", type=Path, help="a (T, D) .npy, e.g. from preprocess.extract")
    ap.add_argument("--duration", type=float, help="clip duration in seconds, with --features")
    ap.add_argument("--weights", type=Path, default=Path("models/detector.pt"))
    ap.add_argument("--samples", type=int, default=1, help="z samples; >1 adds uncertainty")
    a = ap.parse_args()
    det = Detector(a.weights)
    if a.features is not None:
        if a.duration is None:
            ap.error("--features needs --duration")
        feats, duration = np.load(a.features, allow_pickle=False), a.duration
    elif a.clip_id:
        duration = load_annotations(a.root, a.split)[a.clip_id]["duration"]
        feats = load_clip(a.root, a.split, a.clip_id, det.use_flow)
    else:
        ap.error("give --clip-id or --features")
    events, _ = det.detect(feats, duration, a.samples)
    print(json.dumps([e.model_dump() for e in events], indent=1))


if __name__ == "__main__":
    main()
