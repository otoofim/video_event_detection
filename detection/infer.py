"""Run the detector over clip features and emit the event stream as JSON.

python -m detection.infer --split test --clip-id 001_uMmV3EgD76U-039 --weights models/detector.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common.device import onnx_providers, torch_device
from common.lolv2t import load_annotations, load_clip, step_times
from common.schema import Event
from detection.events import events_from_probs


class Detector:
    """Per-step classifier over either a PyTorch checkpoint (.pt) or an ONNX file (.onnx)."""

    def __init__(self, weights: str | Path, device: str = "auto"):
        self.weights = str(weights)
        if self.weights.endswith(".onnx"):
            import onnxruntime as ort

            self._sess = ort.InferenceSession(self.weights, providers=onnx_providers(device))
            self._model = None
            self.use_flow = True
        else:
            import torch

            from detection.model import FeatureEventNet

            self._device = torch_device(device)
            ckpt = torch.load(self.weights, map_location=self._device)
            self._model = FeatureEventNet(
                ckpt["in_dim"], len(ckpt["classes"]), ckpt["hidden"], ckpt["dropout"]
            )
            self._model.load_state_dict(ckpt["state_dict"])
            self._model.to(self._device).eval()
            self._sess = None
            self.use_flow = ckpt["use_flow"]
            self.aggregator = ckpt["aggregator"]

    def probs(self, feats: np.ndarray) -> np.ndarray:
        """``(T, D)`` clip features -> ``(T, C)`` per-step class probabilities."""
        if self._model is not None:
            import torch

            with torch.inference_mode():
                x = torch.from_numpy(feats).T[None].to(self._device)
                logits = self._model(x)[0].T.cpu().numpy()
        else:
            logits = self._sess.run(None, {"features": feats[None].astype(np.float32)})[0][0]
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def detect(self, feats: np.ndarray, duration: float) -> tuple[list[Event], np.ndarray]:
        """Events and per-step probabilities for one clip."""
        p = self.probs(feats)
        agg = getattr(self, "aggregator", {"min_run": 2, "min_conf": 0.5})
        return events_from_probs(p, step_times(len(p), duration), **agg), p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data/lolv2t"))
    ap.add_argument("--split", choices=["train", "val", "test"], default="test")
    ap.add_argument("--clip-id", required=True)
    ap.add_argument("--weights", type=Path, default=Path("models/detector.pt"))
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    a = ap.parse_args()
    det = Detector(a.weights, a.device)
    ann = load_annotations(a.root, a.split)[a.clip_id]
    feats = load_clip(a.root, a.split, a.clip_id, det.use_flow)
    events, _ = det.detect(feats, ann["duration"])
    print(json.dumps([e.model_dump() for e in events], indent=1))


if __name__ == "__main__":
    main()
