"""Clip features -> detector (ONNX) -> events, optionally with sampled uncertainty.

Needs only onnxruntime + numpy (no torch), so the slim service image stays small. The video path
(``detect_video``) additionally needs the ``preprocess`` package and its extractor.
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


def _session(path: Path, device: str) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1  # many small calls: avoid oversubscription across workers
    return ort.InferenceSession(str(path), opts, providers=onnx_providers(device))


def _softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def step_uncertainty(mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Per-step uncertainty: the sampled std of the predicted (highest mean) class."""
    return std[np.arange(len(mean)), mean.argmax(axis=1)]


class Models:
    """Shared, read-only detector session (+ an optional video feature extractor)."""

    def __init__(
        self, model_dir: str | Path, extractor=None, seed: int = 0, device: str | None = None
    ):
        device = device or os.environ.get("DEVICE", "auto")
        d = Path(model_dir)
        self.meta = json.loads((d / "meta.json").read_text())
        m = self.meta["detector"]
        self.det = _session(d / "detector.onnx", device)
        self.in_dim, self.classes = m["in_dim"], m["classes"]
        self.variational, self.factor, self.latent = m["variational"], m["factor"], m["latent"]
        self.extractor = extractor
        self._rng = np.random.default_rng(seed)

    def step_probs(self, feats: np.ndarray, samples: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """``(T, D)`` features -> mean and std of ``(T, C)`` per-step probabilities.

        ``samples == 1`` (or a deterministic model) uses ``z = mu`` and returns a zero std.
        """
        x = feats[None].astype(np.float32)
        feed = {"features": x}
        if not self.variational:
            p = _softmax(self.det.run(None, feed)[0][0])
            return p, np.zeros_like(p)
        tz = -(-len(feats) // self.factor)
        if samples <= 1:
            noise = np.zeros((1, tz, self.latent), np.float32)
        else:
            x = np.repeat(x, samples, axis=0)
            noise = self._rng.standard_normal((samples, tz, self.latent)).astype(np.float32)
        p = _softmax(self.det.run(None, {"features": x, "noise": noise})[0])
        return p.mean(0), (p.std(0) if samples > 1 else np.zeros_like(p[0]))

    def detect(
        self,
        feats: np.ndarray,
        duration: float | None = None,
        samples: int = 1,
        times: np.ndarray | None = None,
    ) -> tuple[list[Event], np.ndarray, np.ndarray | None]:
        """Events, mean step probabilities and (when ``samples > 1``) ``(T, C)`` per-step std."""
        mean, std = self.step_probs(feats, samples)
        if times is None:
            times = step_times(len(feats), duration)
        cfg = self.meta["detector"]["aggregator"]
        unc = std if samples > 1 and self.variational else None
        return events_from_probs(mean, times, **cfg, std=unc), mean, unc

    def detect_video(
        self, path: str | Path, chunking: str = "scene", chunk_s: float = 30.0, samples: int = 1
    ) -> dict:
        """Preprocess a video, detect per chunk; event times are offset to the full video."""
        import time

        if self.extractor is None:
            raise RuntimeError("no feature extractor configured")
        t0 = time.perf_counter()
        chunks = self.extractor.extract_video(path, chunking, chunk_s)
        t1 = time.perf_counter()
        events, info = [], []
        for c in chunks:
            info.append(
                {"start": c["start"], "duration": c["duration"], "steps": len(c["features"])}
            )
            if not len(c["features"]):
                continue
            ev, _, _ = self.detect(c["features"], samples=samples, times=c["times"])
            for e in ev:
                e.timestamp = round(e.timestamp + c["start"], 3)
            events.extend(ev)
        t2 = time.perf_counter()
        return {
            "events": events,
            "chunks": info,
            "latency_ms": {"preprocess_ms": (t1 - t0) * 1000, "model_ms": (t2 - t1) * 1000},
        }


def load_extractor():
    """The video extractor configured by environment variables, or ``None`` if unavailable.

    ``EXTRACTOR_BACKEND`` (``mmaction2`` | ``torch``), ``RGB_CHECKPOINT``, ``FLOW_CHECKPOINT``
    (paths or URLs; default the dataset's checkpoints), ``TRUST_CHECKPOINT=1`` to allow full
    unpickling. ``DEVICE`` (``auto`` | ``cpu`` | ``cuda``) applies to the models and the extractor.
    Leave ``EXTRACTOR_BACKEND`` unset for a features-only service.
    """
    backend = os.environ.get("EXTRACTOR_BACKEND")
    if not backend:
        return None
    from preprocess.config import CHECKPOINTS
    from preprocess.extract import Extractor

    return Extractor.from_checkpoints(
        rgb=os.environ.get("RGB_CHECKPOINT", CHECKPOINTS["rgb"]),
        flow=os.environ.get("FLOW_CHECKPOINT", CHECKPOINTS["flow"]),
        backend=backend,
        trust=os.environ.get("TRUST_CHECKPOINT") == "1",
        device=os.environ.get("DEVICE", "auto"),
    )
