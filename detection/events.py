"""Turn per-step class probabilities into discrete events."""

from __future__ import annotations

import numpy as np

from common.schema import CLASSES, Event


class EventAggregator:
    """Emit one event per run of identical step predictions.

    An event fires once a run of the same non-background class reaches ``min_run`` steps, and is
    stamped with the first step of the run, so a span of 10 steps yields one event. Given a
    per-step ``std`` (from sampling the latent), the event's ``uncertainty`` is the mean std of its
    class over the run so far.
    """

    def __init__(self, min_run: int = 2, min_conf: float = 0.5, classes: list[str] | None = None):
        self.min_run, self.min_conf = min_run, min_conf
        self.classes = classes or CLASSES
        self._cls = 0
        self._start = 0.0
        self._len = 0
        self._conf = 0.0
        self._unc = 0.0
        self._fired = False

    def update(self, ts: float, probs: np.ndarray, std: np.ndarray | None = None) -> list[Event]:
        cls = int(np.argmax(probs))
        conf = float(probs[cls])
        if conf < self.min_conf:
            cls = 0
        if cls != self._cls:
            self._cls, self._start, self._len, self._conf, self._fired = cls, ts, 0, 0.0, False
            self._unc = 0.0
        self._len += 1
        self._conf += conf
        if std is not None:
            self._unc += float(std[cls])
        if cls != 0 and not self._fired and self._len >= self.min_run:
            self._fired = True
            return [
                Event(
                    timestamp=round(self._start, 3),
                    event_type=self.classes[cls],
                    confidence=min(self._conf / self._len, 1.0),
                    uncertainty=None if std is None else self._unc / self._len,
                )
            ]
        return []


def events_from_probs(
    probs: np.ndarray,
    times: np.ndarray,
    min_run: int = 2,
    min_conf: float = 0.5,
    std: np.ndarray | None = None,
) -> list[Event]:
    """Run an aggregator over a whole ``(T, C)`` probability sequence (``std``: same shape)."""
    agg = EventAggregator(min_run, min_conf)
    stds = std if std is not None else [None] * len(probs)
    return [
        e for t, p, s in zip(times, probs, stds, strict=True) for e in agg.update(float(t), p, s)
    ]
