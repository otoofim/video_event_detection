"""Turn per-frame class probabilities into discrete events (stateful, usable on a live stream)."""

from __future__ import annotations

import numpy as np

from common.schema import CLASSES, Event
from common.sim import CENTER_BOX, KILL_BOX, H, W


class EventAggregator:
    """Emit one event per run of identical frame predictions.

    An event fires once a run of the same non-background class reaches ``min_run`` frames, and is
    stamped with the first frame of the run, so a banner that lasts 6 frames yields one event.
    """

    def __init__(
        self,
        min_run: int = 2,
        min_conf: float = 0.5,
        classes: list[str] | None = None,
        hud: bool = True,
    ):
        self.min_run, self.min_conf = min_run, min_conf
        self.classes, self.hud = classes or CLASSES, hud
        self._cls = 0
        self._start = 0.0
        self._len = 0
        self._conf = 0.0
        self._fired = False

    def update(self, ts: float, probs: np.ndarray, frame: np.ndarray | None = None) -> list[Event]:
        cls = int(np.argmax(probs))
        conf = float(probs[cls])
        if conf < self.min_conf:
            cls = 0
        if cls != self._cls:
            self._cls, self._start, self._len, self._conf, self._fired = cls, ts, 0, 0.0, False
        self._len += 1
        self._conf += conf
        if cls != 0 and not self._fired and self._len >= self.min_run:
            self._fired = True
            return [self._make(cls, frame)]
        return []

    def _make(self, cls: int, frame: np.ndarray | None) -> Event:
        return make_event(cls, self._start, self._conf / self._len, frame, self.classes, self.hud)


def make_event(
    cls: int,
    ts: float,
    conf: float,
    frame: np.ndarray | None = None,
    classes: list[str] | None = None,
    hud: bool = True,
) -> Event:
    """``hud=False`` (feature-based detectors) skips the synthetic-HUD bbox/team lookup."""
    etype = (classes or CLASSES)[cls]
    if not hud:
        return Event(timestamp=round(ts, 3), event_type=etype, confidence=min(conf, 1.0))
    box = KILL_BOX if etype == "kill" else CENTER_BOX
    team = kill_team(frame) if etype == "kill" and frame is not None else None
    return Event(
        timestamp=round(ts, 3),
        event_type=etype,
        confidence=min(conf, 1.0),
        bbox=[float(v) for v in box],
        team=team,
    )


def kill_team(frame: np.ndarray) -> str:
    """Team colour of the kill banner, read from the HUD slot (template localisation)."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = KILL_BOX
    sx, sy = w / W, h / H
    roi = (
        frame[int(y1 * sy) : int(y2 * sy), int(x1 * sx) : int(x2 * sx)].reshape(-1, 3).astype(float)
    )
    return "red" if np.median(roi[:, 0]) > np.median(roi[:, 2]) else "blue"
