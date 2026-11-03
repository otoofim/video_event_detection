"""Event stream -> fixed-length windows of per-second features.

numpy only (no torch) so the serving image can use it. ``StreamFeaturizer`` is the single source
of truth: training builds windows with it offline, the live pipeline feeds it event by event.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from common.schema import Annotation, Anomaly, Event

FEATURES = ["kills", "round_start", "round_end", "in_round", "since_last_kill"]
N_FEATURES = len(FEATURES)
SINCE_CAP_S = 20.0


class StreamFeaturizer:
    """Accumulates events into 1-second bins and emits a feature row when a bin closes.

    ``lag`` keeps a bin open slightly past its end so late-arriving events (detection fires a
    couple of frames after a banner appears) still land in the right bin.
    """

    def __init__(self, bin_s: float = 1.0, lag: float = 0.0):
        self.bin_s, self.lag = bin_s, lag
        self._cur = 0
        self._pending: list[Event] = []
        self._in_round = False
        self._anchor = 0.0  # time of last kill or round start

    def add_event(self, event: Event) -> None:
        self._pending.append(event)

    def advance(self, now: float) -> list[np.ndarray]:
        rows = []
        while (self._cur + 1) * self.bin_s <= now - self.lag + 1e-9:
            rows.append(self._close())
        return rows

    def _close(self) -> np.ndarray:
        end = (self._cur + 1) * self.bin_s
        kills = rs = re_ = 0
        rest = []
        for e in sorted(self._pending, key=lambda e: e.timestamp):
            if e.timestamp >= end:
                rest.append(e)
            elif e.event_type == "kill":
                kills += 1
                self._anchor = e.timestamp
            elif e.event_type == "round_start":
                rs, self._in_round, self._anchor = 1, True, e.timestamp
            elif e.event_type == "round_end":
                re_, self._in_round = 1, False
        self._pending = rest
        since = min((end - self._anchor) / SINCE_CAP_S, 1.0) if self._in_round else 0.0
        self._cur += 1
        return np.array([kills, rs, re_, float(self._in_round), since], dtype=np.float32)


def events_to_bins(events: list[Event], duration: float, bin_s: float = 1.0) -> np.ndarray:
    """Offline featurization of a whole match -> ``(T, N_FEATURES)``."""
    f = StreamFeaturizer(bin_s)
    for e in events:
        f.add_event(e)
    rows = f.advance(duration)
    return np.stack(rows) if rows else np.zeros((0, N_FEATURES), dtype=np.float32)


def make_windows(bins: np.ndarray, window: int, stride: int) -> tuple[np.ndarray, np.ndarray]:
    """Slide a window over ``bins`` -> ``(N, window, F)`` and their start bin indices."""
    starts = np.arange(0, max(len(bins) - window, -1) + 1, stride)
    if len(starts) == 0:
        return np.zeros((0, window, bins.shape[1]), dtype=np.float32), starts
    return np.stack([bins[s:s + window] for s in starts]), starts


def window_labels(starts: np.ndarray, window: int, anomalies: list[Anomaly],
                  bin_s: float = 1.0, min_overlap_s: float = 2.0) -> np.ndarray:
    """1 where a window overlaps an annotated anomaly by at least ``min_overlap_s`` seconds."""
    y = np.zeros(len(starts), dtype=np.int64)
    for i, s in enumerate(starts):
        a, b = s * bin_s, (s + window) * bin_s
        for an in anomalies:
            if min(b, an.end) - max(a, an.start) >= min_overlap_s:
                y[i] = 1
    return y


def load_events(path: str | Path) -> list[Event]:
    return [Event(**d) for d in json.loads(Path(path).read_text())]


def load_match(i: int, annotations: Path, events_dir: Path | None) -> tuple[list[Event], Annotation]:
    """Events for match ``i``: detector output from ``events_dir``, else the ground truth."""
    ann = Annotation.model_validate_json((annotations / f"match_{i:04d}.json").read_text())
    if events_dir is None:
        return ann.events, ann
    return load_events(events_dir / f"match_{i:04d}.json"), ann
