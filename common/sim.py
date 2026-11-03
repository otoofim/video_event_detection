"""Synthetic arena-game footage with ground-truth events.

Real esports VODs are licence-encumbered and large, so the repo ships a seeded generator that
renders a top-down arena match with a HUD: a kill-feed banner (top right), a centre banner for
round start / round end, and distractor HUD elements. Matches are written as mp4 plus a JSON
annotation. A fraction of matches get injected irregular play (kill bursts and kill droughts),
which is what the temporal model has to find.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from common.schema import Annotation, Anomaly, Event

W, H, FPS = 160, 96, 10
BANNER_SECONDS = {"kill": 0.6, "round_start": 1.5, "round_end": 1.5}
TEAM_COLORS = {"red": (200, 40, 40), "blue": (40, 90, 215)}
BANNER_COLORS = {"round_start": (40, 180, 80), "round_end": (235, 170, 30)}
KILL_BOX = (96, 4, 156, 16)  # x1, y1, x2, y2 (before jitter)
CENTER_BOX = (30, 38, 130, 56)


@dataclass
class Match:
    events: list[Event]
    anomalies: list[Anomaly]
    duration: float


def simulate_match(seed: int, anomalous: bool = False, n_rounds: int = 4) -> Match:
    rng = np.random.default_rng(seed)
    events: list[Event] = []
    anomalies: list[Anomaly] = []
    bad_rounds = set()
    if anomalous:
        bad_rounds = set(rng.choice(n_rounds, size=int(rng.integers(1, 3)), replace=False).tolist())
    t = 2.0
    for r in range(n_rounds):
        events.append(Event(timestamp=round(t, 2), event_type="round_start", confidence=1.0))
        t += BANNER_SECONDS["round_start"] + 0.5
        if r in bad_rounds and rng.random() < 0.5:
            kind, length = "drought", float(rng.uniform(45, 60))
            anomalies.append(
                Anomaly(start=round(t, 2), end=round(t + length, 2), score=1.0, kind=kind)
            )
        elif r in bad_rounds:
            length = float(rng.uniform(22, 30))
            n = int(rng.integers(6, 9))
            b0 = t + float(rng.uniform(3, 8))
            times = [b0 + i * float(rng.uniform(0.8, 1.1)) for i in range(n)]
            anomalies.append(
                Anomaly(
                    start=round(times[0], 2), end=round(times[-1] + 1, 2), score=1.0, kind="burst"
                )
            )
            for ts in times:
                events.append(_kill(ts, rng))
        else:
            length = float(rng.uniform(22, 32))
            n = int(rng.integers(4, 8))
            times, cur = [], t + float(rng.uniform(2, 4))
            for _ in range(n):
                cur += float(rng.uniform(2.2, 5.0))
                if cur < t + length - 1:
                    times.append(cur)
            for ts in times:
                events.append(_kill(ts, rng))
        t += length
        events.append(Event(timestamp=round(t, 2), event_type="round_end", confidence=1.0))
        t += BANNER_SECONDS["round_end"] + 2.5
    events.sort(key=lambda e: e.timestamp)
    return Match(events=events, anomalies=anomalies, duration=round(t, 2))


def _kill(ts: float, rng: np.random.Generator) -> Event:
    team = "red" if rng.random() < 0.5 else "blue"
    return Event(timestamp=round(ts, 2), event_type="kill", confidence=1.0, team=team)


def frame_labels(events: list[Event], n_frames: int, fps: float = FPS) -> np.ndarray:
    """Per-frame class index (0 = background) derived from event onsets and banner durations."""
    from common.schema import CLASS_TO_IDX

    labels = np.zeros(n_frames, dtype=np.int64)
    for e in events:
        a = int(round(e.timestamp * fps))
        b = a + int(round(BANNER_SECONDS[e.event_type] * fps))
        labels[a : min(b, n_frames)] = CLASS_TO_IDX[e.event_type]
    return labels


class _Renderer:
    def __init__(self, match: Match, seed: int):
        self.rng = np.random.default_rng(seed + 10_000)
        self.n_frames = int(match.duration * FPS) + FPS
        self.match = match
        n_players = 10
        steps = self.rng.normal(0, 1.2, size=(self.n_frames, n_players, 2))
        pos = np.cumsum(steps, axis=0) + self.rng.uniform([10, 25], [150, 90], size=(n_players, 2))
        # reflect into the arena
        self.pos = np.abs(((pos - [0, 20]) % (2 * np.array([W, H - 20]))) - np.array([W, H - 20]))
        self.pos = np.abs(self.pos - np.array([W, H - 20])) + [0, 20]
        self.teams = [TEAM_COLORS["red"]] * 5 + [TEAM_COLORS["blue"]] * 5
        self.active = {}
        for e in match.events:
            a = int(round(e.timestamp * FPS))
            b = a + int(round(BANNER_SECONDS[e.event_type] * FPS))
            for f in range(a, b):
                self.active[f] = e
        self.score = [0, 0]

    def frame(self, i: int) -> np.ndarray:
        rng = self.rng
        img = np.full((H, W, 3), 38, dtype=np.uint8)
        for x in range(0, W, 16):
            cv2.line(img, (x, 20), (x, H), (46, 46, 46), 1)
        for y in range(20, H, 16):
            cv2.line(img, (0, y), (W, y), (46, 46, 46), 1)
        for p, c in zip(self.pos[i], self.teams, strict=True):
            cv2.circle(img, (int(p[0]), int(p[1])), 3, c, -1)
        # distractor HUD: scoreboard (top-left) and chat bars (bottom-left), team-coloured on purpose
        cv2.rectangle(img, (4, 3), (40, 14), (70, 70, 70), -1)
        cv2.rectangle(img, (6, 5), (6 + int(rng.integers(4, 12)), 12), TEAM_COLORS["red"], -1)
        cv2.rectangle(img, (26, 5), (26 + int(rng.integers(4, 12)), 12), TEAM_COLORS["blue"], -1)
        for k in range(3):
            c = (
                list(TEAM_COLORS.values())[int(rng.integers(0, 2))]
                if rng.random() < 0.5
                else (90, 90, 90)
            )
            cv2.rectangle(img, (3, 70 + k * 7), (3 + int(rng.integers(12, 30)), 74 + k * 7), c, -1)
        e = self.active.get(i)
        if e is not None:
            jx, jy = (int(v) for v in rng.integers(-1, 2, size=2))
            if e.event_type == "kill":
                x1, y1, x2, y2 = KILL_BOX
                cv2.rectangle(
                    img, (x1 + jx, y1 + jy), (x2 + jx, y2 + jy), TEAM_COLORS[e.team or "red"], -1
                )
                cv2.rectangle(
                    img,
                    (x1 + 3 + jx, y1 + 3 + jy),
                    (x1 + 20 + jx, y2 - 3 + jy),
                    (240, 240, 240),
                    -1,
                )
                cv2.rectangle(
                    img,
                    (x2 - 20 + jx, y1 + 3 + jy),
                    (x2 - 4 + jx, y2 - 3 + jy),
                    (240, 240, 240),
                    -1,
                )
            else:
                x1, y1, x2, y2 = CENTER_BOX
                cv2.rectangle(
                    img, (x1 + jx, y1 + jy), (x2 + jx, y2 + jy), BANNER_COLORS[e.event_type], -1
                )
                n = 3 if e.event_type == "round_start" else 5
                for k in range(n):
                    cx = x1 + 10 + k * (80 // n)
                    cv2.rectangle(
                        img, (cx + jx, y1 + 5 + jy), (cx + 6 + jx, y2 - 5 + jy), (245, 245, 245), -1
                    )
        img = img.astype(np.float32) * float(rng.uniform(0.9, 1.1)) + rng.normal(0, 4, img.shape)
        return np.clip(img, 0, 255).astype(np.uint8)


def render_match(match: Match, out_path: Path, seed: int) -> Annotation:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    r = _Renderer(match, seed)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    if not writer.isOpened():
        raise RuntimeError("cv2.VideoWriter could not open mp4v output")
    for i in range(r.n_frames):
        writer.write(cv2.cvtColor(r.frame(i), cv2.COLOR_RGB2BGR))
    writer.release()
    return Annotation(
        video=out_path.name,
        fps=FPS,
        duration=r.n_frames / FPS,
        width=W,
        height=H,
        events=match.events,
        anomalies=match.anomalies,
    )
