"""LoL-V2T (Tanaka & Simo-Serra, CVPRW 2021) loading and weak event labelling.

LoL-V2T has 9,723 League of Legends Worlds clips with 62,677 timestamped commentary sentences and
precomputed per-clip features (``<clip_id>_rgb.pkl`` / ``<clip_id>_flow.pkl``); the raw video is not
distributed. It has *no event labels*, so events are derived from the commentary by keyword rules
(``weak_events``). They are noisy: commentary lags the action and ASR text is imperfect, so every
metric computed against them measures agreement with the commentary, not with the game.

Get the data with ``python -m data.scripts.download_lolv2t``. https://tsunehiko.github.io/lol-v2t/
"""

from __future__ import annotations

import json
import pickle
import re
from pathlib import Path

import numpy as np

from common.schema import CLASSES, EVENT_TYPES, Event

# Later entries win where labelled spans overlap (kill is the most specific).
_RULES = {
    "teamfight": r"\b(team ?fights?|fight(s|ing)?|skirmish|engage[sd]?|all[- ]in|ace)\b",
    "objective": r"\b(baron|dragon|drake|herald|towers?|turrets?|inhibitors?|nexus|elder)\b",
    "kill": (
        r"\b(kill(s|ed|ing)?|slain|first blood|double kill|triple kill|quadra kill|penta ?kill|"
        r"picks? (him|her|them|off)|takes? (him|her|them) down|executed|shut ?down|dies|dead)\b"
    ),
}
RULES = {k: re.compile(v, re.IGNORECASE) for k, v in _RULES.items()}

SPLIT_FILES = {"train": "training.json", "val": "validation.json", "test": "testing.json"}
SPLIT_DIRS = {
    "train": ["training", "train"],
    "val": ["validation", "valid", "val"],
    "test": ["testing", "test"],
}


class _SafeUnpickler(pickle.Unpickler):
    """Only numpy array reconstruction is allowed: the feature files come from a third party."""

    _ALLOWED = {
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy", "ndarray"),
        ("numpy", "dtype"),
    }

    def find_class(self, module: str, name: str):
        if (module, name) not in self._ALLOWED:
            raise pickle.UnpicklingError(f"blocked global in feature file: {module}.{name}")
        return super().find_class(module, name)


def load_features(path: str | Path) -> np.ndarray:
    """Load one ``*_rgb.pkl`` / ``*_flow.pkl`` as a float32 ``(T, D)`` array (no code execution)."""
    with open(path, "rb") as f:
        arr = _SafeUnpickler(f).load()
    arr = np.asarray(arr)
    if arr.dtype.hasobject or arr.ndim not in (2, 3) or (arr.ndim == 3 and arr.shape[0] != 1):
        raise ValueError(f"unexpected feature array {arr.dtype} {arr.shape} in {path}")
    return arr.reshape(-1, arr.shape[-1]).astype(np.float32, copy=False)


def _resample(x: np.ndarray, t: int) -> np.ndarray:
    """Nearest-neighbour resample of a ``(T0, D)`` sequence to ``t`` steps."""
    return x[np.linspace(0, len(x) - 1, t).round().astype(int)]


def split_dir(root: Path, split: str) -> Path:
    for name in SPLIT_DIRS[split]:
        if (root / name).is_dir():
            return root / name
    raise FileNotFoundError(
        f"no feature directory for split '{split}' under {root} (tried {SPLIT_DIRS[split]})"
    )


def load_clip(root: Path, split: str, clip_id: str, use_flow: bool = True) -> np.ndarray:
    """RGB (+ flow, resampled to the RGB length) features of a clip -> ``(T, D)``."""
    d = split_dir(root, split)
    rgb = load_features(d / f"{clip_id}_rgb.pkl")
    if not use_flow:
        return rgb
    return np.concatenate(
        [rgb, _resample(load_features(d / f"{clip_id}_flow.pkl"), len(rgb))], axis=1
    )


def load_annotations(root: Path, split: str) -> dict[str, dict]:
    """``{clip_id: {"duration", "timestamps": [[s, e]], "sentences": [str]}}``."""
    return json.loads((root / "annotations" / SPLIT_FILES[split]).read_text())


def available_clips(
    root: Path, split: str, ann: dict[str, dict], use_flow: bool = True
) -> list[str]:
    """Annotated clips whose feature files are on disk (so partial downloads still work)."""
    d = split_dir(root, split)
    return sorted(
        c
        for c in ann
        if (d / f"{c}_rgb.pkl").exists() and (not use_flow or (d / f"{c}_flow.pkl").exists())
    )


def weak_events(clip: dict) -> list[Event]:
    """Keyword events from a clip's commentary, stamped at the start of the sentence's segment."""
    events = []
    for (start, _), text in zip(clip["timestamps"], clip["sentences"], strict=True):
        for etype, pat in RULES.items():
            if pat.search(text):
                events.append(
                    Event(timestamp=round(float(start), 3), event_type=etype, confidence=1.0)
                )
    return sorted(events, key=lambda e: e.timestamp)


def step_times(n_steps: int, duration: float) -> np.ndarray:
    """Centre time of each feature step, assuming steps are uniform over the clip."""
    return (np.arange(n_steps) + 0.5) * duration / n_steps


def step_labels(
    events: list[Event],
    n_steps: int,
    duration: float,
    span_s: float = 3.0,
    offset_s: float = 0.0,
) -> np.ndarray:
    """Per-step class index: an event labels the steps in ``[t + offset_s, t + offset_s + span_s)``."""
    t = step_times(n_steps, duration)
    labels = np.zeros(n_steps, dtype=np.int64)
    # Apply lowest priority first so the earlier type in EVENT_TYPES (kill) wins overlaps.
    for e in sorted(events, key=lambda e: -EVENT_TYPES.index(e.event_type)):
        start = e.timestamp + offset_s
        labels[(t >= start) & (t < start + span_s)] = CLASSES.index(e.event_type)
    return labels


def phase_label(step_labels_in_window: np.ndarray) -> int:
    """Play phase of a window: the highest-priority event class present, else background (0)."""
    present = set(np.unique(step_labels_in_window).tolist()) - {0}
    return min(present) if present else 0  # class index order == priority order
