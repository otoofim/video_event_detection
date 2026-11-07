"""Frame sampling: a port of mmaction2 0.x ``UntrimmedSampleFrames``."""

from __future__ import annotations

import numpy as np


def sample_indices(total_frames: int, interval: int, clip_len: int) -> np.ndarray:
    """``(steps, clip_len)`` frame indices: clips centred at ``interval // 2 + interval * k``.

    Indices past the last frame are clipped to it, as mmaction2 does.
    """
    centres = np.arange(interval // 2, total_frames, interval)
    offsets = np.arange(-(clip_len // 2), clip_len - clip_len // 2)
    return np.clip(centres[:, None] + offsets[None, :], 0, max(total_frames - 1, 0))


def step_times(n_steps: int, interval: int, fps: float) -> np.ndarray:
    """Time in seconds of each step's centre frame."""
    return (interval // 2 + interval * np.arange(n_steps)) / fps
