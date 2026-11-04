"""Video decoding (OpenCV), resized the way the dataset's rawframes were."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def read_frames(
    path: str | Path, width: int, height: int, start: int = 0, end: int | None = None
) -> tuple[np.ndarray, float]:
    """Frames ``[start, end)`` as ``(N, height, width, 3)`` uint8 RGB, plus the frame rate."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"cannot open video {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
    if fps <= 0:
        raise ValueError(f"video {path} reports no frame rate")
    if start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    i = start
    while end is None or i < end:
        ok, f = cap.read()
        if not ok:
            break
        f = cv2.resize(f, (width, height), interpolation=cv2.INTER_LINEAR)
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        i += 1
    cap.release()
    if not frames:
        return np.zeros((0, height, width, 3), dtype=np.uint8), fps
    return np.stack(frames), fps


def frame_count(path: str | Path) -> tuple[int, float]:
    """Number of frames and frame rate, counted by decoding (container counts can be wrong)."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"cannot open video {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    n = 0
    while cap.grab():
        n += 1
    cap.release()
    return n, fps
