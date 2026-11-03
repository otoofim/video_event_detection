"""Video ingestion: decoding, resizing and tensor preprocessing."""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(3, 1, 1)
STD = np.array([0.25, 0.25, 0.25], dtype=np.float32).reshape(3, 1, 1)


def video_info(path: str | Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return info


def iter_frames(
    path: str | Path, size: tuple[int, int] | None = None, stride: int = 1
) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield ``(frame_index, timestamp_seconds, rgb_uint8_frame)``; ``size`` is (w, h)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    idx = 0
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                if size is not None and (frame.shape[1], frame.shape[0]) != tuple(size):
                    frame = cv2.resize(frame, tuple(size), interpolation=cv2.INTER_AREA)
                yield idx, idx / fps, frame
            idx += 1
    finally:
        cap.release()


def decode_image(data: bytes) -> np.ndarray:
    """Decode encoded image bytes (jpeg/png) to an RGB uint8 array."""
    arr = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("could not decode image")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def preprocess(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """RGB uint8 HxWx3 -> normalized float32 3xHxW at ``size`` = (w, h)."""
    if (frame.shape[1], frame.shape[0]) != tuple(size):
        frame = cv2.resize(frame, tuple(size), interpolation=cv2.INTER_AREA)
    x = frame.astype(np.float32).transpose(2, 0, 1) / 255.0
    return (x - MEAN) / STD
