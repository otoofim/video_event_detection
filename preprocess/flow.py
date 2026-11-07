"""TVL1 optical flow, stored and normalised the way denseflow + mmaction2 0.x did."""

from __future__ import annotations

import numpy as np

from preprocess.config import ExtractConfig


def tvl1(prev_rgb: np.ndarray, next_rgb: np.ndarray) -> np.ndarray:
    """Dual TVL1 flow between two RGB frames -> ``(H, W, 2)`` float32 (x, y) in pixels."""
    import cv2

    if not hasattr(cv2, "optflow"):
        raise ImportError("TVL1 needs opencv-contrib-python(-headless) for cv2.optflow")
    a = cv2.cvtColor(prev_rgb, cv2.COLOR_RGB2GRAY)
    b = cv2.cvtColor(next_rgb, cv2.COLOR_RGB2GRAY)
    return cv2.optflow.DualTVL1OpticalFlow_create().calc(a, b, None)


def quantize_flow(flow: np.ndarray, bound: float) -> np.ndarray:
    """denseflow's 8-bit encoding: ``[-bound, bound]`` px -> ``[0, 255]``, clipped."""
    q = np.round((np.clip(flow, -bound, bound) + bound) * 255.0 / (2 * bound))
    return q.astype(np.uint8)


def jpeg_roundtrip(img: np.ndarray, quality: int) -> np.ndarray:
    """Encode and decode as JPEG (the dataset's rawframes are JPEG files)."""
    import cv2

    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)


def flow_input(qflow: np.ndarray, cfg: ExtractConfig) -> np.ndarray:
    """``(L, H, W, 2)`` uint8 flow frames -> ``(2L, H, W)`` normalised network input.

    Channels are interleaved ``x0, y0, x1, y1, ...`` as mmaction2's ``NCHW_Flow`` format makes them.
    """
    x = (qflow.astype(np.float32) - cfg.flow_mean) / cfg.flow_std
    return x.transpose(0, 3, 1, 2).reshape(-1, *x.shape[1:3])
