"""Detector output -> windows of per-second class probabilities, labelled from the captions.

numpy only (no torch) so the serving image can use it.
"""

from __future__ import annotations

import numpy as np

from common.lolv2t import phase_label


def bin_probs(
    probs: np.ndarray, times: np.ndarray, duration: float, bin_s: float = 1.0
) -> np.ndarray:
    """Average ``(T, C)`` step probabilities into ``ceil(duration / bin_s)`` fixed bins -> ``(B, C)``.

    Bins with no step (clips shorter than a bin boundary) repeat background-free zeros.
    """
    n_bins = max(int(np.ceil(duration / bin_s)), 1)
    idx = np.minimum((times // bin_s).astype(int), n_bins - 1)
    sums = np.zeros((n_bins, probs.shape[1]))
    counts = np.zeros(n_bins)
    np.add.at(sums, idx, probs)
    np.add.at(counts, idx, 1.0)
    return (sums / np.maximum(counts, 1)[:, None]).astype(np.float32)


def make_windows(bins: np.ndarray, window: int, stride: int) -> tuple[np.ndarray, np.ndarray]:
    """Slide a window over ``bins`` -> ``(N, window, C)`` and their start bin indices."""
    starts = np.arange(0, len(bins) - window + 1, stride)
    if len(starts) == 0:
        return np.zeros((0, window, bins.shape[1]), dtype=np.float32), starts
    return np.stack([bins[s : s + window] for s in starts]), starts


def window_phases(
    labels: np.ndarray,
    times: np.ndarray,
    starts: np.ndarray,
    window: int,
    bin_s: float = 1.0,
) -> np.ndarray:
    """Caption-derived phase of each window from the per-step caption labels ``labels``."""
    out = np.zeros(len(starts), dtype=np.int64)
    for i, s in enumerate(starts):
        sel = (times >= s * bin_s) & (times < (s + window) * bin_s)
        out[i] = phase_label(labels[sel]) if sel.any() else 0
    return out
