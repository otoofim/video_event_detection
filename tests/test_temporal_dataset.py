import numpy as np

from common.lolv2t import phase_label
from temporal.dataset import bin_probs, make_windows, window_phases


def test_bin_probs_averages_steps_into_seconds():
    probs = np.array([[1, 0], [0, 1], [0, 1], [0, 1]], dtype=float)
    times = np.array([0.25, 0.75, 1.25, 1.75])
    out = bin_probs(probs, times, duration=2.0)
    assert out.shape == (2, 2)
    assert out[0].tolist() == [0.5, 0.5] and out[1].tolist() == [0.0, 1.0]


def test_windows_and_short_clips():
    w, starts = make_windows(np.zeros((10, 4), dtype=np.float32), 6, 2)
    assert w.shape == (3, 6, 4) and starts.tolist() == [0, 2, 4]
    short, s = make_windows(np.zeros((3, 4), dtype=np.float32), 6, 2)
    assert len(short) == 0 and len(s) == 0


def test_window_phase_uses_caption_labels_with_priority():
    times = np.arange(10) + 0.5
    labels = np.zeros(10, dtype=np.int64)
    labels[1], labels[7] = 3, 2  # teamfight at 1.5 s, objective at 7.5 s
    starts = np.array([0, 4])
    assert window_phases(labels, times, starts, window=6).tolist() == [3, 2]
    assert phase_label(np.array([3, 3, 0])) == 3
    assert phase_label(np.array([0, 0])) == 0
    assert phase_label(np.array([1, 2, 3])) == 1  # kill beats everything
