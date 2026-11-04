import numpy as np
import pytest

from common.schema import Event
from detection.events import EventAggregator
from detection.metrics import confusion_matrix, event_prf, per_class_prf


def onehot(i, n=4):
    p = np.full(n, 0.01)
    p[i] = 0.97
    return p


def test_run_yields_single_event_at_run_start():
    agg = EventAggregator(min_run=2)
    seq = [0, 0, 1, 1, 1, 1, 0, 0]
    out = [e for k, c in enumerate(seq) for e in agg.update(k / 10, onehot(c))]
    assert len(out) == 1
    assert out[0].event_type == "kill" and abs(out[0].timestamp - 0.2) < 1e-9


def test_single_frame_blip_is_suppressed():
    agg = EventAggregator(min_run=2)
    assert [e for k, c in enumerate([0, 2, 0, 0]) for e in agg.update(k / 10, onehot(c))] == []


def test_back_to_back_kills_with_gap_fire_twice():
    agg = EventAggregator(min_run=2)
    seq = [1, 1, 1, 0, 0, 1, 1, 1]
    assert len([e for k, c in enumerate(seq) for e in agg.update(k / 10, onehot(c))]) == 2


def test_low_confidence_is_background():
    agg = EventAggregator(min_run=1, min_conf=0.9)
    assert agg.update(0.0, np.array([0.1, 0.6, 0.2, 0.1])) == []


def test_event_prf_tolerance():
    gt = [Event(timestamp=1.0, event_type="kill", confidence=1)]
    near = [Event(timestamp=1.3, event_type="kill", confidence=1)]
    far = [Event(timestamp=2.0, event_type="kill", confidence=1)]
    assert event_prf(near, gt)["kill"]["f1"] == 1.0
    assert event_prf(far, gt)["kill"]["f1"] == 0.0


def test_confusion_and_prf():
    cm = confusion_matrix(np.array([0, 1, 1, 2]), np.array([0, 1, 0, 2]))
    assert cm.sum() == 4
    assert per_class_prf(cm)["kill"]["recall"] == 0.5


def test_event_uncertainty_is_mean_std_of_its_class_over_the_run():
    agg = EventAggregator(min_run=2)
    std = np.array([0.0, 0.2, 0.0, 0.0])
    assert agg.update(0.0, onehot(1), std) == []
    out = agg.update(0.1, onehot(1), std * 2)
    assert len(out) == 1 and out[0].uncertainty == pytest.approx(0.3)
    assert EventAggregator(min_run=1).update(0.0, onehot(1))[0].uncertainty is None
