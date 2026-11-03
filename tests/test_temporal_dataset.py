import numpy as np

from common.schema import Anomaly, Event
from temporal.dataset import StreamFeaturizer, events_to_bins, make_windows, window_labels


def ev(t, kind):
    return Event(timestamp=t, event_type=kind, confidence=1.0)


EVENTS = [ev(1.2, "round_start"), ev(3.5, "kill"), ev(3.9, "kill"), ev(8.0, "round_end")]


def test_bins_count_events_and_round_state():
    bins = events_to_bins(EVENTS, 10.0)
    assert bins.shape == (10, 5)
    assert bins[3, 0] == 2 and bins[1, 1] == 1 and bins[8, 2] == 1
    assert bins[2, 3] == 1 and bins[9, 3] == 0


def test_streaming_matches_offline_with_lag():
    stream = StreamFeaturizer(lag=0.5)
    rows = []
    for e in EVENTS:
        stream.add_event(e)
    for t in np.arange(0, 11, 0.1):
        rows += stream.advance(float(t))
    assert np.allclose(np.stack(rows), events_to_bins(EVENTS, 10.0)[: len(rows)])


def test_late_event_lands_in_open_bin():
    f = StreamFeaturizer(lag=0.5)
    assert f.advance(1.2) == []  # bin 0 stays open until 1.5 s
    f.add_event(ev(0.9, "kill"))
    assert f.advance(1.6)[0][0] == 1


def test_windows_and_labels():
    bins = np.zeros((40, 5), dtype=np.float32)
    w, starts = make_windows(bins, 32, 4)
    assert w.shape == (3, 32, 5) and starts.tolist() == [0, 4, 8]
    y = window_labels(starts, 32, [Anomaly(start=34, end=38, score=1)])
    assert y.tolist() == [0, 1, 1]


def test_short_stream_has_no_windows():
    w, _ = make_windows(np.zeros((10, 5), dtype=np.float32), 32, 4)
    assert len(w) == 0
