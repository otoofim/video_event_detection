import os
import pickle

import numpy as np
import pytest

from common.lolv2t import (
    LOL_CLASSES,
    _resample,
    load_features,
    step_labels,
    step_times,
    weak_events,
)
from common.schema import Event


def test_weak_events_from_commentary():
    clip = {
        "duration": 20.0,
        "timestamps": [[0.0, 5.0], [5.0, 12.0], [12.0, 20.0]],
        "sentences": [
            "welcome to the rift",
            "he gets the first blood on the top lane",
            "baron is up and the team fight breaks out",
        ],
    }
    ev = [(e.timestamp, e.event_type) for e in weak_events(clip)]
    assert (5.0, "kill") in ev and (12.0, "objective") in ev and (12.0, "teamfight") in ev
    assert not any(t == 0.0 for t, _ in ev)


def test_step_labels_kill_wins_overlap_and_background_default():
    evs = [
        Event(timestamp=2.0, event_type="teamfight", confidence=1),
        Event(timestamp=2.0, event_type="kill", confidence=1),
    ]
    y = step_labels(evs, 10, 10.0, span_s=2.0)  # steps are 1 s wide, centres at 0.5, 1.5, ...
    assert y[0] == 0 and y[2] == LOL_CLASSES.index("kill") and y[3] == LOL_CLASSES.index("kill")
    assert y[5] == 0
    assert step_times(4, 8.0).tolist() == [1.0, 3.0, 5.0, 7.0]


def test_load_features_reads_numpy_pickle(tmp_path):
    arr = np.arange(40, dtype=np.float32).reshape(1, 5, 8)
    p = tmp_path / "a_rgb.pkl"
    p.write_bytes(pickle.dumps(arr, protocol=3))
    out = load_features(p)
    assert out.shape == (5, 8) and np.array_equal(out, arr[0])


def test_load_features_blocks_code_execution(tmp_path):
    class Evil:
        def __reduce__(self):
            return (os.system, ("echo pwned",))

    p = tmp_path / "evil.pkl"
    p.write_bytes(pickle.dumps(Evil()))
    with pytest.raises(pickle.UnpicklingError):
        load_features(p)


def test_resample_length():
    assert _resample(np.arange(10).reshape(10, 1), 4).shape == (4, 1)


def test_feature_net_shapes_and_padding_collate():
    torch = pytest.importorskip("torch")
    from detection.seq_model import FeatureEventNet
    from detection.train_lolv2t import IGNORE, collate

    net = FeatureEventNet(16, hidden=8)
    assert net(torch.randn(2, 16, 30)).shape == (2, len(LOL_CLASSES), 30)
    x, y = collate(
        [
            (torch.zeros(5, 16), torch.zeros(5, dtype=torch.long)),
            (torch.zeros(3, 16), torch.zeros(3, dtype=torch.long)),
        ]
    )
    assert x.shape == (2, 16, 5) and y[1, 4].item() == IGNORE
