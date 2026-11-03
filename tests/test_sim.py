import numpy as np

from common.schema import CLASS_TO_IDX
from common.sim import FPS, frame_labels, render_match, simulate_match
from common.video import iter_frames, video_info


def test_simulation_is_deterministic():
    a, b = simulate_match(3, anomalous=True), simulate_match(3, anomalous=True)
    assert [e.model_dump() for e in a.events] == [e.model_dump() for e in b.events]


def test_normal_match_has_no_anomalies_and_balanced_rounds():
    m = simulate_match(1, anomalous=False)
    assert m.anomalies == []
    types = [e.event_type for e in m.events]
    assert types.count("round_start") == types.count("round_end") == 4


def test_anomalous_match_has_anomalies():
    assert simulate_match(2, anomalous=True).anomalies


def test_frame_labels_mark_banner_frames():
    m = simulate_match(0)
    labels = frame_labels(m.events, int(m.duration * FPS))
    first = m.events[0]
    assert labels[int(round(first.timestamp * FPS))] == CLASS_TO_IDX[first.event_type]
    assert labels[0] == 0


def test_render_roundtrip(tmp_path):
    m = simulate_match(0, n_rounds=1)
    ann = render_match(m, tmp_path / "m.mp4", seed=0)
    info = video_info(tmp_path / "m.mp4")
    assert (info["width"], info["height"]) == (ann.width, ann.height)
    frames = [f for _, _, f in iter_frames(tmp_path / "m.mp4")]
    assert len(frames) > 0 and frames[0].dtype == np.uint8
