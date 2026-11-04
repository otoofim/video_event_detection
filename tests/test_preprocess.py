"""Preprocessing: frame sampling, flow encoding, checkpoint mapping and a tiny fixture video."""

import numpy as np
import pytest

from preprocess.config import ExtractConfig
from preprocess.flow import flow_input, quantize_flow
from preprocess.sampling import sample_indices, step_times


def test_sampling_reproduces_109_steps_for_a_23_6_s_clip_at_60_fps():
    idx = sample_indices(1416, interval=13, clip_len=1)
    assert idx.shape == (109, 1)
    assert idx[0, 0] == 6 and idx[1, 0] == 19  # centres at 13 // 2 + 13 k
    assert len(sample_indices(1180, 13, 1)) == 91  # the paper's 0.26 s = 13 frames at 50 fps


def test_flow_clip_is_centred_and_clipped_to_the_last_frame():
    idx = sample_indices(20, interval=13, clip_len=5)
    assert idx.tolist() == [[4, 5, 6, 7, 8], [17, 18, 19, 19, 19]]


def test_step_times_follow_frame_rate():
    assert step_times(3, interval=13, fps=60.0).tolist() == pytest.approx([0.1, 19 / 60, 32 / 60])


def test_quantize_flow_matches_denseflow_bound():
    q = quantize_flow(np.array([[[-30.0, -20.0], [0.0, 20.0]]]), bound=20.0)
    assert q.dtype == np.uint8 and q.tolist() == [[[0, 0], [128, 255]]]


def test_flow_input_interleaves_x_and_y_and_normalises():
    cfg = ExtractConfig()
    q = np.zeros((2, 4, 4, 2), dtype=np.uint8)  # L=2 flow frames
    q[0, ..., 0], q[0, ..., 1], q[1, ..., 0], q[1, ..., 1] = 0, 64, 128, 255
    x = flow_input(q, cfg)
    assert x.shape == (4, 4, 4)
    assert x[:, 0, 0].tolist() == pytest.approx([-1.0, -0.5, 0.0, 127 / 128])


def test_mmaction_checkpoint_keys_map_onto_torchvision_resnet():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from preprocess.networks import build_resnet50, mmaction_to_torchvision

    ref = build_resnet50(in_channels=10)
    keys = {
        "backbone.conv1.conv.weight": "conv1.weight",
        "backbone.conv1.bn.running_mean": "bn1.running_mean",
        "backbone.layer1.0.conv2.bn.weight": "layer1.0.bn2.weight",
        "backbone.layer1.0.conv3.conv.weight": "layer1.0.conv3.weight",
        "backbone.layer2.0.downsample.conv.weight": "layer2.0.downsample.0.weight",
        "backbone.layer2.0.downsample.bn.bias": "layer2.0.downsample.1.bias",
    }
    mapped = mmaction_to_torchvision({k: torch.zeros(1) for k in [*keys, "cls_head.fc_cls.weight"]})
    assert mapped.keys() == set(keys.values())
    assert set(keys.values()) <= set(ref.state_dict())
    assert ref.conv1.in_channels == 10


@pytest.fixture(scope="module")
def tiny_video(tmp_path_factory):
    cv2 = pytest.importorskip("cv2")
    path = tmp_path_factory.mktemp("video") / "tiny.mp4"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 60.0, (64, 48))
    for i in range(40):  # a square moving right: flow is non-zero
        f = np.zeros((48, 64, 3), dtype=np.uint8)
        f[16:32, i : i + 16] = 255
        w.write(f)
    w.release()
    return path


def test_read_frames_resizes_and_reports_fps(tiny_video):
    from preprocess.video import read_frames

    frames, fps = read_frames(tiny_video, width=32, height=24)
    assert frames.shape == (40, 24, 32, 3) and frames.dtype == np.uint8
    assert fps == pytest.approx(60.0)


def test_extract_fixture_video_shapes_and_times(tiny_video):
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from preprocess.extract import Extractor

    cfg = ExtractConfig(width=32, height=24, crop=16)
    ex = Extractor.random_weights(cfg)
    out = ex.extract_video(tiny_video, chunking="none")
    assert len(out) == 1
    c = out[0]
    assert c["features"].shape == (3, 4096)  # ceil((40 - 6) / 13) steps, RGB 2048 + flow 2048
    assert c["features"].dtype == np.float32 and np.isfinite(c["features"]).all()
    assert c["times"] == pytest.approx([6 / 60, 19 / 60, 32 / 60])
    assert c["start"] == 0.0 and c["duration"] == pytest.approx(40 / 60)


def test_fixed_chunks_offset_their_start_times(tiny_video):
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from preprocess.extract import Extractor

    ex = Extractor.random_weights(ExtractConfig(width=32, height=24, crop=16))
    out = ex.extract_video(tiny_video, chunking="fixed", chunk_s=20 / 60)
    assert [c["start"] for c in out] == pytest.approx([0.0, 20 / 60])
    assert all(c["features"].shape == (2, 4096) for c in out)  # 20 frames -> 2 steps each


def test_fixed_chunk_boundaries():
    from preprocess.chunking import fixed_chunks

    assert fixed_chunks(100, fps=10.0, chunk_s=4.0) == [(0, 40), (40, 80), (80, 100)]
    assert fixed_chunks(100, fps=10.0, chunk_s=4.0, min_frames=30) == [(0, 40), (40, 100)]


def test_scene_chunks_split_at_a_hard_cut(tmp_path):
    cv2 = pytest.importorskip("cv2")
    pytest.importorskip("scenedetect")
    from preprocess.chunking import scene_chunks

    path = tmp_path / "cut.mp4"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 48))
    for i in range(60):  # 30 frames of one static scene, then a different one
        w.write(np.full((48, 64, 3), 40 if i < 30 else 220, dtype=np.uint8))
    w.release()
    assert scene_chunks(path, threshold=15, min_scene_len=10) == [(0, 30), (30, 60)]
    assert scene_chunks(path, threshold=15, min_scene_len=100) == [(0, 60)]


def _to_mmaction_name(k: str) -> str:
    import re

    k = re.sub(r"^conv1\.", "conv1.conv.", k)
    k = re.sub(r"^bn1\.", "conv1.bn.", k)
    k = re.sub(r"^(layer\d+\.\d+)\.conv(\d)\.", r"\1.conv\2.conv.", k)
    k = re.sub(r"^(layer\d+\.\d+)\.bn(\d)\.", r"\1.conv\2.bn.", k)
    k = re.sub(r"^(layer\d+\.\d+)\.downsample\.0\.", r"\1.downsample.conv.", k)
    k = re.sub(r"^(layer\d+\.\d+)\.downsample\.1\.", r"\1.downsample.bn.", k)
    return "backbone." + k


def test_torch_backend_loads_an_mmaction_checkpoint_strictly(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from preprocess.networks import build_resnet50, load_network

    torch.manual_seed(0)
    ref = build_resnet50(in_channels=10).eval()
    sd = {_to_mmaction_name(k): v for k, v in ref.state_dict().items()}
    sd["cls_head.fc_cls.weight"] = torch.zeros(200, 2048)
    torch.save({"state_dict": sd, "meta": {"epoch": 150}}, tmp_path / "flow.pth")
    net = load_network(str(tmp_path / "flow.pth"), in_channels=10, backend="torch")
    x = torch.randn(2, 10, 64, 64)
    with torch.inference_mode():
        assert torch.allclose(net(x), ref(x)) and net(x).shape == (2, 2048)
