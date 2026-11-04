import io
import json

import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from serving.app import create_app  # noqa: E402


@pytest.fixture(scope="module")
def client(model_dir):
    with TestClient(create_app(str(model_dir))) as c:
        yield c


def npy(arr):
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


FEATS = np.random.default_rng(0).standard_normal((60, 16)).astype(np.float32)


def test_health_and_meta(client):
    assert client.get("/health").json() == {"status": "ok"}
    meta = client.get("/meta").json()["detector"]
    assert meta["arch"] == "vunet" and meta["factor"] == 4 and meta["latent"] == 4
    assert client.get("/meta").json()["video_input"] is False


def test_detect_features(client):
    r = client.post("/detect_features?duration=12.0", files={"file": ("f.npy", npy(FEATS))})
    assert r.status_code == 200
    body = r.json()
    assert all(e["event_type"] in {"kill", "objective", "teamfight"} for e in body["events"])
    assert "uncertainty" not in body or body["uncertainty"] is None


def test_detect_features_with_samples_reports_uncertainty(client):
    r = client.post(
        "/detect_features?duration=12.0&samples=8", files={"file": ("f.npy", npy(FEATS))}
    )
    body = r.json()
    assert r.status_code == 200 and len(body["uncertainty"]) == 60
    assert all(u >= 0 for u in body["uncertainty"])
    assert all(e["uncertainty"] is not None for e in body["events"])


def test_analyze_endpoint_is_gone(client):
    assert client.post("/analyze", files={"file": ("f.npy", npy(FEATS))}).status_code == 404


@pytest.mark.parametrize("length", [1, 37, 41])
def test_onnx_handles_any_length(client, length):
    r = client.post("/detect_features?duration=5", files={"file": ("f.npy", npy(FEATS[:length]))})
    assert r.status_code == 200


def test_validates_input(client):
    bad = np.zeros((10, 3), dtype=np.float32)
    assert client.post("/detect_features", files={"file": ("f.npy", npy(bad))}).status_code == 400
    assert client.post("/detect_features", files={"file": ("f.npy", b"junk")}).status_code == 400
    r = client.post("/detect_features?samples=0", files={"file": ("f.npy", npy(FEATS))})
    assert r.status_code == 422


def test_detect_video_needs_an_extractor(client):
    r = client.post("/detect_video", files={"file": ("v.mp4", b"\x00" * 10)})
    assert r.status_code == 501


def _tiny_video(path, frames=60, fps=60.0):
    cv2 = pytest.importorskip("cv2")
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 48))
    for i in range(frames):
        f = np.zeros((48, 64, 3), dtype=np.uint8)
        f[16:32, i % 48 : i % 48 + 16] = 255
        w.write(f)
    w.release()
    return path.read_bytes()


def test_detect_video_offsets_events_by_chunk_start(video_model_dir, tmp_path):
    pytest.importorskip("torchvision")
    from preprocess.config import ExtractConfig
    from preprocess.extract import Extractor

    ex = Extractor.random_weights(ExtractConfig(width=32, height=24, crop=16))
    data = _tiny_video(tmp_path / "v.mp4")
    with TestClient(create_app(str(video_model_dir), extractor=ex)) as c:
        r = c.post("/detect_video?chunking=fixed&chunk_s=0.5", files={"file": ("v.mp4", data)})
        bad = c.post("/detect_video", files={"file": ("v.mp4", b"not a video")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert [ch["start"] for ch in body["chunks"]] == pytest.approx([0.0, 0.5])
    assert [ch["steps"] for ch in body["chunks"]] == [2, 2]  # 30 frames each -> 2 steps
    for e in body["events"]:
        assert 0.0 <= e["timestamp"] <= 1.0
    assert {"preprocess_ms", "model_ms"} <= set(body["latency_ms"])
    assert bad.status_code == 400
    json.dumps(body)


def test_step_uncertainty_is_std_of_the_predicted_class():
    from serving.pipeline import step_uncertainty

    mean = np.array([[0.7, 0.1, 0.1, 0.1], [0.1, 0.2, 0.6, 0.1]])
    std = np.array([[0.01, 0.3, 0.0, 0.0], [0.2, 0.0, 0.05, 0.0]])
    assert step_uncertainty(mean, std).tolist() == [0.01, 0.05]
