import asyncio

import cv2
import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from common.sim import render_match, simulate_match  # noqa: E402
from common.video import iter_frames  # noqa: E402
from serving.app import create_app  # noqa: E402
from serving.batching import MicroBatcher  # noqa: E402


@pytest.fixture(scope="module")
def client(model_dir):
    with TestClient(create_app(str(model_dir))) as c:
        yield c


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    p = tmp_path_factory.mktemp("clip") / "m.mp4"
    render_match(simulate_match(0, n_rounds=1), p, seed=0)
    return p


def jpeg(frame):
    return cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))[1].tobytes()


def test_health_and_meta(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/meta").json()["temporal"]["window"] == 32


def test_detect_returns_distribution(client, clip):
    frame = next(iter_frames(clip))[2]
    r = client.post("/detect", files={"file": ("f.jpg", jpeg(frame), "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    assert set(body["probs"]) == {"background", "kill", "round_start", "round_end"}
    assert abs(sum(body["probs"].values()) - 1) < 1e-4


def test_detect_rejects_garbage(client):
    r = client.post("/detect", files={"file": ("f.jpg", b"not an image", "image/jpeg")})
    assert r.status_code == 400


def test_analyze_runs(client):
    ev = [{"timestamp": 1.0, "event_type": "round_start", "confidence": 1.0}]
    r = client.post("/analyze", json={"events": ev, "duration": 60.0})
    assert r.status_code == 200 and "anomalies" in r.json()


def test_stream_websocket_replies_per_frame(client, clip):
    frames = [f for _, _, f in iter_frames(clip)][:25]
    with client.websocket_connect("/stream?fps=10") as ws:
        for i, f in enumerate(frames):
            ws.send_bytes(jpeg(f))
            msg = ws.receive_json()
            assert msg["frame"] == i and {"events", "anomalies", "latency_ms"} <= set(msg)


def test_microbatcher_coalesces_concurrent_requests():
    async def run():
        b = MicroBatcher(lambda x: x * 2, max_batch=8, max_wait_ms=20)
        await b.start()
        outs = await asyncio.gather(*(b.submit(np.full(3, i, dtype=np.float32)) for i in range(8)))
        await b.stop()
        return outs, b.batch_sizes

    outs, sizes = asyncio.run(run())
    assert [o[0] for o in outs] == [2 * i for i in range(8)]
    assert max(sizes) > 1


def _npy(arr):
    import io

    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


def test_detect_features(client):
    feats = np.random.default_rng(0).standard_normal((40, 16)).astype(np.float32)
    r = client.post("/detect_features?duration=12.0", files={"file": ("f.npy", _npy(feats))})
    assert r.status_code == 200
    assert all(e["event_type"] in {"kill", "objective", "teamfight"} for e in r.json()["events"])


def test_detect_features_validates_shape(client):
    bad = np.zeros((10, 3), dtype=np.float32)
    assert client.post("/detect_features", files={"file": ("f.npy", _npy(bad))}).status_code == 400
    assert client.post("/detect_features", files={"file": ("f.npy", b"junk")}).status_code == 400
