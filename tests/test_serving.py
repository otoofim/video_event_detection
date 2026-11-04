import asyncio
import io

import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from serving.app import create_app  # noqa: E402
from serving.batching import MicroBatcher  # noqa: E402


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
    assert client.get("/meta").json()["temporal"]["window"] == 6


def test_detect_features(client):
    r = client.post("/detect_features?duration=12.0", files={"file": ("f.npy", npy(FEATS))})
    assert r.status_code == 200
    assert all(e["event_type"] in {"kill", "objective", "teamfight"} for e in r.json()["events"])


def test_analyze_returns_events_and_phases(client):
    r = client.post("/analyze?duration=12.0", files={"file": ("f.npy", npy(FEATS))})
    body = r.json()
    assert r.status_code == 200 and {"events", "phases", "latency_ms"} <= set(body)
    # 12 s clip, 6 s windows, stride 2 -> windows start at 0, 2, 4, 6
    assert [p["start"] for p in body["phases"]] == [0.0, 2.0, 4.0, 6.0]
    assert all(0 <= p["confidence"] <= 1 for p in body["phases"])


def test_short_clip_has_no_phases(client):
    r = client.post("/analyze?duration=3.0", files={"file": ("f.npy", npy(FEATS[:10]))})
    assert r.status_code == 200 and r.json()["phases"] == []


@pytest.mark.parametrize("endpoint", ["/detect_features", "/analyze"])
def test_validates_input(client, endpoint):
    bad = np.zeros((10, 3), dtype=np.float32)
    assert client.post(endpoint, files={"file": ("f.npy", npy(bad))}).status_code == 400
    assert client.post(endpoint, files={"file": ("f.npy", b"junk")}).status_code == 400


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
