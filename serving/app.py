"""FastAPI inference service.

    uvicorn serving.app:app --port 8000          # MODEL_DIR=models by default

POST /detect_features  clip features (.npy, shape (T, D)) -> events
POST /analyze          clip features -> events + play phase per window (full pipeline)
GET  /meta, /health

The detector sees a whole clip at once (its convolutions use context on both sides), so the
service works on clips, not frame by frame; concurrent /analyze calls share a micro-batcher for
the fixed-length temporal windows.
"""

from __future__ import annotations

import asyncio
import io
import os
import time
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile

from serving.batching import MicroBatcher
from serving.pipeline import Models


def create_app(
    model_dir: str | None = None, max_batch: int = 16, max_wait_ms: float = 3.0
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.models = Models(model_dir or os.environ.get("MODEL_DIR", "models"))
        app.state.tmp_batcher = MicroBatcher(app.state.models.phase_probs, max_batch, max_wait_ms)
        await app.state.tmp_batcher.start()
        yield
        await app.state.tmp_batcher.stop()

    app = FastAPI(title="video_event_detection", lifespan=lifespan)

    async def _read_features(file: UploadFile, duration: float) -> np.ndarray:
        m = app.state.models
        try:
            feats = np.load(io.BytesIO(await file.read()), allow_pickle=False)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"not a .npy array: {exc}") from exc
        if feats.ndim != 2 or feats.shape[1] != m.in_dim or len(feats) == 0 or duration <= 0:
            raise HTTPException(
                status_code=400, detail=f"expected (T>0, {m.in_dim}) features and duration>0"
            )
        return feats

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/meta")
    def meta():
        return app.state.models.meta

    @app.post("/detect_features")
    async def detect_features(file: UploadFile = File(...), duration: float = 1.0):
        feats = await _read_features(file, duration)
        t0 = time.perf_counter()
        events, _ = await asyncio.get_running_loop().run_in_executor(
            None, app.state.models.detect, feats, duration
        )
        return {
            "events": [e.model_dump() for e in events],
            "latency_ms": (time.perf_counter() - t0) * 1000,
        }

    @app.post("/analyze")
    async def analyze(file: UploadFile = File(...), duration: float = 1.0):
        feats = await _read_features(file, duration)
        m = app.state.models
        t0 = time.perf_counter()
        loop = asyncio.get_running_loop()
        events, probs = await loop.run_in_executor(None, m.detect, feats, duration)
        windows, starts = m.windows(probs, duration)
        phases = []
        if len(windows):
            scores = np.stack(
                await asyncio.gather(*(app.state.tmp_batcher.submit(w) for w in windows))
            )
            phases = m.phases(scores, starts)
        return {
            "events": [e.model_dump() for e in events],
            "phases": phases,
            "latency_ms": (time.perf_counter() - t0) * 1000,
        }

    return app


app = create_app()
