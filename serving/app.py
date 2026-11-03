"""FastAPI inference service.

    uvicorn serving.app:app --port 8000          # MODEL_DIR=models by default

POST /detect   one frame (jpeg/png) -> class probabilities + event (micro-batched across requests)
POST /analyze  finished event stream -> irregular-play windows
WS   /stream   binary jpeg frames in, JSON per frame out (events + anomalies as they occur)
"""
from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from common.schema import CLASSES, Event
from common.video import decode_image
from detection.events import make_event
from serving.batching import MicroBatcher
from serving.pipeline import Models, StreamSession, analyze_events


class AnalyzeRequest(BaseModel):
    events: list[Event]
    duration: float


def create_app(model_dir: str | None = None, max_batch: int = 16, max_wait_ms: float = 3.0) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.models = Models(model_dir or os.environ.get("MODEL_DIR", "models"))
        app.state.det_batcher = MicroBatcher(app.state.models.detect_probs, max_batch, max_wait_ms)
        app.state.tmp_batcher = MicroBatcher(app.state.models.score_windows, 8, max_wait_ms)
        await app.state.det_batcher.start()
        await app.state.tmp_batcher.start()
        yield
        await app.state.det_batcher.stop()
        await app.state.tmp_batcher.stop()

    app = FastAPI(title="esports-event-detection", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/meta")
    def meta():
        return app.state.models.meta

    @app.post("/detect")
    async def detect(file: UploadFile = File(...), timestamp: float = 0.0):
        t0 = time.perf_counter()
        try:
            rgb = decode_image(await file.read())
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        m = app.state.models
        probs = await app.state.det_batcher.submit(m.preprocess(rgb))
        cls = int(probs.argmax())
        event = None
        if cls != 0:
            event = make_event(cls, timestamp, float(probs[cls]), rgb)
        return {"probs": dict(zip(CLASSES, map(float, probs), strict=True)),
                "event": event.model_dump() if event else None,
                "latency_ms": (time.perf_counter() - t0) * 1000}

    @app.post("/analyze")
    async def analyze(req: AnalyzeRequest):
        anomalies = analyze_events(app.state.models, req.events, req.duration)
        return {"anomalies": [a.model_dump() for a in anomalies]}

    @app.websocket("/stream")
    async def stream(ws: WebSocket, fps: float = 10.0):
        await ws.accept()
        m = app.state.models
        sess = StreamSession(m, fps)
        try:
            while True:
                data = await ws.receive_bytes()
                t0 = time.perf_counter()
                rgb = decode_image(data)
                probs = await app.state.det_batcher.submit(m.preprocess(rgb))
                events = sess.on_probs(probs, rgb)
                windows = sess.due_windows()
                anomalies = []
                if windows:
                    scores = await app.state.tmp_batcher.submit(windows[-1][2])
                    anomalies = sess.flags(windows[-1:], np.atleast_1d(scores))
                await ws.send_json({
                    "frame": sess.frame_idx - 1, "timestamp": (sess.frame_idx - 1) / fps,
                    "events": [e.model_dump() for e in events],
                    "anomalies": [a.model_dump() for a in anomalies],
                    "latency_ms": (time.perf_counter() - t0) * 1000,
                })
        except WebSocketDisconnect:
            return

    return app


app = create_app()
