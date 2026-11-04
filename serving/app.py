"""FastAPI inference service.

    uvicorn serving.app:app --port 8000          # MODEL_DIR=models by default

POST /detect_features  clip features (.npy, shape (T, D)) -> events (+ uncertainty with samples>1)
POST /detect_video     a video file -> preprocessing -> events with full-video timestamps
GET  /meta, /health

The detector sees a whole clip at once (its convolutions use context on both sides), so the
service works on clips, not frame by frame. ``/detect_video`` needs the feature extractor
(``EXTRACTOR_BACKEND``, full image only) and is dominated by preprocessing: optical flow and two
ResNet-50 passes take seconds per clip on CPU.
"""

from __future__ import annotations

import asyncio
import io
import os
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile

from serving.pipeline import Models, load_extractor, step_uncertainty

_UNSET = object()


def create_app(model_dir: str | None = None, extractor=_UNSET) -> FastAPI:
    """``extractor``: a ``preprocess.extract.Extractor``; by default read from the environment."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        ex = load_extractor() if extractor is _UNSET else extractor
        app.state.models = Models(model_dir or os.environ.get("MODEL_DIR", "models"), ex)
        yield

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
        return {**app.state.models.meta, "video_input": app.state.models.extractor is not None}

    @app.post("/detect_features")
    async def detect_features(
        file: UploadFile = File(...),
        duration: float = 1.0,
        samples: int = Query(1, ge=1, le=64, description="z samples; >1 adds uncertainty"),
    ):
        feats = await _read_features(file, duration)
        t0 = time.perf_counter()
        events, mean, std = await asyncio.get_running_loop().run_in_executor(
            None, app.state.models.detect, feats, duration, samples
        )
        unc = None if std is None else step_uncertainty(mean, std).round(4).tolist()
        return {
            "events": [e.model_dump() for e in events],
            "uncertainty": unc,
            "latency_ms": (time.perf_counter() - t0) * 1000,
        }

    @app.post("/detect_video")
    async def detect_video(
        file: UploadFile = File(...),
        chunking: str = Query("scene", pattern="^(scene|fixed|none)$"),
        chunk_s: float = Query(30.0, gt=0),
        samples: int = Query(1, ge=1, le=64),
    ):
        m = app.state.models
        if m.extractor is None:
            raise HTTPException(
                status_code=501, detail="video input needs the full image (EXTRACTOR_BACKEND)"
            )
        suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / f"upload{suffix}"
            path.write_bytes(await file.read())
            try:
                out = await asyncio.get_running_loop().run_in_executor(
                    None, m.detect_video, path, chunking, chunk_s, samples
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=f"cannot read video: {exc}") from exc
            except ImportError as exc:  # e.g. scene chunking without scenedetect
                raise HTTPException(status_code=501, detail=str(exc)) from exc
        out["events"] = [e.model_dump() for e in out["events"]]
        return out

    return app


app = create_app()
