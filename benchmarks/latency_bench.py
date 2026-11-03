"""Latency / throughput benchmarks.

    # 1. model level: PyTorch vs ONNX Runtime, batch sizes 1..32
    python -m benchmarks.latency_bench model --torch models/detector.pt --onnx models/detector.onnx

    # 2. service level: concurrent HTTP load against a running server
    uvicorn serving.app:app --port 8000 &
    python -m benchmarks.latency_bench http --url http://localhost:8000 --video data/raw/match_0000.mp4 \
        --concurrency 1 4 16 --requests 400

Writes ``benchmarks/results/<mode>.json`` and ``<mode>.md`` (p50/p95/p99, frames/s).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import platform
import time
from pathlib import Path

import numpy as np

from common.video import iter_frames, preprocess

RESULTS = Path("benchmarks/results")


def pct(samples_ms: list[float]) -> dict[str, float]:
    a = np.asarray(samples_ms)
    return {"p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95)),
            "p99": float(np.percentile(a, 99)), "mean": float(a.mean())}


def _write(name: str, rows: list[dict], cols: list[str]) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{name}.json").write_text(json.dumps(
        {"host": platform.processor() or platform.machine(), "python": platform.python_version(),
         "rows": rows}, indent=1))
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join(f"{r[c]:.2f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + " |")
    (RESULTS / f"{name}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def sample_frame(video: Path) -> np.ndarray:
    return next(iter_frames(video))[2]


def bench_model(a: argparse.Namespace) -> None:
    from detection.infer import Detector

    rng = np.random.default_rng(0)
    backends = {"pytorch": Detector(a.torch), "onnxruntime": Detector(a.onnx)}
    size = backends["pytorch"].size
    rows = []
    for bs in a.batch_sizes:
        x = rng.standard_normal((bs, 3, size[1], size[0])).astype(np.float32)
        for name, det in backends.items():
            for _ in range(a.warmup):
                det.probs(x)
            ts = []
            for _ in range(a.iters):
                t0 = time.perf_counter()
                det.probs(x)
                ts.append((time.perf_counter() - t0) * 1000)
            p = pct(ts)
            rows.append({"backend": name, "batch": bs, **p, "frames_per_s": bs / (p["mean"] / 1000)})
    _write("model", rows, ["backend", "batch", "p50", "p95", "p99", "frames_per_s"])


async def _http_load(url: str, jpeg: bytes, concurrency: int, n: int) -> dict:
    import httpx

    lat: list[float] = []
    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(base_url=url, timeout=30) as client:
        async def one() -> None:
            async with sem:
                t0 = time.perf_counter()
                r = await client.post("/detect", files={"file": ("f.jpg", jpeg, "image/jpeg")})
                r.raise_for_status()
                lat.append((time.perf_counter() - t0) * 1000)

        for _ in range(20):  # warmup
            await one()
        lat.clear()
        t0 = time.perf_counter()
        await asyncio.gather(*(one() for _ in range(n)))
        wall = time.perf_counter() - t0
    return {**pct(lat), "frames_per_s": n / wall}


def bench_http(a: argparse.Namespace) -> None:
    import cv2

    frame = sample_frame(a.video)
    jpeg = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))[1].tobytes()
    rows = []
    for c in a.concurrency:
        r = asyncio.run(_http_load(a.url, jpeg, c, a.requests))
        rows.append({"concurrency": c, **r})
    _write("http", rows, ["concurrency", "p50", "p95", "p99", "frames_per_s"])


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    m = sub.add_parser("model")
    m.add_argument("--torch", type=Path, default=Path("models/detector.pt"))
    m.add_argument("--onnx", type=Path, default=Path("models/detector.onnx"))
    m.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 16, 32])
    m.add_argument("--iters", type=int, default=200)
    m.add_argument("--warmup", type=int, default=20)
    h = sub.add_parser("http")
    h.add_argument("--url", default="http://localhost:8000")
    h.add_argument("--video", type=Path, default=Path("data/raw/match_0000.mp4"))
    h.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 16])
    h.add_argument("--requests", type=int, default=400)
    a = ap.parse_args()
    {"model": bench_model, "http": bench_http}[a.mode](a)


if __name__ == "__main__":
    main()
