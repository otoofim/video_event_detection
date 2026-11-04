"""Latency / throughput benchmarks (random features, so no dataset download is needed).

    # 1. model level: PyTorch vs ONNX Runtime for the detector, clip lengths x batch sizes
    python -m benchmarks.latency_bench model --torch models/detector.pt --onnx models/detector.onnx

    # 2. service level: concurrent HTTP load against a running server
    uvicorn serving.app:app --port 8000 &
    python -m benchmarks.latency_bench http --url http://localhost:8000 --concurrency 1 4 16

Writes ``benchmarks/results/<mode>.json`` and ``<mode>.md`` (p50/p95/p99, clips/s).
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import platform
import time
from pathlib import Path

import numpy as np

RESULTS = Path("benchmarks/results")


def pct(samples_ms: list[float]) -> dict[str, float]:
    a = np.asarray(samples_ms)
    return {
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "mean": float(a.mean()),
    }


def _write(name: str, rows: list[dict], cols: list[str]) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{name}.json").write_text(
        json.dumps(
            {
                "host": platform.processor() or platform.machine(),
                "python": platform.python_version(),
                "rows": rows,
            },
            indent=1,
        )
    )
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        cells = (f"{r[c]:.2f}" if isinstance(r[c], float) else str(r[c]) for c in cols)
        lines.append("| " + " | ".join(cells) + " |")
    (RESULTS / f"{name}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def bench_model(a: argparse.Namespace) -> None:
    import torch

    from detection.infer import Detector

    in_dim = torch.load(a.torch, map_location="cpu")["in_dim"]
    rng = np.random.default_rng(0)
    backends = {"pytorch": Detector(a.torch), "onnxruntime": Detector(a.onnx)}
    rows = []
    for steps in a.steps:
        x = rng.standard_normal((steps, in_dim)).astype(np.float32)
        for name, det in backends.items():
            for _ in range(a.warmup):
                det.probs(x)
            ts = []
            for _ in range(a.iters):
                t0 = time.perf_counter()
                det.probs(x)
                ts.append((time.perf_counter() - t0) * 1000)
            p = pct(ts)
            rows.append({"backend": name, "steps": steps, **p, "clips_per_s": 1000 / p["mean"]})
    _write("model", rows, ["backend", "steps", "p50", "p95", "p99", "clips_per_s"])


async def _http_load(url: str, payload: bytes, endpoint: str, concurrency: int, n: int) -> dict:
    import httpx

    lat: list[float] = []
    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(base_url=url, timeout=60) as client:

        async def one() -> None:
            async with sem:
                t0 = time.perf_counter()
                r = await client.post(f"{endpoint}?duration=20", files={"file": ("f.npy", payload)})
                r.raise_for_status()
                lat.append((time.perf_counter() - t0) * 1000)

        for _ in range(10):  # warmup
            await one()
        lat.clear()
        t0 = time.perf_counter()
        await asyncio.gather(*(one() for _ in range(n)))
        wall = time.perf_counter() - t0
    return {**pct(lat), "clips_per_s": n / wall}


def bench_http(a: argparse.Namespace) -> None:
    import httpx

    in_dim = httpx.get(f"{a.url}/meta").json()["detector"]["in_dim"]
    buf = io.BytesIO()
    np.save(buf, np.random.default_rng(0).standard_normal((a.steps[0], in_dim)).astype(np.float32))
    rows = []
    for c in a.concurrency:
        r = asyncio.run(_http_load(a.url, buf.getvalue(), a.endpoint, c, a.requests))
        rows.append({"endpoint": a.endpoint, "concurrency": c, **r})
    _write("http", rows, ["endpoint", "concurrency", "p50", "p95", "p99", "clips_per_s"])


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    m = sub.add_parser("model")
    m.add_argument("--torch", type=Path, default=Path("models/detector.pt"))
    m.add_argument("--onnx", type=Path, default=Path("models/detector.onnx"))
    m.add_argument("--steps", type=int, nargs="+", default=[50, 110, 250])
    m.add_argument("--iters", type=int, default=100)
    m.add_argument("--warmup", type=int, default=10)
    h = sub.add_parser("http")
    h.add_argument("--url", default="http://localhost:8000")
    h.add_argument("--endpoint", default="/analyze", choices=["/detect_features", "/analyze"])
    h.add_argument("--steps", type=int, nargs="+", default=[110], help="clip length in steps")
    h.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 16])
    h.add_argument("--requests", type=int, default=200)
    a = ap.parse_args()
    {"model": bench_model, "http": bench_http}[a.mode](a)


if __name__ == "__main__":
    main()
