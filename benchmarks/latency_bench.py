"""Latency / throughput benchmarks. The model and the preprocessing are measured separately.

    # 1. model only: PyTorch vs ONNX Runtime, clip lengths x z samples (random features)
    python -m benchmarks.latency_bench model --torch models/detector.pt --onnx models/detector.onnx

    # 2. preprocessing only: video -> features, per stage, on a synthetic video
    python -m benchmarks.latency_bench preprocess --seconds 10

    # 3. service level: concurrent HTTP load against a running server (features endpoint)
    uvicorn serving.app:app --port 8000 &
    python -m benchmarks.latency_bench http --url http://localhost:8000 --concurrency 1 4 16

Writes ``benchmarks/results/<mode>.json`` and ``<mode>.md`` (p50/p95/p99, clips/s).

Preprocessing (TVL1 flow plus two ResNet-50 passes per step) costs seconds per clip on CPU and
dominates ``/detect_video``; low-latency numbers apply to the model on precomputed features only.
The preprocessing benchmark uses untrained weights: the compute is identical to the real networks.
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

    in_dim = torch.load(a.torch, map_location="cpu", weights_only=True)["in_dim"]
    rng = np.random.default_rng(0)
    backends = {"pytorch": Detector(a.torch), "onnxruntime": Detector(a.onnx)}
    rows = []
    for steps in a.steps:
        x = rng.standard_normal((steps, in_dim)).astype(np.float32)
        for name, det in backends.items():
            for samples in a.samples if det.variational else [1]:

                def run(det=det, samples=samples, x=x):
                    return det.probs(x) if samples == 1 else det.sample(x, samples)

                for _ in range(a.warmup):
                    run()
                ts = []
                for _ in range(a.iters):
                    t0 = time.perf_counter()
                    run()
                    ts.append((time.perf_counter() - t0) * 1000)
                p = pct(ts)
                rows.append(
                    {
                        "backend": name,
                        "steps": steps,
                        "samples": samples,
                        **p,
                        "clips_per_s": 1000 / p["mean"],
                    }
                )
    _write("model", rows, ["backend", "steps", "samples", "p50", "p95", "p99", "clips_per_s"])


def synthetic_video(path: Path, seconds: float, fps: float, size: tuple[int, int]) -> Path:
    """A moving-pattern video (the content does not change the cost of flow or ResNet-50)."""
    import cv2

    w, h = size
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    yy, xx = np.mgrid[0:h, 0:w]
    for i in range(int(seconds * fps)):
        f = ((np.sin((xx + 3 * i) / 17.0) + np.cos((yy - 2 * i) / 23.0)) * 60 + 128).astype(
            np.uint8
        )
        out.write(np.stack([f, np.roll(f, i, 1), f[::-1]], axis=-1))
    out.release()
    return path


def bench_preprocess(a: argparse.Namespace) -> None:
    import tempfile

    import preprocess.extract as ex_mod
    from preprocess.extract import Extractor
    from preprocess.sampling import sample_indices

    ex = Extractor.random_weights()
    c = ex.cfg
    with tempfile.TemporaryDirectory() as d:
        video = synthetic_video(Path(d) / "bench.mp4", a.seconds, a.fps, (1280, 720))
        t0 = time.perf_counter()
        frames = np.stack(list(ex_mod.iter_frames(video, c.width, c.height)))
        t_decode = time.perf_counter() - t0
    n = len(frames)
    t0 = time.perf_counter()
    rgb_in = ex._rgb_inputs(frames, sample_indices(n, c.frame_interval, c.rgb_clip_len))
    t1 = time.perf_counter()
    flow_in = ex._flow_inputs(frames, sample_indices(n, c.frame_interval, c.flow_clip_len))
    t2 = time.perf_counter()
    ex._run(ex.rgb_net, rgb_in)
    t3 = time.perf_counter()
    ex._run(ex.flow_net, flow_in)
    t4 = time.perf_counter()
    stages = {
        "decode + resize": t_decode,
        "rgb inputs": t1 - t0,
        "tvl1 flow + inputs": t2 - t1,
        "rgb resnet50": t3 - t2,
        "flow resnet50": t4 - t3,
    }
    total = sum(stages.values())
    rows = [
        {"stage": k, "seconds": v, "ms_per_step": 1000 * v / len(rgb_in), "share": v / total}
        for k, v in stages.items()
    ]
    rows.append(
        {
            "stage": "total",
            "seconds": total,
            "ms_per_step": 1000 * total / len(rgb_in),
            "share": 1.0,
        }
    )
    print(
        f"{a.seconds:.0f} s of {a.fps:.0f} fps video, {len(rgb_in)} steps; "
        f"{total / a.seconds:.2f} s of compute per s of video"
    )
    _write("preprocess", rows, ["stage", "seconds", "ms_per_step", "share"])


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
    m.add_argument("--samples", type=int, nargs="+", default=[1, 16], help="z samples")
    pp = sub.add_parser("preprocess")
    pp.add_argument("--seconds", type=float, default=10.0, help="synthetic video length")
    pp.add_argument("--fps", type=float, default=60.0)
    h = sub.add_parser("http")
    h.add_argument("--url", default="http://localhost:8000")
    h.add_argument("--endpoint", default="/detect_features", choices=["/detect_features"])
    h.add_argument("--steps", type=int, nargs="+", default=[110], help="clip length in steps")
    h.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 16])
    h.add_argument("--requests", type=int, default=200)
    a = ap.parse_args()
    {"model": bench_model, "preprocess": bench_preprocess, "http": bench_http}[a.mode](a)


if __name__ == "__main__":
    main()
