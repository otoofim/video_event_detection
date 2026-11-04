"""Turn a video into LoL-V2T-style feature sequences (RGB + flow TSN features per step).

    python -m preprocess.extract video.mp4 --out features.npy                 # scene chunks
    python -m preprocess.extract video.mp4 --out features.npy --chunking none # one clip

Writes one ``(T, 4096)`` float32 ``.npy`` per chunk (``features.npy`` when there is one chunk,
``features-000.npy``, ... otherwise) and ``features.json`` listing each chunk's start time,
duration and step times. Feature layout matches ``common.lolv2t.load_clip``: RGB then flow.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch

from preprocess.chunking import fixed_chunks, scene_chunks
from preprocess.config import CHECKPOINTS, ExtractConfig
from preprocess.flow import flow_input, jpeg_roundtrip, quantize_flow, tvl1
from preprocess.sampling import sample_indices, step_times
from preprocess.video import frame_count


def _center_crop(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    y, x = max((h - size) // 2, 0), max((w - size) // 2, 0)
    return img[y : y + size, x : x + size]


def iter_frames(path: str | Path, width: int, height: int) -> Iterator[np.ndarray]:
    """Decode a video once, yielding resized RGB uint8 frames."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"cannot open video {path}")
    try:
        while True:
            ok, f = cap.read()
            if not ok:
                return
            f = cv2.resize(f, (width, height), interpolation=cv2.INTER_LINEAR)
            yield cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


class Extractor:
    """RGB and flow TSN networks plus the dataset's frame pipeline."""

    def __init__(
        self, cfg: ExtractConfig, rgb_net: torch.nn.Module, flow_net: torch.nn.Module
    ):
        self.cfg, self.rgb_net, self.flow_net = cfg, rgb_net.eval(), flow_net.eval()

    @classmethod
    def from_checkpoints(
        cls,
        cfg: ExtractConfig | None = None,
        rgb: str = CHECKPOINTS["rgb"],
        flow: str = CHECKPOINTS["flow"],
        backend: str = "mmaction2",
        trust: bool = False,
    ) -> Extractor:
        from preprocess.networks import load_network

        cfg = cfg or ExtractConfig()
        return cls(
            cfg,
            load_network(rgb, 3 * cfg.rgb_clip_len, backend, trust),
            load_network(flow, 2 * cfg.flow_clip_len, backend, trust),
        )

    @classmethod
    def random_weights(cls, cfg: ExtractConfig | None = None, seed: int = 0) -> Extractor:
        """Untrained networks with the right shapes: for tests and timing, not for features."""
        from preprocess.networks import build_resnet50

        cfg = cfg or ExtractConfig()
        torch.manual_seed(seed)
        return cls(cfg, build_resnet50(3 * cfg.rgb_clip_len), build_resnet50(2 * cfg.flow_clip_len))

    # -- per-clip pipeline -------------------------------------------------------------------
    def _rgb_inputs(self, frames: np.ndarray, idx: np.ndarray) -> np.ndarray:
        c = self.cfg
        mean, std = np.array(c.rgb_mean, np.float32), np.array(c.rgb_std, np.float32)
        out = []
        for row in idx:
            chans = []
            for i in row:
                f = frames[i]
                if c.jpeg_roundtrip:  # rawframes were JPEG; cv2 encodes BGR
                    f = jpeg_roundtrip(f[..., ::-1], c.jpeg_quality)[..., ::-1]
                f = (_center_crop(f, c.crop).astype(np.float32) - mean) / std
                chans.append(f.transpose(2, 0, 1))
            out.append(np.concatenate(chans))
        return np.stack(out)

    def _flow_inputs(self, frames: np.ndarray, idx: np.ndarray) -> np.ndarray:
        c = self.cfg
        last = max(len(frames) - 2, 0)  # flow frame j is frames j -> j + 1
        cache: dict[int, np.ndarray] = {}
        out = []
        for row in idx:
            stack = []
            for j in np.minimum(row, last):
                if j not in cache:
                    nxt = frames[min(j + 1, len(frames) - 1)]
                    q = quantize_flow(tvl1(frames[j], nxt), c.flow_bound)
                    if c.jpeg_roundtrip:  # denseflow writes x and y as separate gray JPEGs
                        q = np.stack(
                            [jpeg_roundtrip(q[..., k], c.jpeg_quality) for k in (0, 1)], -1
                        )
                    cache[j] = _center_crop(q, c.crop)
                stack.append(cache[j])
            out.append(flow_input(np.stack(stack), c))
        return np.stack(out)

    @torch.inference_mode()
    def _run(self, net: torch.nn.Module, x: np.ndarray) -> np.ndarray:
        bs = self.cfg.batch_size
        parts = [net(torch.from_numpy(x[i : i + bs])).numpy() for i in range(0, len(x), bs)]
        return np.concatenate(parts).astype(np.float32)

    def features(self, frames: np.ndarray, fps: float) -> dict:
        """``(N, H, W, 3)`` RGB frames of one clip -> ``{"features": (T, 4096), "times": (T,)}``.

        Both modalities use the clip's frame count, as the dataset did, so they have the same T.
        """
        c = self.cfg
        n = len(frames)
        if n == 0:
            return {"features": np.zeros((0, 4096), np.float32), "times": np.zeros(0)}
        rgb_idx = sample_indices(n, c.frame_interval, c.rgb_clip_len)
        flow_idx = sample_indices(n, c.frame_interval, c.flow_clip_len)
        rgb = self._run(self.rgb_net, self._rgb_inputs(frames, rgb_idx))
        flow = self._run(self.flow_net, self._flow_inputs(frames, flow_idx))
        return {
            "features": np.concatenate([rgb, flow], axis=1),
            "times": step_times(len(rgb_idx), c.frame_interval, fps),
        }

    # -- whole video ---------------------------------------------------------------------------
    def chunks(
        self, path: str | Path, chunking: str = "scene", chunk_s: float = 30.0
    ) -> tuple[list[tuple[int, int]], float]:
        n, fps = frame_count(path)
        if chunking == "none":
            return [(0, n)], fps
        if chunking == "fixed":
            return fixed_chunks(n, fps, chunk_s), fps
        if chunking == "scene":
            return scene_chunks(path, self.cfg.scene_threshold, self.cfg.min_scene_len), fps
        raise ValueError(f"unknown chunking {chunking!r}")

    def extract_video(
        self, path: str | Path, chunking: str = "scene", chunk_s: float = 30.0
    ) -> list[dict]:
        """Features for every chunk of a video; ``start`` is the chunk's offset in seconds."""
        bounds, fps = self.chunks(path, chunking, chunk_s)
        out: list[dict] = []
        frames = iter_frames(path, self.cfg.width, self.cfg.height)
        pos = 0
        for start, end in bounds:
            for _ in range(start - pos):  # skip frames between chunks
                next(frames, None)
            buf = [f for _, f in zip(range(end - start), frames, strict=False)]
            pos = start + len(buf)
            clip = np.stack(buf) if buf else np.zeros((0, self.cfg.height, self.cfg.width, 3))
            r = self.features(clip, fps)
            out.append(
                {
                    "start": start / fps,
                    "duration": len(buf) / fps,
                    "fps": fps,
                    "features": r["features"],
                    "times": r["times"],
                }
            )
        return out


def save(chunks: list[dict], out: Path) -> Path:
    """Write the ``.npy`` file(s) and the ``.json`` index; returns the index path."""
    out.parent.mkdir(parents=True, exist_ok=True)
    index = []
    for k, c in enumerate(chunks):
        p = out if len(chunks) == 1 else out.with_name(f"{out.stem}-{k:03d}{out.suffix}")
        np.save(p, c["features"])
        index.append(
            {
                "file": p.name,
                "start": c["start"],
                "duration": c["duration"],
                "fps": c["fps"],
                "steps": len(c["features"]),
                "times": [round(float(t), 4) for t in c["times"]],
            }
        )
    meta = out.with_suffix(".json")
    meta.write_text(
        json.dumps({"chunks": index, "note": "all scenes kept (no 'useful' filter)"}, indent=1)
    )
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("video", type=Path)
    ap.add_argument("--out", type=Path, default=Path("features.npy"))
    ap.add_argument("--chunking", choices=["scene", "fixed", "none"], default="scene")
    ap.add_argument("--chunk-s", type=float, default=30.0, help="chunk length for --chunking fixed")
    ap.add_argument("--backend", choices=["mmaction2", "torch"], default="mmaction2")
    ap.add_argument("--rgb-checkpoint", default=CHECKPOINTS["rgb"])
    ap.add_argument("--flow-checkpoint", default=CHECKPOINTS["flow"])
    ap.add_argument(
        "--trust-checkpoint", action="store_true", help="allow full unpickling of checkpoints"
    )
    a = ap.parse_args()
    ex = Extractor.from_checkpoints(
        rgb=a.rgb_checkpoint, flow=a.flow_checkpoint, backend=a.backend, trust=a.trust_checkpoint
    )
    chunks = ex.extract_video(a.video, a.chunking, a.chunk_s)
    print(f"{len(chunks)} chunk(s) -> {save(chunks, a.out)}")


if __name__ == "__main__":
    main()
