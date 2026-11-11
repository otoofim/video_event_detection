"""Clip features cached as one fp16 file per split and served from device memory.

python -m detection.gpu_data --splits train val      # one-time build into <root>/cache
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.nn.utils.rnn import pad_sequence

from common.lolv2t import (
    CLASSES,
    available_clips,
    load_annotations,
    load_clip,
    step_labels,
    step_times,
    weak_events,
)
from detection.model import IGNORE

NC = len(CLASSES)


def cache_paths(root: Path, split: str, use_flow: bool) -> tuple[Path, Path]:
    tag = "rgbflow" if use_flow else "rgb"
    return root / "cache" / f"{split}_{tag}.f16", root / "cache" / f"{split}_{tag}.json"


def build_cache(root: Path, split: str, use_flow: bool, workers: int = 8) -> None:
    ann = load_annotations(root, split)
    ids = available_clips(root, split, ann, use_flow)
    bin_path, meta_path = cache_paths(root, split, use_flow)
    bin_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = bin_path.with_suffix(".tmp")
    lens, dim = [], 0
    with ThreadPoolExecutor(workers) as ex, open(tmp, "wb") as f:
        for i, x in enumerate(ex.map(lambda c: load_clip(root, split, c, use_flow), ids)):
            if not np.isfinite(x).all() or np.abs(x).max() > 6e4:
                raise ValueError(f"clip {ids[i]} does not fit in float16")
            f.write(x.astype(np.float16).tobytes())
            lens.append(len(x))
            dim = x.shape[1]
            if i % 500 == 0:
                print(f"{split}: {i}/{len(ids)} clips", flush=True)
    tmp.rename(bin_path)
    meta_path.write_text(json.dumps({"ids": ids, "lens": lens, "dim": dim}))
    print(f"{split}: cached {len(ids)} clips -> {bin_path}", flush=True)


class GpuSplit:
    """A split read from its cache. ``resident=True`` keeps all features on the device."""

    def __init__(self, root: Path, split: str, cfg: dict, device, resident: bool = False):
        bin_path, meta_path = cache_paths(root, split, cfg["use_flow"])
        if not meta_path.exists():
            build_cache(root, split, cfg["use_flow"])
        meta = json.loads(meta_path.read_text())
        self.ids, self.dim = meta["ids"], meta["dim"]
        self.lens = np.array(meta["lens"])
        self.offsets = np.concatenate(([0], np.cumsum(self.lens)))
        self.mm = np.memmap(bin_path, dtype=np.float16, mode="r").reshape(-1, self.dim)
        self.device = device
        ann = load_annotations(root, split)
        offset = cfg.get("label_offset_s", 0.0)
        self.y, self.times, self.gt = [], [], []
        for cid, n in zip(self.ids, self.lens, strict=True):
            a = ann[cid]
            ev = weak_events(a)
            y = step_labels(ev, int(n), a["duration"], cfg["label_span_s"], offset)
            self.y.append(torch.from_numpy(y).to(device))
            self.times.append(step_times(int(n), a["duration"]))
            g: dict[int, list[float]] = {k: [] for k in range(1, NC)}
            for e in ev:
                g[CLASSES.index(e.event_type)].append(round(e.timestamp + offset, 3))
            self.gt.append({k: sorted(v) for k, v in g.items()})
        self.x = None
        if resident:
            self.x = torch.empty((len(self.mm), self.dim), dtype=torch.float16, device=device)
            step = 100_000
            for s in range(0, len(self.mm), step):
                chunk = np.ascontiguousarray(self.mm[s : s + step])
                self.x[s : s + len(chunk)] = torch.from_numpy(chunk).to(device)

    def __len__(self) -> int:
        return len(self.ids)

    def clip(self, i: int) -> torch.Tensor:
        a, b = int(self.offsets[i]), int(self.offsets[i + 1])
        if self.x is not None:
            return self.x[a:b]
        return torch.from_numpy(np.array(self.mm[a:b])).to(self.device)

    def batch(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        """Zero-padded ``(N, D, T)`` float32 features and ``(N, T)`` labels (``IGNORE`` on padding)."""
        x = pad_sequence([self.clip(int(i)) for i in idx], batch_first=True)
        y = pad_sequence([self.y[int(i)] for i in idx], batch_first=True, padding_value=IGNORE)
        return x.transpose(1, 2).float(), y


class EvalSet:
    """Clips of equal length grouped without padding, in the layout ``weight_search.evaluate`` reads."""

    def __init__(self, split: GpuSplit, idx):
        idx = [int(i) for i in idx]
        by_len: dict[int, list[int]] = {}
        for i in idx:
            by_len.setdefault(int(split.lens[i]), []).append(i)
        self.groups = []
        for members in by_len.values():
            x = torch.stack([split.clip(i) for i in members]).transpose(1, 2).contiguous()
            y = torch.stack([split.y[i] for i in members])
            self.groups.append(([split.ids[i] for i in members], x, y))
        self.times = {split.ids[i]: split.times[i] for i in idx}
        self.gt = {split.ids[i]: split.gt[i] for i in idx}
        self.n = len(idx)

    def with_labels(self, split: GpuSplit) -> EvalSet:
        """Same clips and device features, labels and event times taken from ``split``.

        Use it to score models trained with different ``label_offset_s`` on one copy of the features.
        """
        index = {cid: i for i, cid in enumerate(split.ids)}
        other = object.__new__(EvalSet)
        other.groups = [
            (ids, x, torch.stack([split.y[index[c]] for c in ids])) for ids, x, _ in self.groups
        ]
        other.times = {c: split.times[index[c]] for c in self.times}
        other.gt = {c: split.gt[index[c]] for c in self.times}
        other.n = self.n
        return other


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--splits", nargs="+", default=["train", "val"], choices=["train", "val", "test"]
    )
    ap.add_argument("--config", type=Path, default=Path("detection/configs/default.yaml"))
    a = ap.parse_args()
    cfg = yaml.safe_load(a.config.read_text())
    for split in a.splits:
        build_cache(Path(cfg["root"]), split, cfg["use_flow"])


if __name__ == "__main__":
    main()
