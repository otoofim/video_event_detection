"""Download LoL-V2T annotations and (optionally) feature archives into ``data/lolv2t``.

    python -m data.scripts.download_lolv2t                        # annotations only (3 MB)
    python -m data.scripts.download_lolv2t --parts annotations val   # + validation features (1.4 GB)
    python -m data.scripts.download_lolv2t --parts annotations val test train   # everything (~16 GB)

Dataset: https://tsunehiko.github.io/lol-v2t/ (Tanaka & Simo-Serra, CVPR Workshops 2021). The page
states no licence and says raw video is withheld under Riot Games' policy: use it for research,
cite the paper, and do not redistribute the files.
"""

from __future__ import annotations

import argparse
import tarfile
import urllib.request
from pathlib import Path

BASE = "https://esslab.jp/~tanaka/dataset/lol-v2t"
PARTS = {
    "annotations": f"{BASE}/annotations/annotations.tar.gz",
    "val": f"{BASE}/val/valid.tar.gz",
    "test": f"{BASE}/test/test.tar.gz",
    "train": f"{BASE}/train/train.tar.gz",
}


def fetch(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as r, open(dest, "wb") as f:
        total, done = int(r.headers.get("Content-Length", 0)), 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r{dest.name}: {done / 1e6:.0f}/{total / 1e6:.0f} MB", end="", flush=True)
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="+", choices=list(PARTS), default=["annotations"])
    ap.add_argument("--out", type=Path, default=Path("data/lolv2t"))
    ap.add_argument("--keep-archives", action="store_true")
    a = ap.parse_args()
    for part in a.parts:
        archive = a.out / f"{part}.tar.gz"
        fetch(PARTS[part], archive)
        with tarfile.open(archive) as tar:
            tar.extractall(a.out, filter="data")  # "data" filter rejects paths outside a.out
        if not a.keep_archives:
            archive.unlink()
        print(f"extracted {part} -> {a.out}")


if __name__ == "__main__":
    main()
