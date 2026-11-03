"""Generate synthetic matches: ``data/raw/match_XXXX.mp4`` + ``data/annotations/match_XXXX.json``.

Usage: python -m data.scripts.generate_synthetic --n 40 --anomalous-frac 0.3
Matches 0..n-1 are deterministic in their index, so any subset is reproducible.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from common.sim import render_match, simulate_match


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--anomalous-frac", type=float, default=0.3)
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--annotations", type=Path, default=Path("data/annotations"))
    a = ap.parse_args()
    a.annotations.mkdir(parents=True, exist_ok=True)
    for i in range(a.start, a.start + a.n):
        # every k-th match is anomalous, giving an exact, seed-independent fraction
        anomalous = (i % 10) < round(a.anomalous_frac * 10)
        m = simulate_match(seed=i, anomalous=anomalous)
        ann = render_match(m, a.raw / f"match_{i:04d}.mp4", seed=i)
        (a.annotations / f"match_{i:04d}.json").write_text(ann.model_dump_json(indent=1))
        print(f"match_{i:04d} anomalous={anomalous} events={len(m.events)} dur={m.duration:.0f}s")


if __name__ == "__main__":
    main()
