"""Confusion-matrix heatmaps from ``detection.score_fast`` results (one figure per model).

    python -m detection.plot_confusion                    # all models in detection/results_scores.json
    python -m detection.plot_confusion --models reg4_c --splits train test

Colour is the row-normalised share (recall of each true class); each cell also shows the raw step
count. Rows are the true class, columns the predicted class.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def draw(ax, step: dict, title: str):
    cm = np.array(step["confusion_matrix"], dtype=float)
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    names = step["classes"]
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=30, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for i in range(len(names)):
        for j in range(len(names)):
            colour = "white" if norm[i, j] > 0.5 else "black"
            ax.text(
                j,
                i,
                f"{int(cm[i, j]):,}\n{norm[i, j]:.0%}",
                ha="center",
                va="center",
                color=colour,
                fontsize=8,
            )
    ax.set_title(
        f"{title}\nacc {step['accuracy']:.3f}  macro P {step['macro_precision']:.3f}  "
        f"R {step['macro_recall']:.3f}",
        fontsize=9,
    )
    return im


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", type=Path, default=Path("detection/results_scores.json"))
    ap.add_argument("--out", type=Path, default=Path("figures"))
    ap.add_argument("--models", nargs="+")
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    ap.add_argument("--which", choices=["step", "step_at_conf"], default="step")
    a = ap.parse_args()

    scores = json.loads(a.scores.read_text())
    a.out.mkdir(parents=True, exist_ok=True)
    for name in a.models or scores:
        fig, axes = plt.subplots(
            1,
            len(a.splits),
            figsize=(5.6 * len(a.splits), 4.9),
            squeeze=False,
            layout="constrained",
        )
        for ax, split in zip(axes[0], a.splits, strict=True):
            im = draw(ax, scores[name][split][a.which], f"{name} | {split}")
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label="share of true class")
        path = a.out / f"confusion_{name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("wrote", path)


if __name__ == "__main__":
    main()
