"""Export trained checkpoints to ONNX (dynamic batch) plus a ``meta.json`` the service reads.

python -m serving.onnx_export --detector models/detector.pt --temporal models/temporal.pt --out models
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from common.schema import CLASSES
from detection.model import EventNet
from detection.seq_model import FeatureEventNet, StepMajor
from temporal.dataset import N_FEATURES
from temporal.model import ConvAE, WindowScorer


def _check(onnx_path: Path, torch_out: np.ndarray, feed: np.ndarray, atol: float = 1e-4) -> None:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    got = sess.run(None, {sess.get_inputs()[0].name: feed})[0]
    if not np.allclose(got, torch_out, atol=atol):
        raise AssertionError(
            f"{onnx_path.name}: ONNX output differs from PyTorch "
            f"(max abs diff {np.abs(got - torch_out).max():.2e})"
        )


def export_detector(ckpt_path: Path, out: Path) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = EventNet(width=ckpt["width"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    w, h = ckpt["size"]
    dummy = torch.randn(2, 3, h, w)
    path = out / "detector.onnx"
    torch.onnx.export(
        model,
        dummy,
        str(path),
        input_names=["frames"],
        output_names=["logits"],
        dynamic_axes={"frames": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    with torch.inference_mode():
        _check(path, model(dummy).numpy(), dummy.numpy())
    return {"size": [w, h], "classes": CLASSES}


def export_temporal(ckpt_path: Path, out: Path) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    ae = ConvAE()
    ae.load_state_dict(ckpt["state_dict"])
    scorer = WindowScorer(ae).eval()
    dummy = torch.randn(3, N_FEATURES, ckpt["window"])
    path = out / "temporal.onnx"
    torch.onnx.export(
        scorer,
        dummy,
        str(path),
        input_names=["windows"],
        output_names=["scores"],
        dynamic_axes={"windows": {0: "batch"}, "scores": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    with torch.inference_mode():
        _check(path, scorer(dummy).numpy(), dummy.numpy())
    return {
        "window": ckpt["window"],
        "stride": ckpt["stride"],
        "threshold": ckpt["threshold"],
        "bin_s": 1.0,
    }


def export_lolv2t(ckpt_path: Path, out: Path) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    net = FeatureEventNet(ckpt["in_dim"], len(ckpt["classes"]), ckpt["hidden"], ckpt["dropout"])
    net.load_state_dict(ckpt["state_dict"])
    model = StepMajor(net).eval()
    dummy = torch.randn(2, 37, ckpt["in_dim"])
    path = out / "lolv2t.onnx"
    torch.onnx.export(
        model,
        dummy,
        str(path),
        input_names=["features"],
        output_names=["logits"],
        dynamic_axes={"features": {0: "batch", 1: "steps"}, "logits": {0: "batch", 1: "steps"}},
        opset_version=17,
        dynamo=False,
    )
    with torch.inference_mode():
        _check(path, model(dummy).numpy(), dummy.numpy())
    return {
        "in_dim": ckpt["in_dim"],
        "classes": ckpt["classes"],
        "use_flow": ckpt["use_flow"],
        "aggregator": ckpt["aggregator"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", type=Path, default=Path("models/detector.pt"))
    ap.add_argument("--temporal", type=Path, default=Path("models/temporal.pt"))
    ap.add_argument(
        "--lolv2t", type=Path, help="optional feature-based LoL-V2T detector checkpoint"
    )
    ap.add_argument("--out", type=Path, default=Path("models"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    meta = {
        "detector": export_detector(a.detector, a.out),
        "temporal": export_temporal(a.temporal, a.out),
    }
    if a.lolv2t:
        meta["lolv2t"] = export_lolv2t(a.lolv2t, a.out)
    (a.out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"exported ONNX models + meta.json to {a.out}")


if __name__ == "__main__":
    main()
