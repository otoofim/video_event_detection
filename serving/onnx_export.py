"""Export a trained detector to ONNX (dynamic batch/length) plus the ``meta.json`` the service reads.

python -m serving.onnx_export --detector models/detector.pt --out models

The variational model gets a second input, ``noise`` ``(N, ceil(T / factor), latent)``: zeros give
the deterministic prediction (``z = mu``), standard-normal draws give samples for uncertainty.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from detection.model import StepMajor, build_model


def _check(path: Path, model: torch.nn.Module, feeds: dict, atol: float = 1e-4) -> None:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    with torch.inference_mode():
        want = model(*[torch.from_numpy(v) for v in feeds.values()]).numpy()
    got = sess.run(None, feeds)[0]
    if got.shape != want.shape or not np.allclose(got, want, atol=atol):
        raise AssertionError(
            f"{path.name}: ONNX output differs from PyTorch for input shapes "
            f"{[v.shape for v in feeds.values()]}"
        )


def load_detector(ckpt_path: Path) -> tuple[torch.nn.Module, dict]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    # checkpoints from before the U-Net have no arch/model_kwargs: they are FeatureEventNet
    kwargs = ckpt.get("model_kwargs") or {"hidden": ckpt["hidden"], "dropout": ckpt["dropout"]}
    net = build_model(ckpt.get("arch", "dilated"), ckpt["in_dim"], len(ckpt["classes"]), **kwargs)
    net.load_state_dict(ckpt["state_dict"])
    return net.eval(), ckpt


def export_detector(ckpt_path: Path, out: Path) -> dict:
    net, ckpt = load_detector(ckpt_path)
    model = StepMajor(net).eval()
    variational = bool(getattr(net, "variational", False))
    factor, latent = getattr(net, "factor", 1), getattr(net, "latent", 0)

    def feeds(n: int, t: int, random_noise: bool = False) -> dict:
        rng = np.random.default_rng(t)
        f = {"features": rng.standard_normal((n, t, ckpt["in_dim"])).astype(np.float32)}
        if variational:
            shape = (n, -(-t // factor), latent)
            f["noise"] = (
                rng.standard_normal(shape) if random_noise else np.zeros(shape)
            ).astype(np.float32)
        return f

    dummy = feeds(2, 37, random_noise=True)
    axes = {"features": {0: "batch", 1: "steps"}, "logits": {0: "batch", 1: "steps"}}
    if variational:
        axes["noise"] = {0: "batch", 1: "latent_steps"}
    path = out / "detector.onnx"
    torch.onnx.export(
        model,
        tuple(torch.from_numpy(v) for v in dummy.values()),
        str(path),
        input_names=list(dummy),
        output_names=["logits"],
        dynamic_axes=axes,
        opset_version=17,
        dynamo=False,
    )
    for n, t, rnd in [(2, 37, True), (1, 1, False), (3, 64, True), (1, 109, False)]:
        _check(path, model, feeds(n, t, rnd))  # other lengths too: the graph must stay dynamic
    return {
        "in_dim": ckpt["in_dim"],
        "classes": ckpt["classes"],
        "use_flow": ckpt["use_flow"],
        "aggregator": ckpt["aggregator"],
        "arch": ckpt.get("arch", "dilated"),
        "variational": variational,
        "factor": factor,
        "latent": latent,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", type=Path, default=Path("models/detector.pt"))
    ap.add_argument("--out", type=Path, default=Path("models"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    meta = {"detector": export_detector(a.detector, a.out)}
    (a.out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"exported detector.onnx + meta.json to {a.out}")


if __name__ == "__main__":
    main()
