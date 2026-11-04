import sys

import pytest

CLASSES = ["background", "kill", "objective", "teamfight"]


def export_detector(d, in_dim: int, arch: str = "vunet"):
    """Randomly initialised (untrained) detector exported to ONNX: tests the plumbing only."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from detection.model import build_model
    from serving.onnx_export import main as export_main

    kwargs = {"hidden": 8, "latent": 4, "levels": 2, "dropout": 0.0}
    torch.manual_seed(0)
    torch.save(
        {
            "state_dict": build_model(arch, in_dim, **kwargs).state_dict(),
            "arch": arch,
            "model_kwargs": kwargs,
            "in_dim": in_dim,
            "classes": CLASSES,
            "use_flow": True,
            "aggregator": {"min_run": 1, "min_conf": 0.3},
        },
        d / "detector.pt",
    )
    argv = sys.argv
    sys.argv = ["x", "--detector", str(d / "detector.pt"), "--out", str(d)]
    try:
        export_main()
    finally:
        sys.argv = argv
    return d


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory):
    return export_detector(tmp_path_factory.mktemp("models"), 16)


@pytest.fixture(scope="session")
def video_model_dir(tmp_path_factory):
    """Detector over the extractor's 4096-d RGB + flow features."""
    return export_detector(tmp_path_factory.mktemp("video_models"), 4096)
