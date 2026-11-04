import pytest

CLASSES = ["background", "kill", "objective", "teamfight"]


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory):
    """Randomly initialised (untrained) models exported to ONNX: tests the plumbing, not accuracy."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    import sys

    from detection.model import FeatureEventNet
    from serving.onnx_export import main as export_main
    from temporal.model import PhaseNet

    d = tmp_path_factory.mktemp("models")
    torch.manual_seed(0)
    torch.save(
        {
            "state_dict": FeatureEventNet(16, hidden=8).state_dict(),
            "in_dim": 16,
            "hidden": 8,
            "dropout": 0.0,
            "classes": CLASSES,
            "use_flow": False,
            "aggregator": {"min_run": 1, "min_conf": 0.3},
        },
        d / "detector.pt",
    )
    torch.save(
        {
            "state_dict": PhaseNet(hidden=8).state_dict(),
            "hidden": 8,
            "window": 6,
            "stride": 2,
            "bin_s": 1.0,
            "classes": CLASSES,
        },
        d / "temporal.pt",
    )
    argv = sys.argv
    sys.argv = [
        "x",
        "--detector",
        str(d / "detector.pt"),
        "--temporal",
        str(d / "temporal.pt"),
        "--out",
        str(d),
    ]
    try:
        export_main()
    finally:
        sys.argv = argv
    return d
