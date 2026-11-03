import pytest


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory):
    """Randomly initialised (untrained) models exported to ONNX: tests the plumbing, not accuracy."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from detection.model import EventNet
    from serving.onnx_export import main as export_main
    from temporal.model import ConvAE

    d = tmp_path_factory.mktemp("models")
    torch.manual_seed(0)
    torch.save({"state_dict": EventNet(width=8).state_dict(), "width": 8, "size": [160, 96]},
               d / "detector.pt")
    torch.save({"state_dict": ConvAE().state_dict(), "window": 32, "stride": 4, "threshold": 0.05},
               d / "temporal.pt")
    import sys

    argv = sys.argv
    sys.argv = ["x", "--detector", str(d / "detector.pt"), "--temporal", str(d / "temporal.pt"),
                "--out", str(d)]
    try:
        export_main()
    finally:
        sys.argv = argv
    return d
