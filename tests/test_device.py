import pytest

from common.device import onnx_providers, torch_device


def test_cpu_is_always_available():
    pytest.importorskip("torch")
    assert torch_device("cpu").type == "cpu"


def test_bad_preference_is_rejected():
    pytest.importorskip("torch")
    with pytest.raises(ValueError):
        torch_device("tpu")


def test_onnx_providers_fall_back_to_cpu():
    pytest.importorskip("onnxruntime")
    assert onnx_providers("cpu") == ["CPUExecutionProvider"]
    assert onnx_providers("auto")[-1] == "CPUExecutionProvider"
