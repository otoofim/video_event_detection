"""Device selection shared by training, inference and serving.

torch is imported lazily so the serving image (onnxruntime only) can use this module.
``pref`` is ``auto`` (GPU when available), ``cpu`` or ``cuda``.
"""

from __future__ import annotations


def torch_device(pref: str = "auto"):
    import torch

    if pref not in ("auto", "cpu", "cuda"):
        raise ValueError(f"device must be auto, cpu or cuda, got {pref!r}")
    if pref == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device 'cuda' requested but CUDA is not available")
    use_cuda = pref == "cuda" or (pref == "auto" and torch.cuda.is_available())
    return torch.device("cuda" if use_cuda else "cpu")


def onnx_providers(pref: str = "auto") -> list[str]:
    """Execution providers in priority order; CPU is always the fallback."""
    import onnxruntime as ort

    if pref not in ("auto", "cpu", "cuda"):
        raise ValueError(f"device must be auto, cpu or cuda, got {pref!r}")
    have = "CUDAExecutionProvider" in ort.get_available_providers()
    if pref == "cuda" and not have:
        raise RuntimeError("device 'cuda' requested but onnxruntime has no CUDA provider")
    if pref == "cpu" or not have:
        return ["CPUExecutionProvider"]
    return ["CUDAExecutionProvider", "CPUExecutionProvider"]
