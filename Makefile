PY ?= .venv/bin/python

.PHONY: install install-gpu data train-detector train-temporal export bench serve test lint all
install:
	uv venv .venv && uv pip install --python $(PY) --extra-index-url https://download.pytorch.org/whl/cpu -e ".[train,serve,dev]"
install-gpu:  # CUDA build of torch (pick the wheel index matching your driver) + onnxruntime-gpu
	uv venv .venv && uv pip install --python $(PY) --extra-index-url https://download.pytorch.org/whl/cu124 -e ".[train,serve,dev]" && uv pip install --python $(PY) onnxruntime-gpu
data:  # annotations + features for all splits (~16 GB)
	$(PY) -m data.scripts.download_lolv2t --parts annotations train val test
train-detector:
	$(PY) -m detection.train
train-temporal:
	$(PY) -m temporal.train
export:
	$(PY) -m serving.onnx_export
bench:
	$(PY) -m benchmarks.latency_bench model
serve:
	MODEL_DIR=models $(PY) -m uvicorn serving.app:app --port 8000
test:
	$(PY) -m pytest -q
lint:
	$(PY) -m ruff check . && $(PY) -m ruff format --check .
all: data train-detector train-temporal export
