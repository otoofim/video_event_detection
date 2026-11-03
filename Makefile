PY ?= .venv/bin/python

.PHONY: install data train-detector infer train-temporal export bench serve test lint all
install:
	uv venv .venv && uv pip install --python $(PY) --extra-index-url https://download.pytorch.org/whl/cpu -e ".[train,serve,dev]"
data:
	$(PY) -m data.scripts.generate_synthetic --n 100 --anomalous-frac 0.3
train-detector:
	$(PY) -m detection.train
infer:
	$(PY) -m detection.infer --video-dir data/raw --weights models/detector.pt --out-dir data/events
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
all: data train-detector infer train-temporal export
