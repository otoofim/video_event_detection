PY ?= .venv/bin/python

.PHONY: install install-extract data train-detector ablation export bench bench-preprocess serve test lint all
install:
	uv venv .venv && uv pip install --python $(PY) --extra-index-url https://download.pytorch.org/whl/cpu -e ".[train,serve,dev]"
install-extract:  # + video preprocessing (torch backend, TVL1, PySceneDetect)
	uv pip install --python $(PY) --extra-index-url https://download.pytorch.org/whl/cpu -e ".[extract]"
data:  # annotations + features for all splits (~16 GB)
	$(PY) -m data.scripts.download_lolv2t --parts annotations train val test
train-detector:  # variational U-Net -> models/detector.pt, detection/results.json
	$(PY) -m detection.train
ablation:  # same data, no variational parts / the dilated-conv net
	$(PY) -m detection.train --arch unet --out models/detector_unet.pt --report detection/results_unet.json
	$(PY) -m detection.train --arch dilated --out models/detector_dilated.pt --report detection/results_dilated.json
export:
	$(PY) -m serving.onnx_export
bench:  # model only: PyTorch vs ONNX Runtime
	$(PY) -m benchmarks.latency_bench model
bench-preprocess:  # video -> features, per stage (untrained weights, same compute)
	$(PY) -m benchmarks.latency_bench preprocess
serve:
	MODEL_DIR=models $(PY) -m uvicorn serving.app:app --port 8000
test:
	$(PY) -m pytest -q
lint:
	$(PY) -m ruff check . && $(PY) -m ruff format --check .
all: data train-detector ablation export
