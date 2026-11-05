# All Python runs through uv: dependencies are pinned in uv.lock (CPU-only PyTorch).
RUN := uv run

.PHONY: install install-extract lock data train-detector ablation export bench bench-preprocess serve test lint all
install:  # .venv with training + serving deps and the dev group
	uv sync --extra train --extra serve
install-extract:  # + video preprocessing (torchvision, OpenCV contrib for TVL1, PySceneDetect)
	uv sync --all-extras
lock:  # after editing pyproject.toml
	uv lock
data:  # annotations + features for all splits (~16 GB)
	$(RUN) -m data.scripts.download_lolv2t --parts annotations train val test
train-detector:  # variational U-Net -> models/detector.pt, detection/results.json
	$(RUN) -m detection.train
ablation:  # same data, no variational parts / the dilated-conv net
	$(RUN) -m detection.train --arch unet --out models/detector_unet.pt --report detection/results_unet.json
	$(RUN) -m detection.train --arch dilated --out models/detector_dilated.pt --report detection/results_dilated.json
export:
	$(RUN) -m serving.onnx_export
bench:  # model only: PyTorch vs ONNX Runtime
	$(RUN) -m benchmarks.latency_bench model
bench-preprocess:  # video -> features, per stage (untrained weights, same compute)
	$(RUN) -m benchmarks.latency_bench preprocess
serve:  # http://localhost:8000, player at /player
	MODEL_DIR=models $(RUN) uvicorn serving.app:app --port 8000
test:
	$(RUN) pytest -q
lint:
	$(RUN) ruff check . && $(RUN) ruff format --check .
all: data train-detector ablation export
