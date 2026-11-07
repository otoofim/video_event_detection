# Slim, features-only image: onnxruntime + numpy, no PyTorch. /detect_video answers 501 here
# (the /player page still works with an events JSON); use serving/Dockerfile.full for video input.
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.11.32 /uv /uvx /bin/
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked --no-dev --extra serve --no-install-project
COPY common ./common
COPY detection/__init__.py detection/events.py ./detection/
COPY serving ./serving
# models/ (detector.onnx, meta.json) is mounted or baked in by the user:
#   docker run -p 8000:8000 -v $PWD/models:/app/models video-event-serving
ENV MODEL_DIR=/app/models PYTHONUNBUFFERED=1 PATH="/app/.venv/bin:$PATH"
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
