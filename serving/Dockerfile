# Inference-only image: onnxruntime + numpy + opencv, no PyTorch.
FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir numpy opencv-python-headless pydantic pyyaml fastapi \
    "uvicorn[standard]" onnxruntime python-multipart websockets
COPY common ./common
COPY detection/__init__.py detection/events.py ./detection/
COPY temporal/__init__.py temporal/dataset.py ./temporal/
COPY serving ./serving
# models/ (detector.onnx, temporal.onnx, meta.json) is mounted or baked in by the user:
#   docker run -p 8000:8000 -v $PWD/models:/app/models esports-serving
ENV MODEL_DIR=/app/models PYTHONUNBUFFERED=1
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
