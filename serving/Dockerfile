# Slim, features-only image: onnxruntime + numpy, no PyTorch. /detect_video answers 501 here;
# use serving/Dockerfile.full for video input.
FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir numpy pydantic pyyaml fastapi "uvicorn[standard]" onnxruntime \
    python-multipart
COPY common ./common
COPY detection/__init__.py detection/events.py ./detection/
COPY serving ./serving
# models/ (detector.onnx, meta.json) is mounted or baked in by the user:
#   docker run -p 8000:8000 -v $PWD/models:/app/models video-event-serving
ENV MODEL_DIR=/app/models PYTHONUNBUFFERED=1
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
