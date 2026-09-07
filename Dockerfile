FROM python:3.11-slim

# opencv-python needs libGL/libglib at import time even though this is a
# headless deployment; curl is used by the HEALTHCHECK below.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first so this (slow, multi-GB - torch and friends
# live here) layer is cached across code-only rebuilds.
COPY requirements.txt .
# torch's default PyPI wheel drags in the full NVIDIA CUDA toolkit
# (cuDNN/cuBLAS/NCCL/Triton/...) - several extra GB nothing here uses, since
# there's no GPU acceleration wired up. Install the CPU-only build first;
# pip will then see the pinned version in requirements.txt as already
# satisfied and leave it alone instead of pulling the CUDA variant.
RUN pip install --no-cache-dir torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY app.py init_db.py wsgi.py yolo11n.pt ./
COPY templates ./templates
COPY static ./static

# instance/ holds the sqlite DB and the persisted session secret key - must
# survive container restarts/rebuilds, see docker-compose.yml's volume mount.
RUN mkdir -p instance

ENV FLASK_ENV=production \
    FLASK_RUN_HOST=0.0.0.0 \
    FLASK_RUN_PORT=8000 \
    PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/healthz || exit 1

CMD ["python", "wsgi.py"]
