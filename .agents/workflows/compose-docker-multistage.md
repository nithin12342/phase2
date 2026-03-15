---
description: Compose multistage Docker image with local H5-OmniFusion checkpoint and Hugging Face API
---

# Docker Multistage Build Workflow

This workflow will build a highly optimized, multistage Docker image for the H5-OmniFusion backend tomorrow. It ensures the `h5_omnifusion` checkpoint directory is embedded into the container, while leveraging the Hugging Face Inference API for the heavy modality-specific feature extractors (which keeps the final Docker image size small!).

## Prerequisites
- A valid Hugging Face API token is required for runtime feature extraction.
- Docker Desktop must be running.

## Build Steps

// turbo
1. Ensure the checkpoint directory exists within the backend build context so Docker can copy it.
```powershell
New-Item -ItemType Directory -Force -Path "backend/ml_pipeline/h5_omnifusion/checkpoints"
New-Item -ItemType Directory -Force -Path "backend/ml_pipeline/h5_omnifusion/config"
New-Item -ItemType Directory -Force -Path "backend/ml_pipeline/h5_omnifusion/src/models"
```

// turbo
2. Copy the active fusion model checkpoint and necessary architecture files into the backend build context.
```powershell
Copy-Item -Path "ml_pipeline/h5_omnifusion/checkpoints/*.pt" -Destination "backend/ml_pipeline/h5_omnifusion/checkpoints/" -Force
Copy-Item -Path "ml_pipeline/h5_omnifusion/config/*.py" -Destination "backend/ml_pipeline/h5_omnifusion/config/" -Force
Copy-Item -Path "ml_pipeline/h5_omnifusion/src/models" -Destination "backend/ml_pipeline/h5_omnifusion/src" -Recurse -Force
```

3. Review the `backend/Dockerfile` to confirm multistage compilation. It is already optimized but double-check that it looks like this:
```dockerfile
# STAGE 1: Dependency Builder
FROM python:3.11-slim AS builder
WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends build-essential
COPY requirements.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /build/wheels -r requirements.txt

# STAGE 2: Lightweight Runner
FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY --from=builder /build/wheels /wheels
COPY --from=builder /build/requirements.txt .
RUN pip install --no-cache --no-index --find-links=/wheels -r requirements.txt && rm -rf /wheels
COPY . .
ENV HF_HOME=/tmp/hf_cache
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

4. Build the heavily optimized Multistage Docker image.
```powershell
cd backend
docker build -t h5-omnifusion-backend:latest .
```

5. Run the Docker container.
**Crucial:** You must pass the Hugging Face token as an environment variable so the container can query the Remote HF Inference API for the text/audio/video models, bypassing the need to download them globally!
```powershell
docker run -d -p 8000:8000 --name h5-backend -e HUGGINGFACE_TOKEN="your-hf-token-here" h5-omnifusion-backend:latest
```

6. Verify the container health.
Confirm that the fusion model successfully loaded inside the Docker container.
```powershell
Invoke-RestMethod -Uri http://localhost:8000/status
```
