# ==========================================
# STAGE 1: BUILDER
# ==========================================
FROM python:3.11-slim AS builder

WORKDIR /build

# 1. Install system compilation tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 2. Copy and build wheels for all Python dependencies
COPY requirements.txt .
# Build wheels instead of installing. This leaves behind all pip caches and source files!
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /build/wheels -r requirements.txt


# ==========================================
# STAGE 2: RUNNER (Final Image)
# ==========================================
FROM python:3.11-slim

WORKDIR /app

# 1. Install ONLY the required runtime C-libraries (No build-essential!)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 2. Copy the pre-compiled wheels from the 'builder' stage
COPY --from=builder /build/wheels /wheels
COPY --from=builder /build/requirements.txt .

# 3. Install from the local wheels, bypassing PyPI and pip cache completely
RUN pip install --no-cache --no-index --find-links=/wheels -r requirements.txt \
    && rm -rf /wheels  # Clean up the wheels after installation!

# 4. Copy backend application code
COPY . .

# 5. Set HuggingFace cache directory
ENV HF_HOME=/tmp/hf_cache

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
