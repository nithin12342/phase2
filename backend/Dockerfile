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
RUN pip wheel --no-cache-dir --wheel-dir /build/wheels --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt


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

# 2. Copy requirements.txt
COPY --from=builder /build/requirements.txt .

# 3. Install from local wheels via bind mount to prevent layer bloat
RUN --mount=type=bind,from=builder,source=/build/wheels,target=/wheels \
    pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt

# 4. Copy backend application code
COPY . .

# 5. Set HuggingFace cache directory
ENV HF_HOME=/tmp/hf_cache

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
