FROM python:3.11-slim

WORKDIR /app

# Install minimal system dependencies for FFmpeg, libsndfile, and OpenCV
# --no-install-recommends speeds up the process by avoiding optional packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements (UNCHANGED — reuses cached pip install layer with torch)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY . .

# Set HuggingFace cache dir
ENV HF_HOME=/tmp/hf_cache

# Expose port 8000
EXPOSE 8000

# Use uvicorn to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
