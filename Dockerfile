FROM python:3.10-slim-bullseye

WORKDIR /app

# Install system dependencies required for Librosa (audio) and OpenCV (video)
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# We copy requirements first to leverage Docker layer caching
COPY backend/requirements_azure.txt .
RUN pip install --no-cache-dir -r requirements_azure.txt

# Copy application source code
# We explicitly copy backend and ml_pipeline to match the expected structure
COPY backend/ ./backend/
COPY ml_pipeline/ ./ml_pipeline/

# Set Python path so backend can import ml_pipeline
ENV PYTHONPATH=/app

# Create necessary directories for persistence
RUN mkdir -p backend/database backend/uploads/images backend/uploads/audio backend/uploads/video backend/uploads/text backend/uploads/tabular
RUN mkdir -p ml_pipeline/h5_omnifusion/pretrained_models
RUN mkdir -p ml_pipeline/h5_omnifusion/checkpoints

# Expose the FASTAPI port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
