
# Use official Python image with stable Debian (bookworm)
FROM python:3.9-slim-bookworm

# Set working directory to /app (root of the shared context)
WORKDIR /app

# Install system dependencies (OpenCV, etc. for video processing)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements
COPY backend/requirements.txt ./backend/requirements.txt

# Install dependencies (upgrade pip first)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r backend/requirements.txt

# Copy source code
# We copy both backend and ml_pipeline to maintain the relative structure
COPY backend/ ./backend/
COPY ml_pipeline/ ./ml_pipeline/

# Set PYTHONPATH so imports work
# 1. /app/backend -> allows "import database" (if run from there)
# 2. /app -> allows "from backend import..." or "from ml_pipeline import..."
ENV PYTHONPATH=/app/backend:/app

# Expose port
EXPOSE 8000

# Run from /app so paths like "backend/uploads" resolve correctly manually or 
# if code uses relative paths from CWD. 
# However, application code seems to use "backend/uploads" relative to execution root.
# We will execute uvicorn on the module "backend.main"
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
