# =============================================================================
# Stage 1: Build Stage — Install dependencies and run pipeline to train model
# =============================================================================
FROM python:3.11-slim AS builder

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install system build dependencies (minimal)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first for layer caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code (excluding dev files via .dockerignore)
COPY . .

# Run the pipeline to train and save the model inside the image
# This makes the image self-contained: no runtime training needed
RUN mkdir -p /app/models /app/data/raw /app/data/processed /app/data/features /app/mlruns && \
    python src/run_pipeline.py && \
    rm -rf /app/mlruns /root/.cache /root/.local /tmp/*


# =============================================================================
# Stage 2: Runtime Stage — Minimal production image
# =============================================================================
FROM python:3.11-slim AS runtime

# Security: run as non-root user
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    APP_USER=appuser \
    APP_UID=1001 \
    APP_GID=1001

WORKDIR /app

# Install ONLY runtime system dependencies (no build tools)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd --system --gid ${APP_GID} ${APP_USER} && \
    useradd --system --create-home --no-log-init --gid ${APP_GID} --uid ${APP_UID} ${APP_USER}

# Copy Python packages from builder (avoids re-installing)
COPY --from=builder /usr/local/lib/python3.11/site-packages/ /usr/local/lib/python3.11/site-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/

# Copy application code and trained model from builder
COPY --from=builder --chown=${APP_USER}:${APP_USER} /app/src /app/src
COPY --from=builder --chown=${APP_USER}:${APP_USER} /app/models /app/models

# Create required directories with proper ownership
RUN mkdir -p /app/data/raw /app/data/processed /app/data/features && \
    chown -R ${APP_USER}:${APP_USER} /app/data

# Switch to non-root user
USER ${APP_USER}

# Expose the API port
EXPOSE 8000

# Health check — FastAPI built-in
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run the FastAPI server with production settings
# --workers 1 because ML models are not thread-safe without careful handling
# --proxy-headers for correct client IP when behind Cloud Run's proxy
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--proxy-headers", "--limit-concurrency", "100"]
