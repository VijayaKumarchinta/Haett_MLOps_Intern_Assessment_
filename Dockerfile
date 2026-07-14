FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_USER=appuser \
    APP_UID=1001 \
    APP_GID=1001

WORKDIR /app

# libgomp1 is required by several numerical and ML packages.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create an unprivileged runtime user.
RUN groupadd --system --gid "${APP_GID}" "${APP_USER}" \
    && useradd \
        --system \
        --create-home \
        --no-log-init \
        --gid "${APP_GID}" \
        --uid "${APP_UID}" \
        "${APP_USER}"

# Dependency installation remains cached when only source code changes.
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy only inference code and already-trained deployment artifacts.
COPY --chown=${APP_USER}:${APP_USER} src ./src
COPY --chown=${APP_USER}:${APP_USER} models ./models

USER ${APP_USER}

EXPOSE 8000

HEALTHCHECK \
    --interval=30s \
    --timeout=5s \
    --start-period=15s \
    --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1

STOPSIGNAL SIGTERM

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--proxy-headers", "--limit-concurrency", "100"]