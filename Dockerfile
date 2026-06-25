# Dockerfile - Multi-stage build for ETL Order Service
# =========================================================
# Stage 1: builder  - installs dependencies into a venv
# Stage 2: runtime  - copies only the venv and app code
#
# Security:
# - Non-root user (appuser:appgroup, UID/GID 1001)
# - No build tools in the runtime image
# - Read-only filesystem compatible (data dir is a volume)
#
# Usage:
# docker build -t etl-service:latest .
# docker run -p 8000:8000 --env-file .env etl-service:latest
# =========================================================

# — Stage 1: Builder ———————————————————————————————————————
FROM python:3.11-slim AS builder

# Set build-time environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VENV_PATH="/opt/venv"

# Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv ${VENV_PATH}
ENV PATH="${VENV_PATH}/bin:${PATH}"

# Upgrade pip in venv
RUN pip install --upgrade pip setuptools wheel

# Copy and install Python dependencies
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# — Stage 2: Runtime ———————————————————————————————————————
FROM python:3.11-slim AS runtime

# Runtime environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VENV_PATH="/opt/venv" \
    APP_HOME="/app" \
    APP_ENV="production" \
    PORT=8000

ENV PATH="${VENV_PATH}/bin:${PATH}"

# Install only runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user and group
RUN groupadd -g 1001 appgroup && \
    useradd -u 1001 -g 1001 appgroup --shell /bin/bash --create-home appuser

# Copy virtual environment from builder
COPY --from=builder --chown=appuser:appgroup ${VENV_PATH} ${VENV_PATH}

# Set working directory
WORKDIR ${APP_HOME}

# Copy application code
COPY --chown=appuser:appgroup app/ ./app/
COPY --chown=appuser:appgroup main.py etl.py ./

# Create data directory for SQLite and FAISS index
RUN mkdir -p ${APP_HOME}/data && \
    chown -R appuser:appgroup ${APP_HOME}/data

# Switch to non-root user
USER appuser

# Expose application port
EXPOSE ${PORT}

# Health check - uses the /healthz liveness endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:${PORT}/healthz || exit 1

# Default command - production uvicorn server
CMD ["uvicorn", "main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info", \
     "--no-access-log"]