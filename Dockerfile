# ---------- build stage ----------
FROM python:3.12-slim AS builder

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    libffi-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for faster package management
RUN pip install uv

WORKDIR /app

# Accept AWS CodeArtifact token for installing private packages
ARG AWS_CODEARTIFACT_TOKEN
ARG CODEARTIFACT_INDEX_URL
# Install dependencies first (cached layer)
COPY pyproject.toml uv.lock ./
COPY libs/ libs/
RUN UV_INDEX_CODEARTIFACT_USERNAME=aws \
    UV_INDEX_CODEARTIFACT_PASSWORD=${AWS_CODEARTIFACT_TOKEN} \
    UV_EXTRA_INDEX_URL=${CODEARTIFACT_INDEX_URL} \
    uv pip install --system .

# ---------- production stage ----------
FROM python:3.12-slim AS production

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PASSLIB_BCRYPT_IDENT="2b" \
    DEBUG=False \
    ENVIRONMENT=production

RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

# Install system dependencies for runtime
RUN apt-get update && apt-get install -y \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Set work directory
WORKDIR /app
RUN chown -R appuser:appuser /app

# Copy application code
COPY . .

RUN mkdir -p /tmp/prometheus_multiproc_dir && \
    chown -R appuser:appuser /tmp/prometheus_multiproc_dir

USER appuser

EXPOSE 8004 9004

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8004/health || exit 1

CMD ["sh" ,"scripts/startup.sh"]
