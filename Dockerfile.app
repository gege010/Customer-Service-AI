# ── Stage 1: Builder ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Frontend only requires a subset of the dependencies (no torch/chromadb).
# Installing the full requirements.txt is safe — Docker layer caching means
# this only re-runs when requirements.txt changes.
RUN pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt


# ── Stage 2: Runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="Customer Service AI"
LABEL description="Streamlit frontend — dark-theme chat UI with real-time token streaming"

COPY --from=builder /install /usr/local

WORKDIR /app

COPY src/app.py ./src/app.py

# API_BASE_URL is overridden at runtime (via docker-compose) to point at the
# backend container.  When running locally without Docker, it stays at the
# default 127.0.0.1:8000.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    API_BASE_URL=http://api:8000

EXPOSE 8501

# Streamlit config: disable browser auto-open and telemetry inside a container
CMD ["streamlit", "run", "src/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
