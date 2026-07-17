FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    FLUKE_MODEL_ARTIFACT_DIR=/app/model-artifact \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /app
RUN pip install --no-cache-dir uv==0.11.8 \
    && groupadd --system fluke \
    && useradd --system --gid fluke --home-dir /app fluke

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY scripts/serve_identifier.py scripts/fetch_model_artifact.py scripts/build_ci_reference_index.py ./scripts/
RUN uv sync --locked --no-dev \
    && uv run --no-sync python scripts/fetch_model_artifact.py --out-dir /app/model-artifact \
    && mkdir -p /app/artifacts/reference-index \
    && chown -R fluke:fluke /app

USER fluke
EXPOSE 4100
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD ["uv", "run", "--no-sync", "python", "-c", "import os, urllib.request; port=os.environ.get('PORT','4100'); urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=3).read()"]
CMD ["uv", "run", "--no-sync", "python", "scripts/serve_identifier.py"]
