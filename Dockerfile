# Stage 1: Build dependencies
FROM python:3.12-slim AS builder

WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM python:3.12-slim AS runtime

WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY satquery/ satquery/
COPY scripts/ scripts/
COPY samples/ samples/
COPY weights/ weights/
COPY web/dist/ web/dist/

# Fail the build, not the air-gapped demo, if the bundled offline assets are
# missing. The acquisition layer resolves place names from a bundled gazetteer
# and selects ISRO layers from a bundled index; if .dockerignore ever swallows
# satquery/acquire/data again, those two features degrade silently at runtime.
# The same check covers the built frontend, which the server serves directly.
RUN python -c "\
from pathlib import Path as P;\
d=P('satquery/acquire/data');\
missing=[n for n in ('india_states.json','bhuvan_layers.json') if not (d/n).exists()];\
assert not missing, 'bundled acquisition indices missing from image: %s' % missing;\
assert P('web/dist/index.html').exists(), 'web/dist/index.html missing: run the frontend build';\
print('bundled offline assets OK:', sorted(x.name for x in d.glob('*.json')), 'and web/dist')"

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser -d /app appuser \
    && mkdir -p /app/data /app/runs && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz')" || exit 1

CMD ["python", "-m", "uvicorn", "satquery.server.main:app", "--host", "0.0.0.0", "--port", "8000"]
