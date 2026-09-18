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
COPY anvesha/ anvesha/
COPY scripts/ scripts/
COPY samples/ samples/
# Committed benchmark evidence (~20 KB). The console's Evaluation tab reads
# artifacts/scorecard.json at request time, so a deployed image with no run
# history still shows the measured numbers instead of an empty card.
COPY artifacts/ artifacts/
COPY weights/ weights/
COPY web/dist/ web/dist/

# Fail the build, not the air-gapped demo, if the bundled offline assets are
# missing. The acquisition layer resolves place names from a bundled gazetteer
# and selects ISRO layers from a bundled index; if .dockerignore ever swallows
# anvesha/acquire/data again, those two features degrade silently at runtime.
# The same check covers the built frontend, which the server serves directly.
RUN python -c "\
from pathlib import Path as P;\
d=P('anvesha/acquire/data');\
missing=[n for n in ('india_states.json','bhuvan_layers.json') if not (d/n).exists()];\
assert not missing, 'bundled acquisition indices missing from image: %s' % missing;\
assert P('web/dist/index.html').exists(), 'web/dist/index.html missing: run the frontend build';
assert P('artifacts/scorecard.json').exists(), 'artifacts/scorecard.json missing: the console Evaluation tab reads it';\
print('bundled offline assets OK:', sorted(x.name for x in d.glob('*.json')), 'and web/dist')"

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser -d /app appuser \
    && mkdir -p /app/data /app/runs && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Honour $PORT when the host injects one (Cloud Run, Render, Railway, Fly,
# Spaces' app_port via the platform); fall back to 8000 for a plain `docker
# run`. Shell form is required for the default expansion, and the healthcheck
# resolves the same variable so it never probes a port the app is not on.
ENV PORT=8000
# /healthz instantiates the four core specialists on its first call to report
# trained-vs-heuristic honestly (~6 s cold on a fast laptop, more on a small
# cloud vCPU; memoised thereafter). A 5 s probe timeout marked a healthy
# container unhealthy during that window, so the budget covers the cold load.
HEALTHCHECK --interval=30s --timeout=60s --start-period=180s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://localhost:%s/healthz' % os.environ.get('PORT','8000'))" || exit 1

CMD python -m uvicorn anvesha.server.main:app --host 0.0.0.0 --port ${PORT:-8000}
