FROM python:3.12-slim

WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY satquery/ satquery/
COPY scripts/ scripts/
COPY samples/ samples/
COPY weights/ weights/
COPY web/dist/ web/dist/

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "satquery.server.main:app", "--host", "0.0.0.0", "--port", "8000"]
