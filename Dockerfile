# Multi-stage build — keeps the final image lean
FROM python:3.11-slim AS base

WORKDIR /app

# System deps (needed by some torch/transformers wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download HuggingFace models into the image at build time.
# This means the container is larger (~1.5GB) but cold-start on Render is fast
# because the model is already on disk — no network download at runtime.
RUN python -c "\
from transformers import pipeline; \
pipeline('text-classification', model='cardiffnlp/twitter-roberta-base-sentiment-latest', top_k=None); \
pipeline('text-classification', model='j-hartmann/emotion-english-distilroberta-base', top_k=None); \
print('Models cached successfully')"

# Copy app source
COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
