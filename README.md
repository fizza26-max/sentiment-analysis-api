# SentimentAPI

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green?logo=fastapi)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

A production-ready REST API that analyses text and returns **sentiment** (positive / negative / neutral) plus a **7-emotion breakdown** (joy, sadness, anger, fear, surprise, disgust, neutral).

100% open-source. No API keys. No paid services. Runs locally or deploys free on Render.

**Live demo:** `https://sentimentapi.onrender.com` *(replace with your URL)*

---

## Features

- **Dual-model inference** — two fine-tuned RoBERTa models run in parallel
- **Confidence scores** for every label, not just the top result
- **Bulk endpoint** — analyse up to 10 texts in a single request
- **Interactive UI** at `/` — paste text and see results with animated score bars
- **Swagger docs** auto-generated at `/docs`
- **Rate limiting** — 30 req/min per IP (configurable)
- **Docker-ready** — models baked into the image at build time for fast cold starts
- **CI/CD** via GitHub Actions

---

## Architecture

```
Browser / curl
      │
      ▼
FastAPI  (app/main.py)
      │
      ├── POST /analyse          ──► AnalysisService
      ├── POST /analyse/bulk     ──► AnalysisService (loop)
      ├── GET  /analyse/demo     ──► AnalysisService (preset text)
      ├── GET  /health
      └── GET  /models
                │
                ▼
         ModelService (singleton, loaded at startup)
                │
      ┌─────────┴──────────┐
      ▼                    ▼
Sentiment pipeline    Emotion pipeline
(RoBERTa, ~500MB)    (DistilRoBERTa, ~300MB)
cardiffnlp/...       j-hartmann/...
```

---

## Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/` | None | Interactive demo UI |
| GET | `/health` | None | Health check (used by Render) |
| GET | `/models` | None | Model info and supported labels |
| POST | `/analyse` | None | Analyse a single text |
| POST | `/analyse/bulk` | None | Analyse up to 10 texts |
| GET | `/analyse/demo` | None | Run on a preset sample (browser-friendly) |
| GET | `/docs` | None | Swagger UI |
| GET | `/redoc` | None | ReDoc API reference |

---

## Quick Start (local)

### Option A — plain Python

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/sentimentapi.git
cd sentimentapi

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
uvicorn app.main:app --reload
```

Open http://localhost:8000 — models download on first run (~800MB, cached after that).

### Option B — Docker

```bash
docker build -t sentimentapi .    # models are baked in at build time
docker run -p 8000:8000 sentimentapi
```

---

## Usage Examples

### Single text

```bash
curl -X POST http://localhost:8000/analyse \
  -H "Content-Type: application/json" \
  -d '{"text": "I absolutely love this product!"}'
```

```json
{
  "text": "I absolutely love this product!",
  "char_count": 31,
  "sentiment": {
    "label": "positive",
    "score": 0.9832,
    "all_scores": [
      {"label": "positive", "score": 0.9832},
      {"label": "neutral",  "score": 0.0121},
      {"label": "negative", "score": 0.0047}
    ]
  },
  "emotions": {
    "dominant_emotion": "joy",
    "score": 0.9211,
    "all_emotions": [
      {"label": "joy",      "score": 0.9211},
      {"label": "surprise", "score": 0.0312},
      ...
    ]
  }
}
```

### Bulk analysis

```bash
curl -X POST http://localhost:8000/analyse/bulk \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Great day!", "I am so tired", "Whatever"]}'
```

---

## Running Tests

```bash
pytest tests/ -v
```

Tests use mocked models — no GPU or internet needed.

---

## Deploying to Render (free)

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → New → Web Service
3. Connect your GitHub repo
4. Select **Docker** as the runtime
5. Set health check path to `/health`
6. Click Deploy

Render will build the Docker image (~5 min) and deploy. Your live URL will be `https://YOUR-APP.onrender.com`.

> **Note:** Free tier services sleep after 15 min idle. First request wakes them up (~30s).
> Fix: add your URL to [UptimeRobot](https://uptimerobot.com) (free) with a 10-min ping.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DEBUG` | `false` | Enable debug logging |
| `RATE_LIMIT_PER_MINUTE` | `30` | Max requests per minute per IP |
| `MAX_TEXT_LENGTH` | `512` | Maximum characters per text input |
| `SENTIMENT_MODEL` | `cardiffnlp/twitter-roberta-base-sentiment-latest` | HuggingFace model ID for sentiment |
| `EMOTION_MODEL` | `j-hartmann/emotion-english-distilroberta-base` | HuggingFace model ID for emotions |

---

## Models Used

| Model | Size | Labels |
|-------|------|--------|
| [cardiffnlp/twitter-roberta-base-sentiment-latest](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest) | ~500MB | positive, negative, neutral |
| [j-hartmann/emotion-english-distilroberta-base](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base) | ~300MB | joy, sadness, anger, fear, surprise, disgust, neutral |

Both models are free, open-source, and run fully locally.

---

## Project Structure

```
sentimentapi/
├── app/
│   ├── api/routes/
│   │   ├── analysis.py     # POST /analyse, /analyse/bulk, /analyse/demo
│   │   └── health.py       # GET /health, /models
│   ├── core/
│   │   └── config.py       # Settings via pydantic-settings
│   ├── schemas/
│   │   └── analysis.py     # Pydantic request/response models
│   ├── services/
│   │   ├── model_service.py    # HuggingFace pipeline loader (singleton)
│   │   └── analysis_service.py # Business logic
│   └── main.py             # App factory + lifespan + demo UI
├── tests/
│   └── test_api.py         # Full endpoint test suite
├── Dockerfile
├── render.yaml
├── requirements.txt
└── .github/workflows/ci.yml
```

---

## Built by

[Your Name] — Python AI Developer, Lahore, Pakistan  
[github.com/yourhandle](https://github.com) | [linkedin.com/in/yourhandle](https://linkedin.com)
