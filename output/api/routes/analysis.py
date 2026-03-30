"""
Analysis routes — the core of the API.

Endpoints:
  POST /analyse          single text → full sentiment + emotion result
  POST /analyse/bulk     up to 10 texts in one request
  GET  /analyse/demo     hit the API with a preset example (easy browser test)
"""
import logging
from fastapi import APIRouter, HTTPException, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.schemas.analysis import (
    AnalyseRequest,
    AnalysisResponse,
    BulkAnalyseRequest,
    BulkAnalysisResponse,
)
from app.services.analysis_service import analyse_text, analyse_bulk
from app.services.model_service import model_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/analyse", tags=["Analysis"])
limiter = Limiter(key_func=get_remote_address)


@router.post(
    "",
    response_model=AnalysisResponse,
    summary="Analyse a single text",
    description=(
        "Submit a text string and receive sentiment (positive/negative/neutral) "
        "plus a 7-emotion breakdown. Max 512 characters."
    ),
)
@limiter.limit("30/minute")
async def analyse(request: Request, body: AnalyseRequest) -> AnalysisResponse:
    """
    Main analysis endpoint. Runs two HuggingFace models:
    - cardiffnlp/twitter-roberta-base-sentiment-latest  → sentiment
    - j-hartmann/emotion-english-distilroberta-base     → emotions
    """
    if not model_service.is_ready():
        raise HTTPException(status_code=503, detail="Models are still loading, try again shortly.")

    try:
        return analyse_text(body.text)
    except Exception as exc:
        logger.exception("Analysis failed for text: %r", body.text[:50])
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(exc)}")


@router.post(
    "/bulk",
    response_model=BulkAnalysisResponse,
    summary="Analyse up to 10 texts at once",
    description="Submit a list of up to 10 texts and get a result for each.",
)
@limiter.limit("10/minute")
async def analyse_bulk_endpoint(
    request: Request, body: BulkAnalyseRequest
) -> BulkAnalysisResponse:
    if not model_service.is_ready():
        raise HTTPException(status_code=503, detail="Models are still loading, try again shortly.")

    try:
        return analyse_bulk(body.texts)
    except Exception as exc:
        logger.exception("Bulk analysis failed")
        raise HTTPException(status_code=500, detail=f"Bulk analysis error: {str(exc)}")


@router.get(
    "/demo",
    response_model=AnalysisResponse,
    summary="Quick demo — no request body needed",
    description="Runs analysis on a preset sample sentence. Great for quick browser/curl tests.",
)
async def demo() -> AnalysisResponse:
    if not model_service.is_ready():
        raise HTTPException(status_code=503, detail="Models are still loading.")

    sample = "I just got promoted at work and I couldn't be happier!"
    return analyse_text(sample)
