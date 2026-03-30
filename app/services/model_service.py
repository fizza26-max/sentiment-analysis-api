"""
Model loader — downloads and caches HuggingFace models at startup.
Both models run fully locally, no API keys needed.
"""
import logging
from transformers import pipeline, Pipeline
from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class ModelService:
    """
    Singleton-style service that holds both inference pipelines.
    Loaded once on app startup via lifespan handler.
    """

    def __init__(self):
        self._sentiment_pipeline: Pipeline | None = None
        self._emotion_pipeline: Pipeline | None = None

    def load(self) -> None:
        """
        Download (first run) or load from local cache both models.
        Logs timing so you can see cold-start behaviour on Render.
        """
        import time

        logger.info("Loading sentiment model: %s", settings.SENTIMENT_MODEL)
        t0 = time.perf_counter()
        self._sentiment_pipeline = pipeline(
            task="text-classification",
            model=settings.SENTIMENT_MODEL,
            top_k=None,          # return all labels with scores
            truncation=True,
            max_length=512,
        )
        logger.info("Sentiment model ready in %.2fs", time.perf_counter() - t0)

        logger.info("Loading emotion model: %s", settings.EMOTION_MODEL)
        t1 = time.perf_counter()
        self._emotion_pipeline = pipeline(
            task="text-classification",
            model=settings.EMOTION_MODEL,
            top_k=None,
            truncation=True,
            max_length=512,
        )
        logger.info("Emotion model ready in %.2fs", time.perf_counter() - t1)

    @property
    def sentiment(self) -> Pipeline:
        if self._sentiment_pipeline is None:
            raise RuntimeError("Models not loaded. Call load() first.")
        return self._sentiment_pipeline

    @property
    def emotion(self) -> Pipeline:
        if self._emotion_pipeline is None:
            raise RuntimeError("Models not loaded. Call load() first.")
        return self._emotion_pipeline

    def is_ready(self) -> bool:
        return self._sentiment_pipeline is not None and self._emotion_pipeline is not None


# Module-level singleton — imported wherever models are needed
model_service = ModelService()
