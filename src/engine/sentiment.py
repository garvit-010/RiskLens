"""
RiskLens – sentiment analysis.

Primary:  ProsusAI/finbert (fp16 on GPU if available, else CPU)
Baseline: VADER (vaderSentiment)

The FinBERT score is:  P(positive) - P(negative)  ∈ [-1, 1]

Models are loaded lazily on first call and cached in module-level globals
so they are only loaded once per process.
"""
from __future__ import annotations

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy globals
# ---------------------------------------------------------------------------
_finbert_pipeline = None
_vader_analyzer   = None
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _load_finbert() -> None:
    global _finbert_pipeline
    if _finbert_pipeline is not None:
        return
    try:
        from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
        from src.config import FINBERT_MODEL, MODELS_CACHE_DIR
        logger.info("Loading FinBERT on %s …", _DEVICE)
        # Check if local snapshot with weights exists for 100% offline loading
        snap_dirs = list((MODELS_CACHE_DIR / "models--ProsusAI--finbert" / "snapshots").glob("*"))
        local_dir = snap_dirs[0] if (snap_dirs and (snap_dirs[0] / "pytorch_model.bin").exists()) else None
        model_source = str(local_dir) if local_dir else FINBERT_MODEL
        local_only = local_dir is not None

        tokenizer = AutoTokenizer.from_pretrained(
            model_source,
            cache_dir=str(MODELS_CACHE_DIR),
            local_files_only=local_only,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_source,
            cache_dir=str(MODELS_CACHE_DIR),
            torch_dtype=torch.float16 if _DEVICE == "cuda" else torch.float32,
            local_files_only=local_only,
        )
        _finbert_pipeline = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            device=0 if _DEVICE == "cuda" else -1,
            top_k=None,                          # return all class scores
            truncation=True,
            max_length=512,
        )
        logger.info("FinBERT loaded.")
    except Exception as exc:
        logger.error("Failed to load FinBERT: %s – falling back to VADER.", exc)
        _finbert_pipeline = None


def _load_vader() -> None:
    global _vader_analyzer
    if _vader_analyzer is not None:
        return
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        _vader_analyzer = SentimentIntensityAnalyzer()
    except ImportError:
        logger.error("vaderSentiment not installed.")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def sentiment_finbert(text: str) -> tuple[float, float]:
    """
    Return (score, confidence) using FinBERT.
    score = P(positive) - P(negative) ∈ [-1, 1]
    confidence = max class probability
    """
    _load_finbert()
    if _finbert_pipeline is None:
        return sentiment_vader(text)

    try:
        results = _finbert_pipeline(text[:512])[0]  # list of {label, score}
        scores = {r["label"].lower(): r["score"] for r in results}
        pos = scores.get("positive", 0.0)
        neg = scores.get("negative", 0.0)
        neu = scores.get("neutral", 0.0)
        net = pos - neg
        confidence = max(pos, neg, neu)
        return float(net), float(confidence)
    except Exception as exc:
        logger.warning("FinBERT inference failed: %s", exc)
        return 0.0, 0.5


def sentiment_finbert_probs(text: str) -> dict[str, float]:
    """Return raw FinBERT class probabilities: {'positive': float, 'negative': float, 'neutral': float}."""
    _load_finbert()
    if _finbert_pipeline is None:
        comp, _ = sentiment_vader(text)
        if comp > 0.05:
            return {"positive": 0.7, "negative": 0.1, "neutral": 0.2}
        elif comp < -0.05:
            return {"positive": 0.1, "negative": 0.7, "neutral": 0.2}
        return {"positive": 0.1, "negative": 0.1, "neutral": 0.8}
    try:
        results = _finbert_pipeline(text[:512])[0]
        return {r["label"].lower(): float(r["score"]) for r in results}
    except Exception as exc:
        logger.warning("FinBERT inference failed: %s", exc)
        return {"positive": 0.0, "negative": 0.0, "neutral": 1.0}


def sentiment_vader(text: str) -> tuple[float, float]:
    """
    Return (score, confidence) using VADER.
    score = compound score ∈ [-1, 1]
    confidence = abs(compound) mapped to [0.5, 1.0]
    """
    _load_vader()
    if _vader_analyzer is None:
        return 0.0, 0.5
    vs = _vader_analyzer.polarity_scores(text)
    compound = float(vs["compound"])
    # Rough confidence proxy: how far from neutral
    confidence = 0.5 + 0.5 * abs(compound)
    return compound, confidence


def get_sentiment(text: str, use_finbert: bool = True) -> dict:
    """
    Return a sentiment dict with both model scores.

    Returns
    -------
    {
      "score": float,       # primary score (FinBERT if available)
      "confidence": float,
      "finbert_score": float,
      "vader_score": float,
      "model_used": str
    }
    """
    fb_score, fb_conf = sentiment_finbert(text)
    vd_score, _       = sentiment_vader(text)

    primary_score = fb_score if (use_finbert and _finbert_pipeline is not None) else vd_score
    model_used    = "finbert" if (use_finbert and _finbert_pipeline is not None) else "vader"

    return {
        "score":        primary_score,
        "confidence":   fb_conf if model_used == "finbert" else 0.5 + 0.5 * abs(vd_score),
        "finbert_score": fb_score,
        "vader_score":  vd_score,
        "model_used":   model_used,
    }
