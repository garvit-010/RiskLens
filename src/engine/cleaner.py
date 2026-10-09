"""
RiskLens – text cleaning and language filtering.

Steps
-----
1. Strip URLs
2. Strip cashtag noise (keep the ticker text, drop the $)
3. Normalize whitespace
4. Drop non-English documents (langdetect, falls back to True if uncertain)
"""
from __future__ import annotations

import logging
import re

from src.schemas import Document

logger = logging.getLogger(__name__)

_URL_RE     = re.compile(r"https?://\S+|www\.\S+")
_CASHTAG_RE = re.compile(r"\$([A-Z]{1,5})")
_MENTION_RE = re.compile(r"@\w+")
_WS_RE      = re.compile(r"\s+")

try:
    from langdetect import detect, LangDetectException
    _LANGDETECT_AVAILABLE = True
except ImportError:
    _LANGDETECT_AVAILABLE = False
    logger.warning("langdetect not installed – language filter disabled.")


def clean_text(text: str) -> str:
    """Return cleaned text with URLs and cashtag noise removed."""
    text = _URL_RE.sub(" ", text)
    text = _CASHTAG_RE.sub(r"\1", text)   # $AAPL → AAPL
    text = _MENTION_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def is_english(text: str) -> bool:
    """Return True if text is detected as English (or if detection fails)."""
    if not _LANGDETECT_AVAILABLE or len(text) < 20:
        return True
    try:
        return detect(text) == "en"
    except Exception:
        return True


def preprocess(doc: Document) -> Document | None:
    """
    Clean the document text and filter non-English content.

    Returns None if the document should be dropped.
    """
    cleaned = clean_text(doc.text)
    if len(cleaned) < 10:
        return None
    if not is_english(cleaned):
        return None
    doc.text = cleaned
    return doc
