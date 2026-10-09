"""
RiskLens – deduplication.

Two-stage dedup:
1. Exact hash match (Document.id is already a content hash).
2. Fuzzy title similarity via rapidfuzz ≥ DEDUP_SIMILARITY_THRESHOLD.
   Catches syndicated news and near-duplicate retweets.

The novelty factor returned here is used in the impact formula:
  - First occurrence → novelty = 1.0
  - Near-duplicate of a recent doc → novelty = 0.3
"""
from __future__ import annotations

import hashlib
import logging
from collections import deque
from typing import Optional

from src.config import DEDUP_SIMILARITY_THRESHOLD
from src.schemas import Document

logger = logging.getLogger(__name__)

try:
    from rapidfuzz import fuzz
    _RF_AVAILABLE = True
except ImportError:
    _RF_AVAILABLE = False
    logger.warning("rapidfuzz not installed – fuzzy dedup disabled; exact-hash only.")


class Deduplicator:
    """
    Stateful deduplicator that tracks the last N documents.

    Parameters
    ----------
    window : int
        Number of recent document texts to keep for fuzzy comparison.
    threshold : float
        Rapidfuzz ratio threshold above which a document is a near-duplicate.
    """

    def __init__(
        self,
        window: int = 200,
        threshold: float = DEDUP_SIMILARITY_THRESHOLD,
    ) -> None:
        self._seen_ids: set[str] = set()
        self._recent_texts: deque[str] = deque(maxlen=window)
        self._threshold = threshold

    def check(self, doc: Document) -> tuple[bool, float]:
        """
        Return (is_duplicate, novelty_factor).

        novelty_factor is 1.0 for new content, 0.3 for near-duplicates.
        Callers should still process near-duplicates for corroboration counting,
        but should NOT emit them as primary signals.
        """
        # Stage 1: exact hash
        if doc.id in self._seen_ids:
            return True, 0.3

        # Stage 2: fuzzy similarity
        if _RF_AVAILABLE and self._recent_texts:
            for prev in self._recent_texts:
                score = fuzz.token_sort_ratio(doc.text, prev) / 100.0
                if score >= self._threshold:
                    logger.debug("Near-duplicate detected (score=%.2f): %.60s", score, doc.text)
                    self._seen_ids.add(doc.id)
                    return True, 0.3

        # New document
        self._seen_ids.add(doc.id)
        self._recent_texts.append(doc.text)
        return False, 1.0
