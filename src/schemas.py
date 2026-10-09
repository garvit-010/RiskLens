"""
RiskLens – unified Document schema.
All ingestion adapters emit this dataclass so downstream code
is source-agnostic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Document:
    """One unit of raw text from any source."""

    id: str                          # deterministic hash of source+text
    source: str                      # "kaggle_news" | "kaggle_tweets" | "newsapi" | "replay"
    timestamp: datetime
    text: str
    url: Optional[str] = None
    author: Optional[str] = None
    lang: Optional[str] = "en"

    # optional enrichments added later in the pipeline
    ticker: Optional[str] = None
    entity_name: Optional[str] = None
    sector: Optional[str] = None
    country: Optional[str] = None


@dataclass
class RiskSignal:
    """Structured risk signal output from the NLP engine."""

    signal_id: str
    timestamp: str                   # ISO-8601
    entity: dict                     # {name, ticker, sector, country}
    sentiment_score: float           # [-1.0, 1.0]  P(pos) - P(neg)
    event_type: str                  # taxonomy label
    impact_score: float              # [1.0, 10.0]
    confidence: float                # [0.0, 1.0]
    sources: list[dict]              # [{source, url, snippet}]
    explanation: dict                # component scores
    model_version: str = "risklens-0.1"

    def to_dict(self) -> dict:
        return {
            "signal_id": self.signal_id,
            "timestamp": self.timestamp,
            "entity": self.entity,
            "sentiment_score": round(self.sentiment_score, 4),
            "event_type": self.event_type,
            "impact_score": round(self.impact_score, 2),
            "confidence": round(self.confidence, 4),
            "sources": self.sources,
            "explanation": self.explanation,
            "model_version": self.model_version,
        }
