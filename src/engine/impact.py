"""
RiskLens – impact score formula.

Formula (transparent and defensible):
  impact = clip(1, 10,
      base_severity[event_type]
      × (0.5 + |sentiment|)
      × source_credibility
      × novelty
      × corroboration_boost
      × entity_materiality)

All component values are returned in the explanation dict so every
number on the dashboard traces back to this formula.

ASSUMPTION: base_severity values are DOCUMENTED DEFAULTS, not empirically
calibrated (see docs/ASSUMPTIONS.md). They are labeled as such everywhere.
"""
from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional

from src.config import CORROBORATION_WINDOW_HOURS

# ---------------------------------------------------------------------------
# Base severity table (ASSUMPTION – documented defaults, not calibrated)
# ---------------------------------------------------------------------------
BASE_SEVERITY: dict[str, float] = {
    "Credit Event":            7.0,
    "Geopolitical":            6.5,
    "Macroeconomic":           6.0,
    "Regulatory/Legal":        5.5,
    "Merger/Acquisition":      5.0,
    "Earnings/Guidance":       4.5,
    "Product Launch":          3.0,
    "Operational/Cyber":       5.5,
    "Supply Chain/Commodity":  5.0,
    "ESG/Climate":             3.5,
    "Other/Noise":             2.0,
}

# ---------------------------------------------------------------------------
# Source credibility weights (ASSUMPTION)
# ---------------------------------------------------------------------------
SOURCE_CREDIBILITY: dict[str, float] = {
    "replay_news":   0.85,
    "replay_social": 0.55,
    "kaggle_news":   0.85,
    "newsapi":       0.90,
    "kaggle_tweets": 0.55,
    "replay":        0.80,
}

# ---------------------------------------------------------------------------
# Entity materiality (ASSUMPTION – proxy: is this in our alias dict?)
# Richer version would use market-cap weighting.
# ---------------------------------------------------------------------------
def _entity_materiality(ticker: Optional[str]) -> float:
    """Financial-sector entities get a slight boost."""
    if ticker is None:
        return 0.8
    financial = {"JPM", "BAC", "GS", "MS", "C", "WFC", "SIVB"}
    return 1.1 if ticker in financial else 1.0


# ---------------------------------------------------------------------------
# Corroboration tracker
# ---------------------------------------------------------------------------
class CorroborationTracker:
    """
    Tracks how many independent sources have reported on the same
    entity/event_type pair within a sliding time window.
    """

    def __init__(self, window_hours: int = CORROBORATION_WINDOW_HOURS) -> None:
        self._window_hours = window_hours
        # key → list of (source, timestamp)
        self._records: dict[str, list[tuple[str, datetime]]] = defaultdict(list)

    def record(self, ticker: Optional[str], event_type: str, source: str, ts: datetime) -> float:
        """
        Register an event and return the corroboration_boost factor.
        boost = 1.0 for first report; rises to 1.5 for 3+ independent sources.
        """
        key = f"{ticker or 'unknown'}::{event_type}"
        now = ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
        cutoff = now.timestamp() - self._window_hours * 3600

        # Prune old records
        self._records[key] = [
            (s, t) for s, t in self._records[key]
            if t.timestamp() >= cutoff
        ]

        # Count unique sources already seen
        seen_sources = {s for s, _ in self._records[key]}
        self._records[key].append((source, now))

        n_independent = len(seen_sources | {source})
        # Boost: 1.0 (1 source), 1.15 (2), 1.30 (3), cap 1.50
        boost = min(1.5, 1.0 + (n_independent - 1) * 0.15)
        return boost


# ---------------------------------------------------------------------------
# Impact score computation
# ---------------------------------------------------------------------------

def compute_impact(
    event_type: str,
    sentiment_score: float,
    source: str,
    novelty: float,
    corroboration_boost: float,
    ticker: Optional[str] = None,
) -> tuple[float, dict]:
    """
    Compute impact score and return (score, explanation_dict).

    All intermediate values are in the explanation dict so they can
    be surfaced in the dashboard and in the signal JSON.
    """
    base   = BASE_SEVERITY.get(event_type, 3.0)
    sent_f = 0.5 + abs(sentiment_score)          # [0.5, 1.5]
    cred   = SOURCE_CREDIBILITY.get(source, 0.7)
    mat    = _entity_materiality(ticker)

    raw = base * sent_f * cred * novelty * corroboration_boost * mat
    score = float(max(1.0, min(10.0, raw)))

    explanation = {
        "base_severity":      base,
        "sentiment_factor":   round(sent_f, 3),
        "credibility":        cred,
        "novelty":            novelty,
        "corroboration":      corroboration_boost,
        "entity_materiality": mat,
        "raw_score":          round(raw, 3),
    }
    return score, explanation
