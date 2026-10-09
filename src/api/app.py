"""
RiskLens – FastAPI signal store.

Endpoints
---------
GET /               → health check
GET /signals        → all signals (optionally filtered)
GET /signals/{ticker} → signals for a specific ticker
POST /process       → process a raw text and return a signal (live mode)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.config import SIGNAL_JSONL_PATH, API_PORT, MODEL_VERSION, IMPACT_THRESHOLD

logger = logging.getLogger(__name__)

app = FastAPI(
    title="RiskLens Signal API",
    description="Explainable News-to-Risk engine for wholesale banking.",
    version=MODEL_VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helper: load signals from JSONL
# ---------------------------------------------------------------------------

def _load_signals(jsonl_path: Path = SIGNAL_JSONL_PATH) -> list[dict]:
    if not jsonl_path.exists():
        return []
    signals = []
    with open(jsonl_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    signals.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return signals


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", tags=["Health"])
def health():
    return {"status": "ok", "version": MODEL_VERSION, "engine": "RiskLens"}


@app.get("/signals", tags=["Signals"])
def get_signals(
    ticker: Optional[str] = Query(None, description="Filter by ticker symbol"),
    event_type: Optional[str] = Query(None, description="Filter by event type"),
    min_impact: Optional[float] = Query(None, description="Minimum impact score"),
    limit: int = Query(100, le=1000, description="Max records to return"),
):
    """Return risk signals, optionally filtered."""
    signals = _load_signals()

    if ticker:
        signals = [s for s in signals if s.get("entity", {}).get("ticker", "").upper() == ticker.upper()]
    if event_type:
        signals = [s for s in signals if s.get("event_type", "").lower() == event_type.lower()]
    if min_impact is not None:
        signals = [s for s in signals if s.get("impact_score", 0) >= min_impact]

    # Most recent first
    signals.sort(key=lambda s: s.get("timestamp", ""), reverse=True)
    return {"count": len(signals[:limit]), "signals": signals[:limit]}


@app.get("/signals/{ticker}", tags=["Signals"])
def get_signals_by_ticker(
    ticker: str,
    limit: int = Query(50, le=500),
):
    """Return signals for a specific ticker."""
    signals = _load_signals()
    filtered = [
        s for s in signals
        if s.get("entity", {}).get("ticker", "").upper() == ticker.upper()
    ]
    filtered.sort(key=lambda s: s.get("timestamp", ""), reverse=True)
    if not filtered:
        raise HTTPException(status_code=404, detail=f"No signals found for ticker '{ticker}'")
    return {"ticker": ticker.upper(), "count": len(filtered[:limit]), "signals": filtered[:limit]}


class ProcessRequest(BaseModel):
    text: str
    source: str = "manual"


@app.post("/process", tags=["Live"])
def process_text(req: ProcessRequest):
    """
    Process raw text through the NLP engine and return a signal.
    This endpoint runs the full pipeline on the fly (requires models loaded).
    """
    try:
        from src.engine.pipeline import RiskEngine
        engine = RiskEngine(write_jsonl=True)
        signal = engine.process_text(req.text, source=req.source)
        if signal is None:
            raise HTTPException(status_code=422, detail="Text was filtered (too short, non-English, or duplicate).")
        return signal.to_dict()
    except Exception as exc:
        logger.exception("Error in /process: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/stats", tags=["Signals"])
def get_stats():
    """Summary statistics over all signals."""
    signals = _load_signals()
    if not signals:
        return {"total": 0}

    from collections import Counter
    event_counts = Counter(s.get("event_type", "Unknown") for s in signals)
    ticker_counts = Counter(s.get("entity", {}).get("ticker", "N/A") for s in signals)
    impact_scores = [s.get("impact_score", 0) for s in signals]

    return {
        "total": len(signals),
        "high_impact": sum(1 for i in impact_scores if i >= IMPACT_THRESHOLD),
        "avg_impact": round(sum(impact_scores) / len(impact_scores), 2),
        "top_event_types": dict(event_counts.most_common(5)),
        "top_tickers": dict(ticker_counts.most_common(5)),
    }
