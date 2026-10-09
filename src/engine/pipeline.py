"""
RiskLens – main pipeline orchestrator.

Wires together:  ingestion → clean → dedup → entity link
              → sentiment → classify → impact → RiskSignal

Signals are written to:
  - data/output/signals.jsonl  (append)
  - returned as Python objects for the API / dashboard
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

from src.config import SIGNAL_JSONL_PATH, MODEL_VERSION, IMPACT_THRESHOLD
from src.engine.cleaner import preprocess
from src.engine.classifier import classify_event
from src.engine.dedup import Deduplicator
from src.engine.entity_linker import link_entity
from src.engine.impact import CorroborationTracker, compute_impact
from src.engine.sentiment import get_sentiment
from src.ingestion.base import BaseAdapter
from src.schemas import Document, RiskSignal

logger = logging.getLogger(__name__)


class RiskEngine:
    """
    End-to-end risk signal pipeline.

    Parameters
    ----------
    use_finbert : bool
        Use FinBERT for sentiment (default True). Falls back to VADER automatically.
    write_jsonl : bool
        Append signals to the JSONL signal store.
    trigger_threshold : float
        Impact score threshold above which Module B is notified.
    """

    def __init__(
        self,
        use_finbert: bool = True,
        write_jsonl: bool = True,
        trigger_threshold: float = IMPACT_THRESHOLD,
    ) -> None:
        self._use_finbert   = use_finbert
        self._write_jsonl   = write_jsonl
        self._threshold     = trigger_threshold
        self._dedup         = Deduplicator()
        self._corroboration = CorroborationTracker()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_adapter(self, adapter: BaseAdapter) -> Iterator[RiskSignal]:
        """Stream all documents from one adapter through the full pipeline."""
        yield from self._process_stream(adapter.stream(), source_name=adapter.name())

    def process_adapters(self, adapters: list[BaseAdapter]) -> Iterator[RiskSignal]:
        """Multiplex multiple adapters in source order."""
        for adapter in adapters:
            yield from self.process_adapter(adapter)

    def process_text(self, text: str, source: str = "manual") -> RiskSignal | None:
        """Process a single text snippet – useful for Q&A / live testing."""
        import hashlib
        doc = Document(
            id=hashlib.sha256(f"{source}::{text}".encode()).hexdigest()[:16],
            source=source,
            timestamp=datetime.now(tz=timezone.utc),
            text=text,
        )
        results = list(self._process_stream([doc], source_name=source))
        return results[0] if results else None

    # ------------------------------------------------------------------
    # Internal pipeline
    # ------------------------------------------------------------------

    def _process_stream(
        self,
        documents: Iterable[Document],
        source_name: str,
    ) -> Iterator[RiskSignal]:
        for doc in documents:
            signal = self._process_one(doc)
            if signal is not None:
                yield signal

    def _process_one(self, doc: Document) -> RiskSignal | None:
        # 1. Clean + language filter
        doc = preprocess(doc)
        if doc is None:
            return None

        # 2. Dedup
        is_dup, novelty = self._dedup.check(doc)
        if is_dup and novelty < 1.0:
            # Still track corroboration but don't emit a new signal
            # (unless this is the first time a source corroborates)
            pass

        # 3. Entity linking
        doc = link_entity(doc)

        # 4. Sentiment
        sent_result = get_sentiment(doc.text, use_finbert=self._use_finbert)

        # 5. Event classification
        event_type, event_conf = classify_event(doc.text)

        # 6. Corroboration
        ts = doc.timestamp if doc.timestamp.tzinfo else doc.timestamp.replace(tzinfo=timezone.utc)
        corroboration_boost = self._corroboration.record(
            doc.ticker, event_type, doc.source, ts
        )

        # 7. Impact score
        impact, explanation = compute_impact(
            event_type=event_type,
            sentiment_score=sent_result["score"],
            source=doc.source,
            novelty=novelty,
            corroboration_boost=corroboration_boost,
            ticker=doc.ticker,
        )

        # 8. Composite confidence
        confidence = round(
            0.6 * sent_result["confidence"] + 0.4 * event_conf, 4
        )

        signal = RiskSignal(
            signal_id=str(uuid.uuid4()),
            timestamp=ts.isoformat(),
            entity={
                "name":    doc.entity_name or doc.ticker or "Unknown",
                "ticker":  doc.ticker or "N/A",
                "sector":  doc.sector  or "Unknown",
                "country": doc.country or "Unknown",
            },
            sentiment_score=sent_result["score"],
            event_type=event_type,
            impact_score=impact,
            confidence=confidence,
            sources=[{
                "source":  doc.source,
                "url":     doc.url or "",
                "snippet": doc.text[:200],
            }],
            explanation={
                **explanation,
                "finbert_score": sent_result["finbert_score"],
                "vader_score":   sent_result["vader_score"],
                "sentiment_model": sent_result["model_used"],
                "event_confidence": event_conf,
            },
            model_version=MODEL_VERSION,
        )

        # 9. Persist to JSONL
        if self._write_jsonl and not is_dup:
            self._append_jsonl(signal)

        # 10. Log high-impact events
        if impact >= self._threshold:
            logger.info(
                "[HIGH IMPACT %.1f] %s | %s | %s",
                impact, signal.entity["ticker"], event_type,
                doc.text[:80],
            )

        return signal

    def _append_jsonl(self, signal: RiskSignal) -> None:
        try:
            SIGNAL_JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(SIGNAL_JSONL_PATH, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(signal.to_dict()) + "\n")
        except Exception as exc:
            logger.error("Failed to write signal JSONL: %s", exc)
