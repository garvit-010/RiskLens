"""
RiskLens – deterministic replay adapter.

Reads a pre-processed JSONL file from /data/raw_samples/replay_*.jsonl
and emits Document records in timestamp order, optionally at an
accelerated rate for demo purposes.

This is the demo backbone:  python run.py  uses this adapter so the
demo is fast, GPU-free, and reproducible.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from src.ingestion.base import BaseAdapter
from src.schemas import Document

logger = logging.getLogger(__name__)


class ReplayAdapter(BaseAdapter):
    """
    Replays cached documents from a JSONL file.

    Parameters
    ----------
    jsonl_path : Path
        Path to the replay JSONL file.
    speed_factor : float
        1.0 = real-time gaps preserved; 0 = emit as fast as possible (demo mode).
    """

    def __init__(self, jsonl_path: Path, speed_factor: float = 0.0) -> None:
        self._path = Path(jsonl_path)
        self._speed = speed_factor

    def name(self) -> str:
        return "replay"

    def stream(self) -> Iterator[Document]:
        if not self._path.exists():
            logger.warning("Replay file not found: %s", self._path)
            return

        records: list[dict] = []
        with open(self._path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        # Sort by timestamp
        records.sort(key=lambda r: r.get("timestamp", ""))

        prev_ts: datetime | None = None
        for rec in records:
            ts_raw = rec.get("timestamp", "")
            try:
                ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
            except Exception:
                ts = datetime.now(tz=timezone.utc)

            # Optional real-time delay simulation
            if self._speed > 0 and prev_ts is not None:
                gap = (ts - prev_ts).total_seconds()
                if gap > 0:
                    time.sleep(gap / self._speed)
            prev_ts = ts

            yield Document(
                id=rec.get("id", ""),
                source=rec.get("source", self.name()),
                timestamp=ts,
                text=rec.get("text", ""),
                url=rec.get("url"),
                author=rec.get("author"),
                lang=rec.get("lang", "en"),
                ticker=rec.get("ticker"),
                entity_name=rec.get("entity_name"),
                sector=rec.get("sector"),
                country=rec.get("country"),
            )


# ---------------------------------------------------------------------------
# Replay Data Generator Helper
# ---------------------------------------------------------------------------
_REPLAY_EVENTS = [
    ("2023-03-08T22:00:00+00:00", "replay_news",
     "SVB Financial Group sells $21B bond portfolio at $1.8B loss and seeks emergency $2.25B capital raise", "SIVB"),
    ("2023-03-08T23:00:00+00:00", "replay_social",
     "$SIVB SVB just dumped their bond portfolio at a huge loss. This is bad. Really bad.", "SIVB"),
    ("2023-03-09T12:00:00+00:00", "replay_news",
     "Silicon Valley Bank CEO urges venture capital clients to remain calm as deposit flight rumors circulate", "SIVB"),
    ("2023-03-09T13:00:00+00:00", "replay_news",
     "Peter Thiel's Founders Fund and top venture firms advise portfolio companies to withdraw deposits from SVB", "SIVB"),
    ("2023-03-09T14:00:00+00:00", "replay_social",
     "VCs telling all their portcos to pull money from $SIVB right now. Classic bank run dynamics.", "SIVB"),
    ("2023-03-09T15:00:00+00:00", "replay_news",
     "SVB Financial stock plunges 60% as bank run fears intensify among tech clients", "SIVB"),
    ("2023-03-09T16:00:00+00:00", "replay_social",
     "$SIVB halted multiple times down 60%. Clients trying to pull tens of billions today. Contagion spreading.", "SIVB"),
    ("2023-03-09T20:00:00+00:00", "replay_news",
     "Silicon Valley Bank clients attempt to withdraw $42B in single day, leaving lender with negative cash balance", "SIVB"),
    ("2023-03-10T16:00:00+00:00", "replay_news",
     "FDIC seizes Silicon Valley Bank in largest US bank failure since 2008 financial crisis", "SIVB"),
    ("2023-03-10T17:00:00+00:00", "replay_social",
     "SVB is gone. FDIC has taken over. $SIVB stock halted indefinitely. Regional banks getting hammered $BAC $JPM", "SIVB"),
    ("2023-03-10T19:00:00+00:00", "replay_social",
     "$JPM $BAC $WFC down 3-5% on SVB contagion fears. Watching credit spreads widen in real time.", "JPM"),
    ("2023-03-12T22:00:00+00:00", "replay_news",
     "Federal regulators guarantee all SVB deposits including uninsured amounts above $250,000", "SIVB"),
    ("2023-03-12T22:30:00+00:00", "replay_news",
     "Signature Bank shut down by New York regulators citing systemic risk concerns", None),
    ("2023-03-12T23:00:00+00:00", "replay_news",
     "Federal Reserve launches Bank Term Funding Program to provide emergency liquidity to eligible banks", None),
    ("2023-03-12T23:30:00+00:00", "replay_social",
     "Fed creating BTFP to backstop banking sector liquidity. Rate path now uncertain $GS $MS", "GS"),
    ("2023-03-13T12:00:00+00:00", "replay_news",
     "Moody's downgrades US banking sector outlook to negative following SVB collapse", None),
    ("2023-03-13T14:00:00+00:00", "replay_news",
     "JPMorgan and Bank of America see surge in deposit inflows as customers flee regional lenders", "JPM"),
    ("2023-03-13T16:00:00+00:00", "replay_news",
     "President Biden assures public that US banking system is safe and regulators will hold executives accountable", None),
    ("2023-03-14T14:00:00+00:00", "replay_news",
     "Goldman Sachs cuts US economic growth forecast citing tightening bank lending standards post-SVB", "GS"),
    ("2023-03-14T17:00:00+00:00", "replay_news",
     "Morgan Stanley warns of potential hit to banking sector from commercial real estate exposure", "MS"),
    ("2023-03-14T19:00:00+00:00", "replay_social",
     "Commercial real estate is the next shoe to drop post-SVB. Watching $WFC $BAC closely.", "WFC"),
    ("2023-03-15T10:00:00+00:00", "replay_news",
     "Credit Suisse shares plunge 24% as Saudi National Bank rules out further financial assistance", None),
    ("2023-03-15T15:00:00+00:00", "replay_social",
     "Credit Suisse CDS spreads hitting record distress levels. European banks under pressure.", None),
    ("2023-03-16T13:00:00+00:00", "replay_news",
     "European Central Bank raises interest rates by 50 basis points despite banking turmoil", None),
    ("2023-03-16T15:00:00+00:00", "replay_social",
     "ECB delivers 50bps hike signaling inflation focus despite banking volatility. Hawkish stance.", None),
    ("2023-03-16T19:00:00+00:00", "replay_news",
     "First Republic Bank receives $30B uninsured deposit rescue from coalition of 11 major US banks", None),
    ("2023-03-16T20:00:00+00:00", "replay_social",
     "First Republic getting $30B deposit lifeline from JPMorgan, BofA, Citi. Systemic relief rally $JPM $BAC", "BAC"),
    ("2023-03-19T18:00:00+00:00", "replay_news",
     "UBS agrees to acquire Credit Suisse for $3.2B in emergency Swiss government-brokered rescue", None),
    ("2023-03-19T20:00:00+00:00", "replay_social",
     "UBS buying Credit Suisse for pennies on the dollar with Swiss SNB backstop. Crisis averted for now.", None),
    ("2023-03-21T14:00:00+00:00", "replay_news",
     "Treasury Secretary Yellen tells bankers that deposit guarantees could be expanded if smaller banks face runs", None),
]


def generate_svb_replay(output_path: Path | None = None) -> None:
    """Generate the deterministic SVB replay JSONL file."""
    import hashlib
    from src.config import RAW_DIR

    out_file = output_path or (RAW_DIR / "replay_svb.jsonl")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for ts_iso, source, text, ticker in _REPLAY_EVENTS:
        doc_id = hashlib.sha256(f"{source}::{text}".encode()).hexdigest()[:16]
        rec = {
            "id":        doc_id,
            "source":    source,
            "timestamp": ts_iso,
            "text":      text,
            "ticker":    ticker,
            "lang":      "en",
        }
        records.append(rec)

    with open(out_file, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
    logger.info("SVB replay file written to %s (%d records)", out_file, len(records))
