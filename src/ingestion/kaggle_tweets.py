"""
RiskLens – Kaggle stock tweets adapter.

Expected CSV columns (flexible):
  - Date / date / created_at / timestamp
  - Tweet / tweet / text / body
  - Ticker / ticker / stock / symbol  (optional cashtag column)
  - url / link  (optional)

Place the downloaded CSV at: data/raw_samples/tweets.csv
"""
from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Iterator

import pandas as pd

from src.ingestion.base import BaseAdapter
from src.schemas import Document

logger = logging.getLogger(__name__)

_DATE_COLS   = ["Date", "date", "created_at", "timestamp", "time"]
_TEXT_COLS   = ["Tweet", "tweet", "text", "body", "content"]
_TICKER_COLS = ["Ticker", "ticker", "stock", "symbol", "Stock"]


def _first(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


_CASHTAG_RE = re.compile(r"\$([A-Z]{1,5})")


class KaggleTweetsAdapter(BaseAdapter):
    """Streams stock tweets from a Kaggle CSV file."""

    def __init__(self, csv_path: Path) -> None:
        self._path = Path(csv_path)

    def name(self) -> str:
        return "kaggle_tweets"

    def stream(self) -> Iterator[Document]:
        if not self._path.exists():
            logger.warning(
                "Tweets CSV not found at %s – yielding 0 documents. "
                "Download from Kaggle and place at data/raw_samples/tweets.csv",
                self._path,
            )
            return

        df = pd.read_csv(self._path, low_memory=False)
        date_col   = _first(df, _DATE_COLS)
        text_col   = _first(df, _TEXT_COLS)
        ticker_col = _first(df, _TICKER_COLS)

        if text_col is None:
            logger.error("Could not find a text column in %s (columns: %s)", self._path, list(df.columns))
            return

        for _, row in df.iterrows():
            text = str(row[text_col]).strip()
            if not text or text.lower() in {"nan", "none"}:
                continue

            ts = datetime.utcnow()
            if date_col:
                try:
                    ts = pd.to_datetime(row[date_col], utc=True).to_pydatetime()
                except Exception:
                    pass

            doc_id = hashlib.sha256(f"kaggle_tweets::{text}".encode()).hexdigest()[:16]

            # Extract ticker from dedicated column or from cashtag in text
            ticker: str | None = None
            if ticker_col and pd.notna(row.get(ticker_col)):
                ticker = str(row[ticker_col]).upper().strip()
            else:
                hits = _CASHTAG_RE.findall(text)
                if hits:
                    ticker = hits[0]

            yield Document(
                id=doc_id,
                source=self.name(),
                timestamp=ts,
                text=text,
                ticker=ticker,
                lang="en",
            )
