"""
RiskLens – Kaggle financial news headlines adapter.

Expected CSV columns (flexible, mapped below):
  - date / Date / published_at / timestamp
  - headline / title / text / news
  - url / link  (optional)
  - author      (optional)

Place the downloaded CSV at: data/raw_samples/news.csv
"""
from __future__ import annotations

import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Iterator

import pandas as pd

from src.ingestion.base import BaseAdapter
from src.schemas import Document

logger = logging.getLogger(__name__)

# Column name aliases – first match wins
_DATE_COLS = ["date", "Date", "published_at", "timestamp", "PublishDate"]
_TEXT_COLS = ["headline", "title", "text", "news", "Headline", "Title"]
_URL_COLS  = ["url", "link", "URL", "Link"]
_AUTH_COLS = ["author", "Author", "source", "Source"]


def _first(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


class KaggleNewsAdapter(BaseAdapter):
    """Streams financial news headlines from a Kaggle CSV file."""

    def __init__(self, csv_path: Path) -> None:
        self._path = Path(csv_path)

    def name(self) -> str:
        return "kaggle_news"

    def stream(self) -> Iterator[Document]:
        if not self._path.exists():
            logger.warning(
                "News CSV not found at %s – yielding 0 documents. "
                "Download from Kaggle and place at data/raw_samples/news.csv",
                self._path,
            )
            return

        df = pd.read_csv(self._path, low_memory=False)
        date_col = _first(df, _DATE_COLS)
        text_col = _first(df, _TEXT_COLS)
        url_col  = _first(df, _URL_COLS)
        auth_col = _first(df, _AUTH_COLS)

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

            doc_id = hashlib.sha256(f"kaggle_news::{text}".encode()).hexdigest()[:16]

            yield Document(
                id=doc_id,
                source=self.name(),
                timestamp=ts,
                text=text,
                url=str(row[url_col]) if url_col and pd.notna(row.get(url_col)) else None,
                author=str(row[auth_col]) if auth_col and pd.notna(row.get(auth_col)) else None,
                lang="en",
            )
