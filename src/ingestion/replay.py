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
                source=self.name(),
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
