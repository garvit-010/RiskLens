"""
RiskLens – NewsAPI adapter STUB.

This class has the correct interface so the engine can accept it,
but it does NOT make real API calls. It raises NotImplementedError
so the absence of a key is immediately obvious.

TODO (next steps): implement _fetch_page() using the requests library
and the NEWSAPI_KEY environment variable.
"""
from __future__ import annotations

import logging
from typing import Iterator

from src.ingestion.base import BaseAdapter
from src.schemas import Document

logger = logging.getLogger(__name__)


class NewsAPIAdapter(BaseAdapter):
    """
    STUB adapter for the NewsAPI live source.

    Not implemented – see TODO above. Listed in README as a
    known next step.
    """

    def __init__(self, api_key: str = "") -> None:
        self._key = api_key

    def name(self) -> str:
        return "newsapi"

    def stream(self) -> Iterator[Document]:
        raise NotImplementedError(
            "NewsAPIAdapter is a stub. "
            "Set NEWSAPI_KEY in .env and implement _fetch_page() to enable live ingestion. "
            "Run without --live to use the replay adapter instead."
        )
