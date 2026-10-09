"""
RiskLens – base adapter interface.
Every source adapter must implement this ABC so downstream code
can treat all sources identically.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator

from src.schemas import Document


class BaseAdapter(ABC):
    """Abstract base for all ingestion adapters."""

    @abstractmethod
    def stream(self) -> Iterator[Document]:
        """Yield Document records one at a time."""
        ...

    @abstractmethod
    def name(self) -> str:
        """Human-readable source name, e.g. 'kaggle_news'."""
        ...
