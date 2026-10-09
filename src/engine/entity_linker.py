"""
RiskLens – entity linking.

Maintains a curated alias dictionary for ~15 S&P 100 tickers.
Each entry maps company names, common aliases, and cashtags to a
canonical entity record {ticker, name, sector, country}.

Fuzzy-match fallback: if no exact match, try rapidfuzz partial_ratio
on the document text against all known aliases.

ASSUMPTION: the alias dictionary covers only ~15 tickers used for the
demo. A production system would integrate with a financial entity DB
(e.g., FactSet, Bloomberg symbology). See docs/ASSUMPTIONS.md.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

from src.schemas import Document

logger = logging.getLogger(__name__)

try:
    from rapidfuzz import process, fuzz
    _RF_AVAILABLE = True
except ImportError:
    _RF_AVAILABLE = False


# ---------------------------------------------------------------------------
# Curated entity registry
# ---------------------------------------------------------------------------
# Format: alias (lower-case) → {ticker, name, sector, country}
_ENTITY_MAP: dict[str, dict] = {}
_CANONICAL: dict[str, dict] = {}   # ticker → full record

_RAW_ENTITIES = [
    # ticker, name, sector, country, aliases (comma-separated)
    ("AAPL",  "Apple Inc.",              "Technology",        "US",
     "apple,apple inc,aapl"),
    ("MSFT",  "Microsoft Corporation",   "Technology",        "US",
     "microsoft,microsoft corp,msft"),
    ("GOOGL", "Alphabet Inc.",           "Technology",        "US",
     "google,alphabet,googl,goog"),
    ("AMZN",  "Amazon.com Inc.",         "Consumer Disc.",    "US",
     "amazon,amazon.com,amzn"),
    ("JPM",   "JPMorgan Chase & Co.",    "Financials",        "US",
     "jpmorgan,jp morgan,chase,jpm,jpmorgan chase"),
    ("BAC",   "Bank of America Corp.",   "Financials",        "US",
     "bank of america,bofa,bac"),
    ("GS",    "Goldman Sachs Group",     "Financials",        "US",
     "goldman sachs,goldman,gs"),
    ("MS",    "Morgan Stanley",          "Financials",        "US",
     "morgan stanley,ms"),
    ("C",     "Citigroup Inc.",          "Financials",        "US",
     "citigroup,citi,citibank,c"),
    ("WFC",   "Wells Fargo & Co.",       "Financials",        "US",
     "wells fargo,wfc"),
    ("SIVB",  "SVB Financial Group",     "Financials",        "US",
     "svb,silicon valley bank,svb financial,sivb"),
    ("XOM",   "Exxon Mobil Corp.",       "Energy",            "US",
     "exxon,exxon mobil,xom"),
    ("CVX",   "Chevron Corporation",     "Energy",            "US",
     "chevron,cvx"),
    ("JNJ",   "Johnson & Johnson",       "Healthcare",        "US",
     "johnson and johnson,j&j,jnj"),
    ("META",  "Meta Platforms Inc.",     "Technology",        "US",
     "meta,facebook,fb,meta platforms,meta"),
]

_CASHTAG_RE = re.compile(r"\$([A-Z]{1,5})")

for _ticker, _name, _sector, _country, _aliases in _RAW_ENTITIES:
    _record = {"ticker": _ticker, "name": _name, "sector": _sector, "country": _country}
    _CANONICAL[_ticker] = _record
    for _alias in _aliases.split(","):
        _ENTITY_MAP[_alias.strip().lower()] = _record
    # also index the ticker itself
    _ENTITY_MAP[_ticker.lower()] = _record

_ALL_ALIASES = list(_ENTITY_MAP.keys())


def _lookup_exact(text_lower: str) -> Optional[dict]:
    """Check if any alias appears as a word in the text."""
    for alias, record in _ENTITY_MAP.items():
        if re.search(r"\b" + re.escape(alias) + r"\b", text_lower):
            return record
    return None


def _lookup_cashtag(text: str) -> Optional[dict]:
    """Extract ticker from $CASHTAG pattern."""
    hits = _CASHTAG_RE.findall(text)
    for hit in hits:
        if hit in _CANONICAL:
            return _CANONICAL[hit]
    return None


def _lookup_fuzzy(text_lower: str) -> Optional[dict]:
    """Fuzzy match text against alias list (threshold 80)."""
    if not _RF_AVAILABLE:
        return None
    result = process.extractOne(
        text_lower, _ALL_ALIASES,
        scorer=fuzz.partial_ratio,
        score_cutoff=80,
    )
    if result:
        return _ENTITY_MAP[result[0]]
    return None


def link_entity(doc: Document) -> Document:
    """
    Attempt to resolve the document to a known entity.
    Populates doc.ticker, doc.entity_name, doc.sector, doc.country in place.
    Returns the doc (whether linked or not).
    """
    # If the doc already carries a ticker from the adapter, check canonical first
    if doc.ticker and doc.ticker in _CANONICAL:
        rec = _CANONICAL[doc.ticker]
        doc.entity_name = rec["name"]
        doc.sector      = rec["sector"]
        doc.country     = rec["country"]
        return doc

    text_lower = doc.text.lower()

    # Priority: cashtag > exact alias > fuzzy
    rec = _lookup_cashtag(doc.text) or _lookup_exact(text_lower) or _lookup_fuzzy(text_lower)

    if rec:
        doc.ticker      = rec["ticker"]
        doc.entity_name = rec["name"]
        doc.sector      = rec["sector"]
        doc.country     = rec["country"]

    return doc
