"""
RiskLens – entity linking.

Maintains a curated alias dictionary for ~15 S&P 100 tickers.
Each entry maps company names, common aliases, and cashtags to a
canonical entity record {ticker, name, sector, country}.

Exact alias matching uses word boundaries.
Fuzzy-match fallback searches only against aliases >= 5 characters using
token_set_ratio at cutoff 90 (skipping 1-2 letter bare abbreviations).

ASSUMPTION: the alias dictionary covers ~15 tickers used for the
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
    ("TSLA",  "Tesla Inc.",              "Consumer Disc.",    "US",
     "tesla,tesla inc,tesla motors,tsla"),
    ("JPM",   "JPMorgan Chase & Co.",    "Financials",        "US",
     "jpmorgan,jp morgan,chase,jpm,jpmorgan chase"),
    ("BAC",   "Bank of America Corp.",   "Financials",        "US",
     "bank of america,bofa,bac"),
    ("GS",    "Goldman Sachs Group",     "Financials",        "US",
     "goldman sachs,goldman"),
    ("MS",    "Morgan Stanley",          "Financials",        "US",
     "morgan stanley"),
    ("C",     "Citigroup Inc.",          "Financials",        "US",
     "citigroup,citi,citibank"),
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
     "meta,facebook,meta platforms"),
]

_CASHTAG_RE = re.compile(r"\$([A-Za-z]{1,5})\b")

for _ticker, _name, _sector, _country, _aliases in _RAW_ENTITIES:
    _record = {"ticker": _ticker, "name": _name, "sector": _sector, "country": _country}
    _CANONICAL[_ticker] = _record
    for _alias in _aliases.split(","):
        _alias_clean = _alias.strip().lower()
        if len(_alias_clean) >= 3:
            _ENTITY_MAP[_alias_clean] = _record
    # Only index ticker as standalone word if it is at least 3 characters
    if len(_ticker) >= 3:
        _ENTITY_MAP[_ticker.lower()] = _record

# Exact matching: longest aliases tested first
_SORTED_EXACT_ALIASES = sorted(_ENTITY_MAP.keys(), key=len, reverse=True)

# Fuzzy matching: skip aliases shorter than 5 characters to avoid spurious substring collisions
_FUZZY_ALIASES = [a for a in _ENTITY_MAP.keys() if len(a) >= 5]


def _lookup_cashtag(text: str) -> Optional[dict]:
    """Extract ticker from $CASHTAG pattern."""
    hits = _CASHTAG_RE.findall(text)
    for hit in hits:
        ticker_upper = hit.upper()
        if ticker_upper in _CANONICAL:
            return _CANONICAL[ticker_upper]
    return None


def _lookup_exact(text_lower: str) -> Optional[dict]:
    """Check if any known alias (len >= 3) appears as a whole word in text."""
    for alias in _SORTED_EXACT_ALIASES:
        pattern = r"\b" + re.escape(alias) + r"\b"
        if re.search(pattern, text_lower):
            return _ENTITY_MAP[alias]
    return None


def _lookup_fuzzy(text_lower: str) -> Optional[dict]:
    """Fuzzy match text against long aliases (>=5 chars) with token_set_ratio and cutoff 90."""
    if not _RF_AVAILABLE or not _FUZZY_ALIASES:
        return None
    result = process.extractOne(
        text_lower,
        _FUZZY_ALIASES,
        scorer=fuzz.token_set_ratio,
        score_cutoff=90,
    )
    if result:
        matched_alias = result[0]
        return _ENTITY_MAP[matched_alias]
    return None


def link_entity(doc: Document) -> Document:
    """
    Attempt to resolve the document to a known entity.
    Populates doc.ticker, doc.entity_name, doc.sector, doc.country in place.
    If no confident match is found, leaves them unset (evaluating to 'N/A' / 'Unknown').
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

    # Priority: cashtag > exact word alias > high-confidence fuzzy (>=5 chars)
    rec = _lookup_cashtag(doc.text) or _lookup_exact(text_lower) or _lookup_fuzzy(text_lower)

    if rec:
        doc.ticker      = rec["ticker"]
        doc.entity_name = rec["name"]
        doc.sector      = rec["sector"]
        doc.country     = rec["country"]

    return doc
