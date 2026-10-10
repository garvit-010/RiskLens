"""
Unit tests for entity linking (src/engine/entity_linker.py).
Asserts that 1-2 letter bare aliases do not trigger false positives,
cashtags resolve properly, and known entities are linked accurately.
"""
from datetime import datetime, timezone
import pytest
from src.schemas import Document
from src.engine.entity_linker import link_entity


def _make_doc(text: str) -> Document:
    return Document(
        id="test-doc-1",
        source="news",
        timestamp=datetime.now(tz=timezone.utc),
        text=text,
    )


def test_no_false_positive_on_cramer():
    doc = _make_doc("Jim Cramer warns of trade war")
    linked = link_entity(doc)
    assert linked.ticker is None
    assert linked.entity_name is None


def test_tesla_fsd_update():
    doc = _make_doc("Tesla FSD update announced for global markets")
    linked = link_entity(doc)
    assert linked.ticker == "TSLA"
    assert linked.entity_name == "Tesla Inc."
    assert linked.sector == "Consumer Disc."


def test_citigroup_downgrades():
    doc = _make_doc("Citigroup downgrades European banks following credit review")
    linked = link_entity(doc)
    assert linked.ticker == "C"
    assert linked.entity_name == "Citigroup Inc."
    assert linked.sector == "Financials"


def test_cashtag_ms():
    doc = _make_doc("$MS reports strong wealth management revenue beat")
    linked = link_entity(doc)
    assert linked.ticker == "MS"
    assert linked.entity_name == "Morgan Stanley"
    assert linked.sector == "Financials"


def test_cashtag_c():
    doc = _make_doc("Investors accumulate $C ahead of quarterly earnings")
    linked = link_entity(doc)
    assert linked.ticker == "C"
    assert linked.entity_name == "Citigroup Inc."


def test_morgan_stanley_full_phrase():
    doc = _make_doc("Morgan Stanley warns of potential $30B hit to banking sector from commercial real estate")
    linked = link_entity(doc)
    assert linked.ticker == "MS"
    assert linked.entity_name == "Morgan Stanley"


def test_generic_market_news_no_ticker():
    doc = _make_doc("Bond yields are falling and that could stunt bank profits across the broader market")
    linked = link_entity(doc)
    assert linked.ticker is None
