"""
RiskLens – unit tests for the impact formula.
Run with: pytest tests/test_impact.py -v
"""
import pytest
from src.engine.impact import compute_impact, BASE_SEVERITY, CorroborationTracker


def test_impact_clipped_to_range():
    """Impact score must always be in [1.0, 10.0]."""
    score, _ = compute_impact(
        event_type="Credit Event",
        sentiment_score=-1.0,
        source="kaggle_news",
        novelty=1.0,
        corroboration_boost=1.5,
        ticker="SIVB",
    )
    assert 1.0 <= score <= 10.0


def test_impact_minimum_noise():
    """Other/Noise with neutral sentiment and low credibility should score low."""
    score, _ = compute_impact(
        event_type="Other/Noise",
        sentiment_score=0.0,
        source="kaggle_tweets",
        novelty=0.3,
        corroboration_boost=1.0,
        ticker=None,
    )
    assert score < 5.0, f"Expected score < 5.0, got {score}"


def test_impact_high_credit_event():
    """High-severity credit event with strong negative sentiment should score high."""
    score, explanation = compute_impact(
        event_type="Credit Event",
        sentiment_score=-0.9,
        source="kaggle_news",
        novelty=1.0,
        corroboration_boost=1.3,
        ticker="SIVB",
    )
    assert score >= 6.0, f"Expected score >= 6.0, got {score}"
    # Explanation must contain all components
    for key in ["base_severity", "sentiment_factor", "credibility", "novelty",
                "corroboration", "entity_materiality"]:
        assert key in explanation


def test_explanation_keys():
    """Explanation dict must have all required keys for traceability."""
    _, explanation = compute_impact(
        event_type="Geopolitical",
        sentiment_score=-0.5,
        source="newsapi",
        novelty=1.0,
        corroboration_boost=1.0,
    )
    required = ["base_severity", "sentiment_factor", "credibility",
                "novelty", "corroboration", "entity_materiality", "raw_score"]
    for key in required:
        assert key in explanation, f"Missing key: {key}"


def test_impact_scales_with_sentiment():
    """Higher |sentiment| should produce higher impact."""
    score_low, _ = compute_impact("Macroeconomic", 0.0, "kaggle_news", 1.0, 1.0)
    score_high, _ = compute_impact("Macroeconomic", -0.9, "kaggle_news", 1.0, 1.0)
    assert score_high > score_low


def test_novelty_reduces_impact():
    """Duplicate (low novelty) should produce lower impact than novel content."""
    score_novel, _ = compute_impact("Credit Event", -0.7, "kaggle_news", 1.0, 1.0)
    score_dup,   _ = compute_impact("Credit Event", -0.7, "kaggle_news", 0.3, 1.0)
    assert score_novel > score_dup


def test_corroboration_tracker():
    """Corroboration boost should increase with multiple independent sources."""
    from datetime import datetime, timezone
    tracker = CorroborationTracker(window_hours=6)
    ts = datetime.now(tz=timezone.utc)

    boost1 = tracker.record("SIVB", "Credit Event", "kaggle_news",   ts)
    boost2 = tracker.record("SIVB", "Credit Event", "kaggle_tweets", ts)
    boost3 = tracker.record("SIVB", "Credit Event", "newsapi",       ts)

    assert boost1 < boost2 < boost3
    assert 1.0 <= boost1 <= 1.5
    assert 1.0 <= boost3 <= 1.5


def test_all_event_types_in_severity_table():
    """Every event type in the taxonomy must have a base severity."""
    from src.engine.classifier import EVENT_TYPES
    for et in EVENT_TYPES:
        assert et in BASE_SEVERITY, f"Missing base severity for: {et}"
