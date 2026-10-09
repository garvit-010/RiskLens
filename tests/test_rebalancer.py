"""
RiskLens – unit tests for Module A Tactical Rebalancer.
Run with: pytest tests/test_rebalancer.py -v
"""
import pandas as pd
import numpy as np
import pytest

from src.modules.rebalancer.rebalancer import Rebalancer, UNIVERSE, BASE_WEIGHT, MIN_W, MAX_W, MAX_DEV

def test_rebalancer_initialization():
    reb = Rebalancer()
    weights_df = reb.compute_weights()
    assert len(weights_df) == len(UNIVERSE)
    # Initial weights should sum to 1.0
    assert np.isclose(weights_df["final_weight"].sum(), 1.0, atol=1e-3)

def test_rebalancer_sentiment_tilt():
    reb = Rebalancer()
    now = pd.Timestamp("2026-10-09 10:00:00")
    # Positive sentiment for AAPL, negative for SIVB
    reb.update("AAPL", 0.9, now)
    reb.update("SIVB", -0.9, now)
    df = reb.compute_weights(now)

    aapl_w = df.loc[df["ticker"] == "AAPL", "final_weight"].values[0]
    sivb_w = df.loc[df["ticker"] == "SIVB", "final_weight"].values[0]

    assert aapl_w > BASE_WEIGHT
    assert sivb_w < BASE_WEIGHT
    assert aapl_w > sivb_w

def test_rebalancer_bounds_constraints():
    reb = Rebalancer()
    now = pd.Timestamp("2026-10-09 10:00:00")
    # Extreme sentiment
    for t in UNIVERSE:
        reb.update(t, 1.0 if t == "AAPL" else -1.0, now)
    df = reb.compute_weights(now)

    assert (df["final_weight"] >= MIN_W - 1e-4).all()
    assert (df["final_weight"] <= MAX_W + 1e-4).all()
    assert np.isclose(df["final_weight"].sum(), 1.0, atol=1e-3)
