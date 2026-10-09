"""
RiskLens Module A – Tactical Index Rebalancer.

15 S&P 100 stocks, starting equal-weight.
Smoothed sentiment per ticker via exponential moving average + time decay.
Weights: w_i ∝ w0 × exp(k × smoothed_sentiment_i), normalized,
         subject to:
           - min weight: 2%
           - max weight: 15%
           - max deviation from base weight: 10 pp
           - per-step turnover cap: 5%

DISCLAIMER: This is a demonstration of sentiment-based signal integration,
NOT a real trading strategy. Backtest results are reported with full caveats.
See docs/ASSUMPTIONS.md.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Optional

# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------
UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "JPM",
    "BAC",  "GS",   "MS",    "C",    "WFC",
    "XOM",  "CVX",  "JNJ",   "META", "SIVB",
]

N = len(UNIVERSE)
BASE_WEIGHT = 1.0 / N            # equal-weight baseline

# ---------------------------------------------------------------------------
# Parameters (ASSUMPTION)
# ---------------------------------------------------------------------------
K          = 0.5    # sensitivity of weights to sentiment
EMA_ALPHA  = 0.3    # EMA decay per observation
DECAY_HALF = 24     # hours; sentiment half-life for time decay
MIN_W      = 0.02   # 2% floor
MAX_W      = 0.15   # 15% cap
MAX_DEV    = 0.10   # max abs deviation from base weight
TURNOVER_CAP = 0.05 # per-step turnover as fraction of portfolio


class Rebalancer:
    """
    Maintains smoothed sentiment per ticker and computes constrained weights.
    """

    def __init__(self) -> None:
        self._ema: dict[str, float]      = {t: 0.0 for t in UNIVERSE}
        self._last_ts: dict[str, Optional[pd.Timestamp]] = {t: None for t in UNIVERSE}
        self._prev_weights: dict[str, float] = {t: BASE_WEIGHT for t in UNIVERSE}
        self._history: list[dict] = []   # {timestamp, weights, sentiments}

    # ------------------------------------------------------------------

    def update(self, ticker: str, sentiment_score: float, timestamp: pd.Timestamp) -> None:
        """
        Update the EMA for one ticker with a new sentiment observation.
        Applies exponential time decay since the last update.
        """
        if ticker not in UNIVERSE:
            return

        last = self._last_ts[ticker]
        if last is not None:
            hours_elapsed = (timestamp - last).total_seconds() / 3600
            decay = np.exp(-np.log(2) * hours_elapsed / DECAY_HALF)
            self._ema[ticker] = decay * self._ema[ticker] + (1 - decay * EMA_ALPHA) * sentiment_score
        else:
            self._ema[ticker] = sentiment_score

        self._last_ts[ticker] = timestamp

    def compute_weights(self, timestamp: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        """
        Compute constrained portfolio weights from current EMA sentiments.
        Returns a DataFrame with {ticker, sentiment, raw_weight, final_weight}.
        """
        sentiments = np.array([self._ema[t] for t in UNIVERSE])

        # Raw weights: w ∝ exp(k × sentiment)
        raw = np.array([BASE_WEIGHT * np.exp(K * s) for s in sentiments])
        raw_norm = raw / raw.sum()

        # Apply constraints iteratively (simple projection)
        w = raw_norm.copy()
        for _ in range(20):
            # Floor/cap
            w = np.clip(w, MIN_W, MAX_W)
            # Max deviation from base
            w = np.clip(w, BASE_WEIGHT - MAX_DEV, BASE_WEIGHT + MAX_DEV)
            # Renormalize
            w = w / w.sum()

        # Turnover cap
        prev = np.array([self._prev_weights[t] for t in UNIVERSE])
        turnover = np.abs(w - prev).sum() / 2
        if turnover > TURNOVER_CAP:
            blend = TURNOVER_CAP / turnover
            w = prev + blend * (w - prev)
            w = w / w.sum()

        self._prev_weights = {t: float(w[i]) for i, t in enumerate(UNIVERSE)}

        df = pd.DataFrame({
            "ticker":       UNIVERSE,
            "sentiment":    [round(float(s), 4) for s in sentiments],
            "raw_weight":   [round(float(r), 4) for r in raw_norm],
            "final_weight": [round(float(wi), 4) for wi in w],
            "tilt_vs_base": [round(float(wi - BASE_WEIGHT), 4) for wi in w],
        })

        if timestamp is not None:
            record = {"timestamp": timestamp, "weights": dict(zip(UNIVERSE, w.tolist()))}
            self._history.append(record)

        return df

    def get_history_df(self) -> pd.DataFrame:
        """Return weight history as a wide DataFrame for the stacked-area chart."""
        if not self._history:
            return pd.DataFrame()
        rows = []
        for rec in self._history:
            row = {"timestamp": rec["timestamp"]}
            row.update(rec["weights"])
            rows.append(row)
        return pd.DataFrame(rows).set_index("timestamp")
