"""
RiskLens – synthetic wholesale portfolio generator.

~300 positions across corporate loans, bonds, IR derivatives, and equity.
FULLY SYNTHETIC with fixed random seed.  Labeled synthetic everywhere.

IMPORTANT: This data is entirely made up for demonstration purposes.
It does not represent any real portfolio, institution, or client.
See docs/ASSUMPTIONS.md for all assumptions baked into this data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from src.config import PORTFOLIO_DIR

RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

# ---------------------------------------------------------------------------
# Universe parameters (ASSUMPTION – distribution shapes are plausible
# but not calibrated to any real data)
# ---------------------------------------------------------------------------
SECTORS   = ["Financials", "Energy", "Technology", "Healthcare", "Industrials",
              "Consumer Disc.", "Materials", "Utilities", "Real Estate", "Telecom"]
COUNTRIES = ["US", "UK", "DE", "FR", "JP", "CN", "IN", "BR", "EM"]
RATINGS   = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC"]
RATING_PD = {"AAA": 0.0001, "AA": 0.0002, "A": 0.0005,
             "BBB": 0.002,  "BB": 0.01,   "B": 0.04,  "CCC": 0.15}
RATING_SPREAD_BPS = {"AAA": 20, "AA": 40, "A": 80,
                     "BBB": 150, "BB": 300, "B": 500, "CCC": 900}


def _sample(choices: list, n: int, p=None) -> list:
    return list(rng.choice(choices, size=n, p=p))


def _generate_loans(n: int = 120) -> pd.DataFrame:
    ratings  = _sample(RATINGS, n, p=[0.02, 0.05, 0.12, 0.30, 0.25, 0.18, 0.08])
    sectors  = _sample(SECTORS, n)
    countries = _sample(COUNTRIES, n, p=[0.45, 0.10, 0.08, 0.07, 0.06, 0.06, 0.06, 0.06, 0.06])
    ead      = rng.uniform(1e6, 50e6, n)
    lgd      = rng.uniform(0.30, 0.65, n)
    maturity = rng.uniform(1, 7, n)
    spread   = np.array([RATING_SPREAD_BPS[r] + rng.normal(0, 20) for r in ratings])
    pd_arr   = np.array([RATING_PD[r] * rng.uniform(0.8, 1.2) for r in ratings])

    return pd.DataFrame({
        "asset_class": "Corporate Loan",
        "position_id": [f"LN{i:04d}" for i in range(n)],
        "rating":   ratings,
        "sector":   sectors,
        "country":  countries,
        "ead":      np.round(ead, 0),
        "lgd":      np.round(lgd, 3),
        "pd":       np.round(pd_arr, 5),
        "spread_bps": np.round(spread, 1),
        "maturity_years": np.round(maturity, 2),
        "mark_to_market": np.round(ead * (1 - pd_arr * lgd), 0),  # simplified
        "synthetic": True,
    })


def _generate_bonds(n: int = 100) -> pd.DataFrame:
    ratings  = _sample(RATINGS, n, p=[0.05, 0.10, 0.20, 0.35, 0.15, 0.10, 0.05])
    sectors  = _sample(SECTORS, n)
    face_val = rng.uniform(500e3, 20e6, n)
    duration = rng.uniform(1.5, 10.0, n)
    convexity = duration ** 1.5 * 0.1 + rng.normal(0, 0.5, n)
    coupon   = np.array([RATING_SPREAD_BPS[r] / 10000 + 0.03 + rng.normal(0, 0.005) for r in ratings])
    spread   = np.array([RATING_SPREAD_BPS[r] + rng.normal(0, 15) for r in ratings])

    return pd.DataFrame({
        "asset_class": "Bond",
        "position_id": [f"BD{i:04d}" for i in range(n)],
        "rating":   ratings,
        "sector":   sectors,
        "country":  _sample(COUNTRIES, n),
        "face_value": np.round(face_val, 0),
        "coupon":   np.round(coupon, 4),
        "duration": np.round(duration, 2),
        "convexity": np.round(np.clip(convexity, 0.1, 200), 2),
        "spread_bps": np.round(spread, 1),
        "mark_to_market": np.round(face_val * (1 - 0.01 * rng.uniform(0, 5, n)), 0),
        "synthetic": True,
    })


def _generate_derivatives(n: int = 40) -> pd.DataFrame:
    """IR swaps (pay fixed) with DV01."""
    notional = rng.uniform(5e6, 100e6, n)
    dv01     = notional * rng.uniform(0.00005, 0.0002, n)  # per bp move

    return pd.DataFrame({
        "asset_class": "IR Swap",
        "position_id": [f"DR{i:04d}" for i in range(n)],
        "rating":   "N/A",
        "sector":   "Financials",
        "country":  _sample(["US", "UK", "DE", "FR", "JP"], n),
        "notional": np.round(notional, 0),
        "dv01":     np.round(dv01, 1),
        "mark_to_market": np.round(rng.uniform(-2e6, 2e6, n), 0),
        "synthetic": True,
    })


def _generate_equities(n: int = 40) -> pd.DataFrame:
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM", "BAC", "GS",
               "MS", "C", "WFC", "XOM", "CVX", "JNJ", "META", "SIVB"]
    selected = rng.choice(tickers, size=n, replace=True)
    market_val = rng.uniform(200e3, 5e6, n)
    beta       = rng.uniform(0.6, 1.8, n)

    return pd.DataFrame({
        "asset_class": "Equity",
        "position_id": [f"EQ{i:04d}" for i in range(n)],
        "rating":   "N/A",
        "sector":   _sample(SECTORS, n),
        "country":  _sample(["US", "UK", "DE", "JP"], n),
        "ticker":   selected,
        "market_value": np.round(market_val, 0),
        "beta":     np.round(beta, 2),
        "mark_to_market": np.round(market_val, 0),
        "synthetic": True,
    })


def generate_portfolio(save: bool = True) -> pd.DataFrame:
    """Generate and optionally save the synthetic portfolio."""
    loans  = _generate_loans(120)
    bonds  = _generate_bonds(100)
    derivs = _generate_derivatives(40)
    equities = _generate_equities(40)

    portfolio = pd.concat([loans, bonds, derivs, equities], ignore_index=True)
    portfolio["mark_to_market"] = portfolio["mark_to_market"].fillna(0)

    if save:
        out = PORTFOLIO_DIR / "portfolio.csv"
        PORTFOLIO_DIR.mkdir(parents=True, exist_ok=True)
        portfolio.to_csv(out, index=False)
        print(f"[SYNTHETIC] Portfolio saved to {out}  ({len(portfolio)} positions)")

    return portfolio


def load_portfolio() -> pd.DataFrame:
    """Load portfolio from CSV; generate if not present."""
    path = PORTFOLIO_DIR / "portfolio.csv"
    if not path.exists():
        return generate_portfolio(save=True)
    return pd.read_csv(path)
