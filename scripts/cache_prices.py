"""
RiskLens – Cache historical price CSVs for Module A backtesting.

Usage:
    python scripts/cache_prices.py

Fetches daily Close/Adj Close for the 15 universe tickers plus SPY
covering the Q1 2023 SVB replay period and saves them to data/raw_samples/prices/.
"""
from pathlib import Path
import sys

import yfinance as yf
import pandas as pd

from src.modules.rebalancer.rebalancer import UNIVERSE
from src.config import RAW_DIR

PRICES_DIR = RAW_DIR / "prices"

TICKERS = list(UNIVERSE) + ["SPY"]
START_DATE = "2023-01-01"
END_DATE = "2023-04-30"


def cache_prices() -> None:
    PRICES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Fetching prices for {len(TICKERS)} tickers: {TICKERS} ({START_DATE} to {END_DATE}) ...")

    success_count = 0
    for ticker in TICKERS:
        try:
            print(f"Downloading {ticker} ...", end=" ", flush=True)
            df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
            if not df.empty:
                out_path = PRICES_DIR / f"{ticker}.csv"
                df.to_csv(out_path)
                print(f"OK ({len(df)} rows -> {out_path.name})")
                success_count += 1
            else:
                print("EMPTY (skipped)")
        except Exception as exc:
            print(f"FAILED: {exc}")

    print(f"\nDone: {success_count}/{len(TICKERS)} price files cached to {PRICES_DIR}")


if __name__ == "__main__":
    cache_prices()
