# RiskLens – Data Directory

## Sources

| Dataset | Source | Status in Repo | License | Location |
|---|---|---|---|---|
| SVB Collapse Replay | Curated synthetic events based on public news (March 8–13, 2023) | **Shipped in repo** (30 records: 18 news, 12 social) | Public / Original | `raw_samples/replay_svb.jsonl` |
| Curated Sentiment Eval | Curated financial headline sample | **Shipped in repo** (60 sentences) | Public / Educational | `raw_samples/phrasebank.csv` |
| Labeled Event Eval | Curated banking risk headline set | **Shipped in repo** (67 headlines, 11 classes) | Public / Educational | `labeled_eval/labels.csv` |
| Synthetic Wholesale Portfolio | Fully synthetic (seed 42) wholesale portfolio | **Shipped in repo** (300 positions) | Synthetic | `portfolio/portfolio.csv` |
| Scenario Shock Library | Documented wholesale risk shocks | **Shipped in repo** (11 risk scenarios) | Documented assumptions | `scenarios/scenarios.yaml` |
| Financial News Headlines | Kaggle (Financial News Headlines) | Optional (adapters ready; requires human Kaggle download) | Public | `raw_samples/news.csv` |
| Stock Tweets | Kaggle (Stock Market Tweets) | Optional (adapters ready; requires human Kaggle download) | Public | `raw_samples/tweets.csv` |
| Price Cache | yfinance market data cache | Optional / Local cache | Public | `raw_samples/prices/` |

## Assumptions & Integrity

1. **Portfolio data is entirely synthetic.** Generated with fixed random seed 42. Distribution shapes are plausible wholesale exposures but not calibrated to any real institution.
2. **SVB replay corpus** contains 30 curated synthetic headlines and tweets inspired by real public events from March 8–13, 2023. Source tags distinguish news (`replay_news`, credibility 0.85) from social commentary (`replay_social`, credibility 0.55).
3. **Kaggle Ingestion Adapters** (`kaggle_news.py`, `kaggle_tweets.py`) are fully implemented and verified, but large multi-thousand-row CSVs require human download due to Kaggle authentication rules.
4. **Scenario shock magnitudes** are documented domain assumptions, not calibrated to empirical abnormal returns. See `docs/ASSUMPTIONS.md`.

## No Proprietary Data

No S&P Global, Crisil, or any client data is used anywhere in this repository.
All data is either synthetic, public-domain, or downloaded by the candidate from public sources.

## How to Download Kaggle Data

1. Create a Kaggle account at kaggle.com
2. Download the datasets and place them at the paths shown in the table above
3. Run `python run.py --precompute` to process them through the engine
