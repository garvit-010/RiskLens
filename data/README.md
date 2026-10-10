# RiskLens – Data Directory

## Sources

| Dataset | Source | Status in Repo | License | Location |
|---|---|---|---|---|
| SVB Collapse Replay | Curated chronological events based on public news (March 8–21, 2023) | **Shipped in repo** (30 records: 19 news, 11 social) | Public / Original | `raw_samples/replay_svb.jsonl` |
| Financial PhraseBank | Official 100% annotator agreement subset (Malo et al.) | **Shipped in repo** (2,264 sentences) | CC BY-SA 3.0 / Research | `raw_samples/phrasebank.csv` |
| Curated Sentiment Eval | Curated financial headline sample | **Shipped in repo** (60 sentences, 95% CI ±6 pp) | Public / Educational | `raw_samples/sentiment_sample_n60.csv` |
| Labeled Event Eval | Curated banking risk headline reference set | **Shipped in repo** (67 headlines, 11 classes, evaluated via out-of-fold stratified CV) | Public / Educational | `labeled_eval/labels.csv` |
| Synthetic Wholesale Portfolio | Fully synthetic (seed 42) wholesale portfolio | **Shipped in repo** (300 positions) | Synthetic | `portfolio/portfolio.csv` |
| Scenario Shock Library | Documented wholesale risk shocks | **Shipped in repo** (11 risk scenarios) | Documented assumptions | `scenarios/scenarios.yaml` |
| Financial News Sample | CNBC Market News Headlines subset | **Shipped in repo** (500 sampled headlines) | Public / Educational | `raw_samples/news.csv` |
| Stock Tweets Sample | Stock Market Tweets subset | **Shipped in repo** (300 sampled tweets) | Public / Educational | `raw_samples/tweets.csv` |
| Price Cache | Historical daily prices (AAPL, TSLA, SIVB, etc.) | **Shipped in repo** (25 tickers for offline backtest) | Public / Yahoo Finance | `raw_samples/prices/` |
| Precomputed Signals | Full neural engine replay output | **Shipped in repo** (820 signals: 30 SVB replay [19 news, 11 social], 496 CNBC news, 294 stock tweets) | Computed | `output/signals.jsonl` |

## Assumptions & Integrity

1. **Portfolio data is entirely synthetic.** Generated with fixed random seed 42. Distribution shapes are plausible wholesale exposures but not calibrated to any real institution.
2. **SVB replay corpus** contains 30 curated chronological headlines and tweets based on verified public events from March 8–21, 2023. Source tags distinguish news (`replay_news`, credibility 0.85) from social commentary (`replay_social`, credibility 0.55).
3. **Kaggle Ingestion Adapters** (`kaggle_news.py`, `kaggle_tweets.py`) are fully implemented and verified. Curated 500-news and 300-tweet samples are shipped in repo for complete offline operation without requiring manual Kaggle downloads.
4. **Scenario shock magnitudes** are documented domain assumptions, not calibrated to empirical abnormal returns. See `docs/ASSUMPTIONS.md`.

## No Proprietary Data

No S&P Global, Crisil, or any client data is used anywhere in this repository.
All data is either synthetic, public-domain, or downloaded by the candidate from public sources.

## How to Download Kaggle Data

1. Create a Kaggle account at kaggle.com
2. Download the datasets and place them at the paths shown in the table above
3. Run `python run.py --precompute` to process them through the engine
