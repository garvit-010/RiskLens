# RiskLens - S&P Global & Crisil Campus Hackathon

**Candidate Name:** [Your Full Name]
**College Email ID:** [your_id@college.ac.in]
**College / Campus:** [Your College Name]
**Demo Video Link:** [YouTube / Unlisted]
**Slide Deck Link (if hosted externally):**

## 1. Project Overview / Problem Statement & Approach

RiskLens is an explainable News-to-Risk engine built for wholesale banking. It ingests unstructured text from multiple sources (financial news headlines and social media stock tweets), runs them through a multi-stage NLP pipeline, and outputs structured, auditable risk signals. Each signal carries a sentiment score, event classification, impact score, confidence rating, and a full explanation trace — so every number is defensible.

The engine feeds two downstream modules: **Module B** (Strategic Portfolio Stress Testing) auto-triggers a credit-aware stress test on a synthetic wholesale portfolio whenever a high-impact event is detected, and **Module A** (Tactical Index Rebalancer) adjusts S&P 100 stock weights based on smoothed sentiment. Both modules share a single signal bus, demonstrating the engine's reusability.

The system is designed to run fully offline using cached signals and pre-computed prices, making demos deterministic and reliable. A `--live` flag enables on-the-fly inference for Q&A. Real-event replay mode (SVB collapse, March 2023) provides a compelling, verifiable demonstration narrative.

## 2. Architecture & Tech Stack

```
[Sources]        [Ingestion]     [Normalize]        [NLP Risk Engine]       [Signal Store + API]
 Kaggle news  →  pluggable    →  clean, lang,   →   entity → ticker    →   JSONL + FastAPI
 Kaggle tweets   adapters        hash + fuzzy       FinBERT sentiment       /signals
 Replay (CSV)    one file        dedup              event classifier              │
 NewsAPI (stub)  per source                         impact + explanation          ├→ Module B: Stress Tester → Streamlit
                                                                                  └→ Module A: Rebalancer    → Streamlit tab
```

**Key frameworks:** Python 3.11, FastAPI, Streamlit, Hugging Face Transformers, sentence-transformers, scikit-learn, pandas, numpy, plotly, python-dotenv

**AI/ML models:**
- Sentiment: `ProsusAI/finbert` (fp16, 110M params) vs VADER baseline
- Event classification: `all-MiniLM-L6-v2` embeddings + logistic regression + keyword fallback
- Both run on CPU or NVIDIA GTX 1650 Ti (4 GB VRAM)

See `docs/architecture.png` for the full system diagram.

## 3. Dataset Used

| Dataset | Source | Nature |
|---|---|---|
| Financial news headlines | Kaggle | Public, ~5,000 rows sampled |
| Stock tweets | Kaggle | Public, ~3,000 rows sampled |
| Sentiment eval | Financial PhraseBank | Public labeled |
| Portfolio | Synthetic (seeded) | Fully synthetic, labeled as such |
| Prices | yfinance (cached CSV) | Public market data |

**All portfolio data is synthetic** with a fixed random seed. No real client data or S&P/Crisil proprietary data is used anywhere.

See `data/README.md` for full source citations, licenses, and assumptions.

## 4. Quickstart & Installation

Runtime: Python 3.11 on Windows 11 (also tested on Ubuntu 22.04)

```bash
git clone <your-repo-url>
cd <repo-folder>
pip install -r requirements.txt
python run.py
```

This starts the FastAPI signal server on port 8000 and the Streamlit dashboard on port 8501 in **cached replay mode** (no GPU, no API keys needed).

```bash
# Live mode (runs NLP engine on the fly — requires GPU or patience)
python run.py --live

# API only
python run.py --api-only

# Dashboard only
python run.py --dashboard-only
```

## 5. Key Results & Domain Impact

| Metric | Value | Baseline Comparison |
|---|---|---|
| FinBERT accuracy (Financial PhraseBank) | **93.3%** | +21.6 pp vs VADER (71.7%) |
| FinBERT macro-F1 | **0.935** | +0.221 vs VADER (0.713) |
| Event classifier macro-F1 (67 banking events) | **0.642** (Accuracy 66%) | +0.022 vs Keyword (0.620 / 61%) |
| Credit Event classification F1 | **0.800** | +0.160 vs Keyword (0.640) |
| Engine throughput (GPU inference) | **5.7 docs/sec** | Real-time streaming capability |

**Domain impact:** A credit risk or portfolio management team could deploy RiskLens to automate the monitoring of thousands of news sources, triage high-impact events within seconds, and immediately quantify the P&L impact on their book — all with a full audit trail. The stress-testing module directly mirrors regulatory requirements (ICAAP, DFAST) for scenario analysis, and the reverse stress test answers the regulator's question: "what would break you?"

See `docs/ASSUMPTIONS.md` for a full list of simplifying assumptions made in this prototype.
