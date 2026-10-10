# RiskLens – Explainable News-to-Risk Engine with Wholesale Banking Stress Testing

**Candidate Name:** `[TODO: human to fill]`  
**College Email ID:** `[TODO: human to fill]`  
**College / Campus:** `[TODO: human to fill]`  
**Demo Video Link:** `[TODO: human to fill]`  
**Slide Deck Link:** `[TODO: human to fill]`  

> **Disclaimer:** Unrelated to the commercial FAIR-based cyber-risk product of the same name.  
> **Note for Jury:** Repository name is formatted as `<college>-<candidate-name>-hackathon` per competition submission guidelines.

---

## 1. Project Overview & Approach

**RiskLens** is an explainable, auditable News-to-Risk intelligence platform built for wholesale banking risk teams. Unstructured textual feeds (financial news and market social commentary) are ingested, cleaned, and processed through a domain-specific multi-stage NLP pipeline. Every signal emitted includes:
- **Calibrated sentiment scoring** (FinBERT domain transformer with VADER rule-based fallback).
- **Taxonomy event classification** (11 wholesale banking risk types via MiniLM embeddings + logistic regression + keyword rules).
- **Auditable impact severity score (1–10)** derived with explicit multiplicative attribution across base event severity, sentiment intensity, source credibility, novelty decay, and corroboration boosts.

The core signal engine simultaneously drives two downstream modules:
1. **Module B (Strategic Wholesale Stress Testing):** When high-impact events occur, auto-triggers a credit-aware stress revaluation across a 300-position wholesale portfolio (corporate loans, bonds, equities, interest rate swaps). Applies sector-specific spread and PD shocks with documented spillover to non-target sectors, and runs a signed reverse stress test to pinpoint breaking thresholds.
2. **Module A (Tactical Index Rebalancer):** Tracks smoothed exponential sentiment across a 15-stock S&P 100 universe, tilting weights away from distressed names and toward defensive quality under strict bounds and turnover constraints.

The entire system operates **100% offline with zero external API key requirements**, loading local model weights from `.model_cache/` and deterministic replay signals.

---

## 2. Architecture & Data Flow

```
[DATA INGESTION]          [NORMALIZATION]           [NLP RISK ENGINE]          [API & DISTRIBUTION]
 SVB Replay News    ──┐    Language Filter (EN)      Entity Linking (S&P 100)   JSONL Signal Cache
 SVB Replay Social  ──┼──> Exact Hash Dedup     ──>  FinBERT (CUDA fp16)    ──> FastAPI (/signals)
 Kaggle News (stub) ──┤    RapidFuzz Similarity      MiniLM Event Classifier           │
 Kaggle Tweets (stub)─┘    Novelty Exponential       Auditable Impact Formula          ├──> Module B: Stress Testing
                           Decay Window              (Base × Sent × Cred × Corr)       └──> Module A: Rebalancer
```

- **Environment & Runtime:** Python 3.11 / 3.12 on **Windows 11**
- **Hardware Acceleration:** NVIDIA GeForce GTX 1650 Ti (4GB VRAM, CUDA fp16) with automated CPU fallback
- **Core Frameworks:** PyTorch, Transformers, Sentence-Transformers, Scikit-Learn, FastAPI, Streamlit, Plotly, Pandas, NumPy

See [docs/architecture.png](docs/architecture.png) for the full high-resolution diagram.

---

## 3. Shipped Repositories & Datasets

Every dataset claimed in this repository corresponds directly to local files in `/data`:

| Dataset | Source / Description | Status in Repository | Shipped Location |
|---|---|---|---|
| **SVB Collapse Replay** | Curated chronological headlines & tweets (March 8–13, 2023) | **Shipped in repo** (30 documents: 18 news, 12 social) | `data/raw_samples/replay_svb.jsonl` |
| **Financial PhraseBank** | Official 100% annotator agreement subset (Malo et al.) | **Shipped in repo** (2,264 sentences) | `data/raw_samples/phrasebank.csv` |
| **Sentiment Eval Sample** | Curated financial headline sample with ground truth | **Shipped in repo** (60 sentences, 95% CI ±6 pp) | `data/raw_samples/sentiment_sample_n60.csv` |
| **Event Classification** | Curated banking distress & corporate headlines (11 classes) | **Shipped in repo** (68 headlines, out-of-fold CV) | `data/labeled_eval/labels.csv` |
| **Financial News Sample** | Sampled CNBC market headlines subset | **Shipped in repo** (500 headlines) | `data/raw_samples/news.csv` |
| **Stock Tweets Sample** | Sampled stock market tweets subset | **Shipped in repo** (300 tweets) | `data/raw_samples/tweets.csv` |
| **Wholesale Portfolio** | Synthetic corporate loans, bonds, swaps, equities (seed 42) | **Shipped in repo** (300 positions, ~$4.39B MTM) | `data/portfolio/portfolio.csv` |
| **Scenario Shock Library** | Calibrated rating spread widening & PD multipliers | **Shipped in repo** (11 wholesale risk scenarios) | `data/scenarios/scenarios.yaml` |
| **Historical Price Cache** | Daily stock prices for 25 tickers (AAPL, TSLA, SIVB, etc.) | **Shipped in repo** (offline backtesting ready) | `data/raw_samples/prices/` |
| **Precomputed Signals** | Fully computed neural signals for instant replay | **Shipped in repo** (30 signals with full audit trail) | `data/output/signals.jsonl` |

> [!NOTE]
> All portfolio positions are synthetic with a fixed seed (seed 42). No proprietary S&P Global, Crisil, or client data is used. See [data/README.md](data/README.md) for full integrity statements.

---

## 4. Quickstart & Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# Install required dependencies
pip install -r requirements.txt

# Run the complete system in deterministic replay mode (no API keys, zero network needed)
python run.py
```

This launches the **FastAPI REST API** on port `8000` and the interactive **Streamlit Dashboard** on port `8501`.

```bash
# Alternative execution modes:
python run.py --live            # On-the-fly neural inference on incoming text
python run.py --dashboard-only  # Streamlit UI only
python run.py --api-only        # Headless FastAPI signal server only
python -m src.eval.run_eval     # Run empirical model benchmark evaluation
```

---

## 5. Key Results & Empirical Findings

### A. Model Performance & Benchmarks (from `docs/results.md`)

| Task / Benchmark | Model | Metric | Baseline Comparison | Empirical Notes |
|---|---|---|---|---|
| **Sentiment Scoring** | **ProsusAI/finbert** (fp16) | **95.6% Accuracy**<br>**0.884 Macro-F1** | VADER (Naive Lexicon):<br>69.8% Accuracy, 0.475 F1 | Evaluated on Financial PhraseBank 100% agreement subset ($n=500$). FinBERT was fine-tuned on this corpus, so results are optimistic. |
| **Event Classification** | **MiniLM-L6-v2 + LR** (Hybrid) | **0.679 Macro-F1**<br>(Credit Event F1: **0.923**) | Keyword Baseline:<br>0.679 Macro-F1 | Evaluated via **Stratified Cross-Validation** on out-of-fold data ($n=68$). MiniLM provides calibrated continuous confidence and unseen generalization. |
| **System Throughput** | FinBERT on GPU | **24.2 docs/sec** | Real-time stream capable | Benchmarked on NVIDIA GTX 1650 Ti (4GB VRAM, CUDA fp16). |

### B. Module B: SVB Collapse Stress Test Revaluation

Selected primary demo trigger:
- **Headline:** *"SVB Financial stock plunges 60% as bank run fears intensify among tech clients"* (March 8, 2023, 15:00 UTC)
- **Signal:** Entity `SIVB` (Financials, US), Event Type: `Credit Event`, Impact Score: `10.0`, Sentiment: `-0.9589`.

```
Portfolio Before Shock:   $4,388,584,715  (~$4.39 Billion)
Portfolio After Shock:    $4,246,551,323  (~$4.25 Billion)
Total Portfolio Loss:     -$142,033,392   (-3.24% Net Loss)
```

- **Asset Class Breakdown:** Corporate Loans account for `-$101,244,094` (71.3% of loss) due to PD doubling and spread widening; Bonds absorb `-$36,220,596` (25.5%); Equities lose `-$4,568,702` (3.2%); Pay-fixed IR Swaps have zero rate move in a pure credit scenario (`$0`).
- **Credit-Aware Sector Targeting:** Positions in the target sector (`Financials`) absorb `-$42,104,976` loss (>2.7× higher than any other sector). Non-target sectors absorb a documented `25%` spillover fraction.
- **Signed Reverse Stress Test:** Using a 15-iteration bisection search, the portfolio breaches a 2.5% loss threshold at **Impact Score 8.85** (`-$109.7M` monetary loss). Favorable scenario moves (gains) are strictly ignored.

### C. Module A: Tactical Index Rebalancer Tilts

Replaying the 30 chronological SVB headlines automatically penalizes distressed credit exposures while reallocating capital to defensive quality:
- **Underweighted:** `SIVB` (-2.14% tilt to 4.53% weight floor), `MS` (-1.05%), `GS` (-0.97%), `WFC` (-0.49%), `JPM` (-0.34%).
- **Overweighted:** Flight-to-safety beneficiaries `C` (+1.48%), `BAC` (+0.58%), and defensive megacaps (+0.37%).

---

## 6. Dashboard & Visual Proof

### Stress Testing Dashboard
![RiskLens Dashboard](docs/dashboard_screenshot.png)

### Out-of-Fold Event Classifier Confusion Matrix
![Event Confusion Matrix](docs/event_confusion.png)

---

## 7. Regulatory Alignment & Domain Grounding

- **Supervisory Stress Testing:** The stress-testing module is **conceptually aligned with scenario analysis practices used in ICAAP and supervisory stress testing**, translating macro narrative shocks into granular rating-bucket spread widenings and counterparty PD migrations.
- **Reverse Stress Testing (SR 12-7):** Solves the supervisory question *"What magnitude of event causes a critical capital breach?"* using continuous bisection search.
- **Auditable Lineage:** Every stressed portfolio valuation traces directly to the underlying headline, timestamp, source credibility weight, and corroboration multiplier.

For complete mathematical definitions and simplifying assumptions, see:
- [docs/ASSUMPTIONS.md](docs/ASSUMPTIONS.md) – Transparent methodology choices and documented parameters.
- [docs/results.md](docs/results.md) – Full empirical classification reports and throughput benchmarks.
- [docs/DEMO_NOTES.md](docs/DEMO_NOTES.md) – Structured 5-minute hackathon jury walkthrough guide.
