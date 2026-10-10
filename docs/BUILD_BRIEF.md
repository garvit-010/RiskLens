# RiskLens: Build Brief for the AI Agent

**Project:** An explainable News-to-Risk engine with wholesale-banking stress testing
**Context:** S&P Global & Crisil Campus Hackathon (individual submission)
**Time budget:** 2 days (about 20 working hours)
**Hardware:** NVIDIA GTX 1650 Ti (4 GB VRAM), consumer laptop
**Language:** Python 3.11

> Read this whole document before writing code. Section 12 holds the rules you must follow throughout.

---

## 1. Goal and what the assessors want

Build a **unified AI/NLP Risk Engine** that ingests unstructured text from at least two sources and outputs structured risk signals. Then build at least one downstream module on top of it. S&P Global and Crisil are credit-risk, ratings, and market-intelligence businesses, so the submission is judged on:

1. **Domain understanding** (first tie-breaker). Show wholesale banking risk literacy: credit, market, counterparty, and concentration risk. Do not present this as just a sentiment model.
2. **Presentation and communication** (second tie-breaker). Clean README, deck, demo, and video.
3. **Working end-to-end prototype.** Text goes in, the engine produces signals, a downstream module consumes them, and a dashboard shows the result.
4. **Defensible design.** A live jury Q&A will probe design choices, so every decision needs a "why" and a known limitation.
5. **Repository hygiene.** See Section 10.

### Required engine output (from the case study)

For a given company or event, each signal must contain:
- **Sentiment Score**, from -1.0 to 1.0
- **Event Classification**, a categorical label
- **Impact Score**, from 1 to 10

Signals must be available to downstream apps via an API or a file. We provide both.

### Downstream modules

- **Module B: Strategic Portfolio Stress Testing (PRIMARY, full priority).** Subscribes to event class and impact score. A high-impact event triggers a stress test on a synthetic wholesale portfolio, with a before/after dashboard.
- **Module A: Tactical Index Rebalancing (small add-on, about 2 hours).** Subscribes to sentiment, adjusts the weights of about 15 S&P 100 stocks, and shows a weights-over-time dashboard. **Drop this first if behind schedule.**

### Delivery constraint conflict (resolved)

The case study says the live demo is at most 5 minutes. The packaging guide says the video is 10 minutes. We therefore produce:
- a tight **5-minute live demo script**, and
- a **7 to 10 minute recorded video** that adds a short architecture and code walkthrough.

---

## 2. How we stand out

Most participants will run FinBERT on headlines and show a bar chart. We differentiate through the following, in priority order:

1. **Credit-aware stress testing.** Shocks are spreads by rating bucket, PD migration, and expected-loss change, not just "equities -10%".
2. **Explainable, auditable signals.** Each signal carries its source, evidence snippet, component scores, confidence, and model version.
3. **Real-event replay mode.** Replay a known event (for example the SVB collapse, March 2023) at accelerated speed. The demo is deterministic and works offline.
4. **Measured improvement over a naive baseline.** Compare FinBERT against VADER on sentiment, and the classifier against keyword-only rules on event classification. The deck explicitly asks for "efficiency gains vs naive approach".
5. **Reverse stress test.** Answer "what event severity breaks the portfolio?" as well as "what happens if event X occurs?"
6. **Honest evidence-based impact score.** A small event study checks the impact score against real abnormal returns. If calibration is not done, say so (see Section 5.4).
7. **Both modules on one engine**, which shows the engine is reusable.

---

## 3. Hardware constraints and decisions

| Constraint | Decision |
|---|---|
| 4 GB VRAM | `ProsusAI/finbert` (110M params) in fp16, batch size 16 to 32. CPU is an acceptable fallback. |
| No large zero-shot models | Skip bart-large-mnli and similar. Use `all-MiniLM-L6-v2` embeddings plus a logistic-regression classifier, with keyword rules as a fallback. Runs on CPU. |
| No time for training | **No fine-tuning.** Pretrained FinBERT is enough. |
| Demo reliability | **Precompute everything offline.** Run the engine once over the replay datasets and cache the signals to JSONL. The demo replays cached signals, so it needs no GPU and has no model-load risk. A `--live` flag runs the engine on the fly for Q&A. |
| No API-key dependence | The system must run **fully offline with no keys**. |

---

## 4. Scope: keep, shrink, cut

| Item | Decision |
|---|---|
| Module B stress test | **Keep, full priority** |
| Module A rebalancer | **Keep as a small dashboard tab.** Drop first if behind. |
| Replay mode with real events | **Keep.** One or two events, for example SVB March 2023. |
| NewsAPI live source | **Cut.** Provide an adapter stub and list it under next steps. |
| Dedup | **Shrink** to hash plus `rapidfuzz` title similarity |
| Event-study calibration | **Shrink** to about 8 to 10 events over a handful of tickers, or use documented default severities labeled as assumptions. Never claim calibration that was not done. |
| Hand-labeled event set | **Shrink** from 150 to about 60 to 80 headlines. The agent may draft labels, but the human reviews and corrects them. |
| Sentiment evaluation | **Keep.** FinBERT vs VADER on Financial PhraseBank or the Kaggle labeled set. |
| Reverse stress test | **Keep** (severity sweep or bisection) |
| SSE stream | **Cut.** JSONL file plus FastAPI `GET /signals` satisfies the requirement. |
| Explainability fields | **Keep** |

---

## 5. Core engine spec (primary deliverable)

### 5.1 Ingestion (at least 2 sources)

- **Source 1, News:** Kaggle financial news headlines dataset, optionally plus a small GDELT sample CSV.
- **Source 2, Social:** Kaggle stock-tweets dataset, replayed as a stream.
- **Replay adapter:** reads timestamped CSV/JSON from `/data` and emits records at a configurable speed. This is the demo backbone.
- A **NewsAPI adapter stub** only: a class with the interface and a TODO, not a working integration.
- All adapters emit one unified `Document` schema:

```
id, source, timestamp, text, url, author, lang
```

Adapters are pluggable: one file per source, one shared interface.

### 5.2 Processing pipeline

1. **Clean + language filter.** Strip URLs and cashtag noise, keep English.
2. **Dedup.** Exact hash plus `rapidfuzz` title similarity. Syndicated news and retweets must not inflate signals. The dedup result feeds the novelty factor.
3. **Entity linking.** Curated alias dictionary for about 15 S&P 100 tickers (company names, cashtags, common aliases), with a fuzzy-match fallback. Attach sector and country.
4. **Sentiment.** `ProsusAI/finbert`, score = P(positive) - P(negative), in [-1, 1]. Baseline: VADER.
5. **Event classification.** `all-MiniLM-L6-v2` embeddings plus logistic regression, with keyword rules as a fallback when confidence is low. Taxonomy aligned to banking risk:
   - Credit Event (downgrade, default, covenant breach)
   - Geopolitical
   - Macroeconomic (rates, inflation, GDP)
   - Regulatory/Legal
   - Merger/Acquisition
   - Earnings/Guidance
   - Product Launch
   - Operational/Cyber
   - Supply Chain/Commodity
   - ESG/Climate
   - Other/Noise
6. **Impact score (1 to 10)** from a transparent formula:

```
impact = clip(1, 10,
    base_severity[event_type]
    * (0.5 + |sentiment|)
    * source_credibility
    * novelty
    * corroboration_boost
    * entity_materiality)
```

   - `corroboration_boost` rises when multiple independent sources report the same event in a time window.
   - Apply **time decay** so old signals fade.
   - Output a **confidence** value.

### 5.3 Output signal schema (JSON, versioned)

```json
{
  "signal_id": "...",
  "timestamp": "...",
  "entity": {"name": "...", "ticker": "...", "sector": "...", "country": "..."},
  "sentiment_score": -0.72,
  "event_type": "Credit Event",
  "impact_score": 8.4,
  "confidence": 0.81,
  "sources": [{"source": "newsapi", "url": "...", "snippet": "..."}],
  "explanation": {
    "base_severity": 7,
    "sentiment_factor": 1.22,
    "credibility": 0.9,
    "novelty": 1.0,
    "corroboration": 1.15
  },
  "model_version": "risklens-0.1"
}
```

Expose signals via:
- JSONL file in `/data/output/signals.jsonl`
- FastAPI: `GET /signals`, `GET /signals/{ticker}`

### 5.4 Impact calibration (honesty rule)

`base_severity` per event type is either:
- **calibrated** from a small yfinance event study (mean absolute abnormal return over 1 to 3 days after about 8 to 10 known events, market-adjusted), with the method documented, **or**
- **a documented default table** explicitly labeled as an assumption.

Never claim calibration that was not done. State exactly which one was used in `docs/ASSUMPTIONS.md` and in the deck.

### 5.5 Evaluation (feeds the "Key Results" slide)

- **Sentiment:** accuracy and macro-F1, FinBERT vs VADER, on Financial PhraseBank or the Kaggle labeled set.
- **Event classification:** per-class F1 and confusion matrix on the 60 to 80 hand-labeled headlines (stored in `/data/labeled_eval`). Compare against the keyword-only baseline.
- **Impact:** correlation between impact score and subsequent absolute abnormal returns (mini study, be honest about sample size).
- **System:** throughput (docs/sec) and latency.
- Save all numbers to `/docs/results.md` and as figures.

---

## 6. Module B: Strategic Stress Testing (primary)

### 6.1 Synthetic wholesale portfolio

About 300 positions, **fully synthetic with a fixed random seed** and labeled synthetic everywhere. Optionally borrow distribution shapes from the Kaggle transactions datasets.

| Asset class | Key fields |
|---|---|
| Corporate loans | rating, sector, country, EAD, spread, maturity, LGD, PD |
| Bonds | coupon, duration, convexity, rating, sector |
| Derivatives | IR swaps with DV01; optionally FX forwards |
| Equity holdings | beta, sector |

Save as CSV in `/data/portfolio`.

### 6.2 Scenario library (event type to shock vector)

A YAML/JSON file in `/data/scenarios`. Each scenario defines shocks to equity, rates, credit spreads by rating bucket, FX, commodities, and PD multipliers. Example mappings:

- **Geopolitical:** equities -8%, oil +15%, EM FX -5%, BBB-and-below spreads +80 bp
- **Credit Event:** the named sector's spreads +150 bp and its PDs x2
- **Macroeconomic:** rates +200 bp, equities -5%

Shock magnitude scales with the impact score (linear, between a floor of 5 and 10). **The trigger threshold is configurable (default: impact > 7).**

### 6.3 Revaluation logic (simplified but defensible)

- **Bonds:** duration/convexity price change for rate and spread moves
- **Loans:** change in expected loss = ΔPD × LGD × EAD, plus spread-based mark-to-market
- **Derivatives:** DV01 for rates, delta where applicable
- **Equity:** beta × equity shock

Label this "simplified first-order approximation, not production-grade."

### 6.4 Dashboard (Streamlit)

- Portfolio value before vs after (waterfall chart)
- Loss breakdown by asset class, sector, and country
- Top 10 loss contributors
- Panel linking to the triggering headline and its signal explanation
- **Severity slider** for what-if analysis
- **Reverse stress test:** the severity or shock level at which losses exceed a threshold (for example 5% of portfolio value), done by sweep or bisection

---

## 7. Module A: Tactical Rebalancer (small add-on)

- 15 S&P 100 stocks, starting equal-weight.
- Smooth sentiment per ticker with an exponential moving average and time decay.
- Weights: `w_i ∝ w0 × exp(k × smoothed_sentiment_i)`, normalize, then apply constraints: max deviation from base weight, per-step turnover cap, min and max weights.
- Dashboard tab: stacked-area chart of weights over time, plus a table of latest tilts and the headlines driving them.
- Quick backtest vs equal-weight using cached yfinance prices. **Report the result honestly, including noise and caveats.** Do not cherry-pick.

---

## 8. Data checklist

Cache everything in `/data` so the demo runs offline. Keep samples small (no large dumps).

| Need | Source | Notes |
|---|---|---|
| News | Kaggle financial news headlines (plus optional GDELT sample) | Human downloads (Kaggle login) |
| Social | Kaggle stock tweets dataset | Human downloads |
| Sentiment eval | Financial PhraseBank or Kaggle "Sentiment Analysis for Financial News" | Human downloads |
| Prices | `yfinance` for about 15 S&P 100 tickers and a few indices | Cache to CSV |
| Portfolio | Synthetic, seeded | Labeled synthetic |

Include a `data/README.md` listing every source, license/terms, and assumptions. **No proprietary or confidential data. No real S&P/Crisil client data.**

---

## 9. Architecture

```
[Sources]          [Ingestion]      [Normalize]        [NLP Risk Engine]       [Signal store + API]
 Kaggle news   →   pluggable    →   clean, lang,   →   entity → ticker    →    JSONL + FastAPI
 Kaggle tweets     adapters         hash + fuzzy       FinBERT sentiment        /signals
 Replay (CSV)      one file         dedup              event classifier              │
 NewsAPI (stub)    per source                          impact + explanation          ├→ Module B: Stress Tester → Streamlit
                                                                                     └→ Module A: Rebalancer    → Streamlit tab
```

Produce a high-resolution `docs/architecture.png` from this.

**Entry point:** `python run.py` starts everything (API plus dashboard) in cached-replay mode. `python run.py --live` runs the engine on the fly.

---

## 10. Repository structure and submission hygiene

```
<college>-<candidate-name>-hackathon/
├── README.md              <-- use the mandatory template below, exactly
├── requirements.txt       <-- pinned versions
├── LICENSE                <-- MIT
├── .env.example           <-- placeholder only; system must not need keys
├── run.py                 <-- single entry point
├── src/
│   ├── ingestion/
│   ├── engine/
│   ├── api/
│   ├── modules/
│   │   ├── stress_test/
│   │   └── rebalancer/
│   ├── dashboard/
│   └── eval/
├── tests/                 <-- impact formula + stress revaluation
├── data/
│   ├── README.md          <-- sources + assumptions
│   ├── raw_samples/
│   ├── portfolio/
│   ├── scenarios/
│   ├── labeled_eval/
│   └── output/
└── docs/
    ├── presentation.pdf   <-- 5 to 7 slides
    ├── architecture.png
    ├── ASSUMPTIONS.md
    └── results.md
```

**Rules:**
- Repo must be **public**.
- Naming: `<college>-<candidate-name>-hackathon`.
- **Incremental commits** with meaningful messages. Do not make one final dump.
- No large binaries or data dumps.
- Demo video on **YouTube (unlisted)**, tested in an incognito window.
- **No Google Drive or OneDrive links** for slides. Put the PDF in the repo.

### Mandatory README template

```markdown
# [Project Title] - S&P Global & Crisil Campus Hackathon

**Candidate Name:** [Your Full Name]
**College Email ID:** [your_id@college.ac.in]
**College / Campus:** [Your College Name]
**Demo Video Link:** [YouTube / Unlisted]
**Slide Deck Link (if hosted externally):**

## 1. Project Overview / Problem Statement & Approach
(2-3 paragraphs: business/technical problem and solution approach)

## 2. Architecture & Tech Stack
- System design and data flow (embed architecture diagram)
- Key frameworks, databases, AI/ML libraries

## 3. Dataset Used
- Source/nature of data (synthetic or public)
- Assumptions made

## 4. Quickstart & Installation
Runtime: [Python 3.11] on [OS tested]

git clone <your-repo-url>
cd <repo-folder>
pip install -r requirements.txt
python run.py

## 5. Key Results & Domain Impact
- What the prototype outputs/demonstrates
- Why it matters from a use case or business standpoint
```

The Quickstart code block must be a real fenced bash block in the final README. The human fills in the name, email, college, and video link.

---

## 11. Schedule (2 days)

### Day 1: engine and stress test

| Hours | Task |
|---|---|
| 0 to 1.5 | Repo skeleton, MIT license, unified `Document` schema, replay adapter, README skeleton. **Commit.** |
| 1.5 to 3.5 | Cleaning, language filter, dedup, alias-dictionary entity linking, FinBERT sentiment. **Commit.** |
| 3.5 to 5.5 | Event classifier (MiniLM + logistic regression + keyword fallback), impact formula with explanation fields, JSONL output, FastAPI `/signals`. **Commit.** |
| 5.5 to 8 | Synthetic portfolio, scenario library, revaluation engine, unit tests (impact formula, revaluation). **Commit.** |
| 8 to 10 | Streamlit stress dashboard: waterfall, breakdowns, top 10 contributors, headline drill-down. |

**Day 1 checkpoint (must pass):** a headline goes in, a signal comes out, a stress test fires, and the dashboard shows it. If this works, the submission is safe.

### Day 2: polish, evaluation, deliverables

| Hours | Task |
|---|---|
| 0 to 1.5 | Severity slider and reverse stress test |
| 1.5 to 3 | Evaluation: FinBERT vs VADER, event classifier F1 and confusion matrix, impact vs abnormal-return mini study. Save to `/docs/results.md`. |
| 3 to 5 | Module A rebalancer, stacked-area chart, quick backtest with honest caveats |
| 5 to 6 | Fresh-clone end-to-end test, pin dependencies, write `docs/ASSUMPTIONS.md` |
| 6 to 8 | Architecture diagram, 7-slide deck to PDF, README per template |
| 8 to 9.5 | Record demo video, rehearse 5-minute live demo, upload to YouTube unlisted |
| 9.5 to 10 | Final check against the submission checklist |

### Priority order if time runs short

1. Engine plus Module B end to end (**Must**)
2. Evaluation numbers and baseline comparison (**Must**)
3. README, architecture diagram, deck, video (**Must**, reserve real time)
4. Reverse stress test and explainability panel (**Should**)
5. Module A (**Should**, drop first)

---

## 12. Rules for the build agent

1. **Working end-to-end beats complete.** After every phase, the system must still run.
2. **Hardware limit:** 4 GB VRAM. Use `ProsusAI/finbert` in fp16 and `all-MiniLM-L6-v2`. No models over about 500M parameters. No training runs.
3. **Offline by default.** Cache all model outputs and all market data. No API keys required for the demo.
4. **One command** starts everything: `python run.py` (with a `--live` flag).
5. **Traceability.** Every signal carries source, snippet, component scores, confidence, and model version. Every dashboard number must trace to a signal.
6. **Label synthetic data** as synthetic. Never present simplified pricing as production-grade.
7. **Never claim** calibration, accuracy, or backtest results that were not actually computed. Report real numbers, including unflattering ones.
8. **Code quality:** short docstrings, type hints, pinned dependencies, light repo. Tests are required for the impact formula and stress revaluation only.
9. **Maintain `docs/ASSUMPTIONS.md`** as you build. The human uses it to answer jury questions. Log every assumption: shock values, severity table, taxonomy, thresholds, data limits.
10. **Small, frequent commits** with meaningful messages.
11. **No real or confidential data**, and nothing from S&P Global or Crisil client material.
12. **Flag, do not guess.** If a dataset is missing or a column differs from expectations, stop and tell the human instead of inventing data.

---

## 13. What the human does (the agent cannot)

- Download the Kaggle datasets (login required) and place them in `/data/raw_samples`.
- Review and correct the event labels the agent drafts.
- Fill in name, email, college, and the video link in the README.
- Record the video, rehearse the live demo, and upload to YouTube as unlisted.
- Verify the access checklist: public repo, unlisted video playing in incognito, no Drive or OneDrive links.
- Be ready to explain the impact formula, shock calibration, and why sentiment is not causality.

---

## 14. Live demo script (5 minutes)

| Time | Content |
|---|---|
| 0:00 to 0:30 | Problem and what RiskLens does |
| 0:30 to 1:00 | Start the app with the README command |
| 1:00 to 2:15 | Replay a real event. Show raw headlines and tweets arriving, then structured signals with explanations. |
| 2:15 to 3:45 | Stress test auto-triggers on a high-impact event. Show before/after, loss breakdown, headline drill-down. |
| 3:45 to 4:30 | Module A rebalancing view on the same stream, then the reverse stress test |
| 4:30 to 5:00 | Evaluation results vs baseline, and domain impact |

Keep a pre-recorded fallback clip in case something breaks live.

## 15. Deck outline (7 slides max, PDF in `/docs`)

1. **Title:** project, candidate name, college
2. **Problem and approach:** the business problem in your own words
3. **Architecture and data flow:** the diagram
4. **Implementation highlights:** key modules and why each tech choice
5. **Key results:** eval metrics, screenshots, improvement over the naive baseline
6. **Domain impact:** how a risk or portfolio team would use it, and how it relates to regulatory stress-testing practice
7. **Limitations and next steps:** synthetic data, simplified pricing, sentiment is not causality, NewsAPI live source

## 16. Final submission checklist

- [ ] Repo is public with an MIT LICENSE
- [ ] README follows the template, with the exact run command and OS tested
- [ ] `requirements.txt` is pinned, and a fresh-clone run works
- [ ] `/data` is small, with a sources and assumptions file
- [ ] `docs/presentation.pdf` (at most 7 slides) and `docs/architecture.png` exist
- [ ] Video is on YouTube (unlisted) and plays in incognito
- [ ] No Drive or OneDrive links
- [ ] Commit history is incremental
- [ ] No proprietary or confidential data anywhere
- [ ] 5-minute live demo rehearsed, with a fallback recording ready