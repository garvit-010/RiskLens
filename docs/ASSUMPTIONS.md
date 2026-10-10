# RiskLens – Assumptions and Design Decisions

> This document must be read before the jury Q&A.
> Every assumption is documented here so we can defend our choices.

---

## 1. Impact Score Base Severity Table

**STATUS: DOCUMENTED DEFAULTS (not empirically calibrated)**

The base severity values were NOT derived from an event study.
They represent reasonable domain-expert estimates for a wholesale banking context:

| Event Type | Base Severity | Rationale |
|---|---|---|
| Credit Event | 7.0 | Default, covenant breach, and rating downgrade are high-materiality events in credit books |
| Geopolitical | 6.5 | Systemic risk; affects FX, commodities, and credit spreads broadly |
| Macroeconomic | 6.0 | Rate changes affect entire portfolio through duration and spread |
| Regulatory/Legal | 5.5 | Fines are bounded but reputational effects persist |
| Merger/Acquisition | 5.0 | Structural change but not necessarily a loss event |
| Earnings/Guidance | 4.5 | Affects equity and credit but often already partially priced in |
| Operational/Cyber | 5.5 | Tail risk: actual incidents can be catastrophic |
| Supply Chain/Commodity | 5.0 | Sector-specific; energy/materials most affected |
| ESG/Climate | 3.5 | Long-dated risk; short-term signal content is lower |
| Product Launch | 3.0 | Minimal credit/market impact |
| Other/Noise | 2.0 | Low confidence events |

**What was NOT done:** We did not regress impact scores against abnormal returns.
An event study requires clean event windows, market data, and sufficient sample size –
all of which would take longer than this hackathon allows. We document this explicitly
and discuss it as a "next step" in the deck.

---

## 2. Scenario Shock Magnitudes and Sector Targeting

**STATUS: DOCUMENTED DOMAIN-INFORMED ASSUMPTIONS & SECTOR SENSITIVITY**

All shock magnitudes (equity %, rate bps, spread bps, PD multipliers) are
domain-informed assumptions for an impact score = 10 based on historical stress events:

| Scenario | Reference events used as inspiration |
|---|---|
| Credit Event spread shocks | Investment-grade spread widening during 2008-09 GFC |
| Geopolitical | Russia-Ukraine invasion (Feb 2022): equity -8%, rates +30bps |
| Macroeconomic | 2022 rate hike cycle: +200bps Feds funds, equity -20% over the year |
| Credit Event PD multiplier | SVB: sector PD roughly doubled post-announcement |

**Scaling & Sector Targeting Rules:**
- **Linear Severity Scaling:** Shock magnitude scales linearly from 0% at impact=5 (floor) to 100% at impact=10:
  $$\text{scale} = \max\left(0, \frac{\text{impact} - 5}{5}\right)$$
  For example, an impact-7.1 event applies exactly $42\%$ of the base shock.
- **Sector Targeting:** When a news signal targets an entity (e.g. SVB $\to$ Financials), the Credit Event scenario applies:
  - **Target sector:** Full PD multiplier and an extra spread widening (default $+100$ bps on top of rating spread shock).
  - **Spillover sectors:** Non-target sectors absorb a documented spillover fraction (default $25\%$ of the spread and credit deterioration), capturing financial contagion without distorting unaffected books.
- **Commodity and FX Exposures:** The demonstration portfolio positions are USD-denominated wholesale credits, bonds, equities, and IR swaps without direct commodity holdings or unhedged FX exposures.

---

## 3. Portfolio Assumptions

- Fully synthetic, fixed seed 42
- 300 positions: 120 corporate loans, 100 bonds, 40 IR swaps, 40 equities
- Rating distribution biased toward investment grade (BBB and above ~52%)
- Loan duration proxy = 3.0 years (assumed; actual loans don't always have explicit duration)
- Bond convexity = duration^1.5 × 0.1 (approximation)
- IR swaps: pay-fixed swap where the bank pays fixed and receives floating. When rates rise ($\Delta \text{rate} > 0$), floating cash flows increase, resulting in positive MTM gain ($\Delta PNL = DV01 \times \Delta \text{rate}_{bps}$).
- Reverse stress testing: Formulated as signed portfolio loss ($pnl\_pct \le -\text{threshold}$), sweeps severity from 0 to 10 and pinpoints the exact breach point via 15-iteration bisection search, returning break impact score, monetary loss, and scaled shock vector (ignoring gains). The 2.5% threshold is an internal portfolio loss limit (5% is not reached under sector-targeted shocks), not a regulatory capital breach.
- No credit valuation adjustment (CVA), no wrong-way risk

---

## 4. Entity Linking

- 15 S&P 100 tickers are covered by the curated alias dictionary
- Other entities are not linked (ticker = "N/A")
- In production: integrate with FactSet/Bloomberg symbology

---

## 5. Model Assumptions & Evaluation Methodology

- **FinBERT (ProsusAI/finbert)**: Specialized domain transformer evaluated on the full 2,264 sentences of the Financial PhraseBank 100% agreement subset. FinBERT achieves 97.2% accuracy / 0.963 macro-F1 under academic argmax mapping, and 87.6% accuracy / 0.876 macro-F1 under the continuous net score threshold mapping ($P(pos) - P(neg)$ at $\pm 0.1$) used in the risk engine. Because FinBERT was fine-tuned on PhraseBank, performance represents an optimistic performance ceiling. Notably, FinBERT captures 98.7% of negative credit signals under the threshold mapping (vs 22.4% for VADER).
- **MiniLM + LR (Hybrid Event Classifier)**: Evaluated rigorously via **Stratified Cross-Validation on out-of-fold predictions** ($n=67$). At n=67 the embedding classifier adds no measurable gain over keyword rules; keyword rules do most of the work. Keywords were refined while reviewing these labels, so the baseline is not fully out-of-sample. In cross-validation at confidence threshold 0.55 across 11 classes, 0 of 67 out-of-fold predictions come from LR and 67 come from keyword fallback rules (0.6793 macro-F1).
- **VADER**: General-purpose lexicon baseline; provided for CPU fallback and benchmarking.

---

## 6. Deduplication & Novelty

- Hash window: 200 documents
- rapidfuzz threshold: 0.85 (token_sort_ratio)
- Novelty decay: Exponential penalty based on elapsed hours ($t_{half} = 24h$) applied to repetitive headlines.

---

## 7. Rebalancer & Date Range Alignment

- **Mixed Eras & Date Ranges:**
  - CNBC News covers 2017 to 2020.
  - Stock Market Tweets cover September 2021 to September 2022.
  - Cached Price Histories (`data/raw_samples/prices/`) cover September 2021 to September 2022.
  - SVB Collapse Replay covers March 8–21, 2023 (curated chronological replay).
- **Backtest Scope:** The Module A index rebalancer backtest executes strictly over the 2021–2022 tweet and price period. *No historical backtest is performed around SVB (March 2023) because historical prices end in September 2022.*
- **Illustrative Nature:** EMA alpha = 0.3, sentiment half-life = 24h, sensitivity $k=0.5$. Sentiment tilt demonstrates automated risk integration, not causal alpha generation.

---

## 8. Evaluation Limitations & Honest Assumptions

- **Sentiment Eval:** Evaluated on all 2,264 sentences of Financial PhraseBank (100% agreement subset). Per-class metrics transparently disclose performance across Negative, Neutral, and Positive polarities under both discrete argmax and engine threshold mappings.
- **Event Classifier Eval:** Evaluated via out-of-fold Stratified Cross-Validation ($n=67$). At n=67 the embedding classifier adds no measurable gain over keyword rules; keyword rules do most of the work. Keywords were refined while reviewing these labels, so the baseline is not fully out-of-sample.
- **Impact Calibration vs Abnormal Returns:** Impact scores use documented domain-informed base severity values and linear multipliers rather than econometric regressions against March 2023 SVB abnormal stock returns (no SVB price data exists in the cache).
- **All results in `docs/results.md` are empirically computed from local runs with zero fabrication.**

---

## 9. What This System Is NOT

- Not a production automated trading execution system
- Not a regulatory-grade DFAST / CCAR capital submission engine
- Not calibrated to proprietary institution or client data
- Not backtested across multi-cycle historical regimes
- Sentiment scores reflect informational distress rather than guaranteed market alpha

All limitations are explicitly stated in the deck, README, and dashboard.
