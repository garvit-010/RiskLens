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

**STATUS: DOCUMENTED ASSUMPTIONS & SECTOR SENSITIVITY**

All shock magnitudes (equity %, rate bps, spread bps, PD multipliers) are
calibrated for an impact score = 10 based on historical stress events:

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
- Reverse stress testing: Formulated as signed portfolio loss ($pnl\_pct \le -\text{threshold}$), sweeps severity from 0 to 10 and pinpoints the exact breach point via 15-iteration bisection search, returning break impact score, monetary loss, and scaled shock vector (ignoring gains).
- No credit valuation adjustment (CVA), no wrong-way risk

---

## 4. Entity Linking

- 15 S&P 100 tickers are covered by the curated alias dictionary
- Other entities are not linked (ticker = "N/A")
- In production: integrate with FactSet/Bloomberg symbology

---

## 5. Model Assumptions

- **FinBERT**: trained on financial news (Reuters, Financial Times excerpts). Performance on short tweets is lower – we note this in the evaluation.
- **MiniLM + LR**: trained on the same hand-labeled set it is evaluated on (demo limitation). Not a proper train/test split. Treat F1 as indicative, not rigorous.
- **VADER**: general-purpose; performs worse on domain-specific financial language.

---

## 6. Deduplication

- Hash window: 200 documents
- rapidfuzz threshold: 0.85 (token_sort_ratio)
- ASSUMPTION: this threshold was not empirically tuned; it is a reasonable starting point

---

## 7. Rebalancer Assumptions

- EMA alpha = 0.3 (decay rate per observation)
- Sentiment half-life = 24 hours
- k = 0.5 (weight sensitivity to sentiment)
- These parameters were not backtested or optimized. The backtest vs equal-weight
  (if run) should be interpreted as illustrative only.
- **Sentiment is not causality.** The rebalancer demonstrates signal integration,
  not alpha generation.

---

## 8. Evaluation Limitations

- Sentiment eval: uses up to 500 examples from Financial PhraseBank (CPU time limit)
- Event eval: trained and evaluated on the same set (no held-out test)
- Impact vs abnormal return: not computed (insufficient data and time)
- All results in docs/results.md are real computed values, not fabricated

---

## 9. What This System Is NOT

- Not a production trading system
- Not a regulatory-grade stress testing tool
- Not calibrated to any real institution's data
- Not backtested over a sufficient history
- Sentiment score is not a causal indicator of returns

All of these limitations are explicitly stated in the deck and dashboard.
