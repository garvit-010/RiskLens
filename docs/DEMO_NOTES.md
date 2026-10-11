# RiskLens – Live Demo Script & Walkthrough Notes

> For S&P Global & Crisil Campus Hackathon Jury Presentation (5-Minute Window).

---

## 1. Selected Demo Trigger Event

| Parameter | Value |
|---|---|
| **Headline** | *"SVB Financial stock plunges 60% as bank run fears intensify among tech clients"* |
| **Timestamp** | `2023-03-09T15:00:00+00:00` |
| **Entity Linked** | SVB Financial Group (`SIVB`) |
| **Entity Sector / Country** | `Financials` / `US` |
| **Event Type** | `Credit Event` |
| **Impact Score** | **10.0** (Maximum Severity) |
| **FinBERT Sentiment Score** | `-0.9589` (Extreme Negative) |
| **Corroboration Factor** | 1.0 (single source window; reaches 1.15 on multi-source confirmation) |

---

## 2. Expected Stress Test Output (Synthetic Wholesale Book)

The portfolio generator initializes 300 positions across 4 wholesale asset classes:

- **Portfolio Mark-to-Market Before Stress:** `$4,388,584,715` (~$4.39 Billion)
- **Stressed Value:** `$4,246,551,323` (~$4.25 Billion)
- **Net Portfolio Loss:** `-$142,033,392` (`-3.24%`)

### Breakdown by Asset Class
- **Corporate Loans:** `-$101,244,094` (71.3% of loss) — PD doubling on Financials loans + extra spread widening.
- **Bonds:** `-$36,220,596` (25.5% of loss) — rating-based spread widening + sector extra shock.
- **Equities:** `-$4,568,702` (3.2% of loss) — beta-scaled equity drawdown.
- **IR Swaps:** `$0.00` — zero interest rate shock in pure Credit Event scenario.

### Sector Targeting & Contagion
- **Target Sector (Financials):** Suffers `-$42,104,976` loss (>2.7× higher than any other sector).
- **Spillover Sectors:** Non-target sectors (Telecom, Industrials, Energy) absorb exactly 25% of the base spread and credit deterioration shock, capturing systemic contagion without artificial book-wide damage.

### Reverse Stress Test Break Points
- **2.0% Loss Limit:** Breached at impact score **8.08**
- **2.5% Loss Limit:** Breached at impact score **8.85**
- **3.0% Loss Limit:** Breached at impact score **9.63**
*(Note: The 2.5% threshold represents an internal portfolio loss limit, not a regulatory capital breach; a 5.0% portfolio loss is not reached under sector-targeted shocks).*

---

## 3. Module A: Tactical Index Rebalancer Tilts

Running the rebalancer across the 30 replay signals produces tactical tilts across the 15 S&P 100 universe stocks (starting from equal-weight 6.67%):

| Ticker | Accumulated Decayed Sentiment | Final Weight | Tilt vs Base | Rationale |
|---|---|---|---|---|
| **SIVB** | -1.20 | **5.28%** | **-1.38%** | Distressed credit penalty (bank run & plunge) |
| **JPM** | -1.00 | **5.51%** | **-1.16%** | Contagion pressure & interbank risk |
| **MS** | -0.94 | **5.58%** | **-1.08%** | Commercial real estate & financing scrutiny |
| **GS** | -0.89 | **5.64%** | **-1.03%** | Growth forecast cuts & lending tightening |
| **WFC** | -0.40 | **6.32%** | **-0.35%** | Contagion & regional bank scrutiny |
| **Megacaps (AAPL, MSFT, GOOGL, AMZN, C, XOM, CVX, JNJ, META)** | 0.00 | **7.01%** | **+0.34%** | Neutral baseline; absorb reallocated capital |
| **BAC** | +0.69 | **8.58%** | **+1.91%** | Documented flight-to-safety deposit inflows |

*Note on Sentiment Column:* Sentiment values reflect exponentially decayed accumulation over consecutive events (which can aggregate outside $[-1, 1]$ when multiple severe events occur before elapsed-time decay resets the EMA). Citigroup (`C`) sits at neutral baseline (+0.34% tilt) rather than false entity-matching inflation.

---

## 4. 5-Minute Live Demo Script

### Minute 0:00 – 1:30: Executive Cockpit (Unified 360° Stakeholder View)
1. Open Streamlit Dashboard (`http://localhost:8501`).
2. Show **Executive Cockpit (Default Tab / Unified View)**:
   - Walk the jury through the single unified dashboard designed for stakeholders to see everything at once.
   - Point out the 6 KPI banner: Ingested signals (819), active SIVB shock, portfolio stress loss (-$4.9M / -4.86%), tactical tilt spread (-1.38% SIVB vs +1.91% BAC), and FinBERT NLP reliability (97.2% Acc / 98.7% Adverse Recall).
   - Show how the **Spotlight Card** links the March 9, 2023 Silicon Valley Bank headline directly to balance sheet stress testing and automated rebalancing.
   - Adjust the **Simulation Severity Slider** to show the entire dashboard re-evaluating dynamically in real-time.
3. Point out independent source tagging (`replay_news` vs `replay_social`) and the corroboration multiplier (1.15 boost when two independent sources confirm).

### Minute 1:00 – 2:30: Credit-Aware Stress Testing (Module B)
1. Navigate to **Stress Testing** tab.
2. Select the demo trigger: `[2023-03-09 15:00] SIVB – Credit Event (Impact 10.0)`.
3. Highlight the KPI banner: Portfolio drops from **$4.39B to $4.25B** (-$142M net P&L, -3.24%).
4. Show the **Waterfall Chart**: Corporate loans bear 71% of the loss.
5. Show **Loss by Sector**: Financials bears -$42.1M loss, demonstrating genuine **credit-aware sector targeting** with documented 25% spillover to other sectors.

### Minute 2:30 – 3:30: Severity Scaling & Reverse Stress Testing
1. Adjust the **Severity Slider**: Point out linear scaling formula ($\text{scale} = (\text{impact}-5)/5$); impact 7.1 applies 42% of base shock, impact 5 applies 0%.
2. Scroll to **Reverse Stress Test**:
   - Set Loss Threshold to **2.5%** (internal portfolio loss limit).
   - Show the bisection search result: breaches at impact **8.85**.
   - Explain signed loss logic: reverse stress only flags true losses ($PnL\% \le -threshold$), ignoring gains.

### Minute 3:30 – 4:30: Tactical Rebalancing (Module A)
1. Switch to **Rebalancer** tab.
2. Review the weight tilt table: SIVB underweighted by -1.38% while stable liquid names receive defensive inflows.
3. Show the dynamic weight stacked area chart illustrating continuous exponential decay over time.

### Minute 4:30 – 5:00: Transparency & Q&A
1. Emphasize zero-API-key offline guarantee.
2. Point jury to `docs/ASSUMPTIONS.md` and `docs/results.md` for empirical cross-validation metrics and documented assumptions.
