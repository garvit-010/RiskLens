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
- **2.0% Loss Threshold:** Breached at impact score **8.08**
- **2.5% Loss Threshold:** Breached at impact score **8.85**
- **3.0% Loss Threshold:** Breached at impact score **9.63**
*(Note: Under un-targeted uniform credit shock across all sectors, a 5.0% loss breach occurs at impact score **7.54**).*

---

## 3. Module A: Tactical Index Rebalancer Tilts

Running the rebalancer across the 30 replay signals produces tactical tilts across the 15 S&P 100 universe stocks (starting from equal-weight 6.67%):

| Ticker | Sentiment Score | Final Weight | Tilt vs Base | Rationale |
|---|---|---|---|---|
| **SIVB** | -2.14 (decayed) | **4.53%** | **-2.14%** | Maximum penalty down to turnover cap boundary |
| **MS** | -0.93 | **5.61%** | **-1.05%** | Commercial real estate exposure warnings |
| **GS** | -0.86 | **5.70%** | **-0.97%** | Growth forecast cuts & lending tightening |
| **WFC** | -0.51 | **6.18%** | **-0.49%** | Contagion & CRE exposure |
| **JPM** | -0.41 | **6.32%** | **-0.34%** | Initial bank contagion pressure |
| **Megacaps (AAPL, MSFT, etc.)** | 0.00 | **7.03%** | **+0.37%** | Neutral reallocation beneficiaries |
| **BAC** | +0.11 | **7.25%** | **+0.58%** | Deposit inflow beneficiary |
| **C** | +0.51 | **8.15%** | **+1.48%** | Emergency rescue & acquisition facilitator |

---

## 4. 5-Minute Live Demo Script

### Minute 0:00 – 1:00: Problem & Ingestion Stream
1. Open Streamlit Dashboard (`http://localhost:8501`).
2. Show **Live Signals Feed**: Explain that raw unstructured news and social feeds are processed offline by FinBERT (`ProsusAI/finbert` fp16 on CUDA) and MiniLM (`all-MiniLM-L6-v2`).
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
   - Set Loss Threshold to **2.5%**.
   - Show the bisection search result: breaches at impact **8.85**.
   - Explain signed loss logic: reverse stress only flags true losses ($PnL\% \le -threshold$), ignoring gains.

### Minute 3:30 – 4:30: Tactical Rebalancing (Module A)
1. Switch to **Rebalancer** tab.
2. Review the weight tilt table: SIVB underweighted by -2.14% while stable liquid names receive defensive inflows.
3. Show the dynamic weight stacked area chart illustrating continuous exponential decay over time.

### Minute 4:30 – 5:00: Transparency & Q&A
1. Emphasize zero-API-key offline guarantee.
2. Point jury to `docs/ASSUMPTIONS.md` and `docs/results.md` for empirical cross-validation metrics and documented assumptions.
