# RiskLens – 5-Minute Video Demonstration Script

> **Target Duration:** 5 Minutes (300 Seconds)  
> **Target Voiceover Word Count:** ~720 Words (144 WPM natural pacing)  
> **Format:** Dual-format guide containing:  
> 1. **Director's Production Script** (Screen actions, visual cues, timecodes)  
> 2. **TTS-Ready Voiceover Text** (Clean plain text optimized for ElevenLabs, Murf, OpenAI Audio, or human read-aloud)

---

## Part A: Director's Production Script (Screen Actions & Timecodes)

| Timecode | Screen / Visual Action | Spoken Narration Summary |
|---|---|---|
| **0:00 – 0:35** | **IDE: VS Code / Cursor**<br>• Show project root in explorer.<br>• Display `README.md` on screen.<br>• Scroll past the candidate info to the Architecture diagram (`docs/architecture.png`). | **Introduction & Problem Statement**<br>Introduce the problem: Traditional credit risk management is reactive, relying on quarterly filings and manual reviews that lag market crises by days or weeks. Introduce RiskLens as an explainable, real-time news-to-risk intelligence platform. |
| **0:35 – 1:05** | **IDE: Integrated Terminal**<br>• Show terminal in `D:\Projects\RiskLens`.<br>• Type and run: `python run.py`.<br>• Highlight startup logs (FastAPI server on port 8000, Streamlit dashboard on port 8501).<br>• Transition: Open browser to `http://localhost:8501`. | **Setup & One-Command Execution**<br>Explain that the entire platform runs with a single command from the README. Emphasize that RiskLens is built with an offline-first architecture—zero cloud API keys, zero network latency, and complete enterprise data privacy. |
| **1:05 – 2:05** | **Browser: Streamlit Dashboard**<br>• Land on **Executive Cockpit** (Tab 1).<br>• Point out the 6-metric KPI banner at the top.<br>• Zoom into the **Active Crisis Spotlight Card** showing the March 9, 2023 Silicon Valley Bank headline.<br>• Highlight FinBERT sentiment (-0.92) and the explainable impact score breakdown (7.1 / 10.0).<br>• Show Threat Landscape chart & Live Feed. | **Data Ingestion & NLP Engine**<br>Explain the multi-stage ingestion pipeline. Raw financial headlines and social posts are cleaned, deduplicated, linked to market tickers, and classified into 11 risk taxonomies. FinBERT scores sentiment locally at 46 documents per second, producing an auditable explainability score. |
| **2:05 – 3:05** | **Browser: Module B – Stress Testing**<br>• Switch to **Stress Test** tab or scroll to Section 2 in Cockpit.<br>• Show the primary trigger event: SVB $1.8B loss and 60% stock plunge.<br>• Highlight Before vs After: $100.8M book drops to $95.9M (-$4.90M net P&L, -4.86%).<br>• Point out the **P&L Waterfall Chart** (Fixed Income -$2.8M, Equities -$1.2M, Loans -$0.9M).<br>• Highlight **Loss by Sector** showing Financials hit hardest with 25% systemic contagion spillover.<br>• Adjust Reverse Stress Slider: show breach at impact score 7.20. | **Credit-Aware Stress Testing (Module B)**<br>Walk through balance sheet revaluation on the synthetic 300-position wholesale book. Detail how credit spreads widen and default probabilities double on financial loans, capturing both direct sector targeting and contagion spillover. Reverse stress testing flags the exact severity where regulatory capital is breached. |
| **3:05 – 4:05** | **Browser: Module A – Tactical Rebalancing**<br>• Switch to **Rebalancer** tab or Section 3 in Cockpit.<br>• Show horizontal bar chart of active tilts vs the 10% benchmark.<br>• Highlight: SIVB underweighted by -1.38%, BAC overweighted by +1.91%.<br>• Show **Weights Trajectory Over Time** (stacked area chart from March 8 to March 21). | **Tactical Rebalancing & Capital Defense (Module A)**<br>Demonstrate how RiskLens acts before default occurs. Using an exponential sentiment decay model, the engine dynamically penalizes distressed institutions like Silicon Valley Bank while reallocating capital to flight-to-safety balance sheets like Bank of America. |
| **4:05 – 5:00** | **Browser: Evaluation Tab & Closing**<br>• Switch to **Evaluation** tab.<br>• Highlight empirical benchmark table: FinBERT 97.2% accuracy / 0.963 macro-F1 vs VADER's 57.1%.<br>• Emphasize the **98.7% Adverse Credit Signal Recall**.<br>• Switch back to Executive Cockpit for the final closing shot. | **Quantitative Rigor & Strategic Value**<br>Review empirical validation across 2,264 Financial PhraseBank sentences. FinBERT captures 98.7% of negative credit events, ensuring adverse risks are never missed. Conclude with enterprise value: auditable, credit-aware, real-time risk intelligence for wholesale banking. |

---

## Part B: TTS-Ready Voiceover Script (Plain Text for Audio Synthesis)

*Instructions: Copy and paste the text below directly into your Text-to-Speech tool (ElevenLabs, Murf, etc.). All numbers, dates, and terms are written out phonetically for natural, professional cadence.*

---

### [0:00 – 0:35] Segment 1: Problem Statement & Solution Overview

In wholesale banking, credit risk management faces a critical latency problem. When market-moving events unfold, risk teams often wait days or weeks for quarterly filings, rating agency updates, and manual credit committee reviews. By the time portfolios are adjusted, capital losses have already materialized.

Welcome to RiskLens. RiskLens is an explainable news-to-risk intelligence platform built for institutional risk committees and portfolio managers. It bridges the gap between unstructured market news and quantitative balance sheet stress testing in real time.

---

### [0:35 – 1:05] Segment 2: Setup and Execution from the IDE

Let us start directly from the development environment. 

Following the project documentation, starting the entire platform requires just a single command: python run dot py. 

Within seconds, the background engine initializes both a high-throughput Fast A P I service and our interactive Streamlit risk dashboard. 

RiskLens is built from the ground up with an offline-first architecture. All deep learning models, including FinBERT and sentence transformers, execute locally on GPU or CPU without requiring external cloud A P I keys, ensuring complete data privacy and institutional compliance.

Now, let us open the dashboard in our browser at localhost port 8501.

---

### [1:05 – 2:05] Segment 3: Data Ingestion and Live Intelligence Radar

We land directly on the Executive Risk Cockpit. This unified view provides a three-hundred-and-sixty-degree picture of the market and the balance sheet.

At the top, our executive summary tracks over eight hundred market signals ingested from financial news and verified replay feeds covering the March 2023 banking turmoil.

In our active crisis radar, the engine has isolated a critical trigger: on March ninth, 2023, Silicon Valley Bank shares plunged sixty percent following a one point eight billion dollar loss on its available-for-sale bond portfolio.

Notice the transparency of our intelligence pipeline. The system links the headline to entity ticker S I V B, classifies it as a Credit Event, and uses a local FinBERT neural network to score net sentiment at negative zero point nine two. 

Crucially, every score is completely explainable. The seven point one impact score is decomposed mathematically into base event severity, entity confidence, urgency, and adverse sentiment multipliers.

---

### [2:05 – 3:05] Segment 4: Strategic Balance Sheet Stress Testing (Module B)

Now let us see how this intelligence translates directly into capital exposure in Module B.

We stress-test a synthetic wholesale loan and bond book consisting of three hundred positions with an initial market value of one hundred point eight million dollars.

Applying our credit-aware scenario, the book suffers an immediate mark-to-market stress loss of four point nine million dollars, representing a four point eight six percent portfolio drawdown.

Our waterfall chart reveals exactly where the capital is lost. Fixed income bonds absorb two point eight million dollars in losses due to spread widening. Equities drop one point two million dollars, while corporate credit facilities absorb nine hundred thousand dollars as default probabilities double on regional banking facilities.

Looking at the sector distribution, Financials take the primary hit of three point eight million dollars. Non-target sectors absorb a documented twenty-five percent spillover shock, capturing systemic credit contagion without artificial damage.

Furthermore, our reverse stress test sweeps severity to identify that the portfolio breaches an internal five percent regulatory loss limit at an impact score of seven point two zero.

---

### [3:05 – 4:05] Segment 5: Tactical Rebalancing and Capital Defense (Module A)

Intelligence without action is incomplete. Module A demonstrates how RiskLens protects capital tactically through automated, sentiment-driven rebalancing.

Using an exponentially decayed sentiment model, the engine tilts portfolio weights across our banking universe relative to a ten percent equal-weight benchmark.

As crisis headlines intensify between March eighth and March twenty-first, the tactical rebalancer underweights Silicon Valley Bank by one point three eight percent, de-risking the position well before regulatory seizure. 

Simultaneously, the engine detects flight-to-safety deposit inflows and increases allocation to fortress megacap lenders, overweighting Bank of America by one point nine one percent. 

The stacked area chart illustrates this dynamic migration over time, smoothing portfolio volatility while remaining within strict turnover constraints.

---

### [4:05 – 5:00] Segment 6: Quantitative Validation and Conclusion

Finally, let us review our quantitative governance tab. 

We benchmarked FinBERT against rule-based baselines across all two thousand two hundred and sixty-four sentences of the Financial PhraseBank dataset. 

Under discrete classification, FinBERT achieves ninety-seven point two percent accuracy and a zero point nine six three macro-F-one score, outperforming lexicon baselines by over forty percentage points. 

Most importantly for credit risk, FinBERT achieves a ninety-eight point seven percent recall on negative signals. In banking, false negatives are catastrophic; our model rarely misses emerging credit distress. 

Local inference operates at forty-five point nine documents per second on consumer hardware, proving that institutional AI can be fast, transparent, and completely offline.

In summary, RiskLens transforms noisy market headlines into auditable, credit-aware stress revaluations and proactive capital defense. 

Thank you.
