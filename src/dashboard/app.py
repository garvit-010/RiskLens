"""
RiskLens – Streamlit Dashboard

Tabs:
  1. 📡 Live Feed       – incoming signals from the replay / live engine
  2. 📊 Stress Test     – Module B: portfolio before/after, waterfall, breakdown
  3. ⚖️  Rebalancer     – Module A: weights over time + headlines driving them
  4. 📈 Evaluation      – FinBERT vs VADER, event classifier metrics

Run with:
    streamlit run src/dashboard/app.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Make sure the project root is on the path
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import (
    SIGNAL_JSONL_PATH, IMPACT_THRESHOLD, PORTFOLIO_DIR, DOCS_DIR
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="RiskLens | News-to-Risk Engine",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main { background: #0d1117; }

    .metric-card {
        background: linear-gradient(135deg, #1a1f2e 0%, #16213e 100%);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.5rem;
    }
    .signal-high  { border-left: 4px solid #ef4444; }
    .signal-med   { border-left: 4px solid #f59e0b; }
    .signal-low   { border-left: 4px solid #22c55e; }

    .entity-tag {
        background: #1e3a5f;
        color: #60a5fa;
        padding: 2px 8px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .event-tag {
        background: #3d1a1a;
        color: #fca5a5;
        padding: 2px 8px;
        border-radius: 9999px;
        font-size: 0.75rem;
    }
    h1, h2, h3 { color: #f0f6fc; }
    .stTabs [data-baseweb="tab"] { font-size: 0.9rem; font-weight: 500; }
    .disclaimer {
        background: #1c1207;
        border: 1px solid #713f12;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        color: #fbbf24;
        font-size: 0.8rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

@st.cache_data(ttl=5)
def load_signals() -> list[dict]:
    if not SIGNAL_JSONL_PATH.exists():
        return []
    signals = []
    with open(SIGNAL_JSONL_PATH, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    signals.append(json.loads(line))
                except Exception:
                    continue
    return sorted(signals, key=lambda s: s.get("timestamp", ""), reverse=True)


@st.cache_data(ttl=60)
def load_portfolio() -> pd.DataFrame:
    from src.modules.stress_test.portfolio import load_portfolio as _lp
    return _lp()


def _impact_color(score: float) -> str:
    if score >= 7:
        return "#ef4444"
    if score >= 5:
        return "#f59e0b"
    return "#22c55e"


def _sentiment_label(score: float) -> str:
    if score > 0.2:
        return "🟢 Positive"
    if score < -0.2:
        return "🔴 Negative"
    return "🟡 Neutral"


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
col_logo, col_title = st.columns([1, 8])
with col_logo:
    st.markdown("## 🔭")
with col_title:
    st.markdown("# RiskLens")
    st.caption("Explainable News-to-Risk Engine | S&P Global & Crisil Campus Hackathon")

st.markdown("---")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Controls")
    auto_refresh = st.toggle("Auto-refresh (5 s)", value=False)
    st.markdown("---")
    impact_threshold = st.slider(
        "Impact trigger threshold",
        min_value=1.0, max_value=10.0,
        value=float(IMPACT_THRESHOLD), step=0.5,
    )
    st.markdown("---")
    st.markdown("### 📂 Data Sources")
    signals = load_signals()
    st.metric("Total signals", len(signals))
    high = [s for s in signals if s.get("impact_score", 0) >= impact_threshold]
    st.metric("High-impact events", len(high), delta_color="inverse")
    st.markdown("---")
    st.markdown(
        '<div class="disclaimer">⚠️ Portfolio data is <b>SYNTHETIC</b>. '
        'All stress-test outputs are for demonstration only.</div>',
        unsafe_allow_html=True,
    )

if auto_refresh:
    time.sleep(5)
    st.rerun()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_feed, tab_stress, tab_rebal, tab_eval = st.tabs([
    "📡 Live Feed",
    "📊 Stress Test",
    "⚖️ Rebalancer",
    "📈 Evaluation",
])

# ===========================================================================
# TAB 1 – Live Feed
# ===========================================================================
with tab_feed:
    st.subheader("📡 Risk Signal Feed")

    if not signals:
        st.info(
            "No signals yet. Run `python run.py` to process the replay dataset, "
            "or use the `/process` API endpoint to submit text."
        )
    else:
        # Summary metrics
        m1, m2, m3, m4 = st.columns(4)
        impact_scores = [s.get("impact_score", 0) for s in signals]
        with m1:
            st.metric("Total Signals", len(signals))
        with m2:
            st.metric("Avg Impact", f"{np.mean(impact_scores):.1f}")
        with m3:
            st.metric("High Impact", sum(1 for i in impact_scores if i >= impact_threshold))
        with m4:
            # most common event type
            from collections import Counter
            top_event = Counter(s.get("event_type", "?") for s in signals).most_common(1)[0][0]
            st.metric("Top Event", top_event)

        # Filters
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            tickers_avail = sorted({s.get("entity", {}).get("ticker", "N/A") for s in signals})
            sel_ticker = st.selectbox("Filter ticker", ["All"] + tickers_avail)
        with col_f2:
            event_types = sorted({s.get("event_type", "?") for s in signals})
            sel_event = st.selectbox("Filter event type", ["All"] + event_types)
        with col_f3:
            min_imp = st.slider("Min impact", 1.0, 10.0, 1.0, 0.5)

        filtered = signals
        if sel_ticker != "All":
            filtered = [s for s in filtered if s.get("entity", {}).get("ticker") == sel_ticker]
        if sel_event != "All":
            filtered = [s for s in filtered if s.get("event_type") == sel_event]
        filtered = [s for s in filtered if s.get("impact_score", 0) >= min_imp]

        # Impact distribution chart
        if filtered:
            df_plot = pd.DataFrame([
                {
                    "event_type": s.get("event_type", "?"),
                    "impact_score": s.get("impact_score", 0),
                    "sentiment_score": s.get("sentiment_score", 0),
                    "ticker": s.get("entity", {}).get("ticker", "N/A"),
                }
                for s in filtered
            ])
            fig = px.scatter(
                df_plot, x="sentiment_score", y="impact_score",
                color="event_type", hover_data=["ticker"],
                title="Sentiment vs Impact Score",
                template="plotly_dark",
                color_discrete_sequence=px.colors.qualitative.Bold,
            )
            fig.add_hline(y=impact_threshold, line_dash="dash",
                          line_color="#ef4444", annotation_text="Trigger threshold")
            st.plotly_chart(fig, use_container_width=True)

        # Signal cards
        st.markdown(f"**Showing {min(50, len(filtered))} of {len(filtered)} signals**")
        for sig in filtered[:50]:
            impact = sig.get("impact_score", 0)
            border = "signal-high" if impact >= 7 else ("signal-med" if impact >= 5 else "signal-low")
            entity = sig.get("entity", {})
            exp = sig.get("explanation", {})
            snippet = (sig.get("sources") or [{}])[0].get("snippet", "")

            with st.expander(
                f"[{sig.get('timestamp', '')[:16]}] "
                f"{entity.get('ticker','?')} | {sig.get('event_type','?')} | "
                f"Impact: {impact:.1f} | {_sentiment_label(sig.get('sentiment_score', 0))}",
                expanded=False,
            ):
                c1, c2, c3 = st.columns(3)
                c1.metric("Impact Score", f"{impact:.1f}")
                c2.metric("Sentiment", f"{sig.get('sentiment_score', 0):.3f}")
                c3.metric("Confidence", f"{sig.get('confidence', 0):.2%}")
                st.markdown(f"> {snippet[:300]}")
                st.json({
                    "entity": entity,
                    "event_type": sig.get("event_type"),
                    "explanation": exp,
                    "model_version": sig.get("model_version"),
                }, expanded=False)


# ===========================================================================
# TAB 2 – Stress Test
# ===========================================================================
with tab_stress:
    st.subheader("📊 Strategic Portfolio Stress Test")
    st.markdown(
        '<div class="disclaimer">⚠️ SYNTHETIC portfolio. Simplified first-order approximation. '
        'Not production-grade. See docs/ASSUMPTIONS.md.</div>',
        unsafe_allow_html=True,
    )

    # Pick trigger signal
    high_signals = [s for s in signals if s.get("impact_score", 0) >= impact_threshold]

    col_s1, col_s2 = st.columns([2, 1])
    with col_s1:
        if high_signals:
            sig_labels = [
                f"[{s['timestamp'][:16]}] {s.get('entity', {}).get('ticker','?')} – "
                f"{s.get('event_type','?')} (Impact {s.get('impact_score',0):.1f})"
                for s in high_signals[:20]
            ]
            sel_idx = st.selectbox("Triggering event", range(len(sig_labels)),
                                   format_func=lambda i: sig_labels[i])
            trigger_signal = high_signals[sel_idx]
        else:
            st.warning("No high-impact signals yet. Using demo Credit Event scenario.")
            trigger_signal = {
                "event_type": "Credit Event",
                "impact_score": 8.5,
                "entity": {"ticker": "SIVB", "name": "SVB Financial Group"},
                "sentiment_score": -0.82,
                "sources": [{"snippet": "SVB Financial collapses – regulators seize assets."}],
            }

    with col_s2:
        severity_override = st.slider(
            "Severity override (impact)",
            min_value=1.0, max_value=10.0,
            value=float(trigger_signal.get("impact_score", 7.0)),
            step=0.1,
        )

    st.markdown("---")

    # Run stress test
    portfolio = load_portfolio()
    from src.modules.stress_test.scenarios import get_scenario
    from src.modules.stress_test.revaluation import run_stress_test, reverse_stress_test

    event_type = trigger_signal.get("event_type", "Credit Event")
    shock = get_scenario(event_type, severity_override)
    result = run_stress_test(portfolio, shock)

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Portfolio Before", f"${result['before_value']:,.0f}")
    k2.metric("Portfolio After", f"${result['after_value']:,.0f}",
              delta=f"${result['total_pnl']:,.0f}",
              delta_color="inverse")
    k3.metric("P&L %", f"{result['pnl_pct']:.2f}%", delta_color="inverse")
    k4.metric("Scenario", event_type)

    # Waterfall chart
    ac_pnl = result["by_asset_class"]
    fig_wf = go.Figure(go.Waterfall(
        name="P&L by Asset Class",
        orientation="v",
        measure=["relative"] * len(ac_pnl) + ["total"],
        x=list(ac_pnl.keys()) + ["Total"],
        y=list(ac_pnl.values()) + [result["total_pnl"]],
        connector={"line": {"color": "#30363d"}},
        increasing={"marker": {"color": "#22c55e"}},
        decreasing={"marker": {"color": "#ef4444"}},
        totals={"marker": {"color": "#60a5fa"}},
    ))
    fig_wf.update_layout(
        title="P&L Waterfall by Asset Class",
        template="plotly_dark",
        height=400,
    )
    st.plotly_chart(fig_wf, use_container_width=True)

    col_sc, col_ct = st.columns(2)

    # By sector
    with col_sc:
        sector_pnl = result["by_sector"]
        df_sec = pd.DataFrame(list(sector_pnl.items()), columns=["Sector", "PnL"])
        fig_sec = px.bar(
            df_sec.sort_values("PnL"), x="PnL", y="Sector",
            orientation="h", title="Loss by Sector",
            color="PnL", color_continuous_scale=["#ef4444", "#f59e0b", "#22c55e"],
            template="plotly_dark",
        )
        st.plotly_chart(fig_sec, use_container_width=True)

    # By country
    with col_ct:
        country_pnl = result["by_country"]
        df_cnt = pd.DataFrame(list(country_pnl.items()), columns=["Country", "PnL"])
        fig_cnt = px.bar(
            df_cnt.sort_values("PnL"), x="PnL", y="Country",
            orientation="h", title="Loss by Country",
            color="PnL", color_continuous_scale=["#ef4444", "#f59e0b", "#22c55e"],
            template="plotly_dark",
        )
        st.plotly_chart(fig_cnt, use_container_width=True)

    # Top 10 losers
    st.subheader("🔺 Top 10 Loss Contributors")
    top_df = pd.DataFrame(result["top_losers"])
    if not top_df.empty:
        top_df["pnl"] = top_df["pnl"].apply(lambda x: f"${x:,.0f}")
        top_df["pnl_pct"] = top_df["pnl_pct"].apply(lambda x: f"{x:.2f}%")
        st.dataframe(top_df, use_container_width=True)

    # Triggering headline
    st.subheader("🗞️ Triggering Signal")
    sources = trigger_signal.get("sources", [])
    if sources:
        st.markdown(f"> {sources[0].get('snippet', '')[:400]}")
    st.json({
        "entity": trigger_signal.get("entity"),
        "event_type": event_type,
        "impact_score": trigger_signal.get("impact_score"),
        "sentiment_score": trigger_signal.get("sentiment_score"),
    }, expanded=False)

    # Reverse stress test
    st.subheader("🔄 Reverse Stress Test")
    loss_threshold = st.slider(
        "Loss threshold for reverse stress (%)",
        min_value=1.0, max_value=20.0, value=5.0, step=0.5,
    )
    base_shock = get_scenario(event_type, 10.0)
    rst = reverse_stress_test(portfolio, base_shock, loss_threshold_pct=loss_threshold)

    if rst["break_impact_score"] is not None:
        st.success(
            f"Portfolio breaches **{loss_threshold:.0f}%** loss at impact score "
            f"**{rst['break_impact_score']:.1f}**"
        )
    else:
        st.info(f"Portfolio does not breach {loss_threshold:.0f}% loss in the sweep (max severity tested).")

    sweep_df = pd.DataFrame(rst["sweep_results"])
    fig_rst = px.line(
        sweep_df, x="impact_score", y="pnl_pct",
        title="P&L % vs Impact Score (Reverse Stress Test)",
        template="plotly_dark",
        labels={"impact_score": "Impact Score", "pnl_pct": "Portfolio Loss %"},
    )
    fig_rst.add_hline(y=-loss_threshold, line_dash="dash", line_color="#ef4444",
                      annotation_text=f"Threshold: -{loss_threshold:.0f}%")
    if rst["break_impact_score"]:
        fig_rst.add_vline(x=rst["break_impact_score"], line_dash="dot",
                          line_color="#f59e0b",
                          annotation_text=f"Break at {rst['break_impact_score']}")
    st.plotly_chart(fig_rst, use_container_width=True)


# ===========================================================================
# TAB 3 – Rebalancer
# ===========================================================================
with tab_rebal:
    st.subheader("⚖️ Tactical Index Rebalancer (Module A)")
    st.markdown(
        '<div class="disclaimer">⚠️ Demonstration only. NOT a real trading strategy. '
        'Sentiment is not causality. See docs/ASSUMPTIONS.md.</div>',
        unsafe_allow_html=True,
    )

    from src.modules.rebalancer.rebalancer import Rebalancer, UNIVERSE, BASE_WEIGHT

    # Build rebalancer state from signals
    reb = Rebalancer()
    for sig in reversed(signals):
        ticker = sig.get("entity", {}).get("ticker")
        ts_raw = sig.get("timestamp", "")
        try:
            ts = pd.Timestamp(ts_raw)
        except Exception:
            ts = pd.Timestamp.now()
        if ticker in UNIVERSE:
            reb.update(ticker, sig.get("sentiment_score", 0), ts)
            reb.compute_weights(timestamp=ts)

    weights_df = reb.compute_weights()
    history_df = reb.get_history_df()

    if not weights_df.empty:
        # Current weights bar
        fig_w = px.bar(
            weights_df.sort_values("final_weight", ascending=True),
            x="final_weight", y="ticker",
            orientation="h",
            color="tilt_vs_base",
            color_continuous_scale=["#ef4444", "#94a3b8", "#22c55e"],
            title="Current Portfolio Weights (sentiment-tilted)",
            template="plotly_dark",
        )
        fig_w.add_vline(x=BASE_WEIGHT, line_dash="dash",
                        line_color="#60a5fa", annotation_text="Equal weight")
        st.plotly_chart(fig_w, use_container_width=True)

        # Weight table
        st.dataframe(weights_df.sort_values("tilt_vs_base", ascending=False),
                     use_container_width=True)

    if not history_df.empty:
        # Stacked area chart over time
        fig_area = px.area(
            history_df.reset_index(),
            x="timestamp", y=UNIVERSE,
            title="Portfolio Weights Over Time",
            template="plotly_dark",
        )
        st.plotly_chart(fig_area, use_container_width=True)
    else:
        st.info("Feed some signals through the engine first to see weight evolution.")


# ===========================================================================
# TAB 4 – Evaluation
# ===========================================================================
with tab_eval:
    st.subheader("📈 Model Evaluation")

    results_path = DOCS_DIR / "results.md"
    if results_path.exists():
        with open(results_path) as f:
            st.markdown(f.read())
    else:
        st.info(
            "Evaluation results not yet generated. "
            "Run `python -m src.eval.run_eval` after placing labeled data in "
            "`data/labeled_eval/labels.csv` and `data/raw_samples/phrasebank.csv`."
        )

    st.subheader("Baseline Comparison Summary")
    # Show any available result tables
    eval_data = {
        "Model": ["FinBERT", "VADER"],
        "Accuracy": ["(run eval)", "(run eval)"],
        "Macro-F1": ["(run eval)", "(run eval)"],
    }
    st.dataframe(pd.DataFrame(eval_data), use_container_width=True)
    st.caption("Run the evaluation module to populate real numbers. See docs/results.md.")
