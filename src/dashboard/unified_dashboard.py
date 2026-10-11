"""
RiskLens – Standalone Unified Stakeholder Dashboard.

A dedicated single-page Streamlit dashboard designed specifically for
executive stakeholders, Chief Risk Officers, and hackathon jurors to see
everything at once on a single screen without clicking through tabs.

Usage:
    streamlit run src/dashboard/unified_dashboard.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

# Make sure project root is on sys.path
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import SIGNAL_JSONL_PATH, IMPACT_THRESHOLD, DOCS_DIR
from src.dashboard.unified_cockpit import render_unified_cockpit

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="RiskLens | Unified Stakeholder Cockpit",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

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


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## Executive Controls")
    auto_refresh = st.toggle("Auto-refresh (5 s)", value=False)
    st.markdown("---")

    impact_threshold = st.slider(
        "Impact trigger threshold",
        min_value=1.0, max_value=10.0,
        value=float(IMPACT_THRESHOLD), step=0.5,
    )

    st.markdown("---")
    st.markdown("### Ingested Corpus")
    signals = load_signals()
    st.metric("Total Ingested Signals", len(signals))
    high = [s for s in signals if s.get("impact_score", 0) >= impact_threshold]
    st.metric("Critical Alerts (>= Threshold)", len(high), delta_color="inverse")

    st.markdown("---")
    st.markdown(
        '<div style="background:#1c1207; border:1px solid #713f12; border-radius:8px; padding:0.6rem 0.8rem; color:#fbbf24; font-size:0.8rem;">'
        '<b>NOTICE:</b> Wholesale loan book (300 positions, seed 42) is synthetic for demonstration. '
        'Revaluations represent first-order MTM approximations.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.caption("RiskLens v1.0 | Offline-First News-to-Risk Engine")

if auto_refresh:
    time.sleep(5)
    st.rerun()

# ---------------------------------------------------------------------------
# Load Metrics & Portfolio
# ---------------------------------------------------------------------------
eval_metrics = None
json_path = DOCS_DIR / "eval_results.json"
if json_path.exists():
    try:
        with open(json_path, encoding="utf-8") as f:
            eval_metrics = json.load(f)
    except Exception:
        eval_metrics = None

portfolio_df = load_portfolio()

# ---------------------------------------------------------------------------
# Render Unified Stakeholder Cockpit
# ---------------------------------------------------------------------------
render_unified_cockpit(
    signals=signals,
    portfolio=portfolio_df,
    impact_threshold=impact_threshold,
    metrics=eval_metrics,
)
