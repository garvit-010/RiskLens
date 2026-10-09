"""
RiskLens – global configuration loaded from .env / environment variables.
All modules import from here so there is one single source of truth.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (two levels up from this file)
_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env", override=False)


# ---------------------------------------------------------------------------
# Runtime flags
# ---------------------------------------------------------------------------
LIVE_MODE: bool = os.getenv("RISKLENS_LIVE_MODE", "false").lower() == "true"

# ---------------------------------------------------------------------------
# Server ports
# ---------------------------------------------------------------------------
STREAMLIT_PORT: int = int(os.getenv("STREAMLIT_PORT", "8501"))
API_PORT: int = int(os.getenv("API_PORT", "8000"))

# ---------------------------------------------------------------------------
# Engine thresholds
# ---------------------------------------------------------------------------
IMPACT_THRESHOLD: float = float(os.getenv("IMPACT_THRESHOLD", "7.0"))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR: Path = _ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw_samples"
OUTPUT_DIR: Path = DATA_DIR / "output"
PORTFOLIO_DIR: Path = DATA_DIR / "portfolio"
SCENARIOS_DIR: Path = DATA_DIR / "scenarios"
LABELED_EVAL_DIR: Path = DATA_DIR / "labeled_eval"
DOCS_DIR: Path = _ROOT / "docs"
MODELS_CACHE_DIR: Path = _ROOT / ".model_cache"

# Ensure critical output dirs exist at import time
for _d in [OUTPUT_DIR, PORTFOLIO_DIR, SCENARIOS_DIR, LABELED_EVAL_DIR,
           DOCS_DIR, MODELS_CACHE_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Model identifiers
# ---------------------------------------------------------------------------
FINBERT_MODEL: str = "ProsusAI/finbert"
MINILM_MODEL: str = "all-MiniLM-L6-v2"
MODEL_VERSION: str = "risklens-0.1"

# ---------------------------------------------------------------------------
# Engine parameters
# ---------------------------------------------------------------------------
DEDUP_SIMILARITY_THRESHOLD: float = 0.85   # rapidfuzz ratio threshold
CORROBORATION_WINDOW_HOURS: int = 6        # hours to look back for corroboration
SIGNAL_JSONL_PATH: Path = OUTPUT_DIR / "signals.jsonl"

# ---------------------------------------------------------------------------
# NewsAPI (stub only – not required)
# ---------------------------------------------------------------------------
NEWSAPI_KEY: str = os.getenv("NEWSAPI_KEY", "")
