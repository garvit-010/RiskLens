"""
RiskLens – single entry point.

Usage
-----
  python run.py                    # cached replay mode (default, no GPU needed)
  python run.py --live             # live NLP engine mode
  python run.py --api-only         # FastAPI server only
  python run.py --dashboard-only   # Streamlit dashboard only
  python run.py --precompute       # run engine over replay data and cache signals
  python run.py --gen-portfolio    # regenerate synthetic portfolio and scenarios
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import threading
from pathlib import Path

# Ensure project root on path
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("risklens")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RiskLens News-to-Risk Engine")
    p.add_argument("--live",           action="store_true", help="Run NLP engine live (not cached replay)")
    p.add_argument("--api-only",       action="store_true", help="Start FastAPI server only")
    p.add_argument("--dashboard-only", action="store_true", help="Start Streamlit dashboard only")
    p.add_argument("--precompute",     action="store_true", help="Process replay data and cache signals")
    p.add_argument("--gen-portfolio",  action="store_true", help="Regenerate synthetic portfolio + scenarios")
    return p.parse_args()


def run_precompute() -> None:
    """Run the engine over replay data and cache signals to JSONL."""
    from src.config import RAW_DIR, SIGNAL_JSONL_PATH
    from src.engine.pipeline import RiskEngine
    from src.ingestion.replay import ReplayAdapter
    from src.ingestion.kaggle_news import KaggleNewsAdapter
    from src.ingestion.kaggle_tweets import KaggleTweetsAdapter

    logger.info("=== RiskLens Precompute Mode ===")
    logger.info("Output: %s", SIGNAL_JSONL_PATH)

    # Clear old signals
    if SIGNAL_JSONL_PATH.exists():
        SIGNAL_JSONL_PATH.unlink()

    engine = RiskEngine(use_finbert=True, write_jsonl=True)
    adapters = []

    # Replay file (SVB event + generated demo data)
    replay_path = RAW_DIR / "replay_svb.jsonl"
    if replay_path.exists():
        adapters.append(ReplayAdapter(replay_path, speed_factor=0))

    # Kaggle datasets (if downloaded)
    news_path   = RAW_DIR / "news.csv"
    tweets_path = RAW_DIR / "tweets.csv"
    if news_path.exists():
        adapters.append(KaggleNewsAdapter(news_path))
    if tweets_path.exists():
        adapters.append(KaggleTweetsAdapter(tweets_path))

    if not adapters:
        logger.warning(
            "No data files found! Run with demo data by downloading:\n"
            "  - Kaggle news → data/raw_samples/news.csv\n"
            "  - Kaggle tweets → data/raw_samples/tweets.csv\n"
            "Or run `python run.py --gen-portfolio` to create the SVB replay file."
        )
        # Generate demo signals for the dashboard to show
        _generate_demo_signals(engine)
        return

    count = 0
    for signal in engine.process_adapters(adapters):
        count += 1
        if count % 100 == 0:
            logger.info("Processed %d documents …", count)

    logger.info("Precompute complete: %d signals written.", count)


def _generate_demo_signals(engine) -> None:
    """Generate synthetic demo signals so the dashboard is not empty."""
    logger.info("Generating demo signals …")
    demo_texts = [
        "SVB Financial collapses as regulators seize assets amid bank run fears",
        "Federal Reserve raises interest rates by 50 basis points to combat inflation",
        "JPMorgan reports record Q1 earnings beating analyst expectations",
        "Silicon Valley Bank faces liquidity crisis after bond portfolio losses",
        "Goldman Sachs downgraded by Moody's citing increased credit risk exposure",
        "Apple unveils new iPhone 15 with advanced AI capabilities",
        "Chevron acquires Hess Corporation in $53 billion all-stock deal",
        "SEC investigates cryptocurrency exchange for market manipulation",
        "China imposes new tariffs on US semiconductors escalating trade war",
        "Microsoft Azure outage disrupts financial services sector globally",
        "ESG regulations tighten as EU mandates carbon disclosure for all banks",
        "Morgan Stanley warns of recession risk as yield curve inverts sharply",
        "Wells Fargo settles mortgage fraud lawsuit for $3.7 billion",
        "Oil prices surge 15% following OPEC production cuts",
        "Bank of America reports covenant breach on commercial real estate portfolio",
    ]
    for text in demo_texts:
        signal = engine.process_text(text, source="demo")
        if signal:
            logger.info(
                "Demo signal: [%.1f] %s | %s",
                signal.impact_score, signal.entity["ticker"], signal.event_type
            )
    logger.info("Demo signals written.")


def start_api_server() -> subprocess.Popen:
    from src.config import API_PORT
    logger.info("Starting FastAPI server on port %d …", API_PORT)
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.api.app:app",
         "--host", "0.0.0.0", "--port", str(API_PORT), "--reload"],
        cwd=str(_ROOT),
    )
    return proc


def start_dashboard() -> subprocess.Popen:
    from src.config import STREAMLIT_PORT
    logger.info("Starting Streamlit dashboard on port %d …", STREAMLIT_PORT)
    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "src/dashboard/app.py",
         "--server.port", str(STREAMLIT_PORT),
         "--server.headless", "true",
         "--browser.gatherUsageStats", "false"],
        cwd=str(_ROOT),
    )
    return proc


def main() -> None:
    args = parse_args()

    if args.gen_portfolio:
        from src.modules.stress_test.portfolio import generate_portfolio
        from src.modules.stress_test.scenarios import save_scenarios_yaml
        from scripts.gen_replay import generate_svb_replay
        generate_portfolio(save=True)
        save_scenarios_yaml()
        generate_svb_replay()
        return

    if args.precompute:
        run_precompute()
        return

    if args.live:
        logger.info("Live mode: engine will run on-the-fly.")
        run_precompute()

    procs = []

    if not args.dashboard_only:
        procs.append(start_api_server())

    if not args.api_only:
        # If no signals yet, precompute them first
        from src.config import SIGNAL_JSONL_PATH
        if not SIGNAL_JSONL_PATH.exists() or SIGNAL_JSONL_PATH.stat().st_size == 0:
            logger.info("No cached signals found – running precompute …")
            run_precompute()
        procs.append(start_dashboard())

    if not procs:
        logger.error("No processes started. Use --api-only or --dashboard-only.")
        return

    logger.info("=" * 60)
    logger.info("RiskLens is running!")
    from src.config import API_PORT, STREAMLIT_PORT
    if not args.dashboard_only:
        logger.info("  API:       http://localhost:%d", API_PORT)
        logger.info("  API docs:  http://localhost:%d/docs", API_PORT)
    if not args.api_only:
        logger.info("  Dashboard: http://localhost:%d", STREAMLIT_PORT)
    logger.info("  Press Ctrl+C to stop.")
    logger.info("=" * 60)

    try:
        for p in procs:
            p.wait()
    except KeyboardInterrupt:
        logger.info("Shutting down …")
        for p in procs:
            p.terminate()


if __name__ == "__main__":
    main()
