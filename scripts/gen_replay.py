"""
Generate the SVB replay JSONL file for the demo.

Simulates ~30 news items and tweets around the SVB collapse
(March 8–13, 2023) in timestamp order.

Run: python scripts/gen_replay.py
  or: python run.py --gen-portfolio  (calls this automatically)
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_OUT  = _ROOT / "data" / "raw_samples" / "replay_svb.jsonl"


_EVENTS = [
    # (hours_offset, source, text, ticker)
    (0,   "kaggle_news",   "SVB Financial Group sells bond portfolio at $1.8B loss to shore up liquidity", "SIVB"),
    (1,   "kaggle_tweets", "$SIVB SVB just dumped their bond portfolio at a huge loss. This is bad. Really bad.", "SIVB"),
    (2,   "kaggle_news",   "Silicon Valley Bank seeks emergency capital raise as deposits flee tech startups", "SIVB"),
    (3,   "kaggle_tweets", "SVB Financial announces equity offering. Banks don't do emergency capital raises unless they're desperate $SIVB", "SIVB"),
    (4,   "kaggle_news",   "Peter Thiel's Founders Fund advises portfolio companies to withdraw SVB deposits", "SIVB"),
    (5,   "kaggle_tweets", "VCs telling all their portcos to pull money from $SIVB right now. Classic bank run dynamics.", "SIVB"),
    (6,   "kaggle_news",   "SVB Financial stock plunges 60% as bank run fears intensify among tech clients", "SIVB"),
    (8,   "kaggle_news",   "FDIC seizes Silicon Valley Bank in largest US bank failure since 2008 financial crisis", "SIVB"),
    (9,   "kaggle_tweets", "SVB is gone. FDIC has taken over. $SIVB stock halted. Ripple effects hitting $BAC $JPM $GS", "SIVB"),
    (10,  "kaggle_news",   "Federal regulators guarantee all SVB deposits including uninsured amounts above $250,000", "SIVB"),
    (11,  "kaggle_tweets", "$JPM $BAC $WFC all down 3-5% on SVB contagion fears. Watching credit spreads widen in real time.", "JPM"),
    (12,  "kaggle_news",   "Signature Bank shut down by New York regulators citing systemic risk concerns", "SIVB"),
    (13,  "kaggle_news",   "Federal Reserve launches Bank Term Funding Program to prevent further bank failures", None),
    (14,  "kaggle_tweets", "Fed creating BTFP to backstop banks. Rate hike path now uncertain $GS $MS", "GS"),
    (15,  "kaggle_news",   "JPMorgan and Bank of America see surge in deposit inflows as customers flee smaller banks", "JPM"),
    (16,  "kaggle_news",   "Credit Suisse faces crisis as largest shareholder rules out additional investment", None),
    (18,  "kaggle_news",   "Goldman Sachs cuts US growth forecast citing tighter lending standards post-SVB", "GS"),
    (20,  "kaggle_news",   "Morgan Stanley warns of potential $30B hit to banking sector from commercial real estate", "MS"),
    (22,  "kaggle_tweets", "CRE exposure is the next shoe to drop post-SVB. Watch $WFC $BAC $C closely.", "WFC"),
    (24,  "kaggle_news",   "Moody's downgrades US banking sector outlook to negative following SVB collapse", None),
    (26,  "kaggle_news",   "First Republic Bank rescue plan announced with $30B deposit injection from major banks", None),
    (28,  "kaggle_tweets", "First Republic getting a lifeline from big banks. Systemic risk contained for now? $BAC $JPM", "BAC"),
    (30,  "kaggle_news",   "Senate Banking Committee announces emergency hearing on SVB failure and regulatory oversight", None),
    (32,  "kaggle_news",   "European Central Bank raises rates 50bps despite banking turmoil signaling inflation priority", None),
    (34,  "kaggle_tweets", "ECB raises 50bps despite SVB contagion fears. Hawkish. Bond markets not happy. $GS $MS", "GS"),
    (36,  "kaggle_news",   "UBS agrees to acquire Credit Suisse for $3.2B in emergency government-brokered deal", None),
    (38,  "kaggle_news",   "Fed pauses rate hike cycle amid banking sector stress and tightening credit conditions", None),
    (40,  "kaggle_tweets", "Fed pausing? Market pricing 100bps of cuts by year end now. Huge pivot. $JPM $MS $GS", "JPM"),
    (42,  "kaggle_news",   "KPMG gave SVB Financial clean audit opinion just two weeks before FDIC seizure", "SIVB"),
    (44,  "kaggle_news",   "Treasury Secretary Yellen: Banking system remains sound but vigilant monitoring continues", None),
]

BASE_TS = datetime(2023, 3, 8, 9, 0, 0, tzinfo=timezone.utc)


def generate_svb_replay() -> None:
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for hours, source, text, ticker in _EVENTS:
        ts = BASE_TS + timedelta(hours=hours)
        doc_id = hashlib.sha256(f"{source}::{text}".encode()).hexdigest()[:16]
        rec = {
            "id":        doc_id,
            "source":    source,
            "timestamp": ts.isoformat(),
            "text":      text,
            "ticker":    ticker,
            "lang":      "en",
        }
        records.append(rec)

    with open(_OUT, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")

    print(f"SVB replay file written: {_OUT}  ({len(records)} records)")


if __name__ == "__main__":
    generate_svb_replay()
