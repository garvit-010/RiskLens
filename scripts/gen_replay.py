"""
Generate the SVB replay JSONL file for the demo.

Historically verified news items and tweets around the SVB collapse and
March 2023 banking turmoil (March 8–21, 2023) in chronological order.

Run: python scripts/gen_replay.py
  or: python run.py --gen-portfolio  (calls this automatically)
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_OUT  = _ROOT / "data" / "raw_samples" / "replay_svb.jsonl"


# Verified chronological events from March 2023 US & European banking turmoil
_EVENTS = [
    # (timestamp_iso, source, text, ticker)
    ("2023-03-08T22:00:00+00:00", "replay_news",
     "SVB Financial Group sells $21B bond portfolio at $1.8B loss and seeks emergency $2.25B capital raise", "SIVB"),
    ("2023-03-08T23:00:00+00:00", "replay_social",
     "$SIVB SVB just dumped their bond portfolio at a huge loss. This is bad. Really bad.", "SIVB"),
    ("2023-03-09T12:00:00+00:00", "replay_news",
     "Silicon Valley Bank CEO urges venture capital clients to remain calm as deposit flight rumors circulate", "SIVB"),
    ("2023-03-09T13:00:00+00:00", "replay_news",
     "Peter Thiel's Founders Fund and top venture firms advise portfolio companies to withdraw deposits from SVB", "SIVB"),
    ("2023-03-09T14:00:00+00:00", "replay_social",
     "VCs telling all their portcos to pull money from $SIVB right now. Classic bank run dynamics.", "SIVB"),
    ("2023-03-09T15:00:00+00:00", "replay_news",
     "SVB Financial stock plunges 60% as bank run fears intensify among tech clients", "SIVB"),
    ("2023-03-09T16:00:00+00:00", "replay_social",
     "$SIVB halted multiple times down 60%. Clients trying to pull tens of billions today. Contagion spreading.", "SIVB"),
    ("2023-03-09T20:00:00+00:00", "replay_news",
     "Silicon Valley Bank clients attempt to withdraw $42B in single day, leaving lender with negative cash balance", "SIVB"),
    ("2023-03-10T16:00:00+00:00", "replay_news",
     "FDIC seizes Silicon Valley Bank in largest US bank failure since 2008 financial crisis", "SIVB"),
    ("2023-03-10T17:00:00+00:00", "replay_social",
     "SVB is gone. FDIC has taken over. $SIVB stock halted indefinitely. Regional banks getting hammered $BAC $JPM", "SIVB"),
    ("2023-03-10T19:00:00+00:00", "replay_social",
     "$JPM $BAC $WFC down 3-5% on SVB contagion fears. Watching credit spreads widen in real time.", "JPM"),
    ("2023-03-12T22:00:00+00:00", "replay_news",
     "Federal regulators guarantee all SVB deposits including uninsured amounts above $250,000", "SIVB"),
    ("2023-03-12T22:30:00+00:00", "replay_news",
     "Signature Bank shut down by New York regulators citing systemic risk concerns", None),
    ("2023-03-12T23:00:00+00:00", "replay_news",
     "Federal Reserve launches Bank Term Funding Program to provide emergency liquidity to eligible banks", None),
    ("2023-03-12T23:30:00+00:00", "replay_social",
     "Fed creating BTFP to backstop banking sector liquidity. Rate path now uncertain $GS $MS", "GS"),
    ("2023-03-13T12:00:00+00:00", "replay_news",
     "Moody's downgrades US banking sector outlook to negative following SVB collapse", None),
    ("2023-03-13T14:00:00+00:00", "replay_news",
     "JPMorgan and Bank of America see surge in deposit inflows as customers flee regional lenders", "JPM"),
    ("2023-03-13T16:00:00+00:00", "replay_news",
     "President Biden assures public that US banking system is safe and regulators will hold executives accountable", None),
    ("2023-03-14T14:00:00+00:00", "replay_news",
     "Goldman Sachs cuts US economic growth forecast citing tightening bank lending standards post-SVB", "GS"),
    ("2023-03-14T17:00:00+00:00", "replay_news",
     "Morgan Stanley warns of potential hit to banking sector from commercial real estate exposure", "MS"),
    ("2023-03-14T19:00:00+00:00", "replay_social",
     "Commercial real estate is the next shoe to drop post-SVB. Watching $WFC $BAC closely.", "WFC"),
    ("2023-03-15T10:00:00+00:00", "replay_news",
     "Credit Suisse shares plunge 24% as Saudi National Bank rules out further financial assistance", None),
    ("2023-03-15T15:00:00+00:00", "replay_social",
     "Credit Suisse CDS spreads hitting record distress levels. European banks under pressure.", None),
    ("2023-03-16T13:00:00+00:00", "replay_news",
     "European Central Bank raises interest rates by 50 basis points despite banking turmoil", None),
    ("2023-03-16T15:00:00+00:00", "replay_social",
     "ECB delivers 50bps hike signaling inflation focus despite banking volatility. Hawkish stance.", None),
    ("2023-03-16T19:00:00+00:00", "replay_news",
     "First Republic Bank receives $30B uninsured deposit rescue from coalition of 11 major US banks", None),
    ("2023-03-16T20:00:00+00:00", "replay_social",
     "First Republic getting $30B deposit lifeline from JPMorgan, BofA, Citi. Systemic relief rally $JPM $BAC", "BAC"),
    ("2023-03-19T18:00:00+00:00", "replay_news",
     "UBS agrees to acquire Credit Suisse for $3.2B in emergency Swiss government-brokered rescue", None),
    ("2023-03-19T20:00:00+00:00", "replay_social",
     "UBS buying Credit Suisse for pennies on the dollar with Swiss SNB backstop. Crisis averted for now.", None),
    ("2023-03-21T14:00:00+00:00", "replay_news",
     "Treasury Secretary Yellen tells bankers that deposit guarantees could be expanded if smaller banks face runs", None),
]


def generate_svb_replay() -> None:
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for ts_iso, source, text, ticker in _EVENTS:
        doc_id = hashlib.sha256(f"{source}::{text}".encode()).hexdigest()[:16]
        rec = {
            "id":        doc_id,
            "source":    source,
            "timestamp": ts_iso,
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
