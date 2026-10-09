"""
RiskLens – event classification.

Primary:  all-MiniLM-L6-v2 embeddings + logistic regression
Fallback: keyword rules (used when classifier confidence < threshold,
          or when the sentence-transformers library is unavailable)

Taxonomy (aligned to wholesale banking risk):
  Credit Event | Geopolitical | Macroeconomic | Regulatory/Legal |
  Merger/Acquisition | Earnings/Guidance | Product Launch |
  Operational/Cyber | Supply Chain/Commodity | ESG/Climate | Other/Noise

The classifier is trained on the hand-labeled evaluation set
(data/labeled_eval/labels.csv).  If that file is not present yet,
only the keyword fallback is used – the pipeline still runs end-to-end.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

from src.config import LABELED_EVAL_DIR, MINILM_MODEL, MODELS_CACHE_DIR

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Taxonomy
# ---------------------------------------------------------------------------
EVENT_TYPES = [
    "Credit Event",
    "Geopolitical",
    "Macroeconomic",
    "Regulatory/Legal",
    "Merger/Acquisition",
    "Earnings/Guidance",
    "Product Launch",
    "Operational/Cyber",
    "Supply Chain/Commodity",
    "ESG/Climate",
    "Other/Noise",
]

# ---------------------------------------------------------------------------
# Keyword rules (fallback)
# ---------------------------------------------------------------------------
_KEYWORD_RULES: dict[str, list[str]] = {
    "Credit Event": [
        "downgrade", "default", "covenant", "bankruptcy", "insolvency",
        "credit rating", "junk", "non-investment grade", "distressed",
        "missed payment", "debt restructuring", "svb", "silicon valley bank",
    ],
    "Geopolitical": [
        "war", "sanctions", "invasion", "military", "conflict", "geopolitical",
        "nato", "russia", "ukraine", "china", "tariff", "trade war",
    ],
    "Macroeconomic": [
        "inflation", "interest rate", "fed", "federal reserve", "gdp",
        "recession", "cpi", "rate hike", "rate cut", "ecb", "boe", "rbi",
    ],
    "Regulatory/Legal": [
        "sec", "fine", "penalty", "lawsuit", "regulation", "compliance",
        "investigation", "probe", "antitrust", "settlement", "doj",
    ],
    "Merger/Acquisition": [
        "acquisition", "merger", "takeover", "buyout", "deal", "acquire",
        "m&a", "purchase", "divest", "spin-off",
    ],
    "Earnings/Guidance": [
        "earnings", "revenue", "profit", "loss", "eps", "guidance",
        "q1", "q2", "q3", "q4", "quarterly", "beat", "miss", "outlook",
    ],
    "Product Launch": [
        "launch", "new product", "release", "unveil", "announce",
        "iphone", "model", "version", "upgrade",
    ],
    "Operational/Cyber": [
        "hack", "cyber", "breach", "outage", "data leak", "ransomware",
        "operational", "system failure", "ddos", "vulnerability",
    ],
    "Supply Chain/Commodity": [
        "supply chain", "shortage", "commodity", "oil", "gas", "chip",
        "semiconductor", "logistics", "freight", "drought", "crop",
    ],
    "ESG/Climate": [
        "esg", "climate", "carbon", "emissions", "sustainability",
        "renewable", "green", "net zero", "biodiversity", "deforestation",
    ],
}


def _keyword_classify(text: str) -> tuple[str, float]:
    """Return (event_type, confidence) using keyword rules."""
    text_lower = text.lower()
    scores: dict[str, int] = {}
    for etype, keywords in _KEYWORD_RULES.items():
        count = sum(1 for kw in keywords if kw in text_lower)
        if count:
            scores[etype] = count

    if not scores:
        return "Other/Noise", 0.4

    best = max(scores, key=scores.__getitem__)
    total = sum(scores.values())
    conf = min(0.75, 0.4 + scores[best] / total * 0.4)
    return best, conf


# ---------------------------------------------------------------------------
# Embedding + logistic regression classifier
# ---------------------------------------------------------------------------
_embedder   = None
_classifier = None

_CLASSIFIER_PATH = MODELS_CACHE_DIR / "event_classifier.pkl"
_CONF_THRESHOLD  = 0.55   # below this → use keyword fallback


def _load_embedder() -> None:
    global _embedder
    if _embedder is not None:
        return
    try:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading %s …", MINILM_MODEL)
        _embedder = SentenceTransformer(MINILM_MODEL, cache_folder=str(MODELS_CACHE_DIR))
        logger.info("MiniLM embedder loaded.")
    except Exception as exc:
        logger.warning("Could not load sentence-transformers: %s – keyword fallback only.", exc)


def _load_classifier() -> None:
    global _classifier
    if _classifier is not None:
        return
    if _CLASSIFIER_PATH.exists():
        try:
            with open(_CLASSIFIER_PATH, "rb") as fh:
                _classifier = pickle.load(fh)
            logger.info("Event classifier loaded from cache.")
        except Exception as exc:
            logger.warning("Could not load classifier: %s", exc)


def train_classifier(texts: list[str], labels: list[str]) -> None:
    """
    Train and cache a logistic regression classifier on labeled data.
    Called by the evaluation module when labeled_eval/labels.csv is present.
    """
    global _classifier
    _load_embedder()
    if _embedder is None:
        logger.error("Embedder unavailable – cannot train classifier.")
        return

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder

    logger.info("Training event classifier on %d examples …", len(texts))
    embeddings = _embedder.encode(texts, show_progress_bar=True, batch_size=32)

    le = LabelEncoder()
    y  = le.fit_transform(labels)

    clf = LogisticRegression(max_iter=1000, C=1.0, multi_class="multinomial")
    clf.fit(embeddings, y)

    _classifier = {"model": clf, "encoder": le}
    _CLASSIFIER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_CLASSIFIER_PATH, "wb") as fh:
        pickle.dump(_classifier, fh)
    logger.info("Classifier saved to %s.", _CLASSIFIER_PATH)


def classify_event(text: str) -> tuple[str, float]:
    """
    Return (event_type, confidence).

    Strategy:
    1. Try MiniLM + logistic regression.
    2. If confidence < threshold or models unavailable → keyword rules.
    """
    _load_embedder()
    _load_classifier()

    if _embedder is not None and _classifier is not None:
        try:
            emb = _embedder.encode([text], show_progress_bar=False)
            clf = _classifier["model"]
            le  = _classifier["encoder"]
            proba = clf.predict_proba(emb)[0]
            best_idx  = int(np.argmax(proba))
            confidence = float(proba[best_idx])
            event_type = le.inverse_transform([best_idx])[0]

            if confidence >= _CONF_THRESHOLD:
                return str(event_type), confidence
            # Blend with keyword result if low confidence
            kw_type, kw_conf = _keyword_classify(text)
            if kw_conf > confidence:
                return kw_type, kw_conf
            return str(event_type), confidence
        except Exception as exc:
            logger.warning("Classifier inference failed: %s", exc)

    return _keyword_classify(text)
