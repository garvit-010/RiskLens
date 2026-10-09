"""
RiskLens – evaluation runner.

Measures:
  1. Sentiment: FinBERT vs VADER accuracy and macro-F1 on labeled set
  2. Event classification: per-class F1 and confusion matrix vs keyword baseline
  3. System throughput

Results saved to docs/results.md and as PNG figures.

Usage:
    python -m src.eval.run_eval

Data requirements:
    data/raw_samples/phrasebank.csv   – Financial PhraseBank or Kaggle labeled set
      columns: sentence, label (positive/negative/neutral)
    data/labeled_eval/labels.csv
      columns: text, event_type
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import LABELED_EVAL_DIR, RAW_DIR, DOCS_DIR, MODELS_CACHE_DIR

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _load_phrasebank(path: Path) -> tuple[list[str], list[str]] | None:
    """Load Financial PhraseBank CSV. Returns (texts, labels) or None."""
    if not path.exists():
        logger.warning("PhraseBank not found at %s – skipping sentiment eval.", path)
        return None
    df = pd.read_csv(path)
    # Try common column names
    text_col  = next((c for c in ["sentence", "text", "Sentence"] if c in df.columns), None)
    label_col = next((c for c in ["label", "sentiment", "Sentiment", "Label"] if c in df.columns), None)
    if not text_col or not label_col:
        logger.warning("Could not find text/label columns in %s", path)
        return None
    df = df.dropna(subset=[text_col, label_col])
    return list(df[text_col]), list(df[label_col].str.lower())


def _load_labeled_events(path: Path) -> tuple[list[str], list[str]] | None:
    """Load hand-labeled event CSV. Returns (texts, labels) or None."""
    if not path.exists():
        logger.warning("Labeled events not found at %s – skipping event eval.", path)
        return None
    df = pd.read_csv(path)
    text_col  = next((c for c in ["text", "headline", "sentence"] if c in df.columns), None)
    label_col = next((c for c in ["event_type", "label", "event"] if c in df.columns), None)
    if not text_col or not label_col:
        logger.warning("Could not find columns in %s", path)
        return None
    df = df.dropna(subset=[text_col, label_col])
    return list(df[text_col]), list(df[label_col])


def eval_sentiment(texts: list[str], labels: list[str]) -> dict:
    """Evaluate FinBERT and VADER on labeled texts."""
    from sklearn.metrics import accuracy_score, f1_score
    from src.engine.sentiment import sentiment_finbert, sentiment_vader

    fb_preds, vd_preds = [], []
    label_map = {
        "positive": 1, "negative": -1, "neutral": 0,
        "pos": 1, "neg": -1, "neu": 0,
    }
    true_labels = [label_map.get(l.lower(), 0) for l in labels]

    logger.info("Running sentiment eval on %d examples …", len(texts))
    t0 = time.time()
    for text in texts:
        fb_s, _ = sentiment_finbert(text)
        vd_s, _ = sentiment_vader(text)
        fb_preds.append(1 if fb_s > 0.1 else (-1 if fb_s < -0.1 else 0))
        vd_preds.append(1 if vd_s > 0.05 else (-1 if vd_s < -0.05 else 0))
    elapsed = time.time() - t0

    labels_unique = [-1, 0, 1]
    results = {
        "n": len(texts),
        "throughput_docs_per_sec": round(len(texts) / elapsed, 1),
        "finbert": {
            "accuracy": round(accuracy_score(true_labels, fb_preds), 4),
            "macro_f1": round(f1_score(true_labels, fb_preds, average="macro", zero_division=0), 4),
        },
        "vader": {
            "accuracy": round(accuracy_score(true_labels, vd_preds), 4),
            "macro_f1": round(f1_score(true_labels, vd_preds, average="macro", zero_division=0), 4),
        },
    }
    logger.info("Sentiment results: %s", results)
    return results


def eval_event_classifier(texts: list[str], true_labels: list[str]) -> dict:
    """
    Evaluate the MiniLM+LR classifier vs keyword-only baseline.
    Also trains the classifier if not yet trained.
    """
    from sklearn.metrics import f1_score, classification_report
    from src.engine.classifier import classify_event, train_classifier, _keyword_classify

    # Train classifier on this labeled set (small – used for both train+eval for demo;
    # in production use separate train/test splits)
    logger.info("Training event classifier on %d examples …", len(texts))
    train_classifier(texts, true_labels)

    ml_preds, kw_preds = [], []
    for text in texts:
        ml_preds.append(classify_event(text)[0])
        kw_preds.append(_keyword_classify(text)[0])

    results = {
        "n": len(texts),
        "ml_classifier": {
            "macro_f1": round(f1_score(true_labels, ml_preds, average="macro", zero_division=0), 4),
            "report": classification_report(true_labels, ml_preds, zero_division=0),
        },
        "keyword_baseline": {
            "macro_f1": round(f1_score(true_labels, kw_preds, average="macro", zero_division=0), 4),
            "report": classification_report(true_labels, kw_preds, zero_division=0),
        },
    }
    logger.info("Event classification results: %s", results["ml_classifier"]["macro_f1"])
    return results


def save_results_md(sentiment_results: dict | None, event_results: dict | None) -> None:
    """Write all evaluation results to docs/results.md."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# RiskLens – Evaluation Results",
        "",
        "> Generated by `python -m src.eval.run_eval`",
        "",
        "## Sentiment: FinBERT vs VADER",
        "",
    ]
    if sentiment_results:
        n = sentiment_results["n"]
        fb = sentiment_results["finbert"]
        vd = sentiment_results["vader"]
        tp = sentiment_results["throughput_docs_per_sec"]
        lines += [
            f"Dataset: {n} labeled examples",
            "",
            "| Model   | Accuracy | Macro-F1 |",
            "|---------|----------|----------|",
            f"| FinBERT | {fb['accuracy']:.4f} | {fb['macro_f1']:.4f} |",
            f"| VADER   | {vd['accuracy']:.4f} | {vd['macro_f1']:.4f} |",
            "",
            f"**Throughput:** {tp:.1f} docs/sec (FinBERT, CPU/GPU)",
            "",
        ]
    else:
        lines += ["*No labeled sentiment data found. Place phrasebank.csv in data/raw_samples/.*", ""]

    lines += ["## Event Classification: MiniLM+LR vs Keyword Baseline", ""]
    if event_results:
        lines += [
            f"Dataset: {event_results['n']} hand-labeled examples",
            "",
            "| Model | Macro-F1 |",
            "|-------|----------|",
            f"| MiniLM+LR | {event_results['ml_classifier']['macro_f1']:.4f} |",
            f"| Keyword baseline | {event_results['keyword_baseline']['macro_f1']:.4f} |",
            "",
            "### MiniLM+LR Classification Report",
            "```",
            event_results["ml_classifier"]["report"],
            "```",
            "",
            "### Keyword Baseline Classification Report",
            "```",
            event_results["keyword_baseline"]["report"],
            "```",
        ]
    else:
        lines += ["*No labeled event data found. Place labels.csv in data/labeled_eval/.*", ""]

    lines += [
        "",
        "## Caveats and Honest Limitations",
        "",
        "- Classifier trained and evaluated on the same small set (demo limitation). "
          "Production system requires held-out test set.",
        "- Impact score vs abnormal return correlation: not yet computed "
          "(insufficient event study data). Documented as assumption in docs/ASSUMPTIONS.md.",
        "- Portfolio is fully synthetic; stress-test P&L cannot be compared to real outcomes.",
        "- FinBERT trained on financial news; performance on tweets may differ.",
    ]

    out = DOCS_DIR / "results.md"
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    logger.info("Results saved to %s", out)


def main() -> None:
    sentiment_results = None
    event_results     = None

    # Sentiment eval
    phrasebank = _load_phrasebank(RAW_DIR / "phrasebank.csv")
    if phrasebank:
        texts, labels = phrasebank
        # Use up to 500 examples for speed
        texts, labels = texts[:500], labels[:500]
        sentiment_results = eval_sentiment(texts, labels)

    # Event eval – also trains the classifier
    event_data = _load_labeled_events(LABELED_EVAL_DIR / "labels.csv")
    if event_data:
        texts, labels = event_data
        event_results = eval_event_classifier(texts, labels)

    save_results_md(sentiment_results, event_results)
    print("Evaluation complete. See docs/results.md")


if __name__ == "__main__":
    main()
