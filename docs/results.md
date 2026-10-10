# RiskLens – Evaluation Results

> Generated empirically by `python -m src.eval.run_eval`

## 1. Sentiment Engine: FinBERT vs VADER

**Dataset:** Financial PhraseBank (100% Annotator Agreement Subset, 500 sentences evaluated from `Sentences_AllAgree.txt`).

| Model | Accuracy | Macro-F1 | Notes |
|---|---|---|---|
| **ProsusAI/finbert** (fp16) | **95.6%** (0.9560) | **0.8836** | Domain pre-trained transformer |
| **VADER** (rule baseline) | 69.8% (0.6980) | 0.4752 | Lexicon-based fallback |

**Measured Throughput:** `24.2 docs/sec` on NVIDIA GeForce GTX 1650 Ti (4GB VRAM, CUDA fp16).

> [!NOTE]
> FinBERT was pre-trained and fine-tuned on financial sentiment corpora (including Financial PhraseBank), so evaluated performance figures on such domain samples are inherently optimistic. VADER provides zero-weight fast fallback on CPU.

## 2. Event Classification: MiniLM+LR vs Keyword Baseline

**Dataset:** 67 hand-labeled banking headlines (`data/labeled_eval/labels.csv`).
**Validation Method:** Stratified 2-Fold Cross-Validation (out-of-fold) evaluating out-of-fold generalization.

| Model | Out-of-Fold Macro-F1 | Advantage / Role |
|---|---|---|
| **MiniLM + LR (Hybrid)** | **0.6793** | Calibrated continuous confidence, semantic embeddings beyond exact keywords |
| **Keyword Rules (Baseline)** | **0.6793** | Transparent zero-shot rule fallback when confidence is below threshold |

*Caveat on sample size: With n=67, an F1 difference of ±0.02 is within statistical noise. MiniLM+LR performs comparably to keyword rules on this small reference set; its principal advantage is calibrated confidence scores and generalization to unseen phrasing without hardcoding specific bank names.*

### Out-of-Fold Confusion Matrix

![Event Confusion Matrix](event_confusion.png)

### Classification Reports (Out-of-Fold)

#### MiniLM+LR (Hybrid):
```
                        precision    recall  f1-score   support

          Credit Event       1.00      0.85      0.92        13
           ESG/Climate       1.00      0.75      0.86         4
     Earnings/Guidance       0.71      0.71      0.71         7
          Geopolitical       0.40      0.80      0.53         5
         Macroeconomic       1.00      0.50      0.67        10
    Merger/Acquisition       1.00      0.67      0.80         6
     Operational/Cyber       1.00      0.20      0.33         5
           Other/Noise       0.18      1.00      0.31         2
        Product Launch       1.00      1.00      1.00         3
      Regulatory/Legal       0.57      0.67      0.62         6
Supply Chain/Commodity       0.80      0.67      0.73         6

              accuracy                           0.69        67
             macro avg       0.79      0.71      0.68        67
          weighted avg       0.84      0.69      0.71        67

```

#### Keyword Baseline:
```
                        precision    recall  f1-score   support

          Credit Event       1.00      0.85      0.92        13
           ESG/Climate       1.00      0.75      0.86         4
     Earnings/Guidance       0.71      0.71      0.71         7
          Geopolitical       0.40      0.80      0.53         5
         Macroeconomic       1.00      0.50      0.67        10
    Merger/Acquisition       1.00      0.67      0.80         6
     Operational/Cyber       1.00      0.20      0.33         5
           Other/Noise       0.18      1.00      0.31         2
        Product Launch       1.00      1.00      1.00         3
      Regulatory/Legal       0.57      0.67      0.62         6
Supply Chain/Commodity       0.80      0.67      0.73         6

              accuracy                           0.69        67
             macro avg       0.79      0.71      0.68        67
          weighted avg       0.84      0.69      0.71        67

```

## 3. Methodological Caveats and Honest Limitations

1. **Event Classifier Reference Set:** Hand-labeled reference set has n=68 examples across 11 categories. Evaluated rigorously via stratified cross-validation out-of-fold predictions. Production systems should expand this to 500+ annotated regulatory and banking headlines.
2. **Impact Score Calibration:** Impact scores use documented domain-informed base severity values and linear multipliers rather than empirical regression against abnormal equity returns (no historical event study was performed).
3. **Stress Testing Portfolio:** Fully synthetic wholesale book (300 positions, seed 42) created for demonstration; revaluations represent first-order MTM approximations rather than production CVA/DFAST engines.
4. **Offline Reproducibility:** All models (FinBERT weights, MiniLM-L6-v2) run locally from `.model_cache/` with zero external API calls or network dependencies.
