# RiskLens: Fix & Hardening Brief for the AI Agent

**Context:** The repo (S&P Global & Crisil Campus Hackathon, individual submission) is structurally complete, but a review found claims that do not match the repo, a weak classifier evaluation, and stress-test gaps. This brief lists exactly what to fix, in priority order.

**Constraints that still apply**
- Hardware: GTX 1650 Ti (4 GB VRAM). No fine-tuning. `ProsusAI/finbert` fp16, `all-MiniLM-L6-v2`.
- System must still run offline in cached replay mode with no API keys: `python run.py`.
- **Honesty rule:** never claim data, accuracy, calibration, or test environments that do not exist. If something cannot be done, document the limitation instead.
- **Flag, do not guess.** If a dataset is missing, stop and tell the human instead of inventing data.
- Keep commits small, with meaningful messages. After each task, `pytest` must pass and `python run.py` must still start.
- Update `docs/ASSUMPTIONS.md` and `docs/results.md` whenever behavior or numbers change.

---

## Task 1: Fix the data claims (HIGHEST PRIORITY)

**Problem:** The README says ~5,000 Kaggle news rows and ~3,000 tweets were sampled. The repo has no `data/raw_samples/news.csv` or `tweets.csv`. All 29 shipped signals come from 30 hand-written SVB replay items. `replay_svb.jsonl` tags those items `kaggle_news` / `kaggle_tweets`, which is misleading.

**Do:**
1. Check whether `data/raw_samples/news.csv` and `tweets.csv` exist locally. **If they do not, ask the human to download them (Kaggle login needed). Do not fabricate them.**
2. If the files are provided: sample small subsets (about 500 news, 300 tweets; keep the repo light), run `python run.py --precompute`, and commit the regenerated `data/output/signals.jsonl`.
3. In `data/raw_samples/replay_svb.jsonl` and `scripts/gen_replay.py`, change the `source` values for the hand-written items to honest labels, for example `replay_news` and `replay_social`. Add both to `SOURCE_CREDIBILITY` in `src/engine/impact.py` (news-like about 0.80 to 0.85, social-like 0.55).
4. **If the Kaggle data is not available by the deadline:** rewrite README section 3 and the Key Results to say plainly that the demo corpus is a curated synthetic SVB replay (30 documents), and that Kaggle adapters are implemented but were not run on full data.
5. The README Dataset table must match exactly what is in `/data`.

**Acceptance:** every dataset claim in `README.md` and `data/README.md` corresponds to a file in the repo.

---

## Task 2: Make corroboration and source credibility actually work

**Problem:** every signal has source `replay`, so `CorroborationTracker` always returns 1.0 and tweet credibility never applies.

**Do:**
1. After Task 1, make sure replay documents carry distinct sources (news-like vs social-like) so the tracker sees independent sources.
2. Confirm in `signals.jsonl` that at least some signals have `corroboration > 1.0` and that social-source signals show credibility 0.55.
3. Add a unit test in `tests/test_impact.py` that feeds the tracker two different sources within the window and asserts the boost is 1.15.

---

## Task 3: Honest event-classifier evaluation

**Problem:** `src/eval/run_eval.py` trains the classifier on the same 67 headlines it evaluates on, and accuracy is still only 66%. Per-class results are nearly identical to the keyword baseline, and most signals carry keyword-level confidence (0.6 or 0.75), so the "ML" model is effectively the keyword fallback. `svb` and `silicon valley bank` are hardcoded keywords, and the labels include SVB headlines.

**Do:**
1. In `eval_event_classifier`, replace the train=test evaluation with **stratified 5-fold cross-validation** (use `StratifiedKFold`; if a class has fewer than 5 examples, fall back to `n_splits = min class count`, or use leave-one-out). Compute macro-F1 and a classification report on the out-of-fold predictions for both MiniLM+LR and the keyword baseline.
2. Report the cross-validated numbers in `docs/results.md` and the README. State the sample size (n=67) and that a difference of about 0.02 macro-F1 is within noise.
3. Embed the full dataset for the final production model only after evaluation (train on all 67 for deployment, evaluate only by CV).
4. Remove the hardcoded `svb` / `silicon valley bank` entries from the Credit Event keywords, or move them to a clearly marked demo-only list. Re-run the eval and report honestly if scores drop.
5. Consider a cheap improvement and measure it: lower `_CONF_THRESHOLD`, or tune the LR `C` inside the CV loop. Only keep a change if CV macro-F1 improves.
6. Add a confusion matrix PNG to `docs/` (`docs/event_confusion.png`).
7. Rewrite README results wording to: "comparable to keyword rules on a small set; main benefits are calibrated confidence and generalization to unseen phrasing" **unless the CV numbers show otherwise**.

**Acceptance:** `docs/results.md` clearly states the evaluation method (CV), n, and caveats. No sentence in the README implies the ML classifier clearly beats keywords unless the CV numbers support it.

---

## Task 4: Honest sentiment evaluation

**Problem:** `data/raw_samples/phrasebank.csv` has 60 rows, and many sentences look hand-written, not from Financial PhraseBank. README labels the result "Financial PhraseBank 93.3%". FinBERT was fine-tuned on PhraseBank, so the test is not independent.

**Do:**
1. Ask the human for the real Financial PhraseBank file (for example `Sentences_AllAgree.txt` or the Kaggle CSV). Do not invent sentences.
2. If provided: evaluate on at least 500 sentences, and state which agreement subset is used (50%, 66%, 75%, or 100% annotator agreement). Report FinBERT and VADER accuracy and macro-F1.
3. In the README and `results.md`, state: sample size, subset, and "FinBERT was fine-tuned on this dataset, so results are optimistic".
4. If the real file is not available: rename the file and label it honestly (`sentiment_sample_n60.csv`), report "n=60, mixed real and hand-written sentences, 95% CI about ±6 points", and remove the "Financial PhraseBank" claim.
5. Fix the throughput mismatch: the README says 5.7 docs/sec, `results.md` says 5.3. Use the number from the latest run in both places, and state the hardware.

---

## Task 5: Make the stress test genuinely credit-aware

**Problem:**
- `ShockVector.sector_overrides` exists but is never used.
- `oil_shock` and `fx_shock_em` are defined but never applied in `revaluation.py`.
- The PD multiplier and spread shocks hit every loan and bond uniformly. An SVB Credit Event multiplies PD on all sectors.
- The swap docstrings contradict each other.
- `reverse_stress_test` uses `abs(pnl_pct)`, so a gain would count as a breach.

**Do:**
1. **Sector targeting.** Pass the triggering signal's entity sector into the stress run (`run_stress_test(portfolio, shock, target_sector=None)`). For Credit Event (and optionally Operational/Cyber, Regulatory/Legal):
   - Apply the full PD multiplier and an extra spread shock (default +100 bp on top of the rating-based shock) to positions in the target sector.
   - Apply a **spillover** at a documented fraction (default 25% of the shock) to all other sectors.
   - Make these parameters live in `scenarios.py`/`scenarios.yaml`, not hardcoded in the engine.
   - Use `sector_overrides` for this, or remove the field if you implement it another way.
2. **Oil and FX.** Either apply them or remove them from the scenario output. Preferred: add a `commodity_beta` column for Energy-sector loans/bonds/equities (oil up means a small positive PnL for Energy and a small negative for Industrials/Consumer Disc.) and an `fx_exposure` for positions with non-US country. Keep it simple and documented. If time is short, delete the unused fields from `scenarios.yaml` and the dashboard, and state in `ASSUMPTIONS.md` that the portfolio has no direct commodity or FX exposure.
3. **Swap comments.** Fix the module docstring and function docstring. A pay-fixed swap gains when rates rise; the bank is the fixed-rate payer, not "receiver". The code (`pnl = dv01 * rate_shock_bps`) is consistent with pay-fixed. Make all comments agree.
4. **Reverse stress test.** Use signed loss (`pnl_pct <= -threshold`) instead of `abs()`. Refine the break point with bisection after the coarse sweep. Return the break impact score, the corresponding scaled shock values, and the monetary loss.
5. **Scaling.** Document in the dashboard and `ASSUMPTIONS.md` that shock severity scales linearly from 0% at impact 5 to 100% at impact 10, so an impact-7.1 event applies about 42% of the base shock.
6. Add tests in `tests/test_revaluation.py`:
   - sector-targeted shock produces a larger loss in the target sector than in others
   - spillover is applied at the documented fraction
   - reverse stress ignores gains
   - a Credit Event on Financials hits Financials loans harder than Utilities loans

**Acceptance:** `run_stress_test` takes a target sector, the dashboard passes it from the trigger signal, and the tests above pass.

---

## Task 6: Clean up misclassified demo signals

**Problem (in shipped `signals.jsonl`):**
- "Morgan Stanley warns of potential $30B hit to banking sector from commercial real estate" is tagged Geopolitical at impact 8.18 (would trigger a geopolitical stress test).
- "Goldman Sachs cuts US growth forecast" is tagged Credit Event (8.54).
- "First Republic Bank rescue plan announced" is tagged Product Launch (keyword "announce").
- "Credit Suisse faces crisis ..." is tagged Other/Noise (2.54) despite sentiment -0.94.
- "VCs telling portcos to pull money from SIVB" is tagged Other/Noise.

**Do:**
1. After Task 3, re-run `python run.py --precompute` and review the replay signals by hand. List each misclassification.
2. Improve keyword coverage using generic banking vocabulary, not demo-specific phrases: add terms such as `bank run`, `bank failure`, `liquidity crisis`, `deposit outflows`, `rescue`, `bailout`, `commercial real estate`, `CRE exposure`, `growth forecast`, `tightening` to the right classes. Reduce overly generic Product Launch keywords (`announce`, `model`, `version`, `release`).
3. Re-run the CV from Task 3 to check these changes do not hurt. Report the final numbers.
4. For any demo-relevant misclassification that remains, list it under "Known limitations" in the deck and README instead of hiding it.
5. Choose the demo trigger event as a correctly classified, high-impact signal, and note the choice in `docs/DEMO_NOTES.md` (event, timestamp, expected stress result).

---

## Task 7: README and repo hygiene

1. Fill the header with placeholders clearly marked for the human: **do not invent the name, email, college, or video link**. Leave them as `[TODO: human to fill]`, and list them in the final report.
2. Replace `<your-repo-url>` and `<repo-folder>` with a bash-fenced block using the real repo URL once the human confirms the final repo name.
3. Remove "(also tested on Ubuntu 22.04)" unless the human confirms it was tested. State only the OS that was actually used.
4. Change "mirrors regulatory requirements (ICAAP, DFAST)" to "conceptually aligned with scenario analysis practices used in ICAAP and supervisory stress testing".
5. Add to Key Results (computed from real runs, not typed from memory):
   - SVB replay: trigger signal, stress P&L in dollars and percent, split by asset class
   - reverse stress result (impact score and shock at 5% loss)
   - Module A: weight tilts and the backtest vs equal-weight with the caveat that it is illustrative
   - one dashboard screenshot saved to `docs/` and embedded
6. Move `TASK.MD` into `docs/` (rename `docs/BUILD_BRIEF.md`) or delete it.
7. Prices: `data/raw_samples/prices/` is empty. Cache about 15 S&P 100 tickers plus SPY from yfinance to small CSVs (needs network; ask the human to run `scripts` if offline). Confirm Module A and its backtest run fully offline from these files, or document clearly that the backtest needs network.
8. `data/README.md`: the labeled events are marked "draft, to be reviewed". Ask the human to confirm they reviewed and corrected `labels.csv`, then update the status line.
9. Add the repo name note: the human will rename the repo to `<college>-<candidate-name>-hackathon`. Also add one README line: "Unrelated to the commercial FAIR-based cyber-risk product of the same name", **or** propose a new name to the human.

---

## Task 8: Final verification (do last, report results)

1. In a fresh virtualenv on a clean clone: `pip install -r requirements.txt` then `python run.py`. Confirm API on 8000 and dashboard on 8501 start with no keys and no network.
2. Run `pytest -q` and report the pass count.
3. Click through the dashboard: signal feed, stress test with the chosen trigger, severity slider, reverse stress test, Module A tab. Report any error.
4. Confirm every number in the README and `docs/results.md` matches the latest run output.
5. Regenerate `docs/presentation.pdf` (7 slides max) and `docs/architecture.png` if numbers or architecture changed, using the existing `scripts/`.
6. Produce a short final report listing: what was fixed, what was not done and why, and the human TODOs.

---

## Human TODOs (the agent cannot do these)

- Download Kaggle news, tweets, and the real Financial PhraseBank file.
- Review and correct `data/labeled_eval/labels.csv`.
- Fill in the name, email, college, repo name, and video link.
- Cache the price CSVs if the agent's environment is offline.
- Record the demo video and rehearse the 5-minute live demo.
- Verify the access checklist: public repo, unlisted YouTube video playing in incognito, no Drive or OneDrive links.

## Priority if time is short

1. Task 1 (false data claims), Task 3 (honest eval), Task 4 (sentiment framing), and Task 7 items 1 to 5. These are the claims a jury can check.
2. Task 5 (sector targeting). This is the main domain-understanding upgrade.
3. Task 6, then Task 2.
4. Task 8, always last.