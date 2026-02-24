# Quick Skim (Reviewer Fast Path)

## What This Project Is

WHSDSC hockey analytics project (Phase 1 submission + audit artifacts).

- Goal: produce
  - `power_rankings.csv`
  - `matchup_predictions.csv`
  - `line_disparity.csv`
  - one submission PNG (`TheIsoLab.png`)
  - methodology report (`methodology.md`)
- Data note: `whl_2025.csv` is competition-provided simulated data (not real-world game results).

## Core Submission Model (What We Actually Submit)

Main submission pipeline: `phase1_submit.py`

- Game-level aggregation from line-level rows (`game_id`)
- OT/SO-aware Bradley-Terry (uses `went_ot`; OT/SO wins downweighted)
- Calibrated logistic regression on BT strength difference for `home_win_prob`
- Random 5-fold stratified CV for primary evaluation (accuracy / log-loss / Brier)
- ID-order stress test (numeric `game_id` sort only; not true chronology)
- Submission schema validation (CSV columns/order/types/nulls/ranges) + final ZIP packaging

## ML Challenger Module (Non-Submission Benchmarking)

ML comparison script: `phase1_ml.py`

- Baseline (fold home-rate)
- BT-only Logistic (same core idea as submission)
- Elastic Net Logistic
- XGBoost (benchmark)
- Blended Ensemble (manual weights; benchmark only)

Important caveat:
- CV metrics are leakage-safe
- Feature importance is computed from full-data refits (interpretability only, not CV evidence)

## Aggressive Experiments (Do Not Submit)

`experiments_non_submission_aggressive/run_aggressive_experiments.py`

- High-capacity / aggressive benchmark playground
- Uses random stratified CV + ID-order stress test
- Model selection now prioritizes probability quality (Brier, then log-loss), not tuned accuracy
- Outputs are explicitly marked `NON_SUBMISSION`

## Files to Read First (5-minute Review)

- `phase1_submit.py` (actual submission logic)
- `output/submission/methodology.md` (final submission narrative)
- `output/submission/model_risk_assessment.md` (risk memo; internal)
- `output/submission/ranking_weight_sensitivity.md` (weight robustness evidence; internal)
- `phase1_ml.py` (ML comparisons + feature-importance caveat)
- `report/report_summary_zh.tex` (full Chinese report source)

## Quick Checks (Fast Repro)

Run syntax checks:

```powershell
python -m py_compile phase1_submit.py phase1_ml.py experiments_non_submission_aggressive/run_aggressive_experiments.py
```

Run submission pipeline (generates `output/submission/*` + final zip):

```powershell
python phase1_submit.py
```

Expected headline metrics (approx):

- Random-CV accuracy: `57.0%`
- Random-CV Brier: `0.2430`
- ID-order stress BT accuracy: `54.7%`
- ID-order stress BT Brier: `0.2524`

Run ML comparison:

```powershell
python phase1_ml.py
```

Expected ML comparison highlights (approx):

- Elastic Net best random-CV accuracy/probability quality
- BT-only remains strong, interpretable submission baseline
- XGBoost weaker calibration

## What Auditors Should Pay Attention To

- Data leakage:
  - fold-wise refits for BT/logistic and ML feature building
- OT/SO handling:
  - `went_ot` weighting consistency across submission + ML BT
- ID-order wording:
  - should be treated as stress test, not real temporal validation
- Submission compliance:
  - CSV schema validation and final ZIP contents
- Documentation honesty:
  - ID-order underperformance vs baseline is explicitly stated
  - feature-importance caveat is explicit
  - ranking weight robustness claims are backed by generated evidence file

## Final Submission Package

Expected zip (`output/TheIsoLab_WHSDSC_Phase1_Submission.zip`) contains exactly:

- `power_rankings.csv`
- `matchup_predictions.csv`
- `line_disparity.csv`
- `TheIsoLab.png`
- `methodology.md`
