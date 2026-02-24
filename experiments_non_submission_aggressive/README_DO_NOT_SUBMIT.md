# NON-SUBMISSION Experimental Lab

This directory is intentionally separated from the official submission pipeline.

- Purpose: aggressive model experiments to chase accuracy.
- Risk: higher overfitting chance and lower interpretability.
- Rule: do **NOT** upload any files from this directory as competition submission artifacts.

Run:

```powershell
python experiments_non_submission_aggressive/run_aggressive_experiments.py
```

Outputs:

- `experiments_non_submission_aggressive/outputs/NON_SUBMISSION_cv_fold_metrics.csv`
- `experiments_non_submission_aggressive/outputs/NON_SUBMISSION_cv_summary.csv`
- `experiments_non_submission_aggressive/outputs/NON_SUBMISSION_aggressive_report.md`
- `experiments_non_submission_aggressive/outputs/NON_SUBMISSION_matchup_predictions_aggressive.csv`
- `experiments_non_submission_aggressive/outputs/NON_SUBMISSION_best_model_meta.json`

