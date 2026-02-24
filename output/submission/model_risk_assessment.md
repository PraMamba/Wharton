# Model Improvement Risk Assessment

## 1) Evidence Summary

### Random Stratified CV (Selection-Friendly)
- BT accuracy: 57.0% +/- 3.2%, log-loss: 0.6797, Brier: 0.2430
- Baseline accuracy: 56.4%, log-loss: 0.6849, Brier: 0.2459

### ID-Order Stress Test (Robustness Check, Not True Time Split)
- BT accuracy: 54.7% +/- 1.1%, log-loss: 0.7047, Brier: 0.2524
- Baseline accuracy: 56.2%, log-loss: 0.6860, Brier: 0.2464
- Delta (BT - baseline): accuracy -1.6%, log-loss +0.0187, Brier +0.0060
- Interpretation: In this ID-order stress test, BT was outperformed by the home-rate baseline on all three metrics. Because numeric `game_id` order is not true chronology in this simulated dataset, treat this as a pessimistic robustness check rather than evidence of deployment failure.

### Bootstrap Uncertainty (Random CV OOF)
- Accuracy delta 95% CI: [-2.4%, +3.7%]
- Log-loss delta 95% CI: [-0.0191, +0.0086]
- Brier delta 95% CI: [-0.0095, +0.0035]

## 2) Improvement Options and Risk-Balanced Decisions

| Option | Potential Benefit | Main Risk | Decision |
|---|---|---|---|
| ID-order stress test (numeric game_id sort) | Checks sensitivity beyond random CV | game_id order may not equal chronology | **Adopted** (robustness check, not a true temporal split) |
| Keep BT-only final model | Interpretability and stable submission format | Might miss small nonlinear lift | **Adopted** (lift from complex models was marginal/unstable) |
| Replace with boosted/ensemble model | Possible accuracy lift in random CV | Higher overfit risk and weaker interpretability/calibration | **Deferred** |
| Tune OT weight via nested CV | Potentially better treatment of OT uncertainty | Computational cost and variance with single-season data | **Deferred** |
| Add regularized BT (ridge/Firth) | Better numerical stability in extreme seasons | Bias introduction and extra assumptions | **Deferred** |
| Add bootstrap confidence bands to outputs | Better communication of ranking/probability uncertainty | More complex narrative for judges | **Adopted** (internal reporting) |

## 3) Official Documentation Used

- Scikit-learn calibration guide: https://scikit-learn.org/stable/modules/calibration.html
- Scikit-learn TimeSeriesSplit: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html
- Scikit-learn nested CV example: https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html
- Cawley & Talbot (JMLR 2010): https://jmlr.org/beta/papers/v11/cawley10a.html
