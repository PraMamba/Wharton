# ML Model Comparison Results

## 5-Fold Cross-Validation Results

| Model | Accuracy | Log-Loss | Brier Score |
|-------|----------|----------|-------------|
| Baseline (fold home-rate) | 56.4% | 0.6849 | 0.2459 |
| BT-only Logistic | 57.0% | 0.6796 | 0.2430 |
| Elastic Net Logistic | 58.0% | 0.6758 | 0.2412 |
| XGBoost | 56.7% | 0.7157 | 0.2574 |
| Blended Ensemble | 57.8% | 0.6827 | 0.2443 |

## Feature Importance Caveat

Feature importance values below come from **full-data refits** of the final models (post-hoc interpretation only). They are not leakage-safe CV estimates and should not be interpreted as out-of-sample effect sizes.

## Feature Importance (Elastic Net |Coefficients|)

| Feature | |Coefficient| |
|---------|-------------|
| win_pct_diff | 0.2989 |
| bt_diff | 0.1039 |
| gd_pg_diff | 0.0000 |
| xgd_pg_diff | 0.0000 |
| xg_share_diff | 0.0000 |
| xgd_per60_diff | 0.0000 |
| shooting_pct_diff | 0.0000 |
| save_pct_diff | 0.0000 |
| pdo_diff | 0.0000 |
| pim_pg_diff | 0.0000 |

## Feature Importance (XGBoost Gain)

| Feature | Importance |
|---------|------------|
| bt_diff | 0.1612 |
| gd_pg_diff | 0.1326 |
| win_pct_diff | 0.1132 |
| xgd_per60_diff | 0.0956 |
| xg_share_diff | 0.0937 |
| save_pct_diff | 0.0893 |
| shooting_pct_diff | 0.0839 |
| pdo_diff | 0.0788 |
| pim_pg_diff | 0.0777 |
| xgd_pg_diff | 0.0740 |

## Key Findings

- **Best accuracy**: Elastic Net Logistic (58.0%)
- **Best calibration (Brier)**: Elastic Net Logistic (0.2412)
- **Home-ice advantage**: Baseline win rate = 56.4%
- **Hockey is inherently unpredictable**: Even the best model achieves modest accuracy
- **Recommendation**: Use Elastic Net Logistic for probability predictions (best calibrated)

## ML Matchup Predictions vs Core Model

| Game | Home | Away | Core Model | Elastic Net ML |
|------|------|------|------------|----------------|
| 1 | brazil | kazakhstan | 0.8524 | 80.4% |
| 2 | netherlands | mongolia | 0.816 | 79.0% |
| 3 | peru | rwanda | 0.794 | 75.9% |
| 4 | thailand | oman | 0.7327 | 70.4% |
| 5 | pakistan | germany | 0.7121 | 68.7% |
| 6 | india | usa | 0.7213 | 69.5% |
| 7 | panama | switzerland | 0.6882 | 66.2% |
| 8 | iceland | canada | 0.6744 | 65.9% |
| 9 | china | france | 0.6693 | 65.8% |
| 10 | philippines | morocco | 0.6452 | 62.6% |
| 11 | ethiopia | saudi_arabia | 0.6401 | 61.1% |
| 12 | singapore | new_zealand | 0.591 | 57.1% |
| 13 | guatemala | south_korea | 0.5971 | 59.4% |
| 14 | uk | mexico | 0.6045 | 58.1% |
| 15 | vietnam | serbia | 0.5414 | 54.6% |
| 16 | indonesia | uae | 0.6332 | 60.2% |
