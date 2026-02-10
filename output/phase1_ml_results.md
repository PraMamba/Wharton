# ML Model Comparison Results

## 5-Fold Cross-Validation Results

| Model | Accuracy | Log-Loss | Brier Score |
|-------|----------|----------|-------------|
| Baseline (fold home-rate) | 56.4% | 0.6849 | 0.2459 |
| BT-only Logistic | 57.3% | 0.6814 | 0.2437 |
| Elastic Net Logistic | 57.0% | 0.6774 | 0.2421 |
| XGBoost | 57.5% | 0.7154 | 0.2575 |
| Blended Ensemble | 56.9% | 0.6833 | 0.2448 |

## Feature Importance (Elastic Net |Coefficients|)

| Feature | |Coefficient| |
|---------|-------------|
| bt_diff | 0.2600 |
| win_pct_diff | 0.2030 |
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
| win_pct_diff | 0.1455 |
| bt_diff | 0.1397 |
| gd_pg_diff | 0.1261 |
| xgd_per60_diff | 0.0921 |
| xg_share_diff | 0.0892 |
| save_pct_diff | 0.0853 |
| shooting_pct_diff | 0.0847 |
| pdo_diff | 0.0793 |
| xgd_pg_diff | 0.0790 |
| pim_pg_diff | 0.0789 |

## Key Findings

- **Best accuracy**: XGBoost (57.5%)
- **Best calibration (Brier)**: Elastic Net Logistic (0.2421)
- **Home-ice advantage**: Baseline win rate = 56.4%
- **Hockey is inherently unpredictable**: Even the best model achieves modest accuracy
- **Recommendation**: Use Elastic Net Logistic for probability predictions (best calibrated)

## ML Matchup Predictions vs Core Model

| Game | Home | Away | Core Model | Elastic Net ML |
|------|------|------|------------|----------------|
