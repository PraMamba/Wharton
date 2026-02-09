# ML Model Comparison Results

## 5-Fold Cross-Validation Results

| Model | Accuracy | Log-Loss | Brier Score |
|-------|----------|----------|-------------|
| Baseline (home=~51%) | 56.4% | 0.6849 | 0.2459 |
| BT-only Logistic | 60.2% | 0.6573 | 0.2326 |
| Elastic Net Logistic | 59.8% | 0.6591 | 0.2334 |
| XGBoost | 57.2% | 0.6899 | 0.2468 |
| Blended Ensemble | 58.6% | 0.6634 | 0.2355 |

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

- **Best accuracy**: BT-only Logistic (60.2%)
- **Best calibration (Brier)**: BT-only Logistic (0.2326)
- **Home-ice advantage**: Baseline win rate = 56.4%
- **Hockey is inherently unpredictable**: Even the best model achieves modest accuracy
- **Recommendation**: Use BT-only Logistic for probability predictions (best calibrated)

## ML Matchup Predictions vs Core Model

| Game | Home | Away | Core Model | Elastic Net ML |
|------|------|------|------------|----------------|
| 1 | brazil | kazakhstan | 0.7653 | 82.9% |
| 2 | netherlands | mongolia | 0.7885 | 82.1% |
| 3 | peru | rwanda | 0.7353 | 78.2% |
| 4 | thailand | oman | 0.7037 | 72.0% |
| 5 | pakistan | germany | 0.6723 | 70.3% |
| 6 | india | usa | 0.6397 | 71.2% |
| 7 | panama | switzerland | 0.6397 | 67.2% |
| 8 | iceland | canada | 0.5709 | 67.3% |
| 9 | china | france | 0.5904 | 67.5% |
| 10 | philippines | morocco | 0.5388 | 63.5% |
| 11 | ethiopia | saudi_arabia | 0.5235 | 61.2% |
| 12 | singapore | new_zealand | 0.4748 | 56.5% |
| 13 | guatemala | south_korea | 0.5556 | 60.1% |
| 14 | uk | mexico | 0.5386 | 57.9% |
| 15 | vietnam | serbia | 0.4327 | 54.6% |
| 16 | indonesia | uae | 0.5576 | 60.5% |
