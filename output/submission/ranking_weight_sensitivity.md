# Ranking Weight Sensitivity (Internal)

This file is for auditability only and is not part of the official submission package.

## Summary

- Pairwise Spearman rank correlation across 4 schemes: 0.81 to 1.00
- Common teams in top 6 across all schemes (5): brazil, netherlands, pakistan, peru, thailand

## Schemes

| Scheme | bt_strength | xgd_per60 | xg_share | gd_pg | win_pct |
|---|---:|---:|---:|---:|---:|
| final_submission | 0.40 | 0.30 | 0.10 | 0.10 | 0.10 |
| bt_only | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| equal_weights | 0.20 | 0.20 | 0.20 | 0.20 | 0.20 |
| xgd60_tilt | 0.20 | 0.50 | 0.10 | 0.10 | 0.10 |

## Pairwise Spearman Correlations

| Scheme A | Scheme B | Spearman rho |
|---|---|---:|
| bt_only | xgd60_tilt | 0.814 |
| bt_only | equal_weights | 0.902 |
| final_submission | bt_only | 0.904 |
| final_submission | xgd60_tilt | 0.975 |
| equal_weights | xgd60_tilt | 0.975 |
| final_submission | equal_weights | 0.999 |