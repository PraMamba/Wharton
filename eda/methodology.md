# WHSDSC 2026 Phase 1 — Submission Report

---

## 1. Process

### Data Cleaning & Transformation (~50 words)
We validated 25,827 line-level records across 1,312 games (32 teams), confirming zero missing values. We aggregated line-level rows to game-level outcomes and stacked home/away records symmetrically for unbiased team statistics. For line-disparity analysis we restricted to first/second offensive lines vs first/second defensive pairings (excluding PP/PK and empty-net segments). Overtime games were flagged for special handling in our strength model.

### Additional Variables (~25 words)
Engineered: xG differential per 60 min, xG share, OT-aware Bradley-Terry strength, strength of schedule, adjusted line xG/60 (controlling for defensive matchup difficulty), GSAx per game (goaltender quality above expected), xG per shot (offensive chance quality), PIM per game and penalty differential (team discipline), special teams rates (PP xGF/60, PK xGA/60), even-strength xGD/60, finishing and goalsaving with Bayesian shrinkage, line depth gap, goalie starter analysis with shrinkage, high-danger chance frequency.

---

## 2. Tools & Techniques

### Software Tools (~50 words)
Python (pandas, numpy, scipy, matplotlib, scikit-learn, xgboost). Bradley-Terry fitted via iterative MLE with OT-downweighting. Logistic regression optimized via scipy. ML challengers (Elastic Net, XGBoost) were used as benchmarks; final submission uses the simpler BT-only model. Three AI assistants (Claude, Gemini, Codex) provided independent cross-validation and methodology critique.

### Statistical Methods (~100 words)
Our approach rests on the **Bradley-Terry paired comparison model**, fitted via maximum likelihood on all 1,312 games. Overtime games receive reduced weight (tuned to 0.0 via nested CV), since OT outcomes are near coin-flips (52.1% home win in OT vs 57.6% in regulation) and carry weaker signal about true team quality.

Win probabilities come from **calibrated logistic regression**: P(home_win) = sigmoid(0.2716 + 0.8103 × BT_strength_diff), with the intercept capturing 56.7% home-ice advantage. We validated this via **5-fold cross-validation** (accuracy 56.9%, fold SD = 2.3%, SE = 1.0%; Brier 0.2424).

We also prototyped multi-feature ML challengers (Elastic Net, boosting) in a separate module. For this submission we use the BT-only logistic for transparency and stable calibration.

---

## 3. Your Predictions

### 1a: Power Rankings & Win Probabilities (~50 words)
Teams ranked by composite of OT-aware Bradley-Terry strength (40%), xGD/60 (30%), xG share (10%), goal differential (10%), and win percentage (10%). BT inherently controls for opponent quality. Win probabilities from calibrated logistic regression on BT strength differential; benchmark ML models offered only marginal lift, so we prioritized transparency and stable probability outputs.

### 1b: Offensive Line Quality Disparity (~50 words)
We computed xG per 60 for each team's first and second offensive lines at even strength. We adjusted for **defensive matchup confounding** (as recommended by the competition workbook)—lines facing elite defensive pairings were normalized relative to league-average difficulty. The adjusted disparity ratio quantifies true lineup imbalance.

### 1c: Visualization Choices (~50 words)
A scatter plot maps adjusted disparity (x) against composite strength (y) with linear regression overlay, Pearson r, and R². Color encodes strength for quick tier identification. Key teams labeled. This directly tests the commissioner's question: do balanced lineups predict success? Our finding: no strong linear relationship.

---

## 4. Insights

### Model Performance Assessment (~50 words)
We used **5-fold cross-validation** (refitting BT + logistic each fold): accuracy 56.9% (±2.3%), log-loss 0.6781, Brier score 0.2424. Calibration bins from out-of-fold predictions compare average predicted probabilities to observed win rates, and baseline comparison (home-win rate only) contextualizes the lift.

**Limitations:** As stated in the workbook, team/line quality is treated as stable across the season; we therefore do not time-weight games or model roster/injury shocks (not provided in data).

### Generative AI Usage (~50 words)
Three AI models collaborated: **Claude** (Anthropic) designed the analysis pipeline and code. **Gemini** (Google) critiqued rankings and suggested goaltending/special-teams features. **Codex** (OpenAI, GPT-5.3) recommended ML challenger validation and Brier score reporting. All outputs were cross-validated; final methodological decisions were human-guided.

---

## Appendix: Results

### Power Rankings (32 teams)

| Rank | Team | Composite | BT Strength | xGD/60 | xG Share | Win% | SOS | GSAx/G | xG/Shot | PIM/G | PenDiff/G |
|------|------|-----------|-------------|--------|----------|------|-----|--------|---------|-------|-----------|
| 1 | brazil | 2.036 | 1.0474 | 0.5993 | 0.551 | 0.707 | -0.0534 | 0.400 | 0.1260 | 14.3 | -0.8 | 0.8706 | 7.7975 | 6.9616 | 0.3981 |
| 2 | thailand | 1.600 | 0.5561 | 0.8651 | 0.571 | 0.610 | -0.0271 | -0.321 | 0.1256 | 14.3 | -0.4 | 0.9147 | 8.5340 | 6.7702 | -0.3272 |
| 3 | netherlands | 1.437 | 0.6405 | 0.4721 | 0.546 | 0.659 | -0.0300 | 0.339 | 0.1142 | 11.3 | +2.7 | 0.1316 | 8.0767 | 6.6498 | 0.3405 |
| 4 | pakistan | 1.281 | 0.4810 | 0.6111 | 0.549 | 0.598 | -0.0281 | 0.262 | 0.1159 | 16.2 | -2.2 | 0.8645 | 8.0758 | 6.9988 | 0.2852 |
| 5 | peru | 1.177 | 0.5160 | 0.3404 | 0.532 | 0.634 | -0.0206 | 0.450 | 0.1058 | 12.7 | -0.0 | 0.2111 | 7.5901 | 6.7222 | 0.4407 |
| 6 | china | 0.855 | 0.2688 | 0.4106 | 0.537 | 0.573 | -0.0119 | 0.341 | 0.1187 | 11.6 | +0.8 | 0.3769 | 7.4706 | 7.2817 | 0.3444 |
| 7 | panama | 0.660 | 0.3166 | 0.1899 | 0.516 | 0.561 | -0.0177 | 0.346 | 0.1152 | 11.3 | +1.3 | -0.0403 | 7.7992 | 7.4213 | 0.3769 |
| 8 | india | 0.598 | 0.4169 | 0.0164 | 0.501 | 0.598 | -0.0311 | 0.548 | 0.1009 | 11.4 | +2.0 | -0.0240 | 6.9436 | 6.9374 | 0.5322 |
| 9 | uk | 0.551 | 0.0961 | 0.4243 | 0.535 | 0.476 | -0.0001 | 0.354 | 0.1216 | 10.2 | +2.2 | 0.0806 | 8.3458 | 7.1186 | 0.3611 |
| 10 | iceland | 0.281 | 0.3682 | -0.2194 | 0.481 | 0.561 | -0.0059 | 0.484 | 0.1109 | 14.7 | -1.9 | -0.0141 | 6.5557 | 6.9957 | 0.5000 |
| 11 | guatemala | 0.256 | -0.0093 | 0.2698 | 0.523 | 0.512 | -0.0004 | -0.039 | 0.1079 | 13.6 | +0.2 | 0.3530 | 7.4723 | 7.3190 | -0.0273 |
| 12 | mexico | 0.176 | -0.2176 | 0.4801 | 0.542 | 0.463 | 0.0163 | -0.369 | 0.1110 | 14.6 | -1.2 | 0.5821 | 7.6709 | 6.8973 | -0.3509 |
| 13 | serbia | 0.146 | -0.0142 | 0.1563 | 0.512 | 0.500 | 0.0007 | 0.002 | 0.1129 | 13.0 | +1.0 | 0.1227 | 7.8230 | 7.7858 | 0.0185 |
| 14 | france | 0.108 | -0.1495 | 0.3768 | 0.532 | 0.451 | -0.0035 | -0.376 | 0.1185 | 11.0 | +3.8 | -0.1695 | 8.0925 | 6.4762 | -0.3490 |
| 15 | ethiopia | 0.035 | 0.2338 | -0.2766 | 0.477 | 0.524 | -0.0001 | 0.191 | 0.1092 | 14.7 | -0.9 | -0.1291 | 7.4092 | 7.5463 | 0.2162 |
| 16 | philippines | -0.051 | 0.1568 | -0.2596 | 0.475 | 0.524 | -0.0118 | 0.611 | 0.1017 | 11.0 | +1.6 | -0.4926 | 7.1271 | 6.8957 | 0.5828 |
| 17 | new_zealand | -0.068 | -0.0748 | 0.0348 | 0.503 | 0.488 | 0.0018 | 0.150 | 0.1109 | 13.9 | -0.3 | 0.0477 | 7.6135 | 7.6319 | 0.1464 |
| 18 | south_korea | -0.077 | -0.1179 | 0.1021 | 0.508 | 0.476 | 0.0011 | -0.619 | 0.1056 | 16.5 | -2.9 | 0.4690 | 7.4316 | 7.5519 | -0.5530 |
| 19 | indonesia | -0.187 | 0.0483 | -0.2631 | 0.475 | 0.500 | 0.0135 | 0.346 | 0.1000 | 10.3 | +3.4 | -0.6195 | 6.9703 | 6.7299 | 0.3373 |
| 20 | canada | -0.221 | -0.1398 | 0.0894 | 0.508 | 0.439 | 0.0320 | -0.383 | 0.1092 | 12.1 | +0.9 | 0.1091 | 7.1872 | 7.0835 | -0.3580 |
| 21 | morocco | -0.287 | -0.2598 | 0.0683 | 0.506 | 0.451 | -0.0124 | -0.309 | 0.1057 | 12.1 | +1.9 | 0.0750 | 6.8122 | 7.9078 | -0.2699 |
| 22 | singapore | -0.339 | 0.1871 | -0.4992 | 0.461 | 0.488 | -0.0355 | 0.378 | 0.1123 | 14.6 | -1.3 | -0.2902 | 8.3415 | 8.7613 | 0.3926 |
| 23 | saudi_arabia | -0.451 | -0.2716 | -0.1388 | 0.487 | 0.476 | 0.0124 | 0.317 | 0.1125 | 12.2 | +1.3 | -0.2490 | 6.6598 | 6.7848 | 0.3324 |
| 24 | germany | -0.537 | -0.2668 | -0.1657 | 0.486 | 0.439 | 0.0187 | -0.256 | 0.1165 | 11.2 | +1.2 | -0.2869 | 7.8265 | 7.5477 | -0.2458 |
| 25 | vietnam | -0.787 | -0.1630 | -0.5375 | 0.454 | 0.476 | 0.0038 | 0.183 | 0.1092 | 14.3 | -0.9 | -0.3216 | 7.1292 | 8.2573 | 0.1775 |
| 26 | switzerland | -0.869 | -0.3755 | -0.3708 | 0.464 | 0.439 | 0.0304 | 0.102 | 0.0981 | 15.0 | -1.5 | -0.2416 | 6.7777 | 7.5290 | 0.0929 |
| 27 | oman | -0.900 | -0.3392 | -0.4079 | 0.467 | 0.427 | 0.0324 | -0.043 | 0.1126 | 18.9 | -6.0 | 0.1636 | 7.1486 | 8.3532 | -0.0399 |
| 28 | uae | -0.914 | -0.4190 | -0.3557 | 0.466 | 0.463 | -0.0046 | -0.093 | 0.1084 | 12.4 | +1.3 | -0.4957 | 6.9396 | 7.7516 | -0.0912 |
| 29 | usa | -0.954 | -0.3866 | -0.4044 | 0.468 | 0.427 | 0.0170 | -0.065 | 0.1140 | 15.7 | -2.4 | -0.2151 | 7.1288 | 8.5074 | -0.0573 |
| 30 | kazakhstan | -1.274 | -0.7627 | -0.2548 | 0.476 | 0.366 | 0.0109 | -0.202 | 0.1056 | 10.4 | +2.0 | -0.6383 | 7.5161 | 7.5050 | -0.1511 |
| 31 | rwanda | -1.397 | -0.7492 | -0.3801 | 0.466 | 0.366 | 0.0617 | -0.108 | 0.1053 | 16.7 | -4.3 | 0.0122 | 6.8046 | 7.9859 | -0.1041 |
| 32 | mongolia | -1.885 | -0.6168 | -0.9483 | 0.411 | 0.329 | 0.0414 | 0.299 | 0.0930 | 13.3 | -0.3 | -0.9810 | 6.2086 | 7.7958 | 0.2966 |

### Round 1 Matchup Predictions

| Game | Home | Away | Home Win Prob | Predicted Winner |
|------|------|------|--------------|------------------|
| 1 | brazil | kazakhstan | 85.4% | brazil |
| 2 | netherlands | mongolia | 78.1% | netherlands |
| 3 | peru | rwanda | 78.8% | peru |
| 4 | thailand | oman | 73.5% | thailand |
| 5 | pakistan | germany | 70.3% | pakistan |
| 6 | india | usa | 71.2% | india |
| 7 | panama | switzerland | 69.6% | panama |
| 8 | iceland | canada | 66.8% | iceland |
| 9 | china | france | 63.7% | china |
| 10 | philippines | morocco | 65.0% | philippines |
| 11 | ethiopia | saudi_arabia | 65.8% | ethiopia |
| 12 | singapore | new_zealand | 61.1% | singapore |
| 13 | guatemala | south_korea | 58.9% | guatemala |
| 14 | uk | mexico | 62.1% | uk |
| 15 | vietnam | serbia | 53.9% | vietnam |
| 16 | indonesia | uae | 66.1% | indonesia |

### Top 10 Line Quality Disparity

| Rank | Team | 1st Line Adj xG/60 | 2nd Line Adj xG/60 | Adj Disparity |
|------|------|--------------------|--------------------|---------------|
| 1 | usa | 2.7298 | 1.9933 | 1.3695 |
| 2 | guatemala | 2.8214 | 2.0705 | 1.3627 |
| 3 | saudi_arabia | 2.2441 | 1.6540 | 1.3568 |
| 4 | uae | 1.9980 | 1.4759 | 1.3538 |
| 5 | france | 2.5591 | 1.9011 | 1.3461 |
| 6 | iceland | 2.5894 | 1.9608 | 1.3206 |
| 7 | singapore | 2.6353 | 2.1015 | 1.2540 |
| 8 | new_zealand | 2.4410 | 1.9751 | 1.2359 |
| 9 | peru | 2.3908 | 1.9826 | 1.2059 |
| 10 | panama | 2.5451 | 2.1195 | 1.2008 |

### Probability Calibration

| Bin Center | Avg Predicted | Observed Win Rate | N |
|------------|--------------|-------------------|---|
| 0.34 | 0.352 | 0.434 | 76 |
| 0.43 | 0.431 | 0.534 | 191 |
| 0.51 | 0.511 | 0.497 | 312 |
| 0.59 | 0.593 | 0.560 | 327 |
| 0.68 | 0.673 | 0.661 | 274 |
| 0.76 | 0.751 | 0.703 | 101 |

### Model Comparison (5-Fold CV)

| Model | Accuracy | Log-Loss | Brier |
|-------|----------|----------|-------|
| Baseline (home=56.4%) | 56.4% | 0.6849 | 0.2459 |
| **OT-Aware BT Logistic** | **56.9%** | **0.6781** | **0.2424** |

*Note: 56.9% is the mean across 5 folds (SD = 2.3%, SE = 1.0%). The baseline (56.4%) falls within the 95% CI, so the accuracy improvement is modest.*

*OT-aware BT logistic chosen for its interpretability, calibration quality, and principled treatment of overtime games.*

### Weight Sensitivity Analysis

Spearman rank correlations between alternative weight schemes:

| Scheme | Current (40/30/10/10/10) | BT-only (100%) | Equal (20% each) | xG-heavy (50% xGD) |
|--------|--------|--------|--------|--------|
| Current (40/30/10/10/10) | 1.000 | 0.922 | 0.998 | 0.973 |
| BT-only (100%) | 0.922 | 1.000 | 0.909 | 0.837 |
| Equal (20% each) | 0.998 | 0.909 | 1.000 | 0.977 |
| xG-heavy (50% xGD) | 0.973 | 0.837 | 0.977 | 1.000 |

Top 6 teams identical across all schemes: **False**

### Ranking Component Correlation Matrix

| | bt_strength | xgd_per60 | xg_share | gd_pg | win_pct |
|---|---|---|---|---|---|
| bt_strength | 1.000 | 0.643 | 0.654 | 0.943 | 0.965 |
| xgd_per60 | 0.643 | 1.000 | 0.998 | 0.703 | 0.672 |
| xg_share | 0.654 | 0.998 | 1.000 | 0.710 | 0.684 |
| gd_pg | 0.943 | 0.703 | 0.710 | 1.000 | 0.946 |
| win_pct | 0.965 | 0.672 | 0.684 | 0.946 | 1.000 |

---
*WHSDSC 2026 | Cross-validated by Claude + Gemini + Codex*
