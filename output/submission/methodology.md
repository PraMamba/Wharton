# WHSDSC 2026 Phase 1 — Submission Report

---

## 1. Process

### Data Cleaning & Transformation (~50 words)
We validated 25,827 line-level records across 1,312 games (32 teams) and confirmed zero missing values. We aggregated rows to game outcomes, preserved home/away identity symmetrically, and dropped tied-score anomalies if any. For line disparity we restricted to first/second offense versus first/second defense, excluding PP/PK and empty-net contexts.

### Additional Variables (~25 words)
Engineered variables: xG differential/60, xG share, OT-aware Bradley-Terry strength, strength of schedule, adjusted line xG/60 ratio, GSAx/game, xG/shot, PIM/game, and penalty differential.

---

## 2. Tools & Techniques

### Software Tools (~50 words)
Python stack: pandas, numpy, scipy, matplotlib, scikit-learn, and xgboost (benchmark only). Bradley-Terry was fit with iterative MLE and overtime downweighting. Logistic regression calibrated BT strength differences into win probabilities. We used 5-fold cross-validation, log-loss, Brier score, and calibration bins before finalizing the simpler BT-only submission model.

### Statistical Methods (~100 words)
Our core model is a **Bradley-Terry paired-comparison model** fit by maximum likelihood on all 1,312 games. Overtime outcomes are weighted 0.5 because they are closer to coin flips (52.1% home OT win versus 57.6% in regulation), so they carry weaker quality signal.

Win probability uses calibrated logistic regression: P(home_win) = sigmoid(0.2724 + 0.9350 * BT_strength_diff). The intercept implies 56.8% baseline home-ice advantage. Leakage-safe 5-fold CV gives accuracy 57.0% +/- 3.2%, Brier 0.2430, and stable calibration. We tested richer ML challengers separately, but kept BT-only logistic for interpretability and reproducibility.

---

## 3. Your Predictions

### 1a: Power Rankings & Win Probabilities (~50 words)
Teams ranked by composite of OT-aware Bradley-Terry strength (40%), xGD/60 (30%), xG share (10%), goal differential (10%), and win percentage (10%). BT inherently controls for opponent quality. Win probabilities from calibrated logistic regression on BT strength differential; benchmark ML models offered only marginal lift, so we prioritized transparency and stable probability outputs.

### 1b: Offensive Line Quality Disparity (~50 words)
We computed xG per 60 for each team's first and second offensive lines at even strength. We adjusted for **defensive matchup confounding** using opponent defensive-pairing tiers (`first_def` vs `second_def`) rather than team-specific defense ratings. The adjusted disparity ratio quantifies lineup imbalance after this tier-level normalization.

### 1c: Visualization Choices (~50 words)
A scatter plot maps adjusted disparity (x) against composite strength (y) with linear regression overlay, Pearson r, and R². Color encodes strength for quick tier identification. Key teams labeled. This directly tests the commissioner's question: do balanced lineups predict success? Our finding: no strong linear relationship.

---

## 4. Insights

### Model Performance Assessment (~50 words)
We used leakage-safe **5-fold stratified CV**, refitting BT and logistic within each fold. Random-CV performance was accuracy 57.0% (+/- 3.2%), log-loss 0.6797, and Brier 0.2430 versus baseline (56.4%, 0.6849, 0.2459). In an **ID-order stress test** (sorting by numeric `game_id`, not true dates), BT was outperformed by the home-rate baseline on all three metrics (accuracy 54.7% vs 56.2%, log-loss 0.7047 vs 0.6860, Brier 0.2524 vs 0.2464). Because `game_id` order is not guaranteed to reflect chronology in this simulated dataset, we treat this as a pessimistic robustness check rather than evidence of real-world failure.

**Limitations:** As stated in the workbook, team/line quality is treated as stable across the season; we therefore do not time-weight games or model roster/injury shocks (not provided in data).

### Generative AI Usage (~50 words)
Three AI tools supported workflow review: **Claude** (pipeline draft), **Gemini** (feature critique), and **Codex** (validation and calibration checks). Their suggestions were treated as hypotheses, then verified through reproducible code and cross-validation. Final model choices, metrics, and submission files were selected by the team.

---

## Appendix: Results

### Power Rankings (32 teams)

| Rank | Team | Composite | BT Strength | xGD/60 | xG Share | Win% | SOS | GSAx/G | xG/Shot | PIM/G | PenDiff/G |
|------|------|-----------|-------------|--------|----------|------|-----|--------|---------|-------|-----------|
| 1 | brazil | 2.073 | 0.9399 | 0.5993 | 0.551 | 0.707 | -0.0498 | 0.400 | 0.1260 | 14.3 | -0.8 |
| 2 | thailand | 1.612 | 0.4840 | 0.8651 | 0.571 | 0.610 | -0.0292 | -0.321 | 0.1256 | 14.3 | -0.4 |
| 3 | netherlands | 1.535 | 0.6409 | 0.4721 | 0.546 | 0.659 | -0.0278 | 0.339 | 0.1142 | 11.3 | +2.7 |
| 4 | pakistan | 1.296 | 0.4244 | 0.6111 | 0.549 | 0.598 | -0.0262 | 0.262 | 0.1159 | 16.2 | -2.2 |
| 5 | peru | 1.273 | 0.5319 | 0.3404 | 0.532 | 0.634 | -0.0205 | 0.450 | 0.1058 | 12.7 | -0.0 |
| 6 | china | 0.915 | 0.2820 | 0.4106 | 0.537 | 0.573 | -0.0103 | 0.341 | 0.1187 | 11.6 | +0.8 |
| 7 | panama | 0.653 | 0.2670 | 0.1899 | 0.516 | 0.561 | -0.0170 | 0.346 | 0.1152 | 11.3 | +1.3 |
| 8 | india | 0.633 | 0.3960 | 0.0164 | 0.501 | 0.598 | -0.0273 | 0.548 | 0.1009 | 11.4 | +2.0 |
| 9 | uk | 0.453 | -0.0119 | 0.4243 | 0.535 | 0.476 | 0.0061 | 0.354 | 0.1216 | 10.2 | +2.2 |
| 10 | guatemala | 0.292 | 0.0210 | 0.2698 | 0.523 | 0.512 | -0.0020 | -0.039 | 0.1079 | 13.6 | +0.2 |
| 11 | iceland | 0.251 | 0.2981 | -0.2194 | 0.481 | 0.561 | -0.0028 | 0.484 | 0.1109 | 14.7 | -1.9 |
| 12 | mexico | 0.201 | -0.1743 | 0.4801 | 0.542 | 0.463 | 0.0121 | -0.369 | 0.1110 | 14.6 | -1.2 |
| 13 | serbia | 0.148 | -0.0135 | 0.1563 | 0.512 | 0.500 | -0.0049 | 0.002 | 0.1129 | 13.0 | +1.0 |
| 14 | france | 0.058 | -0.1806 | 0.3768 | 0.532 | 0.451 | 0.0008 | -0.376 | 0.1185 | 11.0 | +3.8 |
| 15 | ethiopia | -0.026 | 0.1520 | -0.2766 | 0.477 | 0.524 | -0.0011 | 0.191 | 0.1092 | 14.7 | -0.9 |
| 16 | new_zealand | -0.063 | -0.0606 | 0.0348 | 0.503 | 0.488 | 0.0024 | 0.150 | 0.1109 | 13.9 | -0.3 |
| 17 | philippines | -0.074 | 0.1193 | -0.2596 | 0.475 | 0.524 | -0.0085 | 0.611 | 0.1017 | 11.0 | +1.6 |
| 18 | south_korea | -0.081 | -0.1083 | 0.1021 | 0.508 | 0.476 | -0.0002 | -0.619 | 0.1056 | 16.5 | -2.9 |
| 19 | indonesia | -0.205 | 0.0300 | -0.2631 | 0.475 | 0.500 | 0.0166 | 0.346 | 0.1000 | 10.3 | +3.4 |
| 20 | morocco | -0.288 | -0.2288 | 0.0683 | 0.506 | 0.451 | -0.0106 | -0.309 | 0.1057 | 12.1 | +1.9 |
| 21 | canada | -0.295 | -0.1896 | 0.0894 | 0.508 | 0.439 | 0.0299 | -0.383 | 0.1092 | 12.1 | +0.9 |
| 22 | saudi_arabia | -0.381 | -0.1724 | -0.1388 | 0.487 | 0.476 | 0.0075 | 0.317 | 0.1125 | 12.2 | +1.3 |
| 23 | singapore | -0.481 | 0.0418 | -0.4992 | 0.461 | 0.488 | -0.0225 | 0.378 | 0.1123 | 14.6 | -1.3 |
| 24 | germany | -0.561 | -0.2526 | -0.1657 | 0.486 | 0.439 | 0.0159 | -0.256 | 0.1165 | 11.2 | +1.2 |
| 25 | vietnam | -0.780 | -0.1274 | -0.5375 | 0.454 | 0.476 | 0.0035 | 0.183 | 0.1092 | 14.3 | -0.9 |
| 26 | uae | -0.806 | -0.2624 | -0.3557 | 0.466 | 0.463 | -0.0090 | -0.093 | 0.1084 | 12.4 | +1.3 |
| 27 | switzerland | -0.832 | -0.2885 | -0.3708 | 0.464 | 0.439 | 0.0258 | 0.102 | 0.0981 | 15.0 | -1.5 |
| 28 | oman | -0.915 | -0.3033 | -0.4079 | 0.467 | 0.427 | 0.0329 | -0.043 | 0.1126 | 18.9 | -6.0 |
| 29 | usa | -0.951 | -0.3295 | -0.4044 | 0.468 | 0.427 | 0.0155 | -0.065 | 0.1140 | 15.7 | -2.4 |
| 30 | kazakhstan | -1.254 | -0.6444 | -0.2548 | 0.476 | 0.366 | 0.0110 | -0.202 | 0.1056 | 10.4 | +2.0 |
| 31 | rwanda | -1.365 | -0.6195 | -0.3801 | 0.466 | 0.366 | 0.0509 | -0.108 | 0.1053 | 16.7 | -4.3 |
| 32 | mongolia | -2.034 | -0.6606 | -0.9483 | 0.411 | 0.329 | 0.0386 | 0.299 | 0.0930 | 13.3 | -0.3 |

### Round 1 Matchup Predictions

| Game | Home | Away | Home Win Prob | Predicted Winner |
|------|------|------|--------------|------------------|
| 1 | brazil | kazakhstan | 85.2% | brazil |
| 2 | netherlands | mongolia | 81.6% | netherlands |
| 3 | peru | rwanda | 79.4% | peru |
| 4 | thailand | oman | 73.3% | thailand |
| 5 | pakistan | germany | 71.2% | pakistan |
| 6 | india | usa | 72.1% | india |
| 7 | panama | switzerland | 68.8% | panama |
| 8 | iceland | canada | 67.4% | iceland |
| 9 | china | france | 66.9% | china |
| 10 | philippines | morocco | 64.5% | philippines |
| 11 | ethiopia | saudi_arabia | 64.0% | ethiopia |
| 12 | singapore | new_zealand | 59.1% | singapore |
| 13 | guatemala | south_korea | 59.7% | guatemala |
| 14 | uk | mexico | 60.5% | uk |
| 15 | vietnam | serbia | 54.1% | vietnam |
| 16 | indonesia | uae | 63.3% | indonesia |

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

Calibration bins are computed on out-of-fold predictions; bins with fewer than 10 games are omitted for stability.

| Bin Center | Avg Predicted | Observed Win Rate | N |
|------------|--------------|-------------------|---|
| 0.31 | 0.334 | 0.407 | 86 |
| 0.44 | 0.447 | 0.512 | 297 |
| 0.56 | 0.565 | 0.532 | 504 |
| 0.69 | 0.680 | 0.666 | 353 |
| 0.81 | 0.790 | 0.712 | 66 |

### Model Comparison (5-Fold CV)

| Model | Accuracy | Log-Loss | Brier |
|-------|----------|----------|-------|
| Baseline (home=56.4%) | 56.4% | 0.6849 | 0.2459 |
| **OT-Aware BT Logistic** | **57.0%** | **0.6797** | **0.2430** |
 
*OT-aware BT logistic chosen for its interpretability, calibration quality, and principled treatment of overtime games.*

---
*WHSDSC 2026 | Cross-validated by Claude + Gemini + Codex*
