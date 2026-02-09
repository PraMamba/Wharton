# Phase 1 — Methodology & Results Report
## WHSDSC 2026 World Hockey League Analytics

---

## 1. Process

### Data Cleaning & Transformation (~50 words)
We verified data integrity across 25,827 line-level records spanning 1,312 games and 32 teams. No missing values were found. We aggregated line-level rows to game-level outcomes by summing goals, xG, shots, and penalty minutes per game, then stacked home/away records to create symmetric team-level statistics.

### Additional Variables (~25 words)
We engineered: xG differential per 60 minutes (xGD/60), xG share, shooting percentage, save percentage, PDO (luck indicator), Bradley-Terry strength ratings, strength of schedule, and adjusted line xG/60.

---

## 2. Tools & Techniques

### Software Tools (~50 words)
We used **Python** with pandas for data manipulation, numpy/scipy for statistical modeling, and matplotlib for visualization. The Bradley-Terry model was fit via iterative maximum likelihood estimation, and logistic regression was optimized using scipy.optimize. All analysis is reproducible from a single script.

### Statistical Methods (~100 words)
Our approach combines multiple methodologies:

1. **Bradley-Terry Model** (40% weight): An iterative pairwise comparison model that estimates each team's latent strength while controlling for opponent quality — stronger teams' wins count less than upsets.

2. **Expected Goals Analysis** (30%): xG differential per 60 minutes of ice time provides a luck-adjusted, time-normalized measure of team quality.

3. **Supplementary Metrics** (30%): xG share, goal differential, and win percentage add breadth. All metrics are z-score normalized and combined via weighted average.

4. **Calibrated Logistic Regression**: Win probabilities are derived from P(home_win) = sigmoid(0.2702 + 0.3466 × strength_diff), capturing home-ice advantage. Training accuracy: 59.8%, log-loss: 0.6606.

---

## 3. Your Predictions

### 1a: Power Rankings & Win Probabilities (~50 words)
Teams are ranked by a composite score blending Bradley-Terry strength (40%), xGD/60 (30%), xG share (15%), goal differential (10%), and win percentage (5%). Matchup probabilities come from logistic regression fit on historical game outcomes, with a home-ice intercept parameter capturing the ~56.7% baseline home-win rate.

### 1b: Offensive Line Quality Disparity (~50 words)
For each team, we computed xG per 60 minutes for first and second offensive lines. Critically, we adjusted for defensive matchup difficulty — a line facing elite defenders should not be penalized versus one facing weaker pairings. The disparity ratio (first ÷ second adjusted xG/60) captures true lineup imbalance.

### 1c: Visualization Choices (~50 words)
We created a scatter plot mapping adjusted line disparity (x-axis) against composite team strength (y-axis), with a linear regression overlay and R² annotation. Color-coding by strength score enables quick identification of team tiers. Key teams are labeled. This directly answers whether balanced lineups predict success.

---

## 4. Insights

### Model Performance Assessment (~50 words)
We evaluated our model using in-sample accuracy (59.8%) and log-loss (0.6606) on 1,312 historical games. The Bradley-Terry component's pairwise accuracy was verified separately. We also examined PDO (shooting% + save%) to identify teams whose records may regress — luck-driven outliers whose future performance may differ.

### Generative AI Usage (~50 words)
We used Claude (Anthropic), Gemini (Google), and Codex (OpenAI) as collaborative tools. Claude led the primary analysis pipeline and feature engineering. Gemini and Codex independently produced alternative rankings for cross-validation. All AI outputs were verified against manual calculations. AI accelerated iteration but all methodological decisions were human-guided.

---

## Appendix: Key Results

### Power Rankings (Top 10)

| Rank | Team | Composite Score | Win% | xG Share | xGD/60 | BT Strength |
|------|------|----------------|------|----------|--------|-------------|
| 1 | brazil | 2.0034 | 0.707 | 0.5512 | 0.0100 | 0.8548 |
| 2 | thailand | 1.6222 | 0.610 | 0.5708 | 0.0144 | 0.4276 |
| 3 | netherlands | 1.5660 | 0.659 | 0.5458 | 0.0079 | 0.6448 |
| 4 | peru | 1.2972 | 0.634 | 0.5322 | 0.0057 | 0.5418 |
| 5 | pakistan | 1.2924 | 0.598 | 0.5494 | 0.0102 | 0.3797 |
| 6 | china | 0.9600 | 0.573 | 0.5366 | 0.0068 | 0.2923 |
| 7 | panama | 0.6243 | 0.561 | 0.5161 | 0.0032 | 0.2343 |
| 8 | india | 0.5922 | 0.598 | 0.5015 | 0.0003 | 0.3793 |
| 9 | uk | 0.4229 | 0.476 | 0.5352 | 0.0071 | -0.0912 |
| 10 | guatemala | 0.3491 | 0.512 | 0.5234 | 0.0045 | 0.0458 |

### Round 1 Matchup Predictions

| Game | Home | Away | Home Win Prob | Predicted Winner |
|------|------|------|--------------|------------------|
| 1 | brazil | kazakhstan | 79.7% | brazil |
| 2 | netherlands | mongolia | 82.6% | netherlands |
| 3 | peru | rwanda | 76.2% | peru |
| 4 | thailand | oman | 75.9% | thailand |
| 5 | pakistan | germany | 71.3% | pakistan |
| 6 | india | usa | 69.0% | india |
| 7 | panama | switzerland | 68.2% | panama |
| 8 | iceland | canada | 60.6% | iceland |
| 9 | china | france | 63.9% | china |
| 10 | philippines | morocco | 57.6% | philippines |
| 11 | ethiopia | saudi_arabia | 58.3% | ethiopia |
| 12 | singapore | new_zealand | 51.5% | singapore |
| 13 | guatemala | south_korea | 60.1% | guatemala |
| 14 | uk | mexico | 57.7% | uk |
| 15 | vietnam | serbia | 48.3% | serbia |
| 16 | indonesia | uae | 60.8% | indonesia |

### Top 10 Offensive Line Quality Disparity

| Rank | Team | 1st Line Adj xG/60 | 2nd Line Adj xG/60 | Disparity Ratio |
|------|------|--------------------|--------------------|-----------------|
| 1 | usa | 0.0455 | 0.0331 | 1.3759 |
| 2 | saudi_arabia | 0.0371 | 0.0271 | 1.3692 |
| 3 | uae | 0.0330 | 0.0242 | 1.3605 |
| 4 | guatemala | 0.0462 | 0.0342 | 1.3528 |
| 5 | france | 0.0425 | 0.0316 | 1.3435 |
| 6 | iceland | 0.0433 | 0.0324 | 1.3360 |
| 7 | singapore | 0.0435 | 0.0347 | 1.2531 |
| 8 | new_zealand | 0.0402 | 0.0329 | 1.2218 |
| 9 | panama | 0.0423 | 0.0352 | 1.1989 |
| 10 | peru | 0.0394 | 0.0330 | 1.1966 |

---
*Generated by WHSDSC 2026 Analytics Pipeline*
