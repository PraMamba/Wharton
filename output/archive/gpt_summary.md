# Phase 1 Outputs

## Power Rankings (Top 10)

| rank | team | strength_score | win_pct | xg_share | xg_diff_pg |
| --- | --- | --- | --- | --- | --- |
| 1 | thailand | 1.8088078707071695 | 0.6097560975609756 | 0.5707546457329667 | 0.891160975609756 |
| 2 | brazil | 1.693864016853313 | 0.7073170731707317 | 0.5512361879556769 | 0.6177158536585371 |
| 3 | pakistan | 1.383677366664943 | 0.5975609756097561 | 0.5494450227668617 | 0.6248951219512195 |
| 4 | netherlands | 1.3802676465529975 | 0.6585365853658537 | 0.5457970235547099 | 0.4962853658536586 |
| 5 | peru | 1.1550230856471442 | 0.6341463414634146 | 0.5322278166680631 | 0.3611743902439024 |
| 6 | china | 1.027479021866136 | 0.573170731707317 | 0.5366310781646968 | 0.42607804878048783 |
| 7 | uk | 0.8562409427981824 | 0.47560975609756095 | 0.5351637831172084 | 0.44281829268292694 |
| 8 | mexico | 0.7124366076524932 | 0.4634146341463415 | 0.5423165411398946 | 0.5023512195121949 |
| 9 | panama | 0.602553687969066 | 0.5609756097560976 | 0.516098562109689 | 0.19587195121951215 |
| 10 | france | 0.4811990270404679 | 0.45121951219512196 | 0.5324614874961037 | 0.3964987804878049 |

## Matchup Probabilities (Round 1)

| game | game_id | home_team | away_team | home_win_prob | predicted_winner |
| --- | --- | --- | --- | --- | --- |
| 1 | game_1 | brazil | kazakhstan | 0.6828 | brazil |
| 2 | game_2 | netherlands | mongolia | 0.7438 | netherlands |
| 3 | game_3 | peru | rwanda | 0.6634 | peru |
| 4 | game_4 | thailand | oman | 0.696 | thailand |
| 5 | game_5 | pakistan | germany | 0.6335 | pakistan |
| 6 | game_6 | india | usa | 0.5936 | india |
| 7 | game_7 | panama | switzerland | 0.6087 | panama |
| 8 | game_8 | iceland | canada | 0.4971 | canada |
| 9 | game_9 | china | france | 0.5406 | china |
| 10 | game_10 | philippines | morocco | 0.4742 | morocco |
| 11 | game_11 | ethiopia | saudi_arabia | 0.4991 | saudi_arabia |
| 12 | game_12 | singapore | new_zealand | 0.4314 | new_zealand |
| 13 | game_13 | guatemala | south_korea | 0.5281 | guatemala |
| 14 | game_14 | uk | mexico | 0.5107 | uk |
| 15 | game_15 | vietnam | serbia | 0.3968 | serbia |
| 16 | game_16 | indonesia | uae | 0.5378 | indonesia |

## Line Disparity Top 10

| team | disparity_ratio |
| --- | --- |
| usa | 1.3643077298926607 |
| saudi_arabia | 1.3627854102063033 |
| guatemala | 1.3556095371485457 |
| uae | 1.3552245051198706 |
| france | 1.340006381103876 |
| iceland | 1.3240769743314706 |
| singapore | 1.2550062744284023 |
| new_zealand | 1.2251181677317453 |
| panama | 1.198958098950572 |
| peru | 1.19781856352069 |

## Phase 1d: Methodology Summary (Draft)

**Process (≈50 words).** We cleaned the season data by verifying game totals, removing no rows, and standardizing fields across home/away records. We aggregated line-level rows to game-level outcomes, then stacked home and away results to compute team-level totals for goals and xG.

**Additional variables (≈25 words).** We engineered xG share, xG differential per game, goal differential per game, and offensive line xG per 60 to support rankings and disparity analysis.

**Tools and techniques (≈50 words).** We used Python with pandas and numpy for aggregation, and matplotlib for visualization. Team strength was built from standardized xG and goal metrics. A calibrated logistic curve converted strength differences into matchup probabilities.

**Statistical methods (≈100 words).** We used summary statistics and z-score normalization to combine multiple performance measures into a single strength score, prioritizing underlying xG performance while retaining outcome-based signals. For matchup probabilities, we fit a single-parameter logistic transformation by minimizing log loss on historical games. This produces monotonic probabilities that map larger strength gaps to higher win likelihoods. For line disparity, we computed xG per 60 by line to normalize for time on ice, then compared first vs second line output. The visualization uses a scatter plot to reveal association between disparity and overall strength.

