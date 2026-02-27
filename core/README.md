# Core Module Documentation

## Overview

The `core/` package provides the foundational analytics infrastructure for the WHSDSC 2026 hockey analytics project. It implements a sophisticated pipeline for team ranking, game outcome prediction, and advanced feature engineering using Bradley-Terry models, ensemble methods, and specialized hockey metrics.

**Key Capabilities:**
- OT-aware Bradley-Terry ranking with L2 regularization
- Bootstrap uncertainty quantification (200 resamples)
- Attack-Defense Poisson modeling with Skellam predictions
- Regularized Adjusted Plus-Minus (RAPM) using sparse Ridge regression
- 10+ specialized feature engineering functions
- Cross-validation with calibration and hyperparameter tuning
- Ensemble ranking with z-score composites

**Architecture Highlights:**
- 1,550 lines across 7 modules
- Vectorized implementations (10-50x speedup via `np.bincount`)
- Sparse matrix construction for RAPM (COO→CSR)
- Configuration externalization via YAML
- Shared codebase for both submission and ML pipelines

---

## Module Reference

### `config.py` (99 lines)
**Purpose:** Centralized configuration management with YAML loading and hardcoded fallback defaults.

**Key Components:**
- `_DEFAULTS`: Comprehensive dict containing all hyperparameters organized by category:
  - `bt`: Bradley-Terry parameters (ot_weight, regularization, max_iter, tolerance)
  - `rankings`: Ensemble weights for 5 components (bt_weight, xgd_weight, etc.)
  - `cv`: Cross-validation settings (n_folds, n_boot_cv)
  - `prediction`: Probability clipping bounds
  - `shrinkage`: Uncertainty shrinkage parameter
  - `high_danger`: Shot danger threshold
  - `bt_tuning`: Nested CV grid search ranges
  - `attack_defense`: Poisson model regularization
  - `rapm`: RAPM regularization and min TOI filtering
  - `augmentation`: Symmetric data augmentation flag
  - `matchup_matrix`: Deployment analysis parameters

- `load_config(path="config.yaml")`: Loads YAML file with deep merge fallback to `_DEFAULTS`
- `CFG`: Module-level singleton providing global config access

**Usage:**
```python
from core.config import CFG

# Access configuration values
ot_weight = CFG["bt"]["ot_weight"]  # 0.2
reg = CFG["bt"]["regularization"]   # 0.03
```

**Design Notes:**
- Deep merge strategy allows partial YAML overrides
- Eliminates hardcoded magic numbers throughout codebase
- Enables hyperparameter tuning without code changes
- No dependencies on other core modules

---

### `data.py` (58 lines)
**Purpose:** Data loading and game-level aggregation from raw line-level records.

**Key Functions:**

#### `load_data(zip_path, output_dir="data")`
Extracts competition data from zip archive.

**Parameters:**
- `zip_path`: Path to `drive-download-20260205T132825Z-1-001.zip`
- `output_dir`: Directory for extracted files (default: "data")

**Returns:**
- `df`: Raw line-level DataFrame (25,827 rows)
- `matchups`: Round 1 matchup schedule DataFrame

**Behavior:**
- Extracts `whl_2025.csv` and `WHSDSC_Rnd1_matchups.xlsx`
- Skips extraction if files already exist
- Handles both zipped and pre-extracted data

#### `build_game_level(df)`
Aggregates line-level records into game-level statistics.

**Parameters:**
- `df`: Raw line-level DataFrame with columns: `game_id`, `home_team`, `away_team`, `toi`, `xg`, `goals`, etc.

**Returns:**
- Game-level DataFrame (1,312 rows) with aggregated stats per game

**Aggregation Logic:**
- Groups by `game_id`
- Sums: `toi`, `xg`, `goals`, `shots`, `penalties_committed` (if present)
- Takes first: `home_team`, `away_team`, `is_ot`
- Validates no draws exist (raises `ValueError` if found)

**Design Notes:**
- No dependencies on other core modules
- Defensive programming: checks for draws before returning
- Handles optional columns gracefully

---

### `stats.py` (113 lines)
**Purpose:** Computes comprehensive per-team season statistics from game-level data.

**Key Functions:**

#### `build_team_stats(game)`
Produces superset of team-level features for both submission and ML pipelines.

**Parameters:**
- `game`: Game-level DataFrame (1,312 rows) from `build_game_level()`

**Returns:**
- Team-level DataFrame (32 rows) indexed by team name with 20+ columns

**Computed Features:**
- **Basic aggregates:** `games`, `wins`, `goals_for`, `goals_against`, `goal_diff`
- **Per-60 rates:** `xgf60`, `xga60`, `xgd60`, `gf60`, `ga60`
- **Advanced metrics:**
  - `xg_share`: xGF / (xGF + xGA)
  - `pdo`: (GF/xGF) + (1 - GA/xGA) - shooting luck + save luck
  - `gsax`: Goals Saved Above Expected = xGA - GA
  - `gsax60`: GSAx per 60 minutes
  - `gsax_shrunk`: Bayesian shrinkage toward league mean (K=500 from config)
- **Optional features** (if columns present):
  - `penalties_committed`, `pen_diff` (penalties drawn - committed)
  - `assists`, `points` (goals + assists)

**Implementation Details:**
- Symmetric home/away stacking doubles sample size (2,624 records)
- Groups by team and aggregates with `.agg()`
- Computes win percentage as `wins / games`
- Applies shrinkage: `(stat * toi + K * league_mean) / (toi + K)`
- Uses `ddof=1` for sample standard deviation in z-scores

**Dependencies:**
- `core.config.CFG` for shrinkage parameter K

**Design Notes:**
- Handles optional columns gracefully with `if "column" in df.columns`
- Defensive division: `np.where(denom != 0, num / denom, 0)`
- Produces superset of features - pipelines select what they need

---

### `bt.py` (200 lines)
**Purpose:** Core ranking algorithms using pairwise comparison models.

**Key Functions:**

#### `fit_bt(game, teams, ot_weight=0.5, max_iter=300, tol=1e-8, regularization=0.0)`
Vectorized Bradley-Terry model with OT downweighting and L2 regularization.

**Algorithm:** Iterative Newton-Raphson optimization
- Maximizes log-likelihood: `Σ log(P(i beats j))` where `P = exp(s_i) / (exp(s_i) + exp(s_j))`
- Gradient: `wins_i - Σ P(i beats j)` for all games involving team i
- Hessian: `-Σ P(i beats j) * (1 - P(i beats j))`
- Update: `strength += 0.5 * delta` where `delta = -gradient / hessian`

**Parameters:**
- `game`: Game-level DataFrame with `home_team`, `away_team`, `home_goals`, `away_goals`, `is_ot`
- `teams`: List of all team names (ensures consistent ordering)
- `ot_weight`: Weight for OT games (default 0.5, tuned to 0.2)
- `regularization`: L2 penalty strength (default 0.0, tuned to 0.03)

**Returns:**
- Dict mapping team name → BT strength (log-scale)

**Performance Optimizations:**
- **Vectorization:** Uses `np.bincount` for gradient/Hessian aggregation (10-50x speedup)
- **Pre-computation:** Index arrays computed once (lines 28-31)
- **Numerical stability:**
  - Hessian clipping to ensure positive definiteness
  - L2 regularization prevents ±inf for undefeated teams
  - Probability clipping to [1e-8, 1-1e-8]

**Convergence:**
- Checks `max(|delta|) < tol` after each iteration
- Warns if max_iter reached without convergence
- Typical convergence: 10-30 iterations

#### `bootstrap_bt(game, teams, n_boot=200, ot_weight=0.5, regularization=0.0, seed=42)`
Bootstrap uncertainty quantification for BT strengths.

**Parameters:**
- Same as `fit_bt()` plus:
- `n_boot`: Number of bootstrap resamples (default 200)
- `seed`: Random seed for reproducibility

**Returns:**
- `bt_mean`: Mean BT strength across resamples
- `bt_std`: Standard deviation (uncertainty)
- `bt_ci_lower`, `bt_ci_upper`: 95% confidence intervals (2.5th, 97.5th percentiles)
- `bt_samples`: Full matrix (n_teams × n_boot) for downstream integration

**Implementation:**
- Resamples games with replacement
- Fits BT model on each resample
- Aggregates statistics across resamples
- Uses `np.random.default_rng(seed)` for reproducibility

#### `compute_sos(game, bt)`
Computes strength of schedule for each team.

**Parameters:**
- `game`: Game-level DataFrame
- `bt`: Dict of BT strengths

**Returns:**
- Dict mapping team → mean opponent BT strength

**Logic:**
- For each team, averages BT strengths of all opponents faced
- Symmetric: considers both home and away games

#### `fit_attack_defense(game, teams, use_xg=True, regularization=0.01)`
Attack-Defense Poisson model using L-BFGS-B optimization.

**Model:** Goals ~ Poisson(attack_i × defense_j)
- Log-likelihood: `Σ goals * log(λ) - λ` where `λ = exp(α_i + δ_j)`
- Regularization: L2 penalty on attack/defense parameters

**Parameters:**
- `use_xg`: If True, models expected goals; if False, actual goals
- `regularization`: L2 penalty strength (default 0.01)

**Returns:**
- Dict with keys: `attack`, `defense` (each mapping team → parameter value)

**Numerical Stability:**
- Clips log-intensities to [-5, 5] to prevent overflow
- Centers parameters post-optimization (mean = 0)

#### `skellam_predict_ad(ad_model, home, away)`
Predicts win probability using Skellam distribution from AD model.

**Parameters:**
- `ad_model`: Dict from `fit_attack_defense()`
- `home`, `away`: Team names

**Returns:**
- `p_home_win`: Probability home team wins

**Algorithm:**
- Computes λ_home = exp(attack_home + defense_away)
- Computes λ_away = exp(attack_away + defense_home)
- Goal differential ~ Skellam(λ_home, λ_away)
- P(home win) = P(GD > 0) = 1 - Skellam.cdf(0)

**Design Notes:**
- No dependencies on other core modules
- All functions are pure (no side effects)
- Extensive use of vectorization for performance
- Proper convergence warnings and numerical stability checks

---

### `model.py` (342 lines)
**Purpose:** High-level modeling utilities that orchestrate BT fitting, calibration, cross-validation, ranking, and prediction.

**Key Functions:**

#### `build_rankings(stats, bt, sos)`
Constructs ensemble rankings using z-score composite of 5 weighted components.

**Parameters:**
- `stats`: Team statistics DataFrame from `build_team_stats()`
- `bt`: BT strength dict from `fit_bt()`
- `sos`: Strength of schedule dict from `compute_sos()`

**Returns:**
- DataFrame with columns: `team`, `composite_score`, `rank`, plus individual z-scores

**Ensemble Formula:**
```
composite = 0.40 × z(BT) + 0.30 × z(xGD/60) + 0.10 × z(xG_share)
          + 0.10 × z(goal_diff) + 0.10 × z(win_pct)
```

**Implementation:**
- Computes z-scores using `ddof=1` (sample std for 32-team population)
- Weights from `CFG["rankings"]`
- Sorts by composite score descending
- Assigns ranks 1-32

#### `fit_logistic(game, bt, method="platt")`
Fits calibration model to convert BT strength differences into win probabilities.

**Parameters:**
- `game`: Game-level DataFrame
- `bt`: BT strength dict
- `method`: "platt" (logistic) or "isotonic" (isotonic regression)

**Returns:**
- `(a, b)`: Calibration parameters where `P(home_win) = sigmoid(a + b × BT_diff)`

**Platt Scaling:**
- Minimizes negative log-likelihood via `minimize_scalar`
- Searches over `a ∈ [-2, 2]` with fixed `b = 1.0`
- Clips probabilities to [1e-8, 1-1e-8] for numerical stability

**Isotonic Regression:**
- Non-parametric monotonic calibration
- Uses `IsotonicRegression` from sklearn

#### `cross_validate_bt(game, teams, n_folds=5, bt_kwargs=None, n_boot=0)`
Performs stratified k-fold cross-validation with optional per-fold bootstrap.

**Parameters:**
- `game`: Game-level DataFrame
- `teams`: List of team names
- `n_folds`: Number of CV folds (default 5)
- `bt_kwargs`: Dict of parameters for `fit_bt()` (e.g., `{"ot_weight": 0.2, "regularization": 0.03}`)
- `n_boot`: Bootstrap resamples per fold (0 = no bootstrap)

**Returns:**
- `CVResult` named tuple with fields:
  - `accuracy`: Mean accuracy across folds
  - `brier`: Mean Brier score
  - `log_loss`: Mean log-loss
  - `ece`: Mean Expected Calibration Error
  - `a`, `b`: Calibration parameters (from full dataset)
  - `fold_metrics`: List of per-fold metric dicts
  - `oof_pred`, `oof_true`: Out-of-fold predictions and labels

**Implementation:**
- Stratified folds preserve class balance (home wins vs away wins)
- Per-fold: fit BT → calibrate → predict → compute metrics
- Optional per-fold bootstrap for leakage-free uncertainty
- Aggregates metrics with mean and std across folds
- Returns OOF predictions for meta-learning

#### `tune_bt_hyperparams(game, teams, n_outer=5, n_inner=3)`
Nested cross-validation for BT hyperparameter tuning.

**Parameters:**
- `n_outer`: Outer CV folds for evaluation
- `n_inner`: Inner CV folds for hyperparameter selection

**Returns:**
- `best_params`: Dict with optimal `ot_weight` and `regularization`
- `best_score`: Brier score achieved

**Grid Search:**
- `ot_weight`: [0.1, 0.2, 0.3, 0.4, 0.5] (from config)
- `regularization`: [0.0, 0.01, 0.03, 0.05, 0.1] (from config)
- Evaluates all 25 combinations
- Selects params minimizing mean Brier score on inner folds

#### `tune_uncertainty_shrink_k(oof_pred, oof_true, bt_std, game)`
Optimizes uncertainty shrinkage parameter for bootstrap integration.

**Parameters:**
- `oof_pred`: Out-of-fold predictions (mean probabilities)
- `oof_true`: True labels
- `bt_std`: Bootstrap standard deviations
- `game`: Game-level DataFrame

**Returns:**
- `best_k`: Optimal shrinkage parameter (typically ~0.83)

**Algorithm:**
- Computes pairwise uncertainties: `sqrt(home_var + away_var)`
- Tries k ∈ [0, 2] to minimize Brier score
- Shrinks uncertainties: `uncertainty_shrunk = k × uncertainty`
- Integrates via: `E[sigmoid(BT_diff + ε)]` where `ε ~ N(0, uncertainty²)`

#### `predict_matchups(matchups, bt, a, b, bt_samples=None, teams=None, bt_std=None, uncertainty_shrink_k=None)`
Generates matchup predictions with optional bootstrap confidence intervals.

**Parameters:**
- `matchups`: DataFrame with `home_team`, `away_team` columns
- `bt`: BT strength dict
- `a`, `b`: Calibration parameters
- `bt_samples`: Bootstrap samples matrix (optional, for CI)
- `bt_std`: Bootstrap standard deviations (optional, for uncertainty integration)
- `uncertainty_shrink_k`: Shrinkage parameter (optional)

**Returns:**
- DataFrame with columns: `home_team`, `away_team`, `home_win_prob`, `ci_lower`, `ci_upper`

**Implementation:**
- Computes BT differences: `bt[home] - bt[away]`
- Applies calibration: `P = sigmoid(a + b × BT_diff)`
- Clips to [0.01, 0.99] (from config)
- If bootstrap samples provided:
  - Computes CI from bootstrap distribution
  - Optionally integrates uncertainty via shrinkage

#### `compute_ece(y_true, y_pred, n_bins=10)`
Computes Expected Calibration Error for probability calibration assessment.

**Parameters:**
- `y_true`: Binary labels (0/1)
- `y_pred`: Predicted probabilities
- `n_bins`: Number of calibration bins (default 10)

**Returns:**
- `ece`: Expected Calibration Error (lower is better)

**Algorithm:**
- Bins predictions into n_bins equal-width intervals
- For each bin: computes |mean(y_pred) - mean(y_true)|
- Weights by bin size
- Returns weighted average of calibration errors

**Dependencies:**
- `core.config.CFG` for various parameters
- `core.bt.{fit_bt, bootstrap_bt}` for model fitting

**Design Notes:**
- Acts as facade providing high-level operations
- Composes lower-level functions from bt.py
- Implements advanced techniques (bootstrap integration, uncertainty shrinkage)
- Returns structured results (named tuples, DataFrames)

---

### `advanced.py` (737 lines)
**Purpose:** Specialized feature engineering from raw line-level data for ML models. Implements 10+ feature extraction functions with sophisticated statistical techniques.

**Common Pattern:**
All feature builders follow a consistent API:
```python
def build_X(df, rec=None, **kwargs):
    if rec is None:
        rec = stack_records(df)  # Symmetric home/away stacking
    # Compute features with per-60 normalization
    # Apply shrinkage/regularization for sparse data
    # Return DataFrame indexed by team
```

**Core Helper Functions:**

#### `stack_records(df)`
Symmetric home/away stacking to double sample size and eliminate bias.

**Transformation:**
- Home records: `{team: home_team, opponent: away_team, ...}`
- Away records: `{team: away_team, opponent: home_team, ...}`
- Concatenates both into unified format (51,654 rows from 25,827)

**Returns:**
- DataFrame with columns: `team`, `opponent`, `toi`, `xg`, `goals`, etc.

#### `_build_advanced_dict(df, rec=None)`
Helper that constructs dict of all advanced features to eliminate redundant lookups.

**Returns:**
- Dict mapping feature name → DataFrame (e.g., `"special_teams"`, `"line_depth"`, `"goalie"`, etc.)

**Usage:**
- Called by `build_ml_features()` to avoid repeated stacking operations
- Enables efficient feature selection (Core/Extended/Full tiers)

**Feature Engineering Functions:**

#### `build_special_teams(df, rec=None)`
Power play, penalty kill, and even-strength per-60 rates.

**Features:**
- `pp_xgf60`, `pp_xga60`: PP expected goals for/against per 60
- `pk_xgf60`, `pk_xga60`: PK expected goals for/against per 60
- `ev_xgf60`, `ev_xga60`: Even-strength expected goals for/against per 60

**Implementation:**
- Filters by `situation` column (PP/PK/EV)
- Aggregates by team
- Normalizes by TOI: `(xg_sum / toi_sum) × 3600`

#### `build_line_depth(df, rec=None)`
Offensive line quality and depth metrics.

**Features:**
- `topline_toi_share`: 1st line TOI / total forward TOI
- `depth_gap`: (1st line xG/60) - (2nd line xG/60)
- `line_balance`: Std dev of xG/60 across all forward lines

**Implementation:**
- Filters to even-strength forwards only
- Groups by team and line number
- Computes per-line xG/60
- Aggregates depth metrics

#### `build_goalie_features(df, rec=None)`
Goalie performance with Bayesian shrinkage.

**Features:**
- `goalie_gsax60`: Goals Saved Above Expected per 60 (shrunk)
- `starter_toi_share`: Primary goalie TOI / total goalie TOI

**Shrinkage:**
- `gsax_shrunk = (gsax × toi + K × 0) / (toi + K)` where K=500
- Shrinks toward league mean (0) for low-TOI goalies

#### `build_high_danger(df, threshold=None, rec=None)`
High-danger shot frequency and quality metrics.

**Features:**
- `hd_shot_freq`: Fraction of shots that are high-danger
- `hd_xg_sum60`: Total xG from high-danger shots per 60
- `p95_max_xg`: 95th percentile of max_xg (shot quality ceiling)

**Implementation:**
- Filters to shots with `max_xg > threshold` (default 0.15 from config)
- Computes frequency and aggregated xG
- Uses `np.percentile` for p95 calculation

#### `residualize_features(team_stats, bt_strength, feature_cols, alpha=0.1)`
Ridge-based residualization to remove BT-correlated components from features.

**Purpose:** Creates decorrelated features that capture information orthogonal to BT strength.

**Parameters:**
- `team_stats`: Team statistics DataFrame
- `bt_strength`: Dict of BT strengths
- `feature_cols`: List of column names to residualize
- `alpha`: Ridge regularization strength (default 0.1)

**Returns:**
- DataFrame with residualized features (suffix `_resid`)

**Algorithm:**
1. For each feature: fit Ridge regression `feature ~ BT_strength`
2. Compute residuals: `residual = feature - prediction`
3. Return residuals as new features

**Use Case:**
- Prevents feature redundancy in ML models
- Captures team characteristics not explained by overall strength

#### `build_rapm_features(df, teams, regularization=100.0, min_toi_unit=200, rec=None)`
Regularized Adjusted Plus-Minus using sparse Ridge regression on segment-level data.

**Features:**
- `rapm_off`: Offensive RAPM (contribution to team xGF)
- `rapm_def`: Defensive RAPM (contribution to team xGA)
- `rapm_net`: Net RAPM (off - def)

**Algorithm:**
1. **Segment extraction:** Groups consecutive records by team-opponent-situation
2. **Unit identification:** Creates 5-player units from line combinations
3. **Sparse matrix construction:**
   - Rows: segments (one per continuous shift)
   - Columns: units (unique 5-player combinations)
   - Values: +1 for offensive unit, -1 for defensive unit
4. **Filtering:** Removes units with TOI < min_toi_unit (200s default)
5. **Ridge regression:** Fits `xG ~ unit_matrix` with L2 penalty
6. **Team aggregation:** Averages unit RAPM weighted by TOI

**Performance:**
- COO matrix construction for efficiency
- Converts to CSR for Ridge fitting
- Vectorized column mapping with `pd.Series.map()`
- Typical dimensions: ~15,000 segments × ~800 units

**Complexity:** O(n_segments × n_units) but sparse (nnz ~ 2 × n_segments)

#### `build_toi_entropy(df, rec=None)`
Shannon entropy of TOI distributions and deployment matrix features.

**Features:**
- `toi_entropy`: Shannon entropy of player TOI distribution (higher = more balanced)
- `top_vs_top`: Fraction of top-line TOI against opponent's top defensive pair
- `top_shelter`: Fraction of top-line TOI against opponent's bottom defensive pair

**Shannon Entropy:**
```python
p = toi / toi.sum()  # Probability distribution
entropy = -sum(p × log(p))
```

**Deployment Matrix:**
- Cross-tabulates offensive line vs defensive pair
- Computes conditional probabilities
- Measures coaching deployment strategies

#### `build_matchup_matrix(df, min_toi=1000, rec=None)`
Matchup exploitation and shutdown metrics with shrinkage fill.

**Features:**
- `exploit_ability`: Success rate when facing weak opponents
- `shutdown_resilience`: Performance when facing strong opponents
- `depth_insurance`: Performance consistency across matchup difficulties

**Implementation:**
1. Bins opponents by quality (top/middle/bottom third)
2. Computes xGD/60 against each bin
3. Applies shrinkage for sparse cells: `K = min_toi / 2`
4. Aggregates exploitation and resilience metrics

**Shrinkage:**
- Fills sparse matchup cells by regressing toward league mean
- Prevents overfitting to small-sample matchups

#### `build_st_expected_value(df, rec=None)`
Special teams expected value from penalty rates.

**Features:**
- `pen_taken60`: Penalties committed per 60
- `pen_drawn60`: Penalties drawn per 60
- `net_st_ev`: Net special teams expected value

**Expected Value Calculation:**
```python
net_st_ev = (pen_drawn60 × pp_value) - (pen_taken60 × pk_cost)
```
where pp_value and pk_cost are league-average xG rates

**Helper:** `_stack_penalty_records(df)` extracts penalty events from raw data

#### `build_volatility_features(df, game_level, rec=None)`
Performance volatility and game context metrics.

**Features:**
- `xgd60_std_game`: Game-to-game std dev of xGD/60
- `gsax_volatility`: Goalie performance consistency
- `blowout_rate`: Fraction of games with |goal_diff| > 3
- `close_game_rate`: Fraction of games with |goal_diff| ≤ 1

**Implementation:**
- Requires both line-level `df` and game-level aggregates
- Computes per-game metrics, then aggregates volatility
- Uses `_make_no_draw_records()` helper to avoid ValueError

**Use Case:**
- Identifies teams with inconsistent performance
- Captures clutch/blowout tendencies

**Dependencies:**
- `core.config.CFG` for multiple parameters

**Design Notes:**
- Largest module (737 lines) with most complex feature engineering
- Consistent API across all 10+ feature builders
- Extensive use of vectorization and sparse matrices
- Proper handling of edge cases (zero TOI, missing teams, sparse matchups)
- Shrinkage/regularization throughout for small-sample stability

---

## Architecture & Data Flow

### Module Dependency Graph

```
config.py (foundation, no dependencies)
    ↓
    ├─→ stats.py (uses CFG for shrinkage)
    ├─→ advanced.py (uses CFG for multiple params)
    └─→ model.py (uses CFG for CV, prediction, rankings)
         ↓
         └─→ bt.py (imported by model.py)

data.py (independent, no internal dependencies)
```

**Key Characteristics:**
- Acyclic dependency graph (no circular dependencies)
- Unidirectional flow from config → domain modules
- Low coupling: only 2 internal dependencies (config, bt)
- High cohesion: each module has focused responsibility

### Data Transformation Pipeline

#### Submission Pipeline Flow
```
Raw line-level (25,827 rows)
    ↓ [data.load_data]
Raw DataFrame + matchup schedule
    ↓ [data.build_game_level]
Game-level (1,312 games)
    ↓ [stats.build_team_stats]
Team statistics (32 teams)
    ↓ [bt.fit_bt, bt.bootstrap_bt, bt.compute_sos]
BT strengths + uncertainty + SOS
    ↓ [model.build_rankings]
Composite rankings (32 teams)
    ↓ [model.cross_validate_bt]
CV metrics + calibration params (a, b)
    ↓ [model.predict_matchups]
Matchup predictions with CI
```

#### ML Pipeline Flow
```
Raw line-level (25,827 rows)
    ↓ [data.load_data, data.build_game_level]
Game-level (1,312 games)
    ↓ [stats.build_team_stats]
Basic team statistics (32 teams)
    ↓ [advanced.stack_records]
Stacked records (51,654 rows)
    ↓ [advanced.build_* functions]
Specialized features (32 teams × 35 features)
    ├─→ special_teams (6 features)
    ├─→ line_depth (3 features)
    ├─→ goalie (2 features)
    ├─→ high_danger (3 features)
    ├─→ rapm (3 features)
    ├─→ toi_entropy (3 features)
    ├─→ matchup_matrix (3 features)
    ├─→ st_expected_value (3 features)
    ├─→ volatility (4 features)
    └─→ residualized (5 features)
    ↓ [ML models]
Predictions + feature importance
```

### Feature Tiers

**Core (5 features):** BT strength, xGD/60, xG share, goal diff, win%
**Extended (22 features):** Core + special teams + line depth + goalie + high danger
**Full (35 features):** Extended + RAPM + entropy + matchup matrix + ST EV + volatility + residuals

### Symmetric Stacking Pattern

Both `stats.py` and `advanced.py` use symmetric home/away stacking:

```python
# Home records
home = df[["home_team", "away_team", "home_goals", ...]].rename(...)

# Away records
away = df[["away_team", "home_team", "away_goals", ...]].rename(...)

# Stack
stacked = pd.concat([home, away], ignore_index=True)
```

**Benefits:**
- Doubles sample size (25,827 → 51,654 records)
- Eliminates home/away bias in aggregations
- Enables symmetric opponent analysis

---

## Key Algorithms & Optimizations

### Bradley-Terry Vectorization

**Problem:** Naive iterative implementation using `iterrows()` is O(n_games × n_teams × n_iter) and extremely slow.

**Solution:** Vectorized aggregation using `np.bincount`

```python
# Pre-compute index arrays once
home_idx = np.array([team_to_idx[t] for t in game["home_team"]])
away_idx = np.array([team_to_idx[t] for t in game["away_team"]])

# Vectorized probability computation
p_home = expit(strength[home_idx] - strength[away_idx])

# Vectorized gradient aggregation (10-50x faster)
grad = np.bincount(home_idx, weights=weights * (1 - p_home), minlength=n)
grad -= np.bincount(away_idx, weights=weights * (1 - p_home), minlength=n)
```

**Speedup:** 10-50x faster than iterative approaches

### RAPM Sparse Matrix Construction

**Problem:** Dense matrix for RAPM would be ~15,000 segments × ~800 units = 12M entries (mostly zeros).

**Solution:** COO → CSR sparse matrix construction

```python
# Build COO matrix (efficient for construction)
row_indices = []  # Segment IDs
col_indices = []  # Unit IDs
values = []       # +1 for offense, -1 for defense

# Convert to CSR (efficient for Ridge fitting)
X_sparse = coo_matrix((values, (row_indices, col_indices))).tocsr()

# Ridge regression on sparse matrix
ridge = Ridge(alpha=regularization)
ridge.fit(X_sparse, y, sample_weight=weights)
```

**Memory reduction:** ~100x (12M → ~30K non-zero entries)

### Bootstrap Integration

**Challenge:** How to integrate bootstrap uncertainty into predictions?

**Naive approach:** `P = sigmoid(E[BT_diff])` - ignores uncertainty

**Correct approach:** `P = E[sigmoid(BT_diff + ε)]` where `ε ~ N(0, σ²)`

**Implementation:**
```python
# Compute uncertainty for each matchup
uncertainty = np.sqrt(bt_std[home]**2 + bt_std[away]**2)

# Shrink uncertainty (empirically tuned k ≈ 0.83)
uncertainty_shrunk = uncertainty_shrink_k * uncertainty

# Integrate via numerical approximation or bootstrap samples
p_home = np.mean([sigmoid(bt_diff + noise) for noise in bootstrap_samples])
```

**Impact:** More accurate probability estimates, especially for uncertain matchups

### Numerical Stability Techniques

1. **Probability clipping:** `[1e-8, 1-1e-8]` prevents `log(0)` errors
2. **Hessian clipping:** Ensures positive definiteness in Newton-Raphson
3. **L2 regularization:** Prevents ±inf for undefeated/winless teams
4. **Log-intensity clipping:** `[-5, 5]` in attack-defense model prevents overflow
5. **Shrinkage:** Bayesian regression toward mean for small samples

---

## Usage Examples

### Basic Submission Pipeline

```python
from core.data import load_data, build_game_level
from core.stats import build_team_stats
from core.bt import fit_bt, bootstrap_bt, compute_sos
from core.model import build_rankings, cross_validate_bt, predict_matchups
from core.config import CFG

# Load data
df, matchups = load_data("drive-download-20260205T132825Z-1-001.zip")
game = build_game_level(df)
teams = sorted(game["home_team"].unique())

# Compute team statistics
stats = build_team_stats(game)

# Fit Bradley-Terry model with bootstrap
bt = fit_bt(game, teams,
            ot_weight=CFG["bt"]["ot_weight"],
            regularization=CFG["bt"]["regularization"])
bt_mean, bt_std, bt_ci_lower, bt_ci_upper, bt_samples = bootstrap_bt(
    game, teams, n_boot=200)
sos = compute_sos(game, bt)

# Build ensemble rankings
rankings = build_rankings(stats, bt, sos)

# Cross-validate and calibrate
cv_result = cross_validate_bt(game, teams, n_folds=5)
print(f"CV Accuracy: {cv_result.accuracy:.3f}")
print(f"CV Brier: {cv_result.brier:.4f}")

# Predict matchups
predictions = predict_matchups(
    matchups, bt, cv_result.a, cv_result.b,
    bt_samples=bt_samples, teams=teams,
    bt_std=bt_std, uncertainty_shrink_k=0.83)
```

### Advanced Feature Engineering

```python
from core.advanced import (
    build_special_teams, build_line_depth, build_goalie_features,
    build_high_danger, build_rapm_features, residualize_features
)

# Extract advanced features
st_features = build_special_teams(df)
depth_features = build_line_depth(df)
goalie_features = build_goalie_features(df)
hd_features = build_high_danger(df, threshold=0.15)
rapm_features = build_rapm_features(df, teams, regularization=100.0)

# Residualize features to remove BT correlation
resid_features = residualize_features(
    stats, bt,
    feature_cols=["xgd60", "xg_share", "gsax60"],
    alpha=0.1)

# Combine all features
ml_features = stats.join([
    st_features, depth_features, goalie_features,
    hd_features, rapm_features, resid_features
])
```

### Hyperparameter Tuning

```python
from core.model import tune_bt_hyperparams, tune_uncertainty_shrink_k

# Tune BT hyperparameters
best_params, best_score = tune_bt_hyperparams(
    game, teams, n_outer=5, n_inner=3)
print(f"Best params: {best_params}")
print(f"Best Brier: {best_score:.4f}")

# Tune uncertainty shrinkage
cv_result = cross_validate_bt(game, teams, n_folds=5, n_boot=200)
best_k = tune_uncertainty_shrink_k(
    cv_result.oof_pred, cv_result.oof_true,
    bt_std, game)
print(f"Optimal shrinkage k: {best_k:.3f}")
```

---

## Design Patterns & Best Practices

### 1. Configuration Pattern
**Centralized configuration with deep merge and fallback defaults**

```python
# config.py
_DEFAULTS = {"bt": {"ot_weight": 0.5, ...}, ...}
CFG = load_config("config.yaml")  # Deep merges with _DEFAULTS

# Usage in other modules
from core.config import CFG
ot_weight = CFG["bt"]["ot_weight"]
```

**Benefits:** Enables hyperparameter tuning without code changes, provides sensible defaults

### 2. Facade Pattern
**model.py acts as high-level facade over lower-level bt.py functions**

```python
# High-level operation
cv_result = cross_validate_bt(game, teams, n_folds=5)

# Internally composes:
# - fit_bt() for each fold
# - fit_logistic() for calibration
# - compute_ece() for evaluation
# - Aggregates results into CVResult named tuple
```

**Benefits:** Simplifies complex workflows, provides clean API

### 3. Strategy Pattern
**Runtime selection of calibration method**

```python
def fit_logistic(game, bt, method="platt"):
    if method == "platt":
        return _fit_platt_scaling(...)
    elif method == "isotonic":
        return _fit_isotonic_regression(...)
```

**Benefits:** Flexible algorithm selection without code duplication

### 4. Template Method Pattern
**Consistent feature engineering pattern across advanced.py**

```python
def build_X(df, rec=None, **kwargs):
    # Step 1: Stack records (if not provided)
    if rec is None:
        rec = stack_records(df)

    # Step 2: Filter/aggregate data
    filtered = rec[rec["condition"]]

    # Step 3: Compute metrics
    metrics = filtered.groupby("team").agg(...)

    # Step 4: Return DataFrame indexed by team
    return metrics
```

**Benefits:** Consistent API, easy to add new features following established pattern

### 5. Functional Core
**Most functions are pure transformations with no side effects**

```python
# Pure function - no side effects, deterministic
def fit_bt(game, teams, ot_weight=0.5, ...):
    # Computes and returns result
    return bt_strength

# Not: modifying global state, writing files, etc.
```

**Benefits:** Composable, testable, parallelizable

### 6. Defensive Programming
**Extensive validation and graceful degradation**

```python
# Check for draws
if (game["home_goals"] == game["away_goals"]).any():
    raise ValueError("Dataset contains draws...")

# Safe division
xgd60 = np.where(toi > 0, (xgf - xga) / toi * 3600, 0)

# Convergence warnings
if not converged:
    warnings.warn("BT model did not converge...")
```

**Benefits:** Catches errors early, prevents silent failures

---

## Performance Characteristics

### Computational Complexity

| Operation | Complexity | Typical Runtime |
|-----------|-----------|-----------------|
| `build_game_level()` | O(n_records) | <1s for 25K records |
| `build_team_stats()` | O(n_games) | <1s for 1.3K games |
| `fit_bt()` | O(n_iter × n_games) | ~0.5s (10-30 iterations) |
| `bootstrap_bt()` | O(n_boot × fit_time) | ~100s (200 resamples) |
| `build_rapm_features()` | O(n_segments × n_units) | ~5s (sparse matrix) |
| `cross_validate_bt()` | O(n_folds × fit_time) | ~2.5s (5 folds) |

### Memory Usage

| Data Structure | Size | Memory |
|----------------|------|--------|
| Raw line-level DataFrame | 25,827 rows × 20 cols | ~4 MB |
| Game-level DataFrame | 1,312 rows × 15 cols | ~200 KB |
| Team statistics | 32 rows × 25 cols | ~10 KB |
| RAPM sparse matrix | 15K × 800 (30K nnz) | ~1 MB |
| Bootstrap samples | 32 × 200 | ~50 KB |

### Bottlenecks & Optimization Opportunities

**Current Bottlenecks:**
1. **Bootstrap (200 resamples):** ~100s sequential execution
   - **Opportunity:** Parallelize with `joblib` → ~20s on 8 cores
2. **RAPM matrix construction:** ~5s for sparse matrix building
   - **Opportunity:** Numba JIT compilation → ~1s
3. **Redundant stacking:** Multiple calls to `stack_records()` if `rec` not passed
   - **Opportunity:** Cache stacked records → eliminate redundant work

**Already Optimized:**
- ✅ Bradley-Terry vectorization (10-50x speedup)
- ✅ Sparse matrices for RAPM (100x memory reduction)
- ✅ Pre-computed index arrays
- ✅ Vectorized pandas operations

---

## Recommended Improvements

### High Priority (Architectural Impact)

**1. Split model.py into focused modules**
Current `model.py` (342 lines) handles multiple responsibilities. Recommended split:
```
core/
  calibration.py  # Platt scaling, isotonic regression
  evaluation.py   # Cross-validation, ECE, metrics
  ranking.py      # Ranking computation
  prediction.py   # Matchup predictions
```
**Benefit:** Improves maintainability, reduces module size to ~80-100 lines each

**2. Add type hints throughout**
```python
# Current
def fit_bt(game, teams, ot_weight=0.5, ...):

# Recommended
def fit_bt(
    game: pd.DataFrame,
    teams: list[str],
    ot_weight: float = 0.5,
    ...
) -> dict[str, float]:
```
**Benefit:** IDE support, type checking, better documentation

**3. Parallelize bootstrap operations**
```python
from joblib import Parallel, delayed

bt_samples = Parallel(n_jobs=-1)(
    delayed(fit_bt)(resample, teams, ot_weight, regularization)
    for resample in resamples
)
```
**Benefit:** 5-8x speedup on multi-core systems

### Medium Priority (Code Quality)

**4. Use dataclasses for structured returns**
```python
from dataclasses import dataclass

@dataclass
class BTResult:
    strengths: dict[str, float]
    convergence_iter: int
    final_delta: float
    converged: bool
```
**Benefit:** Type safety, better IDE support, clearer API

**5. Extract constants to eliminate magic numbers**
```python
# constants.py
SECONDS_PER_HOUR = 3600
AVERAGE_PP_DURATION_SECONDS = 120
BT_STEP_SIZE = 0.5
HESSIAN_CLIP_THRESHOLD = -1e-6
```
**Benefit:** Centralized constants, easier to tune

**6. Decompose long functions**
`build_rapm_features()` (197 lines) should be split:
- `_construct_rapm_matrix()` - sparse matrix construction
- `_fit_rapm_ridge()` - Ridge regression
- `_aggregate_unit_rapm()` - team-level aggregation
- `build_rapm_features()` - orchestration

**Benefit:** Easier to test, understand, and maintain

### Low Priority (Polish)

**7. Add module-level docstrings**
```python
"""
bt.py - Bradley-Terry ranking models

Implements OT-aware Bradley-Terry with L2 regularization,
bootstrap uncertainty quantification, and attack-defense Poisson variants.
"""
```

**8. Custom exception classes**
```python
class BTConvergenceError(Exception):
    """Raised when Bradley-Terry model fails to converge"""
    pass
```

**9. Logging framework**
```python
import logging
logger = logging.getLogger(__name__)
logger.warning("BT model did not converge...")
```

---

## Testing

The core package has comprehensive test coverage:

**Unit Tests:** `tests/test_core.py` (59 tests)
- BT convergence and correctness
- OT downweighting validation
- Regularization shrinkage
- Bootstrap CI coverage
- Team statistics aggregation
- Config loading and merging
- RAPM sparse matrix construction
- Feature engineering functions
- Residualization decorrelation
- Volatility metrics

**Smoke Tests:** `tests/test_smoke.py`
- End-to-end pipeline integration
- Requires data files to run

**CI/CD:** `.github/workflows/test.yml`
- Python 3.10, 3.11, 3.12
- Runs on every push/PR

**Run tests:**
```bash
python -m pytest tests/ -v
python -m pytest tests/test_core.py::test_bt_convergence -v
```

---

## Summary

The `core/` package provides a robust, high-performance foundation for hockey analytics with:

**Strengths:**
- ✅ Exceptional vectorization (10-50x speedup)
- ✅ Sophisticated algorithms (BT, RAPM, bootstrap)
- ✅ Clean module separation and low coupling
- ✅ Comprehensive feature engineering (10+ functions)
- ✅ Configuration externalization
- ✅ Extensive test coverage (59 unit tests)
- ✅ Numerical stability throughout

**Architecture Quality:** 8.5/10

**Primary Improvement Opportunities:**
1. Split `model.py` into smaller modules
2. Add type hints for better IDE support
3. Parallelize bootstrap operations

The codebase demonstrates strong software engineering practices with a focus on computational efficiency, making it well-suited for both competition submissions and production ML pipelines.

---

**Last Updated:** 2026-02-27
**Total Lines:** 1,550 across 7 modules
**Test Coverage:** 59 unit tests + smoke tests
