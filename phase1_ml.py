"""
WHSDSC 2026 Phase 1 — ML Enhancement Module
=============================================
Adds ML challenger models per Codex (GPT-5.3) recommendation:
  - Baseline: Home-ice only (~50.7%)
  - Core model: Bradley-Terry logistic regression (current)
  - Challenger 1: Elastic Net logistic regression (regularized, interpretable)
  - Challenger 2: XGBoost/GradientBoosting (high accuracy)
  - Challenger 3: Blended ensemble (Elastic Net + XGBoost)
  - Challenger 4: Poisson/Skellam model (count-based)

Reports: Accuracy, Log-loss, Brier score, Calibration curves
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.special import expit

from core.data import load_data, build_game_level
from core.bt import fit_bt, fit_attack_defense, skellam_predict_ad
from core.stats import build_team_stats
from core.advanced import (build_special_teams, build_line_depth,
                           build_goalie_features, build_high_danger,
                           residualize_features, build_rapm_features,
                           build_toi_entropy, build_matchup_matrix,
                           build_st_expected_value, build_volatility_features,
                           stack_records)
from core.config import CFG

from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import brier_score_loss, log_loss, accuracy_score
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*lbfgs.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*max_iter.*", category=ConvergenceWarning)

OUTPUT_DIR = "output"

# ── Feature tiers ────────────────────────────────────────────────────
# Core: used by BT-only and base Elastic Net (5 features)
# Removed xg_share_diff (r=0.83 with ev_xgd60_diff) and save_pct_diff (r=0.85 with gsax60_shrunk_diff)
CORE_FEATURES = [
    "bt_diff", "ev_xgd60_diff",
    "gsax60_shrunk_diff", "fin60_shrunk_diff", "pp_edge",
]

# Extended: residualized features replace raw collinear ones (5 core + 17 resid = 22 features)
EXTENDED_FEATURES = CORE_FEATURES + [
    "ev_xgd60_resid_diff", "gsax60_shrunk_resid_diff",
    "fin60_shrunk_resid_diff", "pp_xgf60_resid_diff",
    "pk_xga60_resid_diff", "depth_gap_resid_diff",
    "high_danger_freq60_resid_diff",
    "shooting_pct_resid_diff",
    # New residualized features
    "off_line_entropy_resid_diff", "def_pairing_entropy_resid_diff",
    "top_vs_top_share_resid_diff", "top_shelter_share_resid_diff",
    "exploit_ability_resid_diff", "shutdown_resilience_resid_diff",
    "net_st_ev_resid_diff", "xgd60_std_game_resid_diff", "gsax_volatility_resid_diff",
]

# Full: add RAPM + other non-residualized features (22 extended + 12/13 = 34/35 features)
# Exclude quality_vs_goalie if symmetric augmentation is enabled (asymmetric feature)
_FULL_BASE = [
    "gd_pg_diff", "mean_max_xg_for_diff", "pen60_diff",
    "starter_toi_share_diff", "quality_vs_goalie",
    "rapm_attack_diff", "rapm_defense_diff", "rapm_topline_effect_diff",
    # New non-residualized features
    "goalie_entropy_diff", "depth_insurance_diff", "pen_drawn60_diff",
    "blowout_rate_diff", "close_game_rate_diff",
]

if CFG.get("augmentation", {}).get("symmetric", False):
    FULL_FEATURES = EXTENDED_FEATURES + [f for f in _FULL_BASE if f != "quality_vs_goalie"]
else:
    FULL_FEATURES = EXTENDED_FEATURES + _FULL_BASE

# Columns to residualize from team_stats
RESID_STAT_COLS = ["gsax60_shrunk", "fin60_shrunk", "shooting_pct"]

# Columns to residualize from advanced DataFrames
RESID_ADV_COLS = {
    "special_teams": ["ev_xgd60", "pp_xgf60", "pk_xga60"],
    "line_depth": ["depth_gap"],
    "high_danger": ["high_danger_freq60"],
    "toi_entropy": ["off_line_entropy", "def_pairing_entropy", "top_vs_top_share", "top_shelter_share"],
    "matchup_matrix": ["exploit_ability", "shutdown_resilience"],
    "st_ev": ["net_st_ev"],
    "volatility": ["xgd60_std_game", "gsax_volatility"],
}


# ───────────────────── FEATURE ENGINEERING ──────────────────────────

def _symmetric_augment(X, y):
    """Symmetric augmentation for tree models: flip sign of features and labels.

    Returns augmented data with doubled size.
    """
    X_aug = np.vstack([X, -X])
    y_aug = np.concatenate([y, 1 - y])
    return X_aug, y_aug


def _build_advanced_dict(special_teams=None, line_depth=None, goalie_feats=None,
                         high_danger=None, toi_entropy=None, matchup_matrix=None,
                         st_ev=None, volatility=None, rapm=None):
    """Build {source_name: DataFrame} dict, omitting None entries."""
    d = {}
    for name, val in [("special_teams", special_teams), ("line_depth", line_depth),
                      ("goalie_feats", goalie_feats), ("high_danger", high_danger),
                      ("toi_entropy", toi_entropy), ("matchup_matrix", matchup_matrix),
                      ("st_ev", st_ev), ("volatility", volatility), ("rapm", rapm)]:
        if val is not None:
            d[name] = val
    return d


def build_ml_features(game, team_stats, bt_strength, advanced_features=None,
                      residualized_stats=None, bt_std=None):
    """Build feature matrix for each game: home_feature - away_feature (differential).

    Supports extended features from advanced DataFrames when provided.
    advanced_features is a dict {source_name: DataFrame} built by _build_advanced_dict().
    """
    if advanced_features is None:
        advanced_features = {}

    # Differential columns from team_stats
    stat_diff_cols = [
        "win_pct", "gd_pg", "xgd_pg", "xg_share", "xgd_per60",
        "shooting_pct", "save_pct", "pim_pg",
        "xg_per_shot_against", "shots_for60",
        "fin60_shrunk", "gsax60_shrunk", "mean_max_xg_for",
    ]
    # Add optional columns if they exist in team_stats
    for col in ["pim60", "pen60"]:
        if col in team_stats.columns:
            stat_diff_cols.append(col)

    # Differential columns from advanced DataFrames — dynamic lookup
    ADV_COLS_MAP = {
        "special_teams": ["ev_xgd60", "pp_xgf60", "pk_xga60"],
        "line_depth": ["depth_gap"],
        "goalie_feats": ["goalie_gsax60", "starter_toi_share"],
        "high_danger": ["high_danger_freq60"],
        "toi_entropy": ["off_line_entropy", "def_pairing_entropy", "top_vs_top_share", "top_shelter_share", "goalie_entropy"],
        "matchup_matrix": ["exploit_ability", "shutdown_resilience", "depth_insurance"],
        "st_ev": ["net_st_ev", "pen_drawn60"],
        "volatility": ["xgd60_std_game", "gsax_volatility", "blowout_rate", "close_game_rate"],
    }
    adv_diff_cols = {}
    for src_name, src_df in advanced_features.items():
        if src_name in ADV_COLS_MAP:
            adv_diff_cols[src_name] = [c for c in ADV_COLS_MAP[src_name] if c in src_df.columns]

    # Residualized columns
    resid_diff_cols = []
    if residualized_stats is not None:
        resid_diff_cols = [c for c in residualized_stats.columns if c.endswith("_resid")]

    # Start building the result DataFrame
    feat_df = game[["game_id", "home_team", "away_team", "home_win"]].copy()

    # BT strength differential (vectorized map)
    bt_series = pd.Series(bt_strength)
    feat_df["bt_diff"] = feat_df["home_team"].map(bt_series).fillna(0) - feat_df["away_team"].map(bt_series).fillna(0)

    # Helper: merge a team-indexed DataFrame for home and away, compute diffs
    def _merge_diffs(source_df, cols, suffix=""):
        for col in cols:
            if col not in source_df.columns:
                continue
            col_series = source_df[col]
            out_name = f"{col}{suffix}_diff"
            feat_df[out_name] = (feat_df["home_team"].map(col_series).fillna(0)
                                 - feat_df["away_team"].map(col_series).fillna(0))

    # Stat differentials
    _merge_diffs(team_stats, stat_diff_cols)

    # Advanced feature differentials
    for src_name, cols in adv_diff_cols.items():
        _merge_diffs(advanced_features[src_name], cols)

    # Residualized feature differentials
    if residualized_stats is not None:
        _merge_diffs(residualized_stats, resid_diff_cols)

    # RAPM feature differentials
    if "rapm" in advanced_features:
        rapm = advanced_features["rapm"]
        _merge_diffs(rapm, ["rapm_attack", "rapm_defense", "rapm_topline_effect"])

    # Interaction features (asymmetric matchups — not pure diffs)
    if "special_teams" in advanced_features:
        st = advanced_features["special_teams"]
        pp_xgf60 = st["pp_xgf60"] if "pp_xgf60" in st.columns else pd.Series(dtype=float)
        pk_xga60 = st["pk_xga60"] if "pk_xga60" in st.columns else pd.Series(dtype=float)
        feat_df["pp_edge"] = (feat_df["home_team"].map(pp_xgf60).fillna(0)
                              - feat_df["away_team"].map(pk_xga60).fillna(0))

        if "pen60" in team_stats.columns:
            pen60 = team_stats["pen60"]
            feat_df["pp_opportunity"] = (feat_df["home_team"].map(pp_xgf60).fillna(0)
                                         * feat_df["away_team"].map(pen60).fillna(0))
            feat_df["pk_stress"] = (feat_df["away_team"].map(pp_xgf60).fillna(0)
                                    * feat_df["home_team"].map(pen60).fillna(0))

    if "goalie_feats" in advanced_features:
        gk = advanced_features["goalie_feats"]
        gk_gsax60 = gk["goalie_gsax60"] if "goalie_gsax60" in gk.columns else pd.Series(dtype=float)
        if "xg_per_shot" in team_stats.columns:
            feat_df["quality_vs_goalie"] = (feat_df["home_team"].map(team_stats["xg_per_shot"]).fillna(0)
                                            - feat_df["away_team"].map(gk_gsax60).fillna(0))
        else:
            feat_df["quality_vs_goalie"] = -feat_df["away_team"].map(gk_gsax60).fillna(0)

    # Uncertainty feature from bootstrap
    if bt_std is not None:
        bt_std_series = pd.Series(bt_std) if isinstance(bt_std, dict) else bt_std
        h_std = feat_df["home_team"].map(bt_std_series).fillna(0)
        a_std = feat_df["away_team"].map(bt_std_series).fillna(0)
        feat_df["uncertainty"] = np.sqrt(h_std**2 + a_std**2)

    # Drop helper columns
    feat_df = feat_df.drop(columns=["home_team", "away_team"])
    return feat_df


# ───────────────────── MODEL COMPARISON ─────────────────────────────

def run_model_comparison(game, teams, df_raw=None, n_splits=None, bt_kwargs=None):
    """
    Leakage-safe 5-fold CV comparison.

    IMPORTANT:
    - Feature engineering (team season stats, BT ratings) MUST be fit on the
      training fold only. Otherwise CV will be overly optimistic.
    - Record-level features (special teams, line depth, goalie) are also
      recomputed per fold from raw data.
    - Scaling MUST also be fit on training fold only.
    """
    if n_splits is None:
        n_splits = CFG["cv"]["n_folds"]

    # Feature tiers
    enet_features = EXTENDED_FEATURES.copy()
    gb_features = FULL_FEATURES.copy()

    y = game["home_win"].values
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=CFG["cv"]["random_state"])
    results = {}

    # Out-of-fold predictions for fair metric computation.
    oof = {
        "Baseline (fold home-rate)": np.zeros(len(game), dtype=float),
        "BT-only Logistic": np.zeros(len(game), dtype=float),
        "Elastic Net Logistic": np.zeros(len(game), dtype=float),
        "Skellam (AD)": np.zeros(len(game), dtype=float),
    }
    if HAS_XGB:
        oof["XGBoost"] = np.zeros(len(game), dtype=float)
    else:
        oof["Gradient Boosting"] = np.zeros(len(game), dtype=float)
    oof["Stacked Ensemble"] = np.zeros(len(game), dtype=float)

    # Per-fold feature importance tracking
    fold_enet_coefs = []
    fold_gb_importances = []

    from sklearn.pipeline import Pipeline

    # Build game_id -> raw data index for leak-safe fold splitting
    if df_raw is not None:
        raw_game_ids = set(df_raw["game_id"].unique())
    else:
        raw_game_ids = set()

    for fold_i, (train_idx, test_idx) in enumerate(cv.split(np.zeros(len(game)), y)):
        train = game.iloc[train_idx].reset_index(drop=True)
        test = game.iloc[test_idx].reset_index(drop=True)

        # Fold-specific feature engineering (no leakage).
        team_stats = build_team_stats(train)
        bt = fit_bt(train, teams, **(bt_kwargs or {}))

        # Record-level features from training fold only
        adv_dict = {}
        resid_stats = None
        if df_raw is not None:
            train_game_ids = set(train["game_id"])
            df_train = df_raw[df_raw["game_id"].isin(train_game_ids)]
            rec_train = stack_records(df_train)  # Pre-compute once per fold
            st_feats = build_special_teams(df_train, rec=rec_train)
            ld_feats = build_line_depth(df_train, rec=rec_train)
            gk_feats = build_goalie_features(df_train, rec=rec_train)
            hd_feats = build_high_danger(df_train, rec=rec_train)
            rapm_feats = build_rapm_features(df_train, teams, rec=rec_train)
            te_feats = build_toi_entropy(df_train, rec=rec_train)
            mm_feats = build_matchup_matrix(df_train, rec=rec_train)
            stev_feats = build_st_expected_value(df_train, rec=rec_train)
            vol_feats = build_volatility_features(df_train, train, rec=rec_train)

            adv_dict = _build_advanced_dict(
                special_teams=st_feats, line_depth=ld_feats,
                goalie_feats=gk_feats, high_danger=hd_feats,
                toi_entropy=te_feats, matchup_matrix=mm_feats,
                st_ev=stev_feats if len(stev_feats) > 0 else None,
                volatility=vol_feats, rapm=rapm_feats)

            # Residualize features (per fold, using fold BT)
            resid_base = team_stats.copy()
            for adv_src_name, adv_cols in RESID_ADV_COLS.items():
                src = adv_dict.get(adv_src_name)
                if src is not None:
                    for col in adv_cols:
                        if col in src.columns:
                            resid_base[col] = src[col]

            all_resid_cols = RESID_STAT_COLS + [c for cols in RESID_ADV_COLS.values() for c in cols]
            valid_cols = [c for c in all_resid_cols if c in resid_base.columns]
            resid_stats = residualize_features(resid_base, bt, valid_cols)

        feat_tr = build_ml_features(train, team_stats, bt, adv_dict, resid_stats)
        feat_te = build_ml_features(test, team_stats, bt, adv_dict, resid_stats)

        # Ensure all feature columns exist (fill missing with 0)
        for col in gb_features:
            if col not in feat_tr.columns:
                feat_tr[col] = 0.0
                feat_te[col] = 0.0

        y_tr = feat_tr["home_win"].values

        # Baseline: predict with training home win-rate.
        p0 = float(y_tr.mean())
        oof["Baseline (fold home-rate)"][test_idx] = p0

        # BT-only logistic (single feature) — use LogisticRegressionCV for fair comparison.
        bt_tr = feat_tr[["bt_diff"]].values
        bt_te = feat_te[["bt_diff"]].values
        lr_bt = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegressionCV(Cs=10, cv=3, max_iter=2000, random_state=42)),
        ])
        lr_bt.fit(bt_tr, y_tr)
        oof["BT-only Logistic"][test_idx] = lr_bt.predict_proba(bt_te)[:, 1]

        # Elastic Net logistic (extended features) with fold-scaling and inner CV.
        X_tr_enet = feat_tr[enet_features].values
        X_te_enet = feat_te[enet_features].values
        enet = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegressionCV(
                Cs=20, cv=5, penalty="elasticnet", solver="saga",
                l1_ratios=[0.01, 0.1, 0.3, 0.5, 0.9], max_iter=5000, random_state=42,
            )),
        ])
        enet.fit(X_tr_enet, y_tr)
        oof["Elastic Net Logistic"][test_idx] = enet.predict_proba(X_te_enet)[:, 1]
        fold_enet_coefs.append(np.abs(enet.named_steps["model"].coef_[0]))

        # XGBoost / fallback GB with hyperparameter tuning (full features).
        # Option A symmetric augmentation: tune on original, refit on augmented.
        X_tr_gb = feat_tr[gb_features].values
        X_te_gb = feat_te[gb_features].values
        use_sym = CFG.get("augmentation", {}).get("symmetric", False)
        if HAS_XGB:
            param_grid = {
                "max_depth": [2, 3, 4],
                "learning_rate": [0.03, 0.05, 0.1],
                "n_estimators": [100, 200, 300],
            }
            xgb_base = xgb.XGBClassifier(
                subsample=0.8, colsample_bytree=0.8,
                reg_alpha=1.0, reg_lambda=2.0,
                eval_metric="logloss", random_state=42,
            )
            gs = GridSearchCV(xgb_base, param_grid, cv=3, scoring="neg_brier_score",
                              n_jobs=-1, refit=not use_sym)
            gs.fit(X_tr_gb, y_tr)
            if use_sym:
                X_aug, y_aug = _symmetric_augment(X_tr_gb, y_tr)
                final_model = xgb.XGBClassifier(
                    **{**xgb_base.get_params(), **gs.best_params_})
                final_model.fit(X_aug, y_aug)
                oof["XGBoost"][test_idx] = final_model.predict_proba(X_te_gb)[:, 1]
                fold_gb_importances.append(final_model.feature_importances_)
            else:
                oof["XGBoost"][test_idx] = gs.predict_proba(X_te_gb)[:, 1]
                fold_gb_importances.append(gs.best_estimator_.feature_importances_)
        else:
            param_grid = {
                "max_depth": [2, 3, 4],
                "learning_rate": [0.03, 0.05, 0.1],
                "n_estimators": [100, 200, 300],
            }
            gb_base = GradientBoostingClassifier(
                subsample=0.8, random_state=42,
            )
            gs = GridSearchCV(gb_base, param_grid, cv=3, scoring="neg_brier_score",
                              n_jobs=-1, refit=not use_sym)
            gs.fit(X_tr_gb, y_tr)
            if use_sym:
                X_aug, y_aug = _symmetric_augment(X_tr_gb, y_tr)
                final_model = GradientBoostingClassifier(
                    **{**gb_base.get_params(), **gs.best_params_})
                final_model.fit(X_aug, y_aug)
                oof["Gradient Boosting"][test_idx] = final_model.predict_proba(X_te_gb)[:, 1]
                fold_gb_importances.append(final_model.feature_importances_)
            else:
                oof["Gradient Boosting"][test_idx] = gs.predict_proba(X_te_gb)[:, 1]
                fold_gb_importances.append(gs.best_estimator_.feature_importances_)

        # Attack-Defense Skellam model (no ML — Poisson-based)
        ad_model = fit_attack_defense(train, teams,
                                       use_xg=CFG["attack_defense"]["use_xg"],
                                       regularization=CFG["attack_defense"]["regularization"])
        mean_toi = train["toi"].mean()
        for ti in test_idx:
            g = game.iloc[ti]
            oof["Skellam (AD)"][ti] = skellam_predict_ad(
                ad_model, g["home_team"], g["away_team"], toi=mean_toi)

    # Stacked Ensemble: cross-fitted logistic stacking meta-learner over all OOF predictions
    stack_models = ["BT-only Logistic", "Elastic Net Logistic", "Skellam (AD)"]
    stack_models.append("XGBoost" if HAS_XGB else "Gradient Boosting")
    stack_X = np.column_stack([oof[m] for m in stack_models])

    # Nested CV for honest OOF predictions
    meta_cv = StratifiedKFold(n_splits=n_splits, shuffle=True,
                              random_state=CFG["cv"]["random_state"] + 100)
    for meta_tr, meta_te in meta_cv.split(stack_X, y):
        meta_fold = LogisticRegressionCV(Cs=10, cv=3, max_iter=2000, random_state=42)
        meta_fold.fit(stack_X[meta_tr], y[meta_tr])
        oof["Stacked Ensemble"][meta_te] = meta_fold.predict_proba(stack_X[meta_te])[:, 1]

    # Refit on all for final predictions
    meta_model = LogisticRegressionCV(Cs=10, cv=3, max_iter=2000, random_state=42)
    meta_model.fit(stack_X, y)
    print(f"  Stacked Ensemble meta-model coefs: {dict(zip(stack_models, meta_model.coef_[0].round(3)))}")

    # Summarize
    def _metrics(p):
        p = np.clip(p, 1e-8, 1 - 1e-8)
        return {
            "accuracy": accuracy_score(y, (p >= 0.5).astype(int)),
            "log_loss": log_loss(y, p),
            "brier": brier_score_loss(y, p),
            "probs": p,
        }

    results["Baseline (fold home-rate)"] = _metrics(oof["Baseline (fold home-rate)"])
    results["BT-only Logistic"] = _metrics(oof["BT-only Logistic"])
    results["Elastic Net Logistic"] = _metrics(oof["Elastic Net Logistic"])
    results["Skellam (AD)"] = _metrics(oof["Skellam (AD)"])
    if HAS_XGB:
        results["XGBoost"] = _metrics(oof["XGBoost"])
    else:
        results["Gradient Boosting"] = _metrics(oof["Gradient Boosting"])
    results["Stacked Ensemble"] = _metrics(oof["Stacked Ensemble"])

    # Aggregate per-fold importance
    fold_importance = {
        "enet_coefs": np.array(fold_enet_coefs) if fold_enet_coefs else None,
        "gb_importances": np.array(fold_gb_importances) if fold_gb_importances else None,
    }

    return results, enet_features, gb_features, fold_importance


# ───────────────────── FEATURE IMPORTANCE ───────────────────────────

def compute_feature_importance(X, y, feature_names):
    """Fit final models and extract feature importance."""
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    # Elastic Net coefficients
    enet = LogisticRegressionCV(
        Cs=20, cv=5, penalty="elasticnet", solver="saga",
        l1_ratios=[0.1, 0.5, 0.9], max_iter=5000, random_state=42
    )
    enet.fit(X_s, y)
    enet_coefs = pd.Series(enet.coef_[0], index=feature_names).abs().sort_values(ascending=False)

    # XGBoost feature importance
    if HAS_XGB:
        xgb_model = xgb.XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=1.0, reg_lambda=2.0,
            eval_metric="logloss", random_state=42,
        )
        xgb_model.fit(X, y)
        xgb_imp = pd.Series(
            xgb_model.feature_importances_, index=feature_names
        ).sort_values(ascending=False)
    else:
        xgb_imp = None

    return enet_coefs, xgb_imp, enet, scaler


# ───────────────────── MATCHUP PREDICTION ───────────────────────────

def predict_matchups_ml(matchups, team_stats, bt_strength, model, scaler, feature_names,
                        advanced_features=None, residualized_stats=None):
    """Predict Round 1 matchups using the best ML model."""
    # Build a minimal "game" DataFrame for build_ml_features
    fake_game = matchups.copy()
    fake_game["home_win"] = 0  # placeholder
    if "game_id" not in fake_game.columns:
        fake_game["game_id"] = range(len(fake_game))

    feat_df = build_ml_features(fake_game, team_stats, bt_strength,
                                advanced_features, residualized_stats)

    # Ensure all feature columns exist
    for col in feature_names:
        if col not in feat_df.columns:
            feat_df[col] = 0.0

    X_new = feat_df[feature_names].values
    X_new_s = scaler.transform(X_new)
    probs = model.predict_proba(X_new_s)[:, 1]

    result = matchups.copy()
    result["home_win_prob_ml"] = probs.round(4)
    result["predicted_winner_ml"] = np.where(
        probs >= 0.5, result["home_team"], result["away_team"]
    )
    return result


# ───────────────────── VISUALIZATION ────────────────────────────────

def create_ml_dashboard(results, y, enet_coefs, xgb_imp, feature_names, output_dir):
    """Create ML comparison dashboard."""
    plt.rcParams.update({
        "font.size": 10, "font.family": "sans-serif",
        "axes.spines.top": False, "axes.spines.right": False,
    })

    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)

    # ── Panel 1: Model comparison bar chart ──
    ax1 = fig.add_subplot(gs[0, 0])
    models = list(results.keys())
    accs = [results[m]["accuracy"] for m in models]
    colors = ["#95a5a6", "#3498db", "#2ecc71", "#f39c12", "#e74c3c", "#9b59b6"][:len(models)]
    bars = ax1.barh(models, accs, color=colors, edgecolor="white", height=0.5)
    ax1.set_xlabel("5-Fold CV Accuracy")
    ax1.set_title("Model Accuracy Comparison", fontweight="bold")
    acc_lo = max(0.0, min(accs) - 0.05)
    acc_hi = min(1.0, max(accs) + 0.05)
    ax1.set_xlim(acc_lo, acc_hi)
    for bar, acc in zip(bars, accs):
        ax1.text(acc + 0.003, bar.get_y() + bar.get_height()/2,
                 f"{acc:.1%}", va="center", fontsize=9)

    # ── Panel 2: Brier + Log-loss comparison ──
    ax2 = fig.add_subplot(gs[0, 1])
    x_pos = np.arange(len(models))
    briers = [results[m]["brier"] for m in models]
    lls = [results[m]["log_loss"] for m in models]
    w = 0.35
    ax2.bar(x_pos - w/2, briers, w, label="Brier Score", color="#3498db", alpha=0.8)
    ax2.bar(x_pos + w/2, lls, w, label="Log-Loss", color="#e74c3c", alpha=0.8)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([m.split("(")[0].strip()[:12] for m in models], rotation=30, ha="right", fontsize=8)
    ax2.set_ylabel("Score (lower = better)")
    ax2.set_title("Brier Score & Log-Loss", fontweight="bold")
    ax2.legend(fontsize=8)

    # ── Panel 3: Calibration curves ──
    ax3 = fig.add_subplot(gs[0, 2])
    non_baseline = [n for n in results if n != "Baseline (fold home-rate)"]
    all_model_probs = np.concatenate([results[name]["probs"] for name in non_baseline])
    cal_lo = max(0.0, np.percentile(all_model_probs, 2) - 0.02)
    cal_hi = min(1.0, np.percentile(all_model_probs, 98) + 0.02)
    ax3.plot([cal_lo, cal_hi], [cal_lo, cal_hi], "k--", alpha=0.5, label="Perfect")
    cal_models = ["BT-only Logistic", "Elastic Net Logistic", "Skellam (AD)",
                  "XGBoost" if HAS_XGB else "Gradient Boosting", "Stacked Ensemble"]
    cal_colors = ["#3498db", "#2ecc71", "#f39c12", "#e74c3c", "#9b59b6"]
    for name, color in zip(cal_models, cal_colors):
        if name not in results:
            continue
        probs = results[name]["probs"]
        bins = np.linspace(cal_lo, cal_hi, 7)
        bin_centers, bin_means = [], []
        for i in range(len(bins)-1):
            mask = (probs >= bins[i]) & (probs < bins[i+1])
            if mask.sum() > 10:
                bin_centers.append(probs[mask].mean())
                bin_means.append(y[mask].mean())
        if bin_centers:
            ax3.plot(bin_centers, bin_means, "o-", color=color, label=name, markersize=5)
    ax3.set_xlabel("Predicted Probability")
    ax3.set_ylabel("Observed Win Rate")
    ax3.set_title("Calibration Curves", fontweight="bold")
    ax3.legend(fontsize=7, loc="upper left")
    ax3.set_xlim(cal_lo, cal_hi)
    ax3.set_ylim(cal_lo, cal_hi + 0.05)

    # ── Panel 4: Elastic Net coefficients ──
    ax4 = fig.add_subplot(gs[1, 0])
    top_enet = enet_coefs.head(10)
    clean_names_e = [n.replace("_diff", "").replace("_", " ").title() for n in top_enet.index]
    ax4.barh(clean_names_e[::-1], top_enet.values[::-1], color="#2ecc71", edgecolor="white")
    ax4.set_xlabel("|Coefficient|")
    ax4.set_title("Elastic Net: Feature Importance", fontweight="bold")

    # ── Panel 5: XGBoost feature importance ──
    ax5 = fig.add_subplot(gs[1, 1])
    if xgb_imp is not None:
        top_xgb = xgb_imp.head(10)
        clean_names_xgb = [n.replace("_diff", "").replace("_", " ").title() for n in top_xgb.index]
        ax5.barh(clean_names_xgb[::-1], top_xgb.values[::-1], color="#e74c3c", edgecolor="white")
        ax5.set_xlabel("Feature Importance (gain)")
        ax5.set_title("XGBoost: Feature Importance", fontweight="bold")
    else:
        ax5.text(0.5, 0.5, "XGBoost not available", ha="center", va="center", transform=ax5.transAxes)

    # ── Panel 6: Summary table ──
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")
    table_data = []
    for name in models:
        r = results[name]
        table_data.append([
            name.split("(")[0].strip()[:18],
            f"{r['accuracy']:.1%}",
            f"{r['log_loss']:.4f}",
            f"{r['brier']:.4f}",
        ])
    table = ax6.table(
        cellText=table_data,
        colLabels=["Model", "Accuracy", "Log-Loss", "Brier"],
        loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)
    # Color best values
    best_acc_idx = np.argmax([results[m]["accuracy"] for m in models])
    best_ll_idx = np.argmin([results[m]["log_loss"] for m in models])
    best_br_idx = np.argmin([results[m]["brier"] for m in models])
    for idx, col in [(best_acc_idx, 1), (best_ll_idx, 2), (best_br_idx, 3)]:
        table[idx + 1, col].set_facecolor("#d5f5e3")
    ax6.set_title("Model Comparison Summary", fontweight="bold", pad=20)

    fig.suptitle("WHSDSC 2026 — ML Model Comparison Dashboard",
                 fontsize=15, fontweight="bold", y=0.98)
    fig.text(0.5, 0.005,
             "5-Fold Stratified CV | Tiered Features: Core(5) + Extended(22) + Full(35) | AD Skellam + Cross-Fitted Stacked Ensemble",
             ha="center", fontsize=8, color="gray")

    out = os.path.join(output_dir, "phase1_ml_comparison.png")
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    return out


def create_diagnostics_dashboard(X_enet, X_gb, y, enet_features, gb_features,
                                 fold_importance, output_dir):
    """Create ML diagnostics dashboard: per-fold importance, correlation heatmap."""
    fig = plt.figure(figsize=(18, 6))
    gs = GridSpec(1, 3, figure=fig, wspace=0.35)

    # Panel 1: Feature correlation heatmap (full feature set)
    ax1 = fig.add_subplot(gs[0, 0])
    corr_matrix = np.corrcoef(X_gb.T)
    clean_names = [n.replace("_diff", "").replace("_", " ").title() for n in gb_features]
    im = ax1.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax1.set_xticks(range(len(gb_features)))
    ax1.set_yticks(range(len(gb_features)))
    ax1.set_xticklabels(clean_names, rotation=45, ha="right", fontsize=6)
    ax1.set_yticklabels(clean_names, fontsize=6)
    plt.colorbar(im, ax=ax1, shrink=0.8)
    ax1.set_title("Feature Correlation (Full Tier)", fontweight="bold")

    # Panel 2: Per-fold Elastic Net coefficients (mean +/- std)
    ax2 = fig.add_subplot(gs[0, 1])
    clean_enet = [n.replace("_diff", "").replace("_", " ").title() for n in enet_features]
    if fold_importance["enet_coefs"] is not None:
        coef_arr = fold_importance["enet_coefs"]
        mean_coefs = coef_arr.mean(axis=0)
        std_coefs = coef_arr.std(axis=0)
        order = np.argsort(mean_coefs)[::-1]
        ax2.barh(
            [clean_enet[i] for i in order][::-1],
            mean_coefs[order][::-1],
            xerr=std_coefs[order][::-1],
            color="#2ecc71", edgecolor="white", capsize=3
        )
        ax2.set_xlabel("|Coefficient| (mean +/- std across folds)")
        ax2.set_title("Per-Fold Elastic Net Importance", fontweight="bold")
    else:
        ax2.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax2.transAxes)

    # Panel 3: Per-fold GB importances (mean +/- std)
    ax3 = fig.add_subplot(gs[0, 2])
    clean_gb = [n.replace("_diff", "").replace("_", " ").title() for n in gb_features]
    if fold_importance["gb_importances"] is not None:
        imp_arr = fold_importance["gb_importances"]
        mean_imp = imp_arr.mean(axis=0)
        std_imp = imp_arr.std(axis=0)
        order = np.argsort(mean_imp)[::-1]
        ax3.barh(
            [clean_gb[i] for i in order][::-1],
            mean_imp[order][::-1],
            xerr=std_imp[order][::-1],
            color="#e74c3c", edgecolor="white", capsize=3
        )
        ax3.set_xlabel("Importance (mean +/- std across folds)")
        ax3.set_title("Per-Fold Boosting Importance", fontweight="bold")
    else:
        ax3.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax3.transAxes)

    fig.suptitle("WHSDSC 2026 — ML Diagnostics", fontsize=14, fontweight="bold", y=1.02)
    out = os.path.join(output_dir, "phase1_ml_diagnostics.png")
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    return out


# ───────────────────── MAIN ─────────────────────────────────────────

def main():
    print("=" * 60)
    print("WHSDSC 2026 — ML Model Comparison")
    print("=" * 60)

    print("\n[1/7] Loading data...")
    df, matchups = load_data()

    print("[2/7] Building game features...")
    game = build_game_level(df)
    teams = sorted(game["home_team"].unique().tolist())
    team_stats = build_team_stats(game)

    # Advanced record-level features (full dataset, for final predictions)
    rec_full = stack_records(df)  # Pre-compute once
    st_feats = build_special_teams(df, rec=rec_full)
    ld_feats = build_line_depth(df, rec=rec_full)
    gk_feats = build_goalie_features(df, rec=rec_full)
    hd_feats = build_high_danger(df, rec=rec_full)

    print("[3/7] Fitting Bradley-Terry...")
    bt = fit_bt(game, teams)

    # Build RAPM and new features on full data for final predictions
    rapm_feats = build_rapm_features(df, teams, rec=rec_full)
    te_feats = build_toi_entropy(df, rec=rec_full)
    mm_feats = build_matchup_matrix(df, rec=rec_full)
    stev_feats = build_st_expected_value(df, rec=rec_full)
    vol_feats = build_volatility_features(df, game, rec=rec_full)

    # Build advanced features dict
    adv_dict = _build_advanced_dict(
        special_teams=st_feats, line_depth=ld_feats,
        goalie_feats=gk_feats, high_danger=hd_feats,
        toi_entropy=te_feats, matchup_matrix=mm_feats,
        st_ev=stev_feats if len(stev_feats) > 0 else None,
        volatility=vol_feats, rapm=rapm_feats)

    # Residualize features on full data for final predictions
    resid_base = team_stats.copy()
    for adv_src_name, adv_cols in RESID_ADV_COLS.items():
        src = adv_dict.get(adv_src_name)
        if src is not None:
            for col in adv_cols:
                if col in src.columns:
                    resid_base[col] = src[col]
    all_resid_cols = RESID_STAT_COLS + [c for cols in RESID_ADV_COLS.values() for c in cols]
    valid_cols = [c for c in all_resid_cols if c in resid_base.columns]
    resid_stats = residualize_features(resid_base, bt, valid_cols)

    print("[4/7] Building ML feature matrix...")
    feat_df = build_ml_features(game, team_stats, bt, adv_dict, resid_stats)

    # Ensure all feature columns exist
    for col in FULL_FEATURES:
        if col not in feat_df.columns:
            feat_df[col] = 0.0

    print(f"  {len(feat_df)} games, {len(feat_df.columns)-2} total columns")

    print("[5/7] Running 5-fold CV model comparison...")
    results, enet_features, gb_features, fold_importance = run_model_comparison(
        game, teams, df_raw=df)

    X_enet = feat_df[enet_features].values
    X_gb = feat_df[gb_features].values
    y = feat_df["home_win"].values

    print(f"\n  {'Model':<25} {'Accuracy':>10} {'Log-Loss':>10} {'Brier':>10}")
    print(f"  {'─'*25} {'─'*10} {'─'*10} {'─'*10}")
    for name, r in results.items():
        print(f"  {name:<25} {r['accuracy']:>9.1%}  {r['log_loss']:>9.4f} {r['brier']:>9.4f}")

    print("\n[6/7] Computing feature importance...")
    enet_coefs, xgb_imp, enet_model, scaler = compute_feature_importance(
        X_enet, y, enet_features)

    print("  Elastic Net top 3:", ", ".join(f"{n}({v:.3f})" for n, v in enet_coefs.head(3).items()))
    if xgb_imp is not None:
        print("  XGBoost top 3:", ", ".join(f"{n}({v:.3f})" for n, v in xgb_imp.head(3).items()))

    print("[7/7] Generating outputs...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Select best model by Brier score for matchup predictions
    best_brier_model = min(
        {k: v for k, v in results.items() if k != "Baseline (fold home-rate)"}.items(),
        key=lambda x: x[1]["brier"]
    )
    print(f"  Best model (Brier): {best_brier_model[0]} ({best_brier_model[1]['brier']:.4f})")

    # ML matchup predictions using best calibrated model (Elastic Net retrained on all data)
    ml_matchups = predict_matchups_ml(matchups, team_stats, bt, enet_model, scaler,
                                      enet_features, adv_dict, resid_stats)
    ml_matchups.to_csv(os.path.join(OUTPUT_DIR, "phase1_ml_matchups.csv"), index=False)

    # Dashboard
    viz = create_ml_dashboard(results, y, enet_coefs, xgb_imp, enet_features, OUTPUT_DIR)

    # Diagnostics dashboard
    diag_viz = create_diagnostics_dashboard(X_enet, X_gb, y, enet_features, gb_features,
                                            fold_importance, OUTPUT_DIR)

    # Results summary
    summary = "# ML Model Comparison Results\n\n"
    summary += "## 5-Fold Cross-Validation Results\n\n"
    summary += "| Model | Accuracy | Log-Loss | Brier Score |\n"
    summary += "|-------|----------|----------|-------------|\n"
    for name, r in results.items():
        summary += f"| {name} | {r['accuracy']:.1%} | {r['log_loss']:.4f} | {r['brier']:.4f} |\n"

    summary += "\n## Feature Tiers\n\n"
    summary += f"- **Core** ({len(CORE_FEATURES)}): {', '.join(CORE_FEATURES)}\n"
    summary += f"- **Extended** ({len(EXTENDED_FEATURES)}): adds {', '.join(set(EXTENDED_FEATURES) - set(CORE_FEATURES))}\n"
    summary += f"- **Full** ({len(FULL_FEATURES)}): adds {', '.join(set(FULL_FEATURES) - set(EXTENDED_FEATURES))}\n"

    summary += "\n## Feature Importance (Elastic Net |Coefficients|)\n\n"
    summary += "| Feature | |Coefficient| |\n|---------|-------------|\n"
    for name, val in enet_coefs.items():
        summary += f"| {name} | {val:.4f} |\n"

    if xgb_imp is not None:
        summary += "\n## Feature Importance (XGBoost Gain)\n\n"
        summary += "| Feature | Importance |\n|---------|------------|\n"
        for name, val in xgb_imp.items():
            summary += f"| {name} | {val:.4f} |\n"

    # Interpretation
    best_model = max(results.items(), key=lambda x: x[1]["accuracy"])
    best_brier = min(results.items(), key=lambda x: x[1]["brier"])
    summary += f"\n## Key Findings\n\n"
    summary += f"- **Best accuracy**: {best_model[0]} ({best_model[1]['accuracy']:.1%})\n"
    summary += f"- **Best calibration (Brier)**: {best_brier[0]} ({best_brier[1]['brier']:.4f})\n"
    summary += f"- **Home-ice advantage**: Baseline win rate = {y.mean():.1%}\n"
    summary += f"- **Hockey is inherently unpredictable**: Even the best model achieves modest accuracy\n"
    summary += f"- **Recommendation**: Use {best_brier[0]} for probability predictions (best calibrated)\n"

    summary += "\n## ML Matchup Predictions vs Core Model\n\n"
    summary += "| Game | Home | Away | Core Model | Elastic Net ML |\n"
    summary += "|------|------|------|------------|----------------|\n"

    # Load core model predictions for comparison
    core_path = os.path.join(OUTPUT_DIR, "phase1_final_matchups.csv")
    if os.path.exists(core_path):
        core = pd.read_csv(core_path)
        for i, (_, m) in enumerate(ml_matchups.iterrows()):
            core_prob = core.iloc[i]["home_win_prob"] if i < len(core) else "N/A"
            summary += (f"| {int(m['game'])} | {m['home_team']} | {m['away_team']} | "
                        f"{core_prob} | {m['home_win_prob_ml']:.1%} |\n")

    with open(os.path.join(OUTPUT_DIR, "phase1_ml_results.md"), "w", encoding="utf-8") as f:
        f.write(summary)

    print(f"\n  Dashboard:  {viz}")
    print(f"  Diagnostics: {diag_viz}")
    print(f"  Matchups:   output/phase1_ml_matchups.csv")
    print(f"  Results:    output/phase1_ml_results.md")
    print("\n" + "=" * 60)
    print("ML comparison complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
