"""Model utilities: calibration, CV, rankings, predictions, and ECE.

Extracted from phase1_submit.py to reduce file size and improve reusability.
"""

import warnings
from collections import namedtuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit
from sklearn.model_selection import StratifiedKFold

from core.config import CFG
from core.bt import fit_bt, bootstrap_bt

CVResult = namedtuple('CVResult', [
    'acc', 'acc_std', 'll', 'brier',
    'base_acc', 'base_ll', 'base_brier',
    'oof_pred', 'oof_true', 'oof_uncertainties',
])


def _platt_neg_ll(params, diffs, y):
    """Platt scaling negative log-likelihood: σ(a + b·Δβ)."""
    a, b = params
    p = np.clip(expit(a + b * diffs), 1e-8, 1 - 1e-8)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()


def build_rankings(stats, bt, sos):
    r = stats.copy()
    r["bt_strength"] = r.index.map(bt)
    r["sos"] = r.index.map(sos)
    components = CFG["rankings"]["weights"]
    for col in components:
        m, s = r[col].mean(), r[col].std(ddof=1)
        if s == 0:
            warnings.warn(f"Zero variance in {col}, z-score will be 0")
        r[f"z_{col}"] = (r[col] - m) / s if s > 0 else 0
    r["composite"] = sum(w * r[f"z_{c}"] for c, w in components.items())
    r = r.sort_values("composite", ascending=False).reset_index()
    r["rank"] = range(1, len(r) + 1)
    return r


def _fit_calibration(diffs_tr, y_tr, diffs_te, method='platt'):
    """Fit calibration model on training BT diffs, predict on test diffs.

    Parameters
    ----------
    method : 'platt' (logistic) or 'isotonic'

    Returns
    -------
    p_te : array of predicted probabilities on test set
    cal_params : (a, b) for platt, or fitted isotonic model
    """
    if method == 'isotonic':
        from sklearn.isotonic import IsotonicRegression
        # Map diffs to initial probabilities for isotonic
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(diffs_tr, y_tr)
        p_te = iso.predict(diffs_te)
        return np.clip(p_te, 1e-8, 1 - 1e-8), iso
    else:
        # Platt scaling: σ(a + b·Δβ)
        res = minimize(lambda p: _platt_neg_ll(p, diffs_tr, y_tr), [0.2, 1.0], method="Nelder-Mead")
        a, b = res.x
        p_te = expit(a + b * diffs_te)
        return np.clip(p_te, 1e-8, 1 - 1e-8), (a, b)


def tune_bt_hyperparams(game, teams, n_outer=5, n_inner=3):
    """Nested CV grid search over BT hyperparameters.

    Outer loop: n_outer folds. Inner loop: for each (ot_weight, reg),
    run n_inner-fold inner CV on outer-train. Select by min mean inner Brier.

    Returns
    -------
    best_ot_weight, best_reg, best_calibration, cv_details_dict
    """
    cfg = CFG["bt_tuning"]
    ot_grid = cfg["ot_weight_grid"]
    reg_grid = cfg["regularization_grid"]

    y = game["home_win"].values
    outer_skf = StratifiedKFold(n_splits=n_outer, shuffle=True,
                                random_state=CFG["cv"]["random_state"])

    # Accumulate mean inner Brier for each hyperparameter combo
    combo_scores = {}
    for ot_w in ot_grid:
        for reg in reg_grid:
            combo_scores[(ot_w, reg)] = []

    for o_fold, (outer_train_idx, _outer_test_idx) in enumerate(outer_skf.split(game, y)):
        outer_train = game.iloc[outer_train_idx].reset_index(drop=True)

        # Inner folds (stratified)
        inner_skf = StratifiedKFold(n_splits=n_inner, shuffle=True,
                                    random_state=CFG["cv"]["random_state"] + o_fold)
        inner_y = outer_train["home_win"].values

        for ot_w in ot_grid:
            for reg in reg_grid:
                inner_briers = []
                for i_train_idx, i_test_idx in inner_skf.split(outer_train, inner_y):
                    i_train = outer_train.iloc[i_train_idx]
                    i_test = outer_train.iloc[i_test_idx]

                    bt = fit_bt(i_train, teams, ot_weight=ot_w, regularization=reg)
                    bt_series = pd.Series(bt)
                    diffs_tr = i_train["home_team"].map(bt_series).values - i_train["away_team"].map(bt_series).values
                    y_tr = i_train["home_win"].values
                    diffs_te = i_test["home_team"].map(bt_series).fillna(0).values - i_test["away_team"].map(bt_series).fillna(0).values
                    y_te = i_test["home_win"].values

                    p_te, _ = _fit_calibration(diffs_tr, y_tr, diffs_te, method='platt')
                    brier = np.mean((p_te - y_te) ** 2)
                    combo_scores[(ot_w, reg)].append(brier)

    # Average across all outer×inner evaluations
    mean_scores = {k: np.mean(v) for k, v in combo_scores.items()}
    best_combo = min(mean_scores, key=mean_scores.get)
    best_ot_weight, best_reg = best_combo

    cv_details = {
        "best_combo": best_combo,
        "best_brier": mean_scores[best_combo],
        "all_scores": mean_scores,
    }

    return best_ot_weight, best_reg, 'platt', cv_details


def _compute_pairwise_uncertainty(home_teams, away_teams, bt_std):
    """Vectorized pairwise uncertainty from per-team BT standard deviations."""
    bt_std_s = pd.Series(bt_std)
    h_var = pd.Series(home_teams).map(bt_std_s).fillna(0).values ** 2
    a_var = pd.Series(away_teams).map(bt_std_s).fillna(0).values ** 2
    return np.sqrt(h_var + a_var)


def tune_uncertainty_shrink_k(oof_pred, oof_true, bt_std, game):
    """Tune uncertainty shrink parameter k on OOF predictions.

    p' = 0.5 + (p - 0.5) * exp(-k * sqrt(Var(β_h) + Var(β_a)))

    Parameters
    ----------
    oof_pred : array of OOF predicted probabilities
    oof_true : array of true labels
    bt_std : dict of team → std of BT strength, OR np.ndarray of
             per-game uncertainties (from fold-level bootstrap)
    game : DataFrame with home_team, away_team (ignored if bt_std is array)

    Returns
    -------
    best_k : optimal shrink parameter
    """
    if isinstance(bt_std, np.ndarray):
        uncertainties = bt_std
    else:
        uncertainties = _compute_pairwise_uncertainty(
            game["home_team"], game["away_team"], bt_std)

    def _shrink_brier(k):
        shrunk = 0.5 + (oof_pred - 0.5) * np.exp(-k * uncertainties)
        return np.mean((shrunk - oof_true) ** 2)

    res = minimize_scalar(_shrink_brier, bounds=(0, 5), method="bounded")
    return res.x

def fit_logistic(game, bt):
    """BT-only logistic (interpretable, stable, easy to audit)."""
    bt_series = pd.Series(bt)
    diffs = game["home_team"].map(bt_series).values - game["away_team"].map(bt_series).values
    y = game["home_win"].values

    res = minimize(lambda p: _platt_neg_ll(p, diffs, y), [0.2, 1.0], method="Nelder-Mead")
    if not res.success:
        warnings.warn(f"Logistic fit: {res.message}")
    a, b = res.x

    p_train = expit(a + b * diffs)
    acc = ((p_train >= 0.5).astype(int) == y).mean()
    ll = _platt_neg_ll(res.x, diffs, y)
    brier = np.mean((p_train - y) ** 2)
    return a, b, acc, ll, brier


def cross_validate_bt(game, teams, n_folds=None, bt_kwargs=None, n_boot=0):
    """5-fold CV on BT-only logistic.

    When n_boot > 0, also computes per-fold bootstrap uncertainties to
    avoid leaking full-dataset bt_std into the shrinkage tuner.
    """
    if n_folds is None:
        n_folds = CFG["cv"]["n_folds"]
    if bt_kwargs is None:
        bt_kwargs = {}
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True,
                          random_state=CFG["cv"]["random_state"])
    accs, lls, briers = [], [], []
    base_accs, base_lls, base_briers = [], [], []
    oof_pred = np.full(len(game), np.nan, dtype=float)
    oof_uncertainties = np.full(len(game), np.nan, dtype=float) if n_boot > 0 else None
    oof_true = game["home_win"].values.astype(int, copy=True)
    for fold, (train_idx, test_idx) in enumerate(skf.split(game, oof_true)):
        train = game.iloc[train_idx]
        test = game.iloc[test_idx]

        # Baseline: predict with training home-win rate (leakage-safe).
        y_tr_base = train["home_win"].values
        y_te_base = test["home_win"].values
        p_base = np.full(len(y_te_base), y_tr_base.mean(), dtype=float)
        p_base = np.clip(p_base, 1e-8, 1 - 1e-8)
        base_accs.append(((p_base >= 0.5).astype(int) == y_te_base).mean())
        base_lls.append(
            -(
                y_te_base * np.log(p_base)
                + (1 - y_te_base) * np.log(1 - p_base)
            ).mean()
        )
        base_briers.append(np.mean((p_base - y_te_base) ** 2))

        bt = fit_bt(train, teams, **bt_kwargs)

        # Per-fold bootstrap for leakage-free uncertainty
        if n_boot > 0:
            _, fold_bt_std, _, _ = bootstrap_bt(
                train, teams, n_boot=n_boot, **bt_kwargs)

        bt_series = pd.Series(bt)
        diffs_tr = train["home_team"].map(bt_series).values - train["away_team"].map(bt_series).values
        y_tr = train["home_win"].values

        res = minimize(lambda p: _platt_neg_ll(p, diffs_tr, y_tr), [0.2, 1.0], method="Nelder-Mead")
        a, b = res.x

        diffs_te = test["home_team"].map(bt_series).fillna(0).values - test["away_team"].map(bt_series).fillna(0).values
        y_te = test["home_win"].values
        p_te = expit(a + b * diffs_te)
        oof_pred[test_idx] = p_te
        if n_boot > 0:
            oof_uncertainties[test_idx] = _compute_pairwise_uncertainty(
                test["home_team"], test["away_team"], fold_bt_std)
        accs.append(((p_te >= 0.5).astype(int) == y_te).mean())
        lls.append(-(y_te * np.log(np.clip(p_te, 1e-8, 1-1e-8)) + (1-y_te) * np.log(np.clip(1-p_te, 1e-8, 1-1e-8))).mean())
        briers.append(np.mean((p_te - y_te) ** 2))
    if np.isnan(oof_pred).any():
        # Should not happen, but keep pipeline resilient.
        fill = float(oof_true.mean())
        oof_pred = np.where(np.isnan(oof_pred), fill, oof_pred)
    return CVResult(
        acc=np.mean(accs),
        acc_std=np.std(accs),
        ll=np.mean(lls),
        brier=np.mean(briers),
        base_acc=np.mean(base_accs),
        base_ll=np.mean(base_lls),
        base_brier=np.mean(base_briers),
        oof_pred=oof_pred,
        oof_true=oof_true,
        oof_uncertainties=oof_uncertainties,
    )


def _compute_boot_probs(bt_samples, teams, matchup_df, a, b):
    """Compute bootstrap probability matrix from BT samples."""
    team_idx = {t: i for i, t in enumerate(teams)}
    home_idx = np.array([team_idx[t] for t in matchup_df["home_team"]])
    away_idx = np.array([team_idx[t] for t in matchup_df["away_team"]])
    boot_probs = np.empty((bt_samples.shape[0], len(matchup_df)))
    for s in range(bt_samples.shape[0]):
        strengths = bt_samples[s]
        boot_probs[s] = expit(a + b * (strengths[home_idx] - strengths[away_idx]))
    return boot_probs


def predict_matchups(matchups, bt, a, b, bt_samples=None, teams=None,
                     bt_std=None, uncertainty_shrink_k=0.0):
    r = matchups.copy()
    r["str_home"] = r["home_team"].map(bt)
    r["str_away"] = r["away_team"].map(bt)
    missing = r[r["str_home"].isna() | r["str_away"].isna()]
    if len(missing) > 0:
        warnings.warn(f"Unknown teams in matchups: {missing[['home_team', 'away_team']].values.tolist()}")
    r["str_diff"] = r["str_home"] - r["str_away"]
    prob_lo, prob_hi = CFG["prediction"]["prob_clip"]
    bootstrap_integrate = CFG["prediction"].get("bootstrap_integrate", False)

    # Compute bootstrap probabilities once (if samples available)
    boot_probs = None
    if bt_samples is not None and teams is not None:
        boot_probs = _compute_boot_probs(bt_samples, teams, r, a, b)
        r["prob_ci_lo"] = np.percentile(boot_probs, 2.5, axis=0).round(4)
        r["prob_ci_hi"] = np.percentile(boot_probs, 97.5, axis=0).round(4)

    # Point estimate
    if boot_probs is not None and bootstrap_integrate:
        p = boot_probs.mean(axis=0)  # E[σ(·)] instead of σ(E[·])
    else:
        p = expit(a + b * r["str_diff"].values)

    # Uncertainty shrink: only when NOT using bootstrap integration
    # (bootstrap integration already marginalizes over parameter uncertainty)
    if (not bootstrap_integrate
            and uncertainty_shrink_k > 0
            and bt_std is not None):
        uncertainties = _compute_pairwise_uncertainty(
            r["home_team"], r["away_team"], bt_std)
        p = 0.5 + (p - 0.5) * np.exp(-uncertainty_shrink_k * uncertainties)

    p = np.clip(p, prob_lo, prob_hi)
    r["home_win_prob"] = p.round(4)
    r["predicted_winner"] = np.where(r["home_win_prob"] >= 0.5, r["home_team"], r["away_team"])

    cols = ["game", "game_id", "home_team", "away_team", "home_win_prob", "predicted_winner"]
    if "prob_ci_lo" in r.columns:
        cols += ["prob_ci_lo", "prob_ci_hi"]
    return r[cols]


def compute_ece(y_true, y_pred, n_bins=10):
    """Expected Calibration Error."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (y_pred >= bins[i]) & (y_pred <= bins[i + 1])
        else:
            mask = (y_pred >= bins[i]) & (y_pred < bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_pred[mask].mean()
        ece += np.abs(bin_acc - bin_conf) * mask.sum()
    return ece / len(y_true)
