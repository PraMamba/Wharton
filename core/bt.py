"""Bradley-Terry model with OT downweighting — vectorized."""

import warnings

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.optimize import minimize as scipy_minimize


def fit_bt(game, teams, ot_weight=0.5, max_iter=300, tol=1e-8, regularization=0.0):
    """OT-aware Bradley-Terry via iterative Newton steps (vectorized).

    Parameters
    ----------
    game : DataFrame with columns home_team, away_team, home_win, went_ot
    teams : list of team names
    ot_weight : weight for OT games (default 0.5)
    max_iter : maximum Newton iterations
    tol : convergence tolerance on max |delta|
    regularization : L2 regularization strength (prevents ±inf for undefeated/winless teams)
    """
    team_idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    strength = np.zeros(n)

    # Pre-compute index arrays once (vectorized setup)
    home_idx = game["home_team"].map(team_idx).values
    away_idx = game["away_team"].map(team_idx).values
    y_arr = game["home_win"].values.astype(float)
    w_arr = np.where(game["went_ot"].values == 1, ot_weight, 1.0)

    for _ in range(max_iter):
        p = expit(strength[home_idx] - strength[away_idx])
        residual = w_arr * (y_arr - p)
        fisher = w_arr * p * (1 - p)

        grad = (np.bincount(home_idx, weights=residual, minlength=n)
                - np.bincount(away_idx, weights=residual, minlength=n))
        hess = -(np.bincount(home_idx, weights=fisher, minlength=n)
                 + np.bincount(away_idx, weights=fisher, minlength=n))

        # L2 regularization
        if regularization > 0:
            grad -= regularization * strength
            hess -= regularization

        hess = np.clip(hess, None, -1e-6)
        delta = -grad / hess
        delta -= delta.mean()
        strength += 0.5 * delta
        if np.max(np.abs(delta)) < tol:
            break
    else:
        warnings.warn(f"BT model did not converge after {max_iter} iterations "
                      f"(delta={np.max(np.abs(delta)):.2e})")

    strength -= strength.mean()
    return dict(zip(teams, strength))


def compute_sos(game, bt):
    """Strength of schedule: mean opponent BT strength (vectorized)."""
    bt_series = pd.Series(bt)
    home_opp_str = game["away_team"].map(bt_series)
    away_opp_str = game["home_team"].map(bt_series)
    teams_col = pd.concat([game["home_team"], game["away_team"]], ignore_index=True)
    opp_str = pd.concat([home_opp_str, away_opp_str], ignore_index=True)
    return pd.DataFrame({"team": teams_col, "os": opp_str}).groupby("team")["os"].mean()


def bootstrap_bt(game, teams, n_boot=200, random_state=42, **bt_kwargs):
    """Bootstrap BT strengths to quantify uncertainty.

    Returns
    -------
    bt_mean : dict of team → mean BT strength
    bt_std : dict of team → std of BT strength
    bt_ci : dict of team → (lo_2.5%, hi_97.5%)
    samples : np.ndarray of shape (n_boot, n_teams) — raw bootstrap samples
    """
    rng = np.random.default_rng(random_state)
    n_games = len(game)
    n_teams = len(teams)
    samples = np.zeros((n_boot, n_teams))

    for b in range(n_boot):
        idx = rng.choice(n_games, size=n_games, replace=True)
        boot_game = game.iloc[idx].reset_index(drop=True)
        bt = fit_bt(boot_game, teams, **bt_kwargs)
        samples[b] = [bt[t] for t in teams]

    bt_mean = dict(zip(teams, samples.mean(axis=0)))
    bt_std = dict(zip(teams, samples.std(axis=0)))
    lo = np.percentile(samples, 2.5, axis=0)
    hi = np.percentile(samples, 97.5, axis=0)
    bt_ci = {t: (lo[i], hi[i]) for i, t in enumerate(teams)}
    return bt_mean, bt_std, bt_ci, samples


def fit_attack_defense(game, teams, use_xg=True, regularization=0.01):
    """Attack-Defense Poisson MLE for goal/xG scoring rates.

    Parameters
    ----------
    game : DataFrame with home_team, away_team, home_goals, away_goals,
           home_xg, away_xg, toi columns
    teams : list of team names
    use_xg : if True use xG; otherwise use goals
    regularization : L2 penalty on attack/defense params

    Returns
    -------
    dict with 'attack', 'defense' (team→float), 'home_adv' (float)
    """
    team_idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)

    h_idx = game["home_team"].map(team_idx).values
    a_idx = game["away_team"].map(team_idx).values
    toi = game["toi"].values / 3600.0  # convert to hours

    if use_xg:
        y_h = game["home_xg"].values.astype(float)
        y_a = game["away_xg"].values.astype(float)
    else:
        y_h = game["home_goals"].values.astype(float)
        y_a = game["away_goals"].values.astype(float)

    # Parameters: A[0..n-1] (attack), D[0..n-1] (defense), H (home advantage)
    # A[0] is free (no sum-to-zero constraint in optimization; we center post-hoc)
    n_params = 2 * n + 1

    def neg_log_lik(params):
        A = params[:n]
        D = params[n:2*n]
        H = params[2*n]

        # Clip log-intensities to [-5, 5]
        log_lam_h = np.clip(H + A[h_idx] - D[a_idx], -5, 5)
        log_lam_a = np.clip(A[a_idx] - D[h_idx], -5, 5)

        lam_h = np.exp(log_lam_h) * toi
        lam_a = np.exp(log_lam_a) * toi

        # Poisson log-likelihood: y*log(lam) - lam - log(y!)
        # We drop the log(y!) constant
        nll = -np.sum(y_h * np.log(lam_h + 1e-10) - lam_h)
        nll -= np.sum(y_a * np.log(lam_a + 1e-10) - lam_a)

        # L2 regularization on A, D
        nll += regularization * (np.sum(A**2) + np.sum(D**2))
        return nll

    x0 = np.zeros(n_params)
    x0[2*n] = 0.05  # small home advantage initial guess

    res = scipy_minimize(neg_log_lik, x0, method="L-BFGS-B",
                         options={"maxiter": 500, "ftol": 1e-10})
    if not res.success:
        warnings.warn(f"AD model: {res.message}")

    A = res.x[:n]
    D = res.x[n:2*n]
    H = res.x[2*n]

    # Center attack and defense
    A -= A.mean()
    D -= D.mean()

    attack = dict(zip(teams, A))
    defense = dict(zip(teams, D))
    return {"attack": attack, "defense": defense, "home_adv": H}


def skellam_predict_ad(ad_model, home, away, toi=3600):
    """Predict P(home win) using fitted attack-defense model via Skellam distribution.

    Parameters
    ----------
    ad_model : dict from fit_attack_defense
    home, away : team names
    toi : game duration in seconds (default 3600)
    """
    from scipy.stats import skellam as skellam_dist

    A = ad_model["attack"]
    D = ad_model["defense"]
    H = ad_model["home_adv"]

    log_lam_h = np.clip(H + A.get(home, 0) - D.get(away, 0), -5, 5)
    log_lam_a = np.clip(A.get(away, 0) - D.get(home, 0), -5, 5)

    t = toi / 3600.0
    lam_h = np.exp(log_lam_h) * t
    lam_a = np.exp(log_lam_a) * t

    # P(home_goals > away_goals)
    p = 1 - skellam_dist.cdf(0, lam_h, lam_a)
    return float(np.clip(p, 0.01, 0.99))
