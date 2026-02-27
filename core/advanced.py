"""Advanced record-level feature engineering.

Functions that operate on raw line-level data (not game-level aggregates)
to produce team-indexed DataFrames for special teams, line depth, goalie
analysis, high-danger shot frequency, residualization, and RAPM.
"""

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from sklearn.linear_model import Ridge

from core.config import CFG

# ── Even-strength line/pairing identifiers ──────────────────────────
_EV_LINES = {"first_off", "second_off"}
_EV_PAIRS = {"first_def", "second_def"}


def stack_records(df):
    """Symmetric home/away stacking of raw record-level data.

    Returns a unified DataFrame with columns:
        team, opp, off_line, def_pairing, opp_off_line, opp_def_pairing,
        goalie, opp_goalie, xg_f, xg_a, goals_f, goals_a, shots_f, shots_a,
        max_xg_f, toi
    """
    home = df[["home_team", "away_team",
               "home_off_line", "home_def_pairing",
               "away_off_line", "away_def_pairing",
               "home_goalie", "away_goalie",
               "home_xg", "away_xg",
               "home_goals", "away_goals",
               "home_shots", "away_shots",
               "home_max_xg", "toi"]].copy()
    home.columns = ["team", "opp",
                    "off_line", "def_pairing",
                    "opp_off_line", "opp_def_pairing",
                    "goalie", "opp_goalie",
                    "xg_f", "xg_a",
                    "goals_f", "goals_a",
                    "shots_f", "shots_a",
                    "max_xg_f", "toi"]

    away = df[["away_team", "home_team",
               "away_off_line", "away_def_pairing",
               "home_off_line", "home_def_pairing",
               "away_goalie", "home_goalie",
               "away_xg", "home_xg",
               "away_goals", "home_goals",
               "away_shots", "home_shots",
               "away_max_xg", "toi"]].copy()
    away.columns = ["team", "opp",
                    "off_line", "def_pairing",
                    "opp_off_line", "opp_def_pairing",
                    "goalie", "opp_goalie",
                    "xg_f", "xg_a",
                    "goals_f", "goals_a",
                    "shots_f", "shots_a",
                    "max_xg_f", "toi"]

    return pd.concat([home, away], ignore_index=True)


# ── Special Teams ───────────────────────────────────────────────────

def build_special_teams(df, rec=None):
    """Compute PP, PK, and EV per-60 rates from record-level data.

    Returns DataFrame indexed by team with columns:
        pp_xgf60, pp_gf60, pp_toi_share,
        pk_xga60, pk_ga60, pk_toi_share,
        ev_xgd60, ev_xgf60, ev_xga60
    """
    if rec is None:
        rec = stack_records(df)
    team_toi = rec.groupby("team")["toi"].sum()

    # Power play: records where team is on PP (off_line == PP_up)
    pp = rec[rec["off_line"] == "PP_up"].groupby("team").agg(
        pp_xgf=("xg_f", "sum"), pp_gf=("goals_f", "sum"), pp_toi=("toi", "sum"))
    pp["pp_xgf60"] = np.where(pp["pp_toi"] > 0, pp["pp_xgf"] / pp["pp_toi"] * 3600, 0)
    pp["pp_gf60"] = np.where(pp["pp_toi"] > 0, pp["pp_gf"] / pp["pp_toi"] * 3600, 0)
    pp["pp_toi_share"] = np.where(team_toi > 0, pp["pp_toi"] / team_toi, 0)

    # Penalty kill: records where team is on PK (off_line == PP_kill_dwn)
    pk = rec[rec["off_line"] == "PP_kill_dwn"].groupby("team").agg(
        pk_xga=("xg_a", "sum"), pk_ga=("goals_a", "sum"), pk_toi=("toi", "sum"))
    pk["pk_xga60"] = np.where(pk["pk_toi"] > 0, pk["pk_xga"] / pk["pk_toi"] * 3600, 0)
    pk["pk_ga60"] = np.where(pk["pk_toi"] > 0, pk["pk_ga"] / pk["pk_toi"] * 3600, 0)
    pk["pk_toi_share"] = np.where(team_toi > 0, pk["pk_toi"] / team_toi, 0)

    # Even strength: both teams using EV lines and EV pairings
    ev_mask = (rec["off_line"].isin(_EV_LINES) &
               rec["opp_off_line"].isin(_EV_LINES))
    ev = rec[ev_mask].groupby("team").agg(
        ev_xgf=("xg_f", "sum"), ev_xga=("xg_a", "sum"), ev_toi=("toi", "sum"))
    ev["ev_xgf60"] = np.where(ev["ev_toi"] > 0, ev["ev_xgf"] / ev["ev_toi"] * 3600, 0)
    ev["ev_xga60"] = np.where(ev["ev_toi"] > 0, ev["ev_xga"] / ev["ev_toi"] * 3600, 0)
    ev["ev_xgd60"] = ev["ev_xgf60"] - ev["ev_xga60"]

    result = pd.DataFrame(index=team_toi.index)
    for sub, cols in [(pp, ["pp_xgf60", "pp_gf60", "pp_toi_share"]),
                      (pk, ["pk_xga60", "pk_ga60", "pk_toi_share"]),
                      (ev, ["ev_xgd60", "ev_xgf60", "ev_xga60"])]:
        result = result.join(sub[cols], how="left")
    return result.fillna(0)


# ── Line Depth ──────────────────────────────────────────────────────

def build_line_depth(df, rec=None):
    """Compute line depth metrics from even-strength records.

    Returns DataFrame indexed by team with columns:
        topline_toi_share, depth_gap, line_xgd60_std,
        toppair_toi_share, pair_xga60_std
    """
    if rec is None:
        rec = stack_records(df)
    ev = rec[rec["off_line"].isin(_EV_LINES) &
             rec["def_pairing"].isin(_EV_PAIRS)].copy()

    # Per-line stats
    line_agg = ev.groupby(["team", "off_line"]).agg(
        xgf=("xg_f", "sum"), xga=("xg_a", "sum"), toi=("toi", "sum"))
    line_agg["xgd60"] = np.where(
        line_agg["toi"] > 0,
        (line_agg["xgf"] - line_agg["xga"]) / line_agg["toi"] * 3600, 0)

    # Per-team total EV TOI
    team_ev_toi = ev.groupby("team")["toi"].sum()

    result = pd.DataFrame(index=team_ev_toi.index)

    # topline_toi_share: TOI share of the most-used offensive line
    line_toi = line_agg["toi"].unstack("off_line", fill_value=0)
    total_line_toi = line_toi.sum(axis=1)
    result["topline_toi_share"] = np.where(
        total_line_toi > 0, line_toi.max(axis=1) / total_line_toi, 0.5)

    # depth_gap: xGD60(first_off) - xGD60(second_off)
    xgd60_unstacked = line_agg["xgd60"].unstack("off_line", fill_value=0)
    first_xgd = xgd60_unstacked.get("first_off", pd.Series(0, index=result.index))
    second_xgd = xgd60_unstacked.get("second_off", pd.Series(0, index=result.index))
    result["depth_gap"] = first_xgd - second_xgd

    # line_xgd60_std: std of per-line xGD60 (balance measure)
    result["line_xgd60_std"] = xgd60_unstacked.std(axis=1, ddof=0).fillna(0)

    # Per-pairing stats
    pair_agg = ev.groupby(["team", "def_pairing"]).agg(
        xga=("xg_a", "sum"), toi=("toi", "sum"))
    pair_agg["xga60"] = np.where(
        pair_agg["toi"] > 0, pair_agg["xga"] / pair_agg["toi"] * 3600, 0)

    pair_toi = pair_agg["toi"].unstack("def_pairing", fill_value=0)
    total_pair_toi = pair_toi.sum(axis=1)
    result["toppair_toi_share"] = np.where(
        total_pair_toi > 0, pair_toi.max(axis=1) / total_pair_toi, 0.5)

    xga60_unstacked = pair_agg["xga60"].unstack("def_pairing", fill_value=0)
    result["pair_xga60_std"] = xga60_unstacked.std(axis=1, ddof=0).fillna(0)

    return result.fillna(0)


# ── Goalie Features ─────────────────────────────────────────────────

def build_goalie_features(df, rec=None):
    """Compute goalie-level features aggregated to team level.

    Returns DataFrame indexed by team with columns:
        goalie_gsax60, starter_toi_share
    """
    if rec is None:
        rec = stack_records(df)
    # Exclude empty net records
    rec = rec[rec["goalie"] != "empty_net"].copy()

    K = CFG["shrinkage"]["K"]

    # Per (team, goalie) aggregation
    gk = rec.groupby(["team", "goalie"]).agg(
        xga=("xg_a", "sum"), ga=("goals_a", "sum"),
        sa=("shots_a", "sum"), toi=("toi", "sum"))
    gk["gsax60"] = np.where(gk["toi"] > 0, (gk["xga"] - gk["ga"]) / gk["toi"] * 3600, 0)

    # League mean GSAx/60 (across all goalies weighted by TOI)
    total_toi = gk["toi"].sum()
    if total_toi > 0:
        league_gsax60 = ((gk["xga"].sum() - gk["ga"].sum()) / total_toi) * 3600
    else:
        league_gsax60 = 0.0

    # Shrinkage
    gk["gsax60_shrunk"] = (gk["gsax60"] * gk["toi"] + league_gsax60 * K) / (gk["toi"] + K)

    # Identify starter (max TOI goalie per team)
    starter_idx = gk.groupby("team")["toi"].idxmax()
    starter = gk.loc[starter_idx].copy()
    starter.index = starter.index.get_level_values("team")

    # Team-level goalie TOI (excluding empty net)
    team_gk_toi = gk.groupby("team")["toi"].sum()

    result = pd.DataFrame(index=team_gk_toi.index)
    result["goalie_gsax60"] = starter["gsax60_shrunk"]
    result["starter_toi_share"] = np.where(
        team_gk_toi > 0, starter["toi"] / team_gk_toi, 1.0)

    return result.fillna(0)


# ── High Danger ─────────────────────────────────────────────────────

def build_high_danger(df, threshold=None, rec=None):
    """Compute high-danger shot creation frequency per 60 min.

    Returns DataFrame indexed by team with columns:
        high_danger_freq60, hd_xg_sum60, p95_max_xg
    """
    if threshold is None:
        threshold = CFG["high_danger"]["threshold"]

    if rec is None:
        rec = stack_records(df)
    rec = rec.copy()
    rec["is_hd"] = (rec["max_xg_f"] >= threshold).astype(int)

    team_agg = rec.groupby("team").agg(
        hd_count=("is_hd", "sum"), toi=("toi", "sum"))

    team_agg["high_danger_freq60"] = np.where(
        team_agg["toi"] > 0, team_agg["hd_count"] / team_agg["toi"] * 3600, 0)

    # Enhanced: sum of xG from high-danger chances per 60 (vectorized)
    hd_rec = rec[rec["is_hd"] == 1]
    hd_xg_sum = hd_rec.groupby("team")["max_xg_f"].sum()
    team_toi = team_agg["toi"]
    team_agg["hd_xg_sum60"] = np.where(
        team_toi > 0, hd_xg_sum.reindex(team_toi.index, fill_value=0) / team_toi * 3600, 0)

    # 95th percentile of max_xg_f per team (vectorized)
    team_agg["p95_max_xg"] = rec.groupby("team")["max_xg_f"].quantile(0.95)

    return team_agg[["high_danger_freq60", "hd_xg_sum60", "p95_max_xg"]]


# ── Residualization ────────────────────────────────────────────────

def residualize_features(team_stats, bt_strength, feature_cols, alpha=1.0):
    """Regress features on BT strength and return residuals.

    For each feature f, fits Ridge(alpha) on BT strength and returns
    the residual (f - predicted). This removes the BT-correlated component.

    Must be called per fold with training-fold BT (leak-safe).

    Parameters
    ----------
    team_stats : DataFrame indexed by team
    bt_strength : dict team → BT strength
    feature_cols : list of column names to residualize
    alpha : Ridge regularization (default 1.0)

    Returns
    -------
    DataFrame with columns renamed to {col}_resid, same index as team_stats
    """
    teams_in = team_stats.index
    bt_arr = np.array([bt_strength.get(t, 0.0) for t in teams_in]).reshape(-1, 1)

    result = pd.DataFrame(index=teams_in)
    for col in feature_cols:
        if col not in team_stats.columns:
            result[f"{col}_resid"] = 0.0
            continue
        f_values = team_stats[col].values.astype(float)
        ridge = Ridge(alpha=alpha, fit_intercept=True)
        ridge.fit(bt_arr, f_values)
        predicted = ridge.predict(bt_arr)
        result[f"{col}_resid"] = f_values - predicted

    return result


# ── RAPM (Regularized Adjusted Plus-Minus) ─────────────────────────

def build_rapm_features(df, teams, regularization=None, min_toi_unit=None, rec=None):
    """Fit Ridge regression on segment-level records to estimate unit-level effects.

    Builds a sparse design matrix from offensive line, defensive pairing,
    and (optionally) goalie identifiers, then fits Ridge to estimate
    per-unit xG impact controlling for matchup quality.

    Parameters
    ----------
    df : raw record-level DataFrame
    teams : list of team names
    regularization : Ridge alpha (default from config)
    min_toi_unit : minimum TOI for a unit to be included (default from config)

    Returns
    -------
    DataFrame indexed by team with columns:
        rapm_attack, rapm_defense, rapm_topline_effect
    """
    if regularization is None:
        regularization = CFG["rapm"]["regularization"]
    if min_toi_unit is None:
        min_toi_unit = CFG["rapm"].get("min_toi_unit", 200)
    include_goalies = CFG["rapm"]["include_goalies"]

    if rec is None:
        rec = stack_records(df)
    n_rows = len(rec)

    # Response: xG differential per 60
    toi_arr = rec["toi"].values.astype(float)
    xg_diff = (rec["xg_f"].values - rec["xg_a"].values).astype(float)

    # Avoid div-by-zero
    safe_toi = np.where(toi_arr > 0, toi_arr, 1.0)
    y = xg_diff / safe_toi * 3600

    # Filter out low-TOI units
    off_unit_toi = rec.groupby(["team", "off_line"])["toi"].sum()
    valid_off_units = set(off_unit_toi[off_unit_toi >= min_toi_unit].index)

    def_unit_toi = rec.groupby(["team", "def_pairing"])["toi"].sum()
    valid_def_units = set(def_unit_toi[def_unit_toi >= min_toi_unit].index)

    # Build column indices for sparse matrix
    # Offensive line units: (team, off_line) — filtered
    off_units = [u for u in set(zip(rec["team"], rec["off_line"])) if u in valid_off_units]
    off_idx = {u: i for i, u in enumerate(off_units)}
    n_off = len(off_units)

    # Defensive pairing units: (team, def_pairing) — filtered
    def_units = [u for u in set(zip(rec["team"], rec["def_pairing"])) if u in valid_def_units]
    def_idx = {u: i for i, u in enumerate(def_units)}
    n_def = len(def_units)

    # Goalie units (optional) — filtered
    if include_goalies:
        gk_unit_toi = rec.groupby(["team", "goalie"])["toi"].sum()
        valid_gk_units = set(gk_unit_toi[gk_unit_toi >= min_toi_unit].index)
        goalie_units = [u for u in set(zip(rec["team"], rec["goalie"])) if u in valid_gk_units]
        goalie_idx = {u: i for i, u in enumerate(goalie_units)}
        n_gk = len(goalie_units)
    else:
        n_gk = 0

    n_cols = n_off + n_def + n_gk + 1  # +1 for home advantage (unused in stacked but keep for consistency)

    # Build COO sparse matrix (vectorized)
    rows_list = []
    cols_list = []
    data_list = []

    team_arr = rec["team"].values
    opp_arr = rec["opp"].values
    off_line_arr = rec["off_line"].values
    def_pair_arr = rec["def_pairing"].values
    opp_off_arr = rec["opp_off_line"].values
    opp_def_arr = rec["opp_def_pairing"].values
    row_range = np.arange(n_rows)

    # Team offensive lines (+1)
    team_off_key = list(zip(team_arr, off_line_arr))
    team_off_col = pd.Series(team_off_key).map(off_idx)
    valid = team_off_col.notna()
    if valid.any():
        rows_list.append(row_range[valid.values])
        cols_list.append(team_off_col[valid].astype(int).values)
        data_list.append(np.ones(valid.sum()))

    # Opponent offensive lines (-1)
    opp_off_key = list(zip(opp_arr, opp_off_arr))
    opp_off_col = pd.Series(opp_off_key).map(off_idx)
    valid = opp_off_col.notna()
    if valid.any():
        rows_list.append(row_range[valid.values])
        cols_list.append(opp_off_col[valid].astype(int).values)
        data_list.append(-np.ones(valid.sum()))

    # Team defensive pairings (+1, offset n_off)
    team_def_key = list(zip(team_arr, def_pair_arr))
    team_def_col = pd.Series(team_def_key).map(def_idx)
    valid = team_def_col.notna()
    if valid.any():
        rows_list.append(row_range[valid.values])
        cols_list.append(team_def_col[valid].astype(int).values + n_off)
        data_list.append(np.ones(valid.sum()))

    # Opponent defensive pairings (-1, offset n_off)
    opp_def_key = list(zip(opp_arr, opp_def_arr))
    opp_def_col = pd.Series(opp_def_key).map(def_idx)
    valid = opp_def_col.notna()
    if valid.any():
        rows_list.append(row_range[valid.values])
        cols_list.append(opp_def_col[valid].astype(int).values + n_off)
        data_list.append(-np.ones(valid.sum()))

    # Goalie columns (optional)
    if include_goalies:
        goalie_arr = rec["goalie"].values
        opp_goalie_arr = rec["opp_goalie"].values

        team_gk_key = list(zip(team_arr, goalie_arr))
        team_gk_col = pd.Series(team_gk_key).map(goalie_idx)
        valid = team_gk_col.notna()
        if valid.any():
            rows_list.append(row_range[valid.values])
            cols_list.append(team_gk_col[valid].astype(int).values + n_off + n_def)
            data_list.append(np.ones(valid.sum()))

        opp_gk_key = list(zip(opp_arr, opp_goalie_arr))
        opp_gk_col = pd.Series(opp_gk_key).map(goalie_idx)
        valid = opp_gk_col.notna()
        if valid.any():
            rows_list.append(row_range[valid.values])
            cols_list.append(opp_gk_col[valid].astype(int).values + n_off + n_def)
            data_list.append(-np.ones(valid.sum()))

    all_rows = np.concatenate(rows_list) if rows_list else np.array([], dtype=int)
    all_cols = np.concatenate(cols_list) if cols_list else np.array([], dtype=int)
    all_data = np.concatenate(data_list) if data_list else np.array([], dtype=float)

    X = coo_matrix(
        (all_data, (all_rows, all_cols)),
        shape=(n_rows, n_cols)
    ).tocsr()

    # Fit Ridge with TOI as sample weight
    ridge = Ridge(alpha=regularization, fit_intercept=False)
    ridge.fit(X, y, sample_weight=toi_arr)
    coefs = ridge.coef_

    # Aggregate to team level (vectorized)
    # Attack: TOI-weighted mean of team's offensive line coefficients
    off_coef_df = pd.DataFrame([
        {"team": k[0], "off_line": k[1], "coef": coefs[v]}
        for k, v in off_idx.items()
    ])
    off_toi_df = rec.groupby(["team", "off_line"])["toi"].sum().reset_index()
    off_toi_df.columns = ["team", "off_line", "unit_toi"]

    rapm_attack = {}
    rapm_topline = {}
    if len(off_coef_df) > 0:
        off_merged = off_coef_df.merge(off_toi_df, on=["team", "off_line"], how="left")
        off_merged["unit_toi"] = off_merged["unit_toi"].fillna(0)
        for t, grp in off_merged.groupby("team"):
            total = grp["unit_toi"].sum()
            if total > 0:
                rapm_attack[t] = (grp["coef"] * grp["unit_toi"]).sum() / total
            else:
                rapm_attack[t] = grp["coef"].mean()
            # Topline effect
            top = grp[grp["off_line"] == "first_off"]
            rapm_topline[t] = float(top["coef"].iloc[0]) if len(top) > 0 else 0.0

    # Defense: TOI-weighted mean of team's defensive pairing coefficients
    def_coef_df = pd.DataFrame([
        {"team": k[0], "def_pairing": k[1], "coef": coefs[n_off + v]}
        for k, v in def_idx.items()
    ])
    def_toi_df = rec.groupby(["team", "def_pairing"])["toi"].sum().reset_index()
    def_toi_df.columns = ["team", "def_pairing", "unit_toi"]

    rapm_defense = {}
    if len(def_coef_df) > 0:
        def_merged = def_coef_df.merge(def_toi_df, on=["team", "def_pairing"], how="left")
        def_merged["unit_toi"] = def_merged["unit_toi"].fillna(0)
        for t, grp in def_merged.groupby("team"):
            total = grp["unit_toi"].sum()
            if total > 0:
                rapm_defense[t] = (grp["coef"] * grp["unit_toi"]).sum() / total
            else:
                rapm_defense[t] = grp["coef"].mean()

    # Fill missing teams with 0
    for t in teams:
        rapm_attack.setdefault(t, 0.0)
        rapm_defense.setdefault(t, 0.0)
        rapm_topline.setdefault(t, 0.0)

    result = pd.DataFrame({
        "rapm_attack": rapm_attack,
        "rapm_defense": rapm_defense,
        "rapm_topline_effect": rapm_topline,
    })
    result.index.name = "team"
    return result


# ── TOI Entropy + Deployment Matrix ───────────────────────────────

def build_toi_entropy(df, rec=None):
    """Compute Shannon entropy of TOI distributions and deployment matrix features.

    Returns DataFrame indexed by team with columns:
        off_line_entropy, def_pairing_entropy, goalie_entropy,
        top_vs_top_share, top_shelter_share
    """
    if rec is None:
        rec = stack_records(df)

    def _shannon(shares):
        s = shares[shares > 0]
        return -np.sum(s * np.log2(s))

    result = pd.DataFrame(index=rec.groupby("team")["toi"].sum().index)

    # Offensive line entropy
    off_toi = rec.groupby(["team", "off_line"])["toi"].sum().unstack(fill_value=0)
    off_total = off_toi.sum(axis=1)
    off_shares = off_toi.div(off_total, axis=0).fillna(0)
    result["off_line_entropy"] = off_shares.apply(_shannon, axis=1)

    # Defensive pairing entropy
    def_toi = rec.groupby(["team", "def_pairing"])["toi"].sum().unstack(fill_value=0)
    def_total = def_toi.sum(axis=1)
    def_shares = def_toi.div(def_total, axis=0).fillna(0)
    result["def_pairing_entropy"] = def_shares.apply(_shannon, axis=1)

    # Goalie entropy (exclude empty_net)
    gk_rec = rec[rec["goalie"] != "empty_net"]
    gk_toi = gk_rec.groupby(["team", "goalie"])["toi"].sum().unstack(fill_value=0)
    gk_total = gk_toi.sum(axis=1)
    gk_shares = gk_toi.div(gk_total, axis=0).fillna(0)
    result["goalie_entropy"] = gk_shares.apply(_shannon, axis=1)

    # Deployment matrix: top_vs_top_share and top_shelter_share
    ev = rec[rec["off_line"].isin(_EV_LINES) & rec["opp_def_pairing"].isin(_EV_PAIRS)]
    ev_team_toi = ev.groupby("team")["toi"].sum()

    # top_vs_top: first_off facing opp first_def
    tvt = ev[(ev["off_line"] == "first_off") & (ev["opp_def_pairing"] == "first_def")]
    tvt_toi = tvt.groupby("team")["toi"].sum()
    result["top_vs_top_share"] = (tvt_toi / ev_team_toi).fillna(0)

    # top_shelter: first_off facing opp second_def
    ts = ev[(ev["off_line"] == "first_off") & (ev["opp_def_pairing"] == "second_def")]
    ts_toi = ts.groupby("team")["toi"].sum()
    result["top_shelter_share"] = (ts_toi / ev_team_toi).fillna(0)

    return result.fillna(0)


# ── Matchup Matrix Features ───────────────────────────────────────

def build_matchup_matrix(df, min_toi=None, rec=None):
    """Compute matchup exploitation and shutdown features with shrinkage fill.

    Returns DataFrame indexed by team with columns:
        exploit_ability, shutdown_resilience, depth_insurance
    """
    if min_toi is None:
        min_toi = CFG.get("matchup_matrix", {}).get("min_toi", 300)

    if rec is None:
        rec = stack_records(df)
    ev = rec[rec["off_line"].isin(_EV_LINES) & rec["opp_def_pairing"].isin(_EV_PAIRS)].copy()

    # League mean xGF/60 for shrinkage
    total_xg = ev["xg_f"].sum()
    total_toi = ev["toi"].sum()
    league_mean_xg60 = (total_xg / total_toi * 3600) if total_toi > 0 else 0.0
    K = min_toi / 2  # shrinkage strength

    # Per (team, off_line, opp_def_pairing) cell
    cell = ev.groupby(["team", "off_line", "opp_def_pairing"]).agg(
        xgf=("xg_f", "sum"), toi=("toi", "sum"))

    # Shrinkage fill for sparse cells
    cell["xgf60"] = (cell["xgf"] + league_mean_xg60 * K / 3600) / (cell["toi"] + K) * 3600
    cell.loc[cell["toi"] <= 0, "xgf60"] = league_mean_xg60

    # exploit_ability: max xGF/60 across matchup cells per team
    exploit = cell.groupby("team")["xgf60"].max()

    # Per (team, def_pairing, opp_off_line) for shutdown
    league_mean_xga60 = league_mean_xg60  # symmetric
    cell_def = ev.groupby(["team", "def_pairing", "opp_off_line"]).agg(
        xga=("xg_a", "sum"), toi=("toi", "sum"))
    cell_def["xga60"] = (cell_def["xga"] + league_mean_xga60 * K / 3600) / (cell_def["toi"] + K) * 3600
    cell_def.loc[cell_def["toi"] <= 0, "xga60"] = league_mean_xga60

    # shutdown_resilience: min xGA/60 across defensive cells per team (lower = better)
    shutdown = cell_def.groupby("team")["xga60"].min()

    # depth_insurance: std of per-line xGF/60 across EV lines
    line_xgf60 = ev.groupby(["team", "off_line"]).agg(
        xgf=("xg_f", "sum"), toi=("toi", "sum"))
    line_xgf60["xgf60"] = np.where(
        line_xgf60["toi"] > 0, line_xgf60["xgf"] / line_xgf60["toi"] * 3600, 0)
    depth_ins = line_xgf60.groupby("team")["xgf60"].std(ddof=0).fillna(0)

    teams = rec.groupby("team")["toi"].sum().index
    result = pd.DataFrame(index=teams)
    result["exploit_ability"] = exploit
    result["shutdown_resilience"] = shutdown
    result["depth_insurance"] = depth_ins
    return result.fillna(0)


# ── Special Teams Expected Value ──────────────────────────────────

def _stack_penalty_records(df):
    """Stack penalty columns from raw data, symmetric home/away.

    Returns DataFrame with columns: game_id, team, opp, pen_committed, opp_pen_committed, toi
    Returns None if penalty columns don't exist in df.
    """
    if "home_penalties_committed" not in df.columns:
        return None
    home = df[["game_id", "home_team", "away_team",
               "home_penalties_committed", "away_penalties_committed", "toi"]].copy()
    home.columns = ["game_id", "team", "opp", "pen_committed", "opp_pen_committed", "toi"]
    away = df[["game_id", "away_team", "home_team",
               "away_penalties_committed", "home_penalties_committed", "toi"]].copy()
    away.columns = ["game_id", "team", "opp", "pen_committed", "opp_pen_committed", "toi"]
    return pd.concat([home, away], ignore_index=True)


def build_st_expected_value(df, rec=None):
    """Compute special teams expected value from penalty rates and ST efficiency.

    Returns DataFrame indexed by team with columns:
        pen_taken60, pen_drawn60, net_st_ev
    """
    pen_rec = _stack_penalty_records(df)
    if pen_rec is None:
        return pd.DataFrame()

    st = build_special_teams(df, rec=rec)

    # Aggregate penalty data per team
    team_pen = pen_rec.groupby("team").agg(
        pen_committed=("pen_committed", "sum"),
        opp_pen_committed=("opp_pen_committed", "sum"),
        toi=("toi", "sum"))

    team_pen["pen_taken60"] = np.where(
        team_pen["toi"] > 0, team_pen["pen_committed"] / team_pen["toi"] * 3600, 0)
    team_pen["pen_drawn60"] = np.where(
        team_pen["toi"] > 0, team_pen["opp_pen_committed"] / team_pen["toi"] * 3600, 0)

    # net_st_ev = pp_xgf60 * (pen_drawn60 * 120/3600) - pk_xga60 * (pen_taken60 * 120/3600)
    # 120s = average PP duration assumption
    pp_duration_frac = 120 / 3600
    result = pd.DataFrame(index=team_pen.index)
    result["pen_taken60"] = team_pen["pen_taken60"]
    result["pen_drawn60"] = team_pen["pen_drawn60"]

    pp_xgf60 = st["pp_xgf60"].reindex(result.index, fill_value=0)
    pk_xga60 = st["pk_xga60"].reindex(result.index, fill_value=0)
    result["net_st_ev"] = (pp_xgf60 * result["pen_drawn60"] * pp_duration_frac -
                           pk_xga60 * result["pen_taken60"] * pp_duration_frac)

    return result.fillna(0)


# ── Volatility / Tail-Risk Features ──────────────────────────────

def build_volatility_features(df, game_level, rec=None):
    """Compute volatility and tail-risk features.

    Parameters
    ----------
    df : raw record-level DataFrame
        Used for xGD and GSAx volatility (requires xG columns not in game_level)
    game_level : game-level aggregated DataFrame (from build_game_level)
        Used for blowout/close rates (only has goals, not xG)

    Note: Both sources are needed because game_level doesn't include xG columns.

    Returns DataFrame indexed by team with columns:
        xgd60_std_game, gsax_volatility, blowout_rate, close_game_rate
    """
    if rec is None:
        rec = stack_records(df)

    # xgd60_std_game: std of per-game xGD/60
    # Use raw df directly since stack_records doesn't include game_id
    home_games = df[["game_id", "home_team", "home_xg", "away_xg", "toi"]].copy()
    home_games.columns = ["game_id", "team", "xgf", "xga", "toi"]
    away_games = df[["game_id", "away_team", "away_xg", "home_xg", "toi"]].copy()
    away_games.columns = ["game_id", "team", "xgf", "xga", "toi"]
    # Aggregate to game level per team
    stacked_games = pd.concat([home_games, away_games], ignore_index=True)
    game_agg = stacked_games.groupby(["team", "game_id"]).agg(
        xgf=("xgf", "sum"), xga=("xga", "sum"), toi=("toi", "sum"))
    game_agg["xgd60"] = np.where(
        game_agg["toi"] > 0, (game_agg["xgf"] - game_agg["xga"]) / game_agg["toi"] * 3600, 0)
    xgd60_std = game_agg.groupby("team")["xgd60"].std(ddof=1).fillna(0)

    # gsax_volatility: std of per-game goalie GSAx/60
    home_gk = df[["game_id", "home_team", "home_goalie", "away_xg", "away_goals", "toi"]].copy()
    home_gk.columns = ["game_id", "team", "goalie", "xga", "ga", "toi"]
    away_gk = df[["game_id", "away_team", "away_goalie", "home_xg", "home_goals", "toi"]].copy()
    away_gk.columns = ["game_id", "team", "goalie", "xga", "ga", "toi"]
    stacked_gk = pd.concat([home_gk, away_gk], ignore_index=True)
    stacked_gk = stacked_gk[stacked_gk["goalie"] != "empty_net"]
    gk_game = stacked_gk.groupby(["team", "game_id"]).agg(
        xga=("xga", "sum"), ga=("ga", "sum"), toi=("toi", "sum"))
    gk_game["gsax60"] = np.where(
        gk_game["toi"] > 0, (gk_game["xga"] - gk_game["ga"]) / gk_game["toi"] * 3600, 0)
    gsax_vol = gk_game.groupby("team")["gsax60"].std(ddof=1).fillna(0)

    # blowout_rate and close_game_rate from game_level
    # Stack home/away from game_level
    home_gl = game_level[["game_id", "home_team", "home_goals", "away_goals"]].copy()
    home_gl.columns = ["game_id", "team", "gf", "ga"]
    away_gl = game_level[["game_id", "away_team", "away_goals", "home_goals"]].copy()
    away_gl.columns = ["game_id", "team", "gf", "ga"]
    gl_stacked = pd.concat([home_gl, away_gl], ignore_index=True)
    gl_stacked["gd_abs"] = (gl_stacked["gf"] - gl_stacked["ga"]).abs()

    team_games = gl_stacked.groupby("team").size()
    blowouts = gl_stacked[gl_stacked["gd_abs"] >= 3].groupby("team").size()
    close_games = gl_stacked[gl_stacked["gd_abs"] <= 1].groupby("team").size()

    # Add-2 smoothing: (count + 2) / (games + 4)
    teams = xgd60_std.index
    result = pd.DataFrame(index=teams)
    result["xgd60_std_game"] = xgd60_std
    result["gsax_volatility"] = gsax_vol.reindex(teams, fill_value=0)
    result["blowout_rate"] = ((blowouts.reindex(teams, fill_value=0) + 2) /
                              (team_games.reindex(teams, fill_value=0) + 4))
    result["close_game_rate"] = ((close_games.reindex(teams, fill_value=0) + 2) /
                                 (team_games.reindex(teams, fill_value=0) + 4))

    return result.fillna(0)
