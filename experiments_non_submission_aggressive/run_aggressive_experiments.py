"""
NON-SUBMISSION AGGRESSIVE EXPERIMENTS
=====================================

This script is intentionally isolated from the submission pipeline.
It runs high-capacity models and aggressive ensembling to chase accuracy.

Outputs are written under:
  experiments_non_submission_aggressive/outputs/
"""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=FutureWarning)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "output"
OUT_DIR = SCRIPT_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

# WARNING: OneHot-encoding 32 teams × 2 sides = 64 sparse features on ~1,050
# training samples per fold creates a high-dimensional setting that is prone to
# overfitting, especially with deep tree models.  This is intentional for the
# "aggressive accuracy chase" experiment but should NOT be used for submission.
CAT_COLS = ["home_team", "away_team"]
NUM_COLS = [
    "bt_diff",
    "abs_bt_diff",
    "bt_diff_sq",
    "sos_diff",
    "win_pct_diff",
    "win_pct_sm_diff",
    "gd_pg_diff",
    "xgd_per60_diff",
    "xg_share_diff",
    "pdo_diff",
    "gsax_pg_diff",
    "xg_per_shot_diff",
    "pim_pg_diff",
    "pen_diff_pg_diff",
]
ALL_COLS = CAT_COLS + NUM_COLS


@dataclass
class FoldPack:
    train_feat: pd.DataFrame
    test_feat: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray
    train_games: pd.DataFrame
    test_games: pd.DataFrame


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_path = DATA_DIR / "whl_2025.csv"
    matchup_path = DATA_DIR / "WHSDSC_Rnd1_matchups.xlsx"
    if not df_path.exists() or not matchup_path.exists():
        raise FileNotFoundError(
            "Missing data files under output/. Run submission pipeline once to extract data."
        )
    df = pd.read_csv(df_path)
    matchups = pd.read_excel(matchup_path)
    return df, matchups


def _assert_game_constant_fields(df: pd.DataFrame, group_col: str, cols: List[str], context: str) -> None:
    """Guard against silent `.agg(first)` assumptions drifting from the raw file."""
    if group_col not in df.columns:
        return
    gb = df.groupby(group_col, dropna=False)
    for col in cols:
        if col not in df.columns:
            continue
        nunq = gb[col].nunique(dropna=False)
        bad = nunq[nunq > 1]
        if bad.empty:
            continue
        sample_ids = ", ".join(str(x) for x in bad.index.tolist()[:3])
        raise ValueError(
            f"{context}: `{col}` is not constant within `{group_col}` "
            f"for {len(bad)} group(s) (e.g., {sample_ids})"
        )


def build_game_level(df: pd.DataFrame) -> pd.DataFrame:
    _assert_game_constant_fields(
        df, "game_id", ["home_team", "away_team", "went_ot"], "build_game_level"
    )
    game = (
        df.groupby("game_id")
        .agg(
            home_team=("home_team", "first"),
            away_team=("away_team", "first"),
            home_goals=("home_goals", "sum"),
            away_goals=("away_goals", "sum"),
            went_ot=("went_ot", "first"),
        )
        .reset_index()
    )
    game = game[game["home_goals"] != game["away_goals"]].reset_index(drop=True)
    game["home_win"] = (game["home_goals"] > game["away_goals"]).astype(int)
    game["gid_num"] = (
        game["game_id"].astype(str).str.extract(r"(\d+)", expand=False).astype(int)
    )
    return game


def build_team_list(game: pd.DataFrame) -> List[str]:
    return sorted(set(game["home_team"].tolist()) | set(game["away_team"].tolist()))


def fit_bt_ot_aware(
    game: pd.DataFrame, teams: List[str], ot_weight: float = 0.5, max_iter: int = 300
) -> Dict[str, float]:
    team_idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    strength = np.zeros(n)
    for _ in range(max_iter):
        grad = np.zeros(n)
        hess = np.zeros(n)
        for _, row in game.iterrows():
            i = team_idx[row["home_team"]]
            j = team_idx[row["away_team"]]
            p = expit(strength[i] - strength[j])
            y = row["home_win"]
            w = ot_weight if row["went_ot"] == 1 else 1.0
            grad[i] += w * (y - p)
            grad[j] -= w * (y - p)
            fisher = w * p * (1 - p)
            hess[i] -= fisher
            hess[j] -= fisher
        hess = np.clip(hess, None, -1e-6)
        delta = -grad / hess
        delta -= delta.mean()
        strength += 0.5 * delta
        if np.max(np.abs(delta)) < 1e-8:
            break
    strength -= strength.mean()
    return dict(zip(teams, strength))


def build_team_stats(game: pd.DataFrame, smooth_alpha: float = 8.0) -> pd.DataFrame:
    home = game[
        ["game_id", "home_team", "away_team", "home_goals", "away_goals", "home_win"]
    ].copy()
    home.columns = ["gid", "team", "opp", "gf", "ga", "win"]
    away = game[
        ["game_id", "away_team", "home_team", "away_goals", "home_goals", "home_win"]
    ].copy()
    away.columns = ["gid", "team", "opp", "gf", "ga", "hw"]
    away["win"] = (away["hw"] == 0).astype(int)
    away = away.drop(columns=["hw"])
    tg = pd.concat([home, away], ignore_index=True)
    s = (
        tg.groupby("team")
        .agg(games=("gid", "count"), wins=("win", "sum"), gf=("gf", "sum"), ga=("ga", "sum"))
        .copy()
    )
    s["win_pct"] = s["wins"] / s["games"]
    global_rate = float(tg["win"].mean())
    s["win_pct_sm"] = (s["wins"] + smooth_alpha * global_rate) / (s["games"] + smooth_alpha)
    s["gd_pg"] = (s["gf"] - s["ga"]) / s["games"]
    return s


def build_line_based_stats(df: pd.DataFrame) -> pd.DataFrame:
    home = df[
        [
            "game_id",
            "home_team",
            "away_team",
            "home_xg",
            "away_xg",
            "home_shots",
            "away_shots",
            "home_penalty_minutes",
            "away_penalty_minutes",
            "toi",
            "home_goals",
            "away_goals",
        ]
    ].copy()
    home.columns = [
        "gid",
        "team",
        "opp",
        "xgf",
        "xga",
        "sf",
        "sa",
        "pm_taken",
        "pm_drawn",
        "toi",
        "gf",
        "ga",
    ]
    away = df[
        [
            "game_id",
            "away_team",
            "home_team",
            "away_xg",
            "home_xg",
            "away_shots",
            "home_shots",
            "away_penalty_minutes",
            "home_penalty_minutes",
            "toi",
            "away_goals",
            "home_goals",
        ]
    ].copy()
    away.columns = [
        "gid",
        "team",
        "opp",
        "xgf",
        "xga",
        "sf",
        "sa",
        "pm_taken",
        "pm_drawn",
        "toi",
        "gf",
        "ga",
    ]
    tg = pd.concat([home, away], ignore_index=True)
    s = (
        tg.groupby("team")
        .agg(
            xgf=("xgf", "sum"),
            xga=("xga", "sum"),
            sf=("sf", "sum"),
            sa=("sa", "sum"),
            toi=("toi", "sum"),
            pm_taken=("pm_taken", "sum"),
            pm_drawn=("pm_drawn", "sum"),
            gf=("gf", "sum"),
            ga=("ga", "sum"),
            games=("gid", "nunique"),
        )
        .copy()
    )
    toi_hours = s["toi"] / 3600.0
    s["xgd_per60"] = np.where(toi_hours > 0, (s["xgf"] - s["xga"]) / toi_hours, 0.0)
    total_xg = s["xgf"] + s["xga"]
    s["xg_share"] = np.where(total_xg > 0, s["xgf"] / total_xg, 0.5)
    s["save_pct"] = np.where(s["sa"] > 0, 1.0 - s["ga"] / s["sa"], 0.0)
    s["shooting_pct"] = np.where(s["sf"] > 0, s["gf"] / s["sf"], 0.0)
    s["pdo"] = s["save_pct"] + s["shooting_pct"]
    s["gsax_pg"] = np.where(s["games"] > 0, (s["xga"] - s["ga"]) / s["games"], 0.0)
    s["xg_per_shot"] = np.where(s["sf"] > 0, s["xgf"] / s["sf"], 0.0)
    s["pim_pg"] = np.where(s["games"] > 0, s["pm_taken"] / s["games"], 0.0)
    s["pen_diff_pg"] = np.where(
        s["games"] > 0, (s["pm_drawn"] - s["pm_taken"]) / s["games"], 0.0
    )
    return s[
        [
            "xgd_per60",
            "xg_share",
            "pdo",
            "gsax_pg",
            "xg_per_shot",
            "pim_pg",
            "pen_diff_pg",
        ]
    ]


def compute_sos(game: pd.DataFrame, bt: Dict[str, float]) -> pd.Series:
    recs: List[dict] = []
    for _, r in game.iterrows():
        recs.append({"team": r["home_team"], "os": bt.get(r["away_team"], 0.0)})
        recs.append({"team": r["away_team"], "os": bt.get(r["home_team"], 0.0)})
    return pd.DataFrame(recs).groupby("team")["os"].mean()


def _safe_get(series: pd.Series, key: str, default: float = 0.0) -> float:
    if key in series.index:
        val = series.loc[key]
        if pd.isna(val):
            return default
        return float(val)
    return default


def build_feature_frame(
    games: pd.DataFrame,
    team_stats: pd.DataFrame,
    line_stats: pd.DataFrame,
    bt: Dict[str, float],
    sos: pd.Series,
    include_target: bool = True,
) -> pd.DataFrame:
    stats = team_stats.join(line_stats, how="left").fillna(0.0)
    out: List[dict] = []
    for _, g in games.iterrows():
        h = g["home_team"]
        a = g["away_team"]
        bt_diff = bt.get(h, 0.0) - bt.get(a, 0.0)
        row = {
            "game_id": g["game_id"],
            "home_team": h,
            "away_team": a,
            "bt_diff": bt_diff,
            "abs_bt_diff": abs(bt_diff),
            "bt_diff_sq": bt_diff * bt_diff,
            "sos_diff": _safe_get(sos, h) - _safe_get(sos, a),
            "win_pct_diff": _safe_get(stats["win_pct"], h) - _safe_get(stats["win_pct"], a),
            "win_pct_sm_diff": _safe_get(stats["win_pct_sm"], h)
            - _safe_get(stats["win_pct_sm"], a),
            "gd_pg_diff": _safe_get(stats["gd_pg"], h) - _safe_get(stats["gd_pg"], a),
            "xgd_per60_diff": _safe_get(stats["xgd_per60"], h)
            - _safe_get(stats["xgd_per60"], a),
            "xg_share_diff": _safe_get(stats["xg_share"], h)
            - _safe_get(stats["xg_share"], a),
            "pdo_diff": _safe_get(stats["pdo"], h) - _safe_get(stats["pdo"], a),
            "gsax_pg_diff": _safe_get(stats["gsax_pg"], h) - _safe_get(stats["gsax_pg"], a),
            "xg_per_shot_diff": _safe_get(stats["xg_per_shot"], h)
            - _safe_get(stats["xg_per_shot"], a),
            "pim_pg_diff": _safe_get(stats["pim_pg"], h) - _safe_get(stats["pim_pg"], a),
            "pen_diff_pg_diff": _safe_get(stats["pen_diff_pg"], h)
            - _safe_get(stats["pen_diff_pg"], a),
        }
        if include_target:
            row["home_win"] = int(g["home_win"])
        out.append(row)
    return pd.DataFrame(out)


def _tune_threshold(y_true: np.ndarray, probs: np.ndarray) -> float:
    y = np.asarray(y_true).astype(int)
    p = np.asarray(probs, dtype=float)
    grid = np.linspace(0.25, 0.75, 501)
    best_t = 0.5
    best_acc = -1.0
    for t in grid:
        acc = float(((p >= t).astype(int) == y).mean())
        if acc > best_acc + 1e-12:
            best_acc = acc
            best_t = t
        elif abs(acc - best_acc) <= 1e-12 and abs(t - 0.5) < abs(best_t - 0.5):
            best_t = t
    return float(best_t)


def _metric_pack(y: np.ndarray, probs: np.ndarray, threshold: float) -> Dict[str, float]:
    p = np.clip(np.asarray(probs, dtype=float), 1e-8, 1 - 1e-8)
    yv = np.asarray(y, dtype=int)
    return {
        "accuracy_05": float(((p >= 0.5).astype(int) == yv).mean()),
        "accuracy_tuned": float(((p >= threshold).astype(int) == yv).mean()),
        "log_loss": float(log_loss(yv, p)),
        "brier": float(brier_score_loss(yv, p)),
    }


def _fit_bt_logistic(
    tr: pd.DataFrame, te: pd.DataFrame, y_train: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    pipe = Pipeline(
        [("scaler", StandardScaler()), ("lr", LogisticRegression(C=2.0, max_iter=5000))]
    )
    Xtr = tr[["bt_diff"]].values
    Xte = te[["bt_diff"]].values
    pipe.fit(Xtr, y_train)
    return pipe.predict_proba(Xtr)[:, 1], pipe.predict_proba(Xte)[:, 1]


def _fit_xgb(tr: pd.DataFrame, te: pd.DataFrame, y_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    prep = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS)],
        remainder="passthrough",
    )
    Xtr = prep.fit_transform(tr[ALL_COLS])
    Xte = prep.transform(te[ALL_COLS])
    model = XGBClassifier(
        n_estimators=400,
        learning_rate=0.04,
        max_depth=4,
        min_child_weight=1,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=4.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=SEED,
        n_jobs=-1,
    )
    model.fit(Xtr, y_train)
    return model.predict_proba(Xtr)[:, 1], model.predict_proba(Xte)[:, 1]


def _fit_cat(tr: pd.DataFrame, te: pd.DataFrame, y_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    Xtr = tr[ALL_COLS].copy()
    Xte = te[ALL_COLS].copy()
    for c in CAT_COLS:
        Xtr[c] = Xtr[c].astype(str)
        Xte[c] = Xte[c].astype(str)
    cat_idx = [0, 1]
    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="Logloss",
        depth=6,
        learning_rate=0.05,
        iterations=700,
        l2_leaf_reg=6.0,
        random_strength=1.0,
        bagging_temperature=0.5,
        border_count=64,
        random_seed=SEED,
        thread_count=-1,
        train_dir=str(OUT_DIR / "catboost_tmp"),
        verbose=False,
    )
    model.fit(Xtr, y_train, cat_features=cat_idx)
    return model.predict_proba(Xtr)[:, 1], model.predict_proba(Xte)[:, 1]


def _optimize_blend(
    y_train: np.ndarray, p_bt: np.ndarray, p_xgb: np.ndarray, p_cat: np.ndarray
) -> Tuple[Tuple[float, float, float], float]:
    best = (-1.0, (0.34, 0.33, 0.33), 0.5)
    ws = np.linspace(0.0, 1.0, 11)
    for w_bt in ws:
        for w_xgb in ws:
            w_cat = 1.0 - w_bt - w_xgb
            if w_cat < -1e-9:
                continue
            if w_cat < 0:
                w_cat = 0.0
            s = w_bt + w_xgb + w_cat
            if s <= 0:
                continue
            w_bt_n = w_bt / s
            w_xgb_n = w_xgb / s
            w_cat_n = w_cat / s
            p = w_bt_n * p_bt + w_xgb_n * p_xgb + w_cat_n * p_cat
            t = _tune_threshold(y_train, p)
            acc = float(((p >= t).astype(int) == y_train).mean())
            if acc > best[0]:
                best = (acc, (w_bt_n, w_xgb_n, w_cat_n), t)
    return best[1], float(best[2])


def _prepare_fold_pack(
    full_df: pd.DataFrame, game: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray
) -> FoldPack:
    train_games = game.iloc[train_idx].reset_index(drop=True)
    test_games = game.iloc[test_idx].reset_index(drop=True)
    teams = build_team_list(train_games)
    bt = fit_bt_ot_aware(train_games, teams, ot_weight=0.5)
    sos = compute_sos(train_games, bt)
    tstats_basic = build_team_stats(train_games)
    train_ids = set(train_games["game_id"].tolist())
    line_part = full_df[full_df["game_id"].isin(train_ids)].copy()
    tstats_line = build_line_based_stats(line_part)
    train_feat = build_feature_frame(train_games, tstats_basic, tstats_line, bt, sos, include_target=True)
    test_feat = build_feature_frame(test_games, tstats_basic, tstats_line, bt, sos, include_target=True)
    return FoldPack(
        train_feat=train_feat,
        test_feat=test_feat,
        y_train=train_feat["home_win"].values.astype(int),
        y_test=test_feat["home_win"].values.astype(int),
        train_games=train_games,
        test_games=test_games,
    )


def run_cv(
    full_df: pd.DataFrame,
    game: pd.DataFrame,
    splitter: Iterable[Tuple[np.ndarray, np.ndarray]],
    split_name: str,
) -> pd.DataFrame:
    records: List[dict] = []
    for fold_id, (train_idx, test_idx) in enumerate(splitter, 1):
        print(f"  [{split_name}] fold {fold_id} ...")
        pack = _prepare_fold_pack(full_df, game, train_idx, test_idx)
        tr, te = pack.train_feat, pack.test_feat
        ytr, yte = pack.y_train, pack.y_test

        p_train_base = np.full(len(ytr), ytr.mean(), dtype=float)
        p_test_base = np.full(len(yte), ytr.mean(), dtype=float)

        p_train_bt, p_test_bt = _fit_bt_logistic(tr, te, ytr)
        p_train_xgb, p_test_xgb = _fit_xgb(tr, te, ytr)
        p_train_cat, p_test_cat = _fit_cat(tr, te, ytr)

        blend_w, blend_t = _optimize_blend(ytr, p_train_bt, p_train_xgb, p_train_cat)
        p_train_blend = blend_w[0] * p_train_bt + blend_w[1] * p_train_xgb + blend_w[2] * p_train_cat
        p_test_blend = blend_w[0] * p_test_bt + blend_w[1] * p_test_xgb + blend_w[2] * p_test_cat

        preds = {
            "Baseline_home_rate": (p_train_base, p_test_base),
            "BT_logistic": (p_train_bt, p_test_bt),
            "XGBoost_aggressive": (p_train_xgb, p_test_xgb),
            "CatBoost_aggressive": (p_train_cat, p_test_cat),
            "Blend_bt_xgb_cat": (p_train_blend, p_test_blend),
        }

        model_thresholds = {}
        for m, (ptr, pte) in preds.items():
            if m == "Blend_bt_xgb_cat":
                thr = blend_t
            else:
                thr = _tune_threshold(ytr, ptr)
            model_thresholds[m] = thr
            rec = {
                "split": split_name,
                "fold": fold_id,
                "model": m,
                "threshold": thr,
                "n_test": int(len(yte)),
            }
            rec.update(_metric_pack(yte, pte, thr))
            if m == "Blend_bt_xgb_cat":
                rec["blend_w_bt"] = blend_w[0]
                rec["blend_w_xgb"] = blend_w[1]
                rec["blend_w_cat"] = blend_w[2]
            records.append(rec)

    return pd.DataFrame(records)


def aggregate_summary(cv_df: pd.DataFrame) -> pd.DataFrame:
    metrics = ["accuracy_05", "accuracy_tuned", "log_loss", "brier"]
    g = cv_df.groupby(["split", "model"], as_index=False)[metrics].mean()
    g_std = cv_df.groupby(["split", "model"], as_index=False)[metrics].std(ddof=0)
    g_std = g_std.rename(columns={m: f"{m}_std" for m in metrics})
    return g.merge(g_std, on=["split", "model"], how="left")


def _as_text_table(df: pd.DataFrame) -> str:
    return "```text\n" + df.to_string(index=False) + "\n```"


def fit_full_and_predict_matchups(
    full_df: pd.DataFrame,
    game: pd.DataFrame,
    matchups: pd.DataFrame,
    best_model: str,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    teams = build_team_list(game)
    bt = fit_bt_ot_aware(game, teams, ot_weight=0.5)
    sos = compute_sos(game, bt)
    tstats_basic = build_team_stats(game)
    tstats_line = build_line_based_stats(full_df)

    train_feat = build_feature_frame(game, tstats_basic, tstats_line, bt, sos, include_target=True)
    ytr = train_feat["home_win"].values.astype(int)

    match_for_feat = matchups[["game", "game_id", "home_team", "away_team"]].copy()
    match_for_feat["home_win"] = 0
    test_feat = build_feature_frame(
        match_for_feat[["game_id", "home_team", "away_team", "home_win"]],
        tstats_basic,
        tstats_line,
        bt,
        sos,
        include_target=False,
    )

    p_train_base = np.full(len(ytr), ytr.mean(), dtype=float)
    p_test_base = np.full(len(test_feat), ytr.mean(), dtype=float)
    p_train_bt, p_test_bt = _fit_bt_logistic(train_feat, test_feat, ytr)
    p_train_xgb, p_test_xgb = _fit_xgb(train_feat, test_feat, ytr)
    p_train_cat, p_test_cat = _fit_cat(train_feat, test_feat, ytr)
    blend_w, blend_t = _optimize_blend(ytr, p_train_bt, p_train_xgb, p_train_cat)
    p_train_blend = blend_w[0] * p_train_bt + blend_w[1] * p_train_xgb + blend_w[2] * p_train_cat
    p_test_blend = blend_w[0] * p_test_bt + blend_w[1] * p_test_xgb + blend_w[2] * p_test_cat

    probs_map = {
        "Baseline_home_rate": (p_train_base, p_test_base),
        "BT_logistic": (p_train_bt, p_test_bt),
        "XGBoost_aggressive": (p_train_xgb, p_test_xgb),
        "CatBoost_aggressive": (p_train_cat, p_test_cat),
        "Blend_bt_xgb_cat": (p_train_blend, p_test_blend),
    }
    if best_model not in probs_map:
        raise ValueError(f"Unknown best model: {best_model}")

    p_train, p_test = probs_map[best_model]
    threshold = blend_t if best_model == "Blend_bt_xgb_cat" else _tune_threshold(ytr, p_train)

    out = matchups.copy()
    out["home_win_prob_aggressive"] = np.clip(p_test, 0.01, 0.99).round(4)
    out["winner_05"] = np.where(
        out["home_win_prob_aggressive"] >= 0.5, out["home_team"], out["away_team"]
    )
    out["winner_tuned"] = np.where(
        out["home_win_prob_aggressive"] >= threshold, out["home_team"], out["away_team"]
    )
    meta = {
        "best_model": best_model,
        "threshold": float(threshold),
        "selection_rule": "random_stratified_5fold: sort by brier asc, log_loss asc, accuracy_05 desc (excluding baseline)",
    }
    if best_model == "Blend_bt_xgb_cat":
        meta["blend_w_bt"] = float(blend_w[0])
        meta["blend_w_xgb"] = float(blend_w[1])
        meta["blend_w_cat"] = float(blend_w[2])
    return out, meta


def main() -> None:
    print("=" * 68)
    print("NON-SUBMISSION AGGRESSIVE ACCURACY LAB")
    print("=" * 68)
    print("Output dir:", OUT_DIR)

    df, matchups = load_data()
    game = build_game_level(df)

    print("[1/4] Random stratified CV (5-fold) ...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    random_df = run_cv(df, game, skf.split(np.zeros(len(game)), game["home_win"].values), "random_stratified_5fold")

    print("[2/4] ID-order CV stress test (5-fold; numeric game_id sort, not true dates) ...")
    game_sorted = game.sort_values("gid_num").reset_index(drop=True)
    tscv = TimeSeriesSplit(n_splits=5)
    time_df = run_cv(df, game_sorted, tscv.split(game_sorted), "id_order_5fold")

    cv_df = pd.concat([random_df, time_df], ignore_index=True)
    summary = aggregate_summary(cv_df)
    cv_path = OUT_DIR / "NON_SUBMISSION_cv_fold_metrics.csv"
    summary_path = OUT_DIR / "NON_SUBMISSION_cv_summary.csv"
    cv_df.to_csv(cv_path, index=False)
    summary.to_csv(summary_path, index=False)

    # Choose aggressive winner by probability quality in random CV.
    # We avoid `accuracy_tuned` for model selection because thresholds are tuned
    # on training predictions inside each fold and are therefore optimistic.
    random_rank = (
        summary[summary["split"] == "random_stratified_5fold"]
        .sort_values(["brier", "log_loss", "accuracy_05"], ascending=[True, True, False])
        .reset_index(drop=True)
    )
    selection_pool = random_rank[random_rank["model"] != "Baseline_home_rate"].reset_index(drop=True)
    if selection_pool.empty:
        raise RuntimeError("No non-baseline models available for aggressive selection.")
    best_model = str(selection_pool.iloc[0]["model"])

    print("[3/4] Fit full-data aggressive winner and predict round-1 matchups ...")
    matchup_pred, meta = fit_full_and_predict_matchups(df, game, matchups, best_model=best_model)
    matchup_path = OUT_DIR / "NON_SUBMISSION_matchup_predictions_aggressive.csv"
    matchup_pred.to_csv(matchup_path, index=False)

    meta_path = OUT_DIR / "NON_SUBMISSION_best_model_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("[4/4] Write markdown summary ...")
    lines: List[str] = []
    lines.append("# NON-SUBMISSION Aggressive Accuracy Report")
    lines.append("")
    lines.append("> WARNING: This folder is for experimentation only. Do NOT submit these files as official competition deliverables.")
    lines.append("")
    lines.append("## Random Stratified 5-Fold (mean)")
    lines.append("")
    lines.append(_as_text_table(random_rank))
    lines.append("")
    lines.append("## ID-Order 5-Fold (mean; numeric `game_id` sort, not true chronology)")
    lines.append("")
    time_rank = (
        summary[summary["split"] == "id_order_5fold"]
        .sort_values(["brier", "log_loss", "accuracy_05"], ascending=[True, True, False])
        .reset_index(drop=True)
    )
    lines.append(_as_text_table(time_rank))
    lines.append("")
    lines.append("## Selected Aggressive Model")
    lines.append("")
    lines.append(f"- Winner by random-CV probability quality (Brier, tie-break log-loss): `{best_model}`")
    lines.append(f"- Matchup output: `{matchup_path.name}`")
    lines.append(f"- Model metadata: `{meta_path.name}`")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- `accuracy_tuned` is diagnostic only: thresholds are tuned on training predictions and can be optimistic.")
    lines.append("- ID-order CV sorts by numeric `game_id`; this is a robustness check, not a true date-based split.")
    lines.append("- This setup is intentionally aggressive and may overfit; compare random CV and ID-order CV before trusting gains.")

    report_path = OUT_DIR / "NON_SUBMISSION_aggressive_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("Best model:", best_model)
    print("Summary:", summary_path)
    print("Report :", report_path)
    print("Matchup:", matchup_path)


if __name__ == "__main__":
    main()
