"""Team season statistics computation."""

import numpy as np
import pandas as pd

from core.config import CFG


def build_team_stats(game):
    """Stack home/away symmetrically into per-team season stats.

    Returns a DataFrame indexed by team name with all derived stats.
    Superset of columns needed by both phase1_submit.py and phase1_ml.py.
    """
    # Determine which optional columns are present in game-level data
    has_pc = "home_pc" in game.columns
    has_assists = "home_assists" in game.columns

    home_cols = ["game_id", "home_team", "away_team", "home_goals", "away_goals",
                 "home_xg", "away_xg", "home_shots", "away_shots", "home_win", "toi",
                 "home_max_xg", "away_max_xg", "home_pm", "away_pm"]
    home_names = ["gid", "team", "opp", "gf", "ga", "xgf", "xga", "sf", "sa", "win", "toi",
                  "max_xg_for", "max_xg_against", "pm_taken", "pm_drawn"]

    away_cols = ["game_id", "away_team", "home_team", "away_goals", "home_goals",
                 "away_xg", "home_xg", "away_shots", "home_shots", "home_win", "toi",
                 "away_max_xg", "home_max_xg", "away_pm", "home_pm"]
    away_names = ["gid", "team", "opp", "gf", "ga", "xgf", "xga", "sf", "sa", "hw", "toi",
                  "max_xg_for", "max_xg_against", "pm_taken", "pm_drawn"]

    if has_pc:
        home_cols += ["home_pc", "away_pc"]
        home_names += ["pc_taken", "pc_drawn"]
        away_cols += ["away_pc", "home_pc"]
        away_names += ["pc_taken", "pc_drawn"]

    if has_assists:
        home_cols += ["home_assists", "away_assists"]
        home_names += ["assists_f", "assists_a"]
        away_cols += ["away_assists", "home_assists"]
        away_names += ["assists_f", "assists_a"]

    home = game[home_cols].copy()
    home.columns = home_names
    away = game[away_cols].copy()
    away.columns = away_names
    away["win"] = (away["hw"] == 0).astype(int)
    away.drop(columns=["hw"], inplace=True)
    tg = pd.concat([home, away], ignore_index=True)

    agg_dict = dict(
        games=("gid", "count"), wins=("win", "sum"),
        gf=("gf", "sum"), ga=("ga", "sum"),
        xgf=("xgf", "sum"), xga=("xga", "sum"),
        sf=("sf", "sum"), sa=("sa", "sum"), toi=("toi", "sum"),
        pm_taken=("pm_taken", "sum"), pm_drawn=("pm_drawn", "sum"),
        max_xg_for=("max_xg_for", "mean"),
        max_xg_against=("max_xg_against", "mean"),
    )
    if has_pc:
        agg_dict["pc_taken"] = ("pc_taken", "sum")
        agg_dict["pc_drawn"] = ("pc_drawn", "sum")
    if has_assists:
        agg_dict["assists_f"] = ("assists_f", "sum")
        agg_dict["assists_a"] = ("assists_a", "sum")

    stats = tg.groupby("team").agg(**agg_dict)

    # --- Original derived features ---
    stats["win_pct"] = stats["wins"] / stats["games"]
    stats["gd_pg"] = (stats["gf"] - stats["ga"]) / stats["games"]
    stats["xgd_pg"] = (stats["xgf"] - stats["xga"]) / stats["games"]
    # Data dictionary: toi is seconds. per60 = per 60 minutes = per 3600 seconds.
    toi_hours = stats["toi"] / 3600
    stats["xgd_per60"] = np.where(toi_hours > 0, (stats["xgf"] - stats["xga"]) / toi_hours, 0)
    total_xg = stats["xgf"] + stats["xga"]
    stats["xg_share"] = np.where(total_xg > 0, stats["xgf"] / total_xg, 0.5)
    stats["save_pct"] = np.where(stats["sa"] > 0, 1 - (stats["ga"] / stats["sa"]), 0)
    stats["shooting_pct"] = np.where(stats["sf"] > 0, stats["gf"] / stats["sf"], 0)
    stats["pdo"] = stats["shooting_pct"] + stats["save_pct"]
    stats["gsax_pg"] = (stats["xga"] - stats["ga"]) / stats["games"]
    stats["xg_per_shot"] = np.where(stats["sf"] > 0, stats["xgf"] / stats["sf"], 0)
    stats["pim_pg"] = stats["pm_taken"] / stats["games"]
    stats["pen_diff_pg"] = (stats["pm_drawn"] - stats["pm_taken"]) / stats["games"]

    # --- New derived features ---
    stats["xg_per_shot_against"] = np.where(stats["sa"] > 0, stats["xga"] / stats["sa"], 0)
    stats["shots_for60"] = np.where(toi_hours > 0, stats["sf"] / toi_hours, 0)
    stats["shots_against60"] = np.where(toi_hours > 0, stats["sa"] / toi_hours, 0)
    stats["shot_diff60"] = stats["shots_for60"] - stats["shots_against60"]
    stats["mean_max_xg_for"] = stats["max_xg_for"]  # already aggregated as mean

    # Per-60 finishing and goalsaving
    stats["fin60"] = np.where(toi_hours > 0, (stats["gf"] - stats["xgf"]) / toi_hours, 0)
    stats["gsax60"] = np.where(toi_hours > 0, (stats["xga"] - stats["ga"]) / toi_hours, 0)

    # Shrinkage (Bayesian regression toward league mean)
    K = CFG["shrinkage"]["K"]
    league_fin60 = np.mean(stats["fin60"].values)
    league_gsax60 = np.mean(stats["gsax60"].values)
    stats["fin60_shrunk"] = (stats["fin60"] * stats["toi"] + league_fin60 * K) / (stats["toi"] + K)
    stats["gsax60_shrunk"] = (stats["gsax60"] * stats["toi"] + league_gsax60 * K) / (stats["toi"] + K)

    # Penalty count per-60 (if available)
    if has_pc:
        stats["pim60"] = np.where(toi_hours > 0, stats["pm_taken"] / toi_hours, 0)
        stats["pen60"] = np.where(toi_hours > 0, stats["pc_taken"] / toi_hours, 0)

    # Assists per-60 (if available)
    if has_assists:
        stats["assists_per60"] = np.where(toi_hours > 0, stats["assists_f"] / toi_hours, 0)

    return stats
