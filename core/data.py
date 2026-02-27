"""Shared data loading and game-level aggregation."""

import os
import zipfile

import numpy as np
import pandas as pd


ZIP_PATH = "drive-download-20260205T132825Z-1-001.zip"


def load_data(zip_path=None, output_dir="output"):
    """Load WHL 2025 data and Round 1 matchups.

    Extracts from the zip if files are not already present.
    """
    zip_path = zip_path or ZIP_PATH
    os.makedirs(output_dir, exist_ok=True)
    for fn in ["whl_2025.csv", "WHSDSC_Rnd1_matchups.xlsx"]:
        out = os.path.join(output_dir, fn)
        if not os.path.exists(out):
            try:
                with zipfile.ZipFile(zip_path) as zf:
                    zf.extract(fn, output_dir)
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"Data file not found: '{zip_path}'. Download the competition "
                    f"data zip and place it in the repo root, or pre-extract "
                    f"CSV/XLSX files to '{output_dir}'."
                ) from None
            except zipfile.BadZipFile:
                raise zipfile.BadZipFile(
                    f"'{zip_path}' is corrupted or not a valid zip file. "
                    f"Please re-download the competition data zip."
                ) from None
    df = pd.read_csv(os.path.join(output_dir, "whl_2025.csv"))
    matchups = pd.read_excel(os.path.join(output_dir, "WHSDSC_Rnd1_matchups.xlsx"))
    return df, matchups


def build_game_level(df):
    """Collapse line-level rows into game-level records."""
    game = df.groupby("game_id").agg(
        home_team=("home_team", "first"), away_team=("away_team", "first"),
        home_goals=("home_goals", "sum"), away_goals=("away_goals", "sum"),
        home_xg=("home_xg", "sum"), away_xg=("away_xg", "sum"),
        home_shots=("home_shots", "sum"), away_shots=("away_shots", "sum"),
        home_max_xg=("home_max_xg", "max"), away_max_xg=("away_max_xg", "max"),
        home_pm=("home_penalty_minutes", "sum"), away_pm=("away_penalty_minutes", "sum"),
        home_pc=("home_penalties_committed", "sum"), away_pc=("away_penalties_committed", "sum"),
        home_assists=("home_assists", "sum"), away_assists=("away_assists", "sum"),
        went_ot=("went_ot", "first"), toi=("toi", "sum"),
    ).reset_index()
    game["home_win"] = (game["home_goals"] > game["away_goals"]).astype(int)
    if (game["home_goals"] == game["away_goals"]).any():
        raise ValueError("Dataset contains draws, but the pipeline assumes a winner is always determined.")
    return game
