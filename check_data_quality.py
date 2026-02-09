import pandas as pd
import numpy as np
import os

def check_data_quality():
    print("--- Loading Data ---")
    try:
        df = pd.read_csv(os.path.join('output', 'whl_2025.csv'))
        matchups = pd.read_excel(os.path.join('output', 'WHSDSC_Rnd1_matchups.xlsx'))
        print("Data loaded successfully.")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    print("\n--- Basic Information ---")
    print(f"Training Data Shape: {df.shape}")
    print(f"Matchups Data Shape: {matchups.shape}")

    print("\n--- Missing Values ---")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("No missing values found in training data.")
    else:
        print(missing[missing > 0])

    print("\n--- Team Name Consistency ---")
    # Check if teams in matchups exist in training data
    train_teams = set(df['home_team'].unique()) | set(df['away_team'].unique())
    matchup_teams = set(matchups['home_team'].unique()) | set(matchups['away_team'].unique())
    
    print(f"Unique teams in training data: {len(train_teams)}")
    print(f"Unique teams in matchups: {len(matchup_teams)}")
    
    unknown_teams = matchup_teams - train_teams
    if unknown_teams:
        print(f"WARNING: The following teams in matchups are not in training data: {unknown_teams}")
    else:
        print("All teams in matchups are present in training data.")

    print("\n--- Data Anomalies Check ---")
    # Check for negative values in metrics that shouldn't be negative
    numeric_cols = ['toi', 'home_goals', 'away_goals', 'home_xg', 'away_xg', 'home_shots', 'away_shots']
    for col in numeric_cols:
        if col in df.columns:
            neg_count = (df[col] < 0).sum()
            if neg_count > 0:
                print(f"WARNING: {col} has {neg_count} negative values.")
            else:
                print(f"{col}: No negative values.")

    # Check consistency: goals vs xG (just a sanity check, xG shouldn't be wildly different from goals on average, but individual variance is expected)
    print("\n--- Logic Checks ---")
    # Check if went_ot is binary
    if set(df['went_ot'].unique()).issubset({0, 1}):
         print("went_ot is binary (0/1).")
    else:
         print(f"WARNING: went_ot contains unexpected values: {df['went_ot'].unique()}")

    # Check if Game IDs are unique per game (sanity check)
    game_counts = df.groupby('game_id').size()
    print(f"Average records per game: {game_counts.mean():.2f}")
    print(f"Min records per game: {game_counts.min()}")
    print(f"Max records per game: {game_counts.max()}")

    # Duplicates
    dupes = df.duplicated().sum()
    print(f"\nDuplicate rows: {dupes}")

if __name__ == "__main__":
    check_data_quality()
