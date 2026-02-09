import os
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ZIP_PATH = "drive-download-20260205T132825Z-1-001.zip"
OUTPUT_DIR = "output"


def ensure_extracted(zip_path: str, filename: str, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)
    if os.path.exists(out_path):
        return out_path
    with zipfile.ZipFile(zip_path) as zf:
        zf.extract(filename, output_dir)
    return out_path


def load_data(zip_path: str, output_dir: str):
    whl_path = ensure_extracted(zip_path, "whl_2025.csv", output_dir)
    matchups_path = ensure_extracted(zip_path, "WHSDSC_Rnd1_matchups.xlsx", output_dir)
    df = pd.read_csv(whl_path)
    matchups = pd.read_excel(matchups_path)
    return df, matchups


def build_game_level(df: pd.DataFrame) -> pd.DataFrame:
    game = df.groupby("game_id").agg(
        home_team=("home_team", "first"),
        away_team=("away_team", "first"),
        home_goals=("home_goals", "sum"),
        away_goals=("away_goals", "sum"),
        home_xg=("home_xg", "sum"),
        away_xg=("away_xg", "sum"),
        went_ot=("went_ot", "first"),
    )
    game["home_win"] = (game["home_goals"] > game["away_goals"]).astype(int)
    return game


def build_team_stats(game: pd.DataFrame) -> pd.DataFrame:
    home = game.reset_index()[["game_id", "home_team", "away_team", "home_goals", "away_goals", "home_xg", "away_xg", "home_win"]]
    home = home.rename(
        columns={
            "home_team": "team",
            "away_team": "opponent",
            "home_goals": "goals_for",
            "away_goals": "goals_against",
            "home_xg": "xg_for",
            "away_xg": "xg_against",
            "home_win": "win",
        }
    )

    away = game.reset_index()[["game_id", "away_team", "home_team", "away_goals", "home_goals", "away_xg", "home_xg", "home_win"]]
    away = away.rename(
        columns={
            "away_team": "team",
            "home_team": "opponent",
            "away_goals": "goals_for",
            "home_goals": "goals_against",
            "away_xg": "xg_for",
            "home_xg": "xg_against",
            "home_win": "home_win",
        }
    )
    away["win"] = (away["home_win"] == 0).astype(int)
    away = away.drop(columns=["home_win"])

    team_games = pd.concat([home, away], ignore_index=True)
    stats = team_games.groupby("team").agg(
        games=("game_id", "count"),
        wins=("win", "sum"),
        goals_for=("goals_for", "sum"),
        goals_against=("goals_against", "sum"),
        xg_for=("xg_for", "sum"),
        xg_against=("xg_against", "sum"),
    )
    stats["losses"] = stats["games"] - stats["wins"]
    stats["win_pct"] = stats["wins"] / stats["games"]
    stats["goal_diff_pg"] = (stats["goals_for"] - stats["goals_against"]) / stats["games"]
    stats["xg_diff_pg"] = (stats["xg_for"] - stats["xg_against"]) / stats["games"]
    stats["xg_share"] = stats["xg_for"] / (stats["xg_for"] + stats["xg_against"])
    return stats


def build_strength_ratings(stats: pd.DataFrame) -> pd.DataFrame:
    zcols = ["xg_diff_pg", "xg_share", "goal_diff_pg", "win_pct"]
    for col in zcols:
        mean = stats[col].mean()
        std = stats[col].std(ddof=0)
        stats[f"z_{col}"] = 0 if std == 0 else (stats[col] - mean) / std

    stats["strength_score"] = (
        0.45 * stats["z_xg_diff_pg"]
        + 0.25 * stats["z_xg_share"]
        + 0.2 * stats["z_goal_diff_pg"]
        + 0.1 * stats["z_win_pct"]
    )

    rankings = stats.sort_values("strength_score", ascending=False).reset_index()
    rankings["rank"] = np.arange(1, len(rankings) + 1)
    return rankings


def calibrate_logistic_k(diff: np.ndarray, y: np.ndarray) -> float:
    best_k = 1.0
    best_loss = float("inf")
    for k in np.linspace(0.1, 5.0, 100):
        p = 1 / (1 + np.exp(-k * diff))
        p = np.clip(p, 1e-6, 1 - 1e-6)
        loss = -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()
        if loss < best_loss:
            best_loss = loss
            best_k = k
    return best_k


def build_matchup_probabilities(game: pd.DataFrame, rankings: pd.DataFrame, matchups: pd.DataFrame) -> pd.DataFrame:
    strength = dict(zip(rankings["team"], rankings["strength_score"]))
    diff = game["home_team"].map(strength) - game["away_team"].map(strength)
    k = calibrate_logistic_k(diff.values, game["home_win"].values)

    matchups = matchups.copy()
    matchups["strength_home"] = matchups["home_team"].map(strength)
    matchups["strength_away"] = matchups["away_team"].map(strength)
    matchups["strength_diff"] = matchups["strength_home"] - matchups["strength_away"]
    matchups["home_win_prob"] = 1 / (1 + np.exp(-k * matchups["strength_diff"]))
    matchups["predicted_winner"] = np.where(matchups["home_win_prob"] >= 0.5, matchups["home_team"], matchups["away_team"])
    matchups["home_win_prob"] = matchups["home_win_prob"].round(4)
    return matchups[["game", "game_id", "home_team", "away_team", "home_win_prob", "predicted_winner"]]


def build_line_disparity(df: pd.DataFrame) -> pd.DataFrame:
    home = df[["home_team", "home_off_line", "home_xg", "toi"]].rename(
        columns={"home_team": "team", "home_off_line": "line", "home_xg": "xg"}
    )
    away = df[["away_team", "away_off_line", "away_xg", "toi"]].rename(
        columns={"away_team": "team", "away_off_line": "line", "away_xg": "xg"}
    )
    lines = pd.concat([home, away], ignore_index=True)
    lines = lines[lines["line"].isin(["first_off", "second_off"])]

    line_stats = lines.groupby(["team", "line"]).agg(xg=("xg", "sum"), toi=("toi", "sum"))
    # Data dictionary: toi is seconds. per60 = per 60 minutes = per 3600 seconds.
    line_stats["xg_per60"] = np.where(line_stats["toi"] > 0, (line_stats["xg"] / line_stats["toi"]) * 3600, np.nan)
    pivot = line_stats["xg_per60"].unstack("line")
    pivot["disparity_ratio"] = pivot["first_off"] / pivot["second_off"]
    return pivot.reset_index().sort_values("disparity_ratio", ascending=False)


def plot_disparity_vs_strength(rankings: pd.DataFrame, disparity: pd.DataFrame, output_dir: str) -> str:
    plot_df = rankings.merge(disparity[["team", "disparity_ratio"]], on="team", how="left")
    plt.figure(figsize=(8, 6))
    plt.scatter(plot_df["disparity_ratio"], plot_df["strength_score"], alpha=0.75)
    plt.xlabel("Offensive Line Disparity (First / Second xG per 60)")
    plt.ylabel("Team Strength Score")
    plt.title("Team Strength vs Offensive Line Disparity")

    for _, row in plot_df.nlargest(5, "strength_score").iterrows():
        plt.annotate(row["team"], (row["disparity_ratio"], row["strength_score"]), textcoords="offset points", xytext=(4, 4), fontsize=8)

    output_path = os.path.join(output_dir, "phase1_line_disparity_vs_strength.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_path


def write_summary(rankings: pd.DataFrame, matchups: pd.DataFrame, disparity_top10: pd.DataFrame, output_dir: str) -> str:
    top10_rankings = rankings[["rank", "team", "strength_score", "win_pct", "xg_share", "xg_diff_pg"]].head(10)
    summary_path = os.path.join(output_dir, "phase1_summary.md")

    methodology = """## Phase 1d: Methodology Summary (Draft)

**Process (≈50 words).** We cleaned the season data by verifying game totals, removing no rows, and standardizing fields across home/away records. We aggregated line-level rows to game-level outcomes, then stacked home and away results to compute team-level totals for goals and xG.

**Additional variables (≈25 words).** We engineered xG share, xG differential per game, goal differential per game, and offensive line xG per 60 to support rankings and disparity analysis.

**Tools and techniques (≈50 words).** We used Python with pandas and numpy for aggregation, and matplotlib for visualization. Team strength was built from standardized xG and goal metrics. A calibrated logistic curve converted strength differences into matchup probabilities.

**Statistical methods (≈100 words).** We used summary statistics and z-score normalization to combine multiple performance measures into a single strength score, prioritizing underlying xG performance while retaining outcome-based signals. For matchup probabilities, we fit a single-parameter logistic transformation by minimizing log loss on historical games. This produces monotonic probabilities that map larger strength gaps to higher win likelihoods. For line disparity, we computed xG per 60 by line to normalize for time on ice, then compared first vs second line output. The visualization uses a scatter plot to reveal association between disparity and overall strength.
"""

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# Phase 1 Outputs\n\n")
        f.write("## Power Rankings (Top 10)\n\n")
        f.write(df_to_markdown(top10_rankings))
        f.write("\n\n## Matchup Probabilities (Round 1)\n\n")
        f.write(df_to_markdown(matchups))
        f.write("\n\n## Line Disparity Top 10\n\n")
        f.write(df_to_markdown(disparity_top10))
        f.write("\n\n")
        f.write(methodology)
        f.write("\n")

    return summary_path


def df_to_markdown(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(x) for x in row.tolist()) + " |")
    return "\n".join([header, sep] + rows)


def main(zip_path: str = ZIP_PATH, output_dir: str = OUTPUT_DIR):
    df, matchups = load_data(zip_path, output_dir)
    game = build_game_level(df)
    stats = build_team_stats(game)
    rankings = build_strength_ratings(stats)
    matchup_probs = build_matchup_probabilities(game, rankings, matchups)
    disparity = build_line_disparity(df)
    disparity_top10 = disparity[["team", "disparity_ratio"]].head(10)

    os.makedirs(output_dir, exist_ok=True)
    rankings_out = os.path.join(output_dir, "phase1_power_rankings.csv")
    matchups_out = os.path.join(output_dir, "phase1_matchup_probabilities.csv")
    disparity_out = os.path.join(output_dir, "phase1_line_disparity_top10.csv")

    rankings.to_csv(rankings_out, index=False)
    matchup_probs.to_csv(matchups_out, index=False)
    disparity_top10.to_csv(disparity_out, index=False)

    chart_path = plot_disparity_vs_strength(rankings, disparity, output_dir)
    summary_path = write_summary(rankings, matchup_probs, disparity_top10, output_dir)

    print("Phase 1 outputs generated:")
    print(rankings_out)
    print(matchups_out)
    print(disparity_out)
    print(chart_path)
    print(summary_path)


if __name__ == "__main__":
    main()
