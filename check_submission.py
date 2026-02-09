import os
import re
import sys

import numpy as np
import pandas as pd


def team_png_basename() -> str:
    name = os.environ.get("TEAM_NAME", "TheIsoLab")
    name = re.sub(r"\s+", "", str(name))
    name = re.sub(r"[^A-Za-z0-9_-]", "", name)
    return name or "TeamName"


def main() -> int:
    out_dir = os.path.join("output", "submission")
    issues: list[str] = []

    # Power rankings
    pr_path = os.path.join(out_dir, "power_rankings.csv")
    if not os.path.exists(pr_path):
        issues.append("missing output/submission/power_rankings.csv")
    else:
        pr = pd.read_csv(pr_path)
        if pr.shape[0] != 32:
            issues.append(f"power_rankings.csv rows={pr.shape[0]} (expected 32)")
        if pr["team"].nunique(dropna=False) != 32:
            issues.append("power_rankings.csv team not unique (expected 32 unique teams)")
        if pr["rank"].isna().any():
            issues.append("power_rankings.csv rank has NaN")
        else:
            try:
                ranks = pr["rank"].astype(int).tolist()
                if sorted(ranks) != list(range(1, 33)):
                    issues.append("power_rankings.csv rank is not exactly 1..32")
            except Exception:
                issues.append("power_rankings.csv rank is not integer-like")
        if "composite" in pr.columns and not pr["composite"].is_monotonic_decreasing:
            issues.append("power_rankings.csv composite not sorted descending")

    # Matchups
    mp_path = os.path.join(out_dir, "matchup_predictions.csv")
    if not os.path.exists(mp_path):
        issues.append("missing output/submission/matchup_predictions.csv")
    else:
        mp = pd.read_csv(mp_path)
        if mp.shape[0] != 16:
            issues.append(f"matchup_predictions.csv rows={mp.shape[0]} (expected 16)")
        if "home_win_prob" not in mp.columns:
            issues.append("matchup_predictions.csv missing home_win_prob column")
        else:
            p = mp["home_win_prob"].to_numpy(dtype=float)
            if np.isnan(p).any():
                issues.append("matchup_predictions.csv has NaN probabilities")
            if ((p < 0) | (p > 1)).any():
                issues.append("matchup_predictions.csv has probabilities outside [0,1]")
        if {"home_team", "away_team", "predicted_winner"}.issubset(mp.columns):
            bad = mp[~mp["predicted_winner"].isin(mp["home_team"].tolist() + mp["away_team"].tolist())]
            if len(bad) > 0:
                issues.append("matchup_predictions.csv has predicted_winner not in {home_team, away_team}")

    # Line disparity
    ld_path = os.path.join(out_dir, "line_disparity.csv")
    if not os.path.exists(ld_path):
        issues.append("missing output/submission/line_disparity.csv")
    else:
        ld = pd.read_csv(ld_path)
        if ld.shape[0] != 32:
            issues.append(f"line_disparity.csv rows={ld.shape[0]} (expected 32)")
        for col in ("first_adj", "second_adj", "adj_ratio"):
            if col not in ld.columns:
                issues.append(f"line_disparity.csv missing {col} column")
                continue
            v = ld[col].to_numpy(dtype=float)
            if np.isnan(v).any():
                issues.append(f"line_disparity.csv {col} has NaN")
            if np.isinf(v).any():
                issues.append(f"line_disparity.csv {col} has Inf")

    # PNG
    png_name = f"{team_png_basename()}.png"
    png_path = os.path.join(out_dir, png_name)
    if not os.path.exists(png_path):
        issues.append(f"missing output/submission/{png_name} (set TEAM_NAME if needed)")
    else:
        size = os.path.getsize(png_path)
        if size > 5 * 1024 * 1024:
            issues.append(f"PNG too large: {png_name} is {size} bytes (>5MB)")

    # Informational warning: stale placeholder
    placeholder = os.path.join(out_dir, "TeamName.png")
    if os.path.exists(placeholder) and os.path.basename(png_path) != "TeamName.png":
        print(f"WARNING: found {placeholder}; upload {png_name} instead.", file=sys.stderr)

    if issues:
        print("ISSUES:")
        for it in issues:
            print(f"- {it}")
        return 1

    print("OK: submission outputs look sane.")
    print(f"Upload PNG: {png_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

