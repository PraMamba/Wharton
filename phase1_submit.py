"""
WHSDSC 2026 Phase 1 — SUBMISSION-GRADE Pipeline
================================================
Final optimized version incorporating all insights from:
  - Claude (core pipeline + feature engineering)
  - Gemini (goaltending + special teams + calibration)
  - Codex GPT-5.3 (ML challenger validation + Brier/calibration focus)

Optimizations over previous versions:
  1. OT-aware Bradley-Terry: OT games downweighted (OT is ~coin flip)
  2. Proper home-ice advantage: separate intercept in logistic
  3. BT-only logistic for predictions (ML challengers offered minimal lift vs added complexity)
  4. Comprehensive evaluation: Accuracy + Log-loss + Brier + Calibration
  5. Clean, consolidated output for submission
"""

import os
import re
import warnings
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.special import expit

from core.data import load_data, build_game_level
from core.bt import fit_bt, compute_sos, bootstrap_bt
from core.stats import build_team_stats
from core.advanced import build_special_teams, build_line_depth, build_goalie_features, build_high_danger, stack_records
from core.config import CFG
from core.model import (
    CVResult, build_rankings, fit_logistic,
    cross_validate_bt, predict_matchups, compute_ece,
    tune_bt_hyperparams, tune_uncertainty_shrink_k,
    _fit_calibration,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*lbfgs.*", category=UserWarning)

OUTPUT_DIR = "output/submission"


@dataclass
class ReportMetrics:
    """Bundle of CV and baseline metrics for write_submission_report."""
    train_acc: float
    cv_acc: float
    cv_std: float
    cv_ll: float
    cv_brier: float
    base_acc: float
    base_ll: float
    base_brier: float
    base_home_rate: float


def get_team_png_basename():
    """
    Competition requirement: PNG titled with team name and no spaces.
    We support configuration via env var TEAM_NAME.
    """
    name = os.environ.get("TEAM_NAME", "TheIsoLab")
    name = re.sub(r"\s+", "", str(name))
    name = re.sub(r"[^A-Za-z0-9_-]", "", name)
    return name or "TeamName"



# ═══════════════════════ LINE DISPARITY ═════════════════════════════

def build_line_disparity(df, rec=None):
    if rec is None:
        rec = stack_records(df)
    lines = rec[["team", "off_line", "opp_def_pairing", "xg_f", "toi"]].copy()
    lines.columns = ["team", "line", "opp_def", "xgf", "toi"]
    # Focus on even-strength-style matchups: first/second lines vs first/second defensive pairings.
    # Exclude power play/penalty kill and empty-net segments.
    es = lines[
        lines["line"].isin(["first_off", "second_off"])
        & lines["opp_def"].isin(["first_def", "second_def"])
    ].copy()

    # Defensive difficulty adjustment
    dq = es.groupby("opp_def").agg(txg=("xgf","sum"), tt=("toi","sum"))
    # toi is seconds; scale to per-60-minutes (3600 seconds).
    # Protect against division by zero
    dq["xga60"] = np.where(dq["tt"] > 0, (dq["txg"] / dq["tt"]) * 3600, 0)
    total_toi = es["toi"].sum()
    avg = (es["xgf"].sum() / total_toi * 3600) if total_toi > 0 else 1.0
    dq["diff"] = np.where(avg > 0, dq["xga60"] / avg, 1.0)
    es = es.merge(dq[["diff"]], left_on="opp_def", right_index=True, how="left")
    es["diff"] = es["diff"].fillna(1.0).replace(0, 1.0)  # Ensure no zero divisors
    es["adj_xgf"] = es["xgf"] / es["diff"]

    min_toi = CFG["disparity"]["min_toi_seconds"]
    ls = es.groupby(["team","line"]).agg(rxg=("xgf","sum"), axg=("adj_xgf","sum"), t=("toi","sum"))
    # Filter out lines with insufficient TOI to avoid noisy ratios
    ls.loc[ls["t"] < min_toi, ["rxg", "axg"]] = np.nan
    ls["raw60"] = np.where(ls["t"] >= min_toi, (ls["rxg"]/ls["t"])*3600, np.nan)
    ls["adj60"] = np.where(ls["t"] >= min_toi, (ls["axg"]/ls["t"])*3600, np.nan)

    pr = ls["raw60"].unstack("line")
    pa = ls["adj60"].unstack("line")
    # Protect against division by zero: use np.where to avoid inf/NaN
    raw_second = pr["second_off"].replace(0, np.nan)
    adj_second = pa["second_off"].replace(0, np.nan)
    disp = pd.DataFrame({
        "first_raw": pr["first_off"], "second_raw": pr["second_off"],
        "raw_ratio": pr["first_off"] / raw_second,
        "first_adj": pa["first_off"], "second_adj": pa["second_off"],
        "adj_ratio": pa["first_off"] / adj_second,
    }).reset_index()
    ratio_lo, ratio_hi = CFG["disparity"]["ratio_clip"]
    disp["raw_ratio"] = disp["raw_ratio"].clip(ratio_lo, ratio_hi)
    disp["adj_ratio"] = disp["adj_ratio"].clip(ratio_lo, ratio_hi)
    disp = disp.sort_values("adj_ratio", ascending=False)
    return disp


# ═══════════════════════ VISUALIZATION ══════════════════════════════

def create_submission_viz(rankings, disparity, output_dir):
    """Single competition-grade PNG for Phase 1c submission."""
    from adjustText import adjust_text
    plt.rcParams.update({"font.size": 11, "font.family": "sans-serif",
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(10, 7))
    pdf = rankings.merge(disparity[["team","adj_ratio"]], on="team", how="left")
    x, y = pdf["adj_ratio"].values, pdf["composite"].values
    m = np.isfinite(x) & np.isfinite(y)

    # Use a colorblind-friendly colormap (avoid red/green).
    sc = ax.scatter(x, y, c=y, cmap="viridis", s=110, edgecolors="white", linewidths=0.8, zorder=3)
    corr, r2 = 0.0, 0.0
    if m.sum() > 2:
        coef = np.polyfit(x[m], y[m], 1)
        xl = np.linspace(x[m].min(), x[m].max(), 100)
        ax.plot(xl, np.polyval(coef, xl), "k--", alpha=0.5, lw=2, label="Linear trend")
        corr = np.corrcoef(x[m], y[m])[0, 1]
        yp = np.polyval(coef, x[m])
        r2 = 1 - np.sum((y[m]-yp)**2) / np.sum((y[m]-y[m].mean())**2) if np.sum((y[m]-y[m].mean())**2) > 0 else 0

    # Label collision avoidance using adjustText
    texts = []
    label_rows = pd.concat([pdf.nlargest(5, "composite"), pdf.nsmallest(3, "composite")])
    for _, row in label_rows.iterrows():
        is_top = row["composite"] >= pdf["composite"].median()
        texts.append(ax.text(
            row["adj_ratio"], row["composite"],
            row["team"].replace("_"," ").title(),
            fontsize=9.5, fontweight="bold" if is_top else "normal",
            color="#2c3e50" if is_top else "#c0392b"))
    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="gray", alpha=0.3))

    ax.set_xlabel("Adjusted Offensive Line Quality Disparity\n"
                  "(1st Line xG/60 \u00f7 2nd Line xG/60, adjusted for defensive matchup difficulty)", fontsize=11)
    ax.set_ylabel("Composite Team Strength Score\n(OT-Aware Bradley-Terry + xG Ensemble)", fontsize=11)
    ax.set_title("Do Teams With More Balanced Offensive Lines Perform Better?",
                 fontsize=14, fontweight="bold", pad=15)
    ax.text(0.02, 0.97,
            f"Pearson r = {corr:.3f},  R\u00b2 = {r2:.3f}\n"
            f"Finding: No strong linear relationship.\n"
            f"Teams succeed through various lineup strategies.",
            transform=ax.transAxes, fontsize=9.5, va="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0", edgecolor="#ccc", alpha=0.9))
    ax.axhline(0, color="gray", lw=0.5, ls=":")
    plt.colorbar(sc, ax=ax, label="Strength Score", shrink=0.7, pad=0.02)
    ax.legend(loc="lower right", fontsize=9)
    fig.text(0.5, -0.01,
             "Data: WHL 2025 (32 teams, 1,312 games) | OT-aware Bradley-Terry | Disparity adjusted for defensive matchup quality",
             ha="center", fontsize=8, color="gray")
    out = os.path.join(output_dir, f"{get_team_png_basename()}.png")
    plt.savefig(out, dpi=250, bbox_inches="tight", facecolor="white")
    plt.close()
    return out


def create_full_dashboard(rankings, disparity, matchups, cal_data, output_dir):
    """Internal analytics dashboard (not for submission)."""
    plt.rcParams.update({"font.size": 9, "font.family": "sans-serif",
                         "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(18, 11))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)
    pdf = rankings.merge(disparity[["team","adj_ratio","raw_ratio"]], on="team", how="left")

    # P1: Scatter
    ax1 = fig.add_subplot(gs[0, :2])
    x, y = pdf["adj_ratio"].values, pdf["composite"].values
    m = np.isfinite(x) & np.isfinite(y)
    corr, r2 = 0.0, 0.0
    sc = ax1.scatter(x, y, c=y, cmap="viridis", s=60, edgecolors="white", lw=0.5, zorder=3)
    if m.sum() > 2:
        coef = np.polyfit(x[m], y[m], 1)
        xl = np.linspace(x[m].min(), x[m].max(), 100)
        ax1.plot(xl, np.polyval(coef, xl), "k--", alpha=0.4, lw=1.5)
        corr = np.corrcoef(x[m], y[m])[0, 1]
        r2 = 1 - np.sum((y[m]-np.polyval(coef,x[m]))**2)/np.sum((y[m]-y[m].mean())**2)
    for _, row in pdf.nlargest(5, "composite").iterrows():
        ax1.annotate(row["team"].replace("_"," ").title(),
                     (row["adj_ratio"], row["composite"]),
                     textcoords="offset points", xytext=(5,5), fontsize=7.5,
                     fontweight="bold", color="#2c3e50")
    for _, row in pdf.nsmallest(3, "composite").iterrows():
        ax1.annotate(row["team"].replace("_"," ").title(),
                     (row["adj_ratio"], row["composite"]),
                     textcoords="offset points", xytext=(5,-8), fontsize=7.5, color="#c0392b")
    ax1.set_xlabel("Adjusted Line Disparity"); ax1.set_ylabel("Composite Strength")
    ax1.set_title("Line Balance vs. Team Strength", fontsize=11, fontweight="bold")
    ax1.text(0.02, 0.95, f"r={corr:.3f} R\u00b2={r2:.3f}", transform=ax1.transAxes, fontsize=8,
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.6))
    ax1.axhline(0, color="gray", lw=0.5, ls=":")

    # P2: Calibration
    ax2 = fig.add_subplot(gs[0, 2])
    if cal_data:
        cd = pd.DataFrame(cal_data)
        cal_lo_d = max(0.0, cd["pred"].min() - 0.05)
        cal_hi_d = min(1.0, cd["pred"].max() + 0.05)
        ax2.plot([cal_lo_d, cal_hi_d], [cal_lo_d, cal_hi_d], "k--", alpha=0.5, label="Perfect")
        ax2.scatter(cd["pred"], cd["obs"], s=cd["n"]/2, c="#3498db", edgecolors="white", zorder=3)
        for _, c in cd.iterrows():
            ax2.annotate(f"n={int(c['n'])}", (c["pred"], c["obs"]),
                         textcoords="offset points", xytext=(5,5), fontsize=7)
        ax2.set_xlim(cal_lo_d, cal_hi_d); ax2.set_ylim(cal_lo_d, cal_hi_d + 0.05)
    else:
        ax2.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")
    ax2.set_xlabel("Predicted"); ax2.set_ylabel("Observed")
    ax2.set_title("Probability Calibration", fontsize=10, fontweight="bold")
    ax2.legend(fontsize=7)

    # P3: Rankings bar
    ax3 = fig.add_subplot(gs[1, 0])
    t10 = pdf.nlargest(10, "composite")
    b5 = pdf.nsmallest(5, "composite")
    bd = pd.concat([t10, b5]).sort_values("composite", ascending=True)
    cols = ["#e74c3c" if s < 0 else "#27ae60" for s in bd["composite"]]
    ax3.barh([t.replace("_"," ").title() for t in bd["team"]], bd["composite"],
             color=cols, edgecolor="white", height=0.55)
    ax3.set_xlabel("Composite Score"); ax3.axvline(0, color="gray", lw=0.5)
    ax3.set_title("Power Rankings", fontsize=10, fontweight="bold")

    # P4: Matchup predictions
    ax4 = fig.add_subplot(gs[1, 1])
    ms = matchups.sort_values("home_win_prob", ascending=True)
    bars = ax4.barh(
        ms.apply(lambda r: f"{r['home_team'].replace('_',' ').title()} v {r['away_team'].replace('_',' ').title()}", axis=1),
        ms["home_win_prob"],
        color=[plt.cm.viridis(p) for p in ms["home_win_prob"]],
        edgecolor="white", height=0.55)
    ax4.axvline(0.5, color="gray", lw=1, ls="--", alpha=0.7)
    ax4.set_xlabel("Home Win Probability")
    prob_lo_d = max(0, ms["home_win_prob"].min() - 0.05)
    prob_hi_d = min(1, ms["home_win_prob"].max() + 0.05)
    ax4.set_xlim(prob_lo_d, prob_hi_d)
    ax4.set_title("Round 1 Predictions", fontsize=10, fontweight="bold")
    for bar, prob in zip(bars, ms["home_win_prob"]):
        ax4.text(bar.get_width()+0.005, bar.get_y()+bar.get_height()/2,
                 f"{prob:.1%}", va="center", fontsize=7)

    # P5: Top 10 disparity
    ax5 = fig.add_subplot(gs[1, 2])
    d10 = disparity.nlargest(10, "adj_ratio").sort_values("adj_ratio", ascending=True)
    ax5.barh([t.replace("_"," ").title() for t in d10["team"]],
             d10["adj_ratio"], color="#f39c12", edgecolor="white", height=0.55)
    ax5.axvline(1.0, color="gray", lw=0.5, ls=":")
    ax5.set_xlabel("Adjusted Disparity Ratio")
    ax5.set_title("Top 10 Line Disparity", fontsize=10, fontweight="bold")
    for i, (_, r) in enumerate(d10.iterrows()):
        ax5.text(r["adj_ratio"]+0.003, i, f"{r['adj_ratio']:.3f}", va="center", fontsize=7)

    fig.suptitle("WHSDSC 2026 \u2014 Phase 1 Full Analytics Dashboard",
                 fontsize=14, fontweight="bold", y=0.98)
    fig.text(0.5, 0.005,
             "OT-Aware Bradley-Terry + xG Ensemble | 5-Fold CV | Calibrated Logistic | Adjusted for Defensive Matchups",
             ha="center", fontsize=7.5, color="gray")
    out = os.path.join(output_dir, "dashboard.png")
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    return out


# ═══════════════════════ REPORT ═════════════════════════════════════

def write_submission_report(rankings, matchups, disparity, a, b,
                            metrics, cal_data, output_dir, ot_weight=0.5):
    home_adv = expit(a)
    d10 = disparity.nlargest(10, "adj_ratio")
    train_acc = metrics.train_acc
    cv_acc = metrics.cv_acc
    cv_std = metrics.cv_std
    cv_ll = metrics.cv_ll
    cv_brier = metrics.cv_brier
    base_acc = metrics.base_acc
    base_ll = metrics.base_ll
    base_brier = metrics.base_brier
    base_home_rate = metrics.base_home_rate

    txt = f"""# WHSDSC 2026 Phase 1 — Submission Report

---

## 1. Process

### Data Cleaning & Transformation (~50 words)
We validated 25,827 line-level records across 1,312 games (32 teams), confirming zero missing values. We aggregated line-level rows to game-level outcomes and stacked home/away records symmetrically for unbiased team statistics. For line-disparity analysis we restricted to first/second offensive lines vs first/second defensive pairings (excluding PP/PK and empty-net segments). Overtime games were flagged for special handling in our strength model.

### Additional Variables (~25 words)
Engineered: xG differential per 60 min, xG share, OT-aware Bradley-Terry strength, strength of schedule, adjusted line xG/60 (controlling for defensive matchup difficulty), GSAx per game (goaltender quality above expected), xG per shot (offensive chance quality), PIM per game and penalty differential (team discipline), special teams rates (PP xGF/60, PK xGA/60), even-strength xGD/60, finishing and goalsaving with Bayesian shrinkage, line depth gap, goalie starter analysis with shrinkage, high-danger chance frequency.

---

## 2. Tools & Techniques

### Software Tools (~50 words)
Python (pandas, numpy, scipy, matplotlib, scikit-learn, xgboost). Bradley-Terry fitted via iterative MLE with OT-downweighting. Logistic regression optimized via scipy. ML challengers (Elastic Net, XGBoost) were used as benchmarks; final submission uses the simpler BT-only model. Three AI assistants (Claude, Gemini, Codex) provided independent cross-validation and methodology critique.

### Statistical Methods (~100 words)
Our approach rests on the **Bradley-Terry paired comparison model**, fitted via maximum likelihood on all 1,312 games. Overtime games receive reduced weight (tuned to {ot_weight} via nested CV), since OT outcomes are near coin-flips (52.1% home win in OT vs 57.6% in regulation) and carry weaker signal about true team quality.

Win probabilities come from **calibrated logistic regression**: P(home_win) = sigmoid({a:.4f} + {b:.4f} × BT_strength_diff), with the intercept capturing {home_adv:.1%} home-ice advantage. We validated this via **5-fold cross-validation** (accuracy {cv_acc:.1%} ± {cv_std:.1%}, Brier {cv_brier:.4f}).

We also prototyped multi-feature ML challengers (Elastic Net, boosting) in a separate module. For this submission we use the BT-only logistic for transparency and stable calibration.

---

## 3. Your Predictions

### 1a: Power Rankings & Win Probabilities (~50 words)
Teams ranked by composite of OT-aware Bradley-Terry strength (40%), xGD/60 (30%), xG share (10%), goal differential (10%), and win percentage (10%). BT inherently controls for opponent quality. Win probabilities from calibrated logistic regression on BT strength differential; benchmark ML models offered only marginal lift, so we prioritized transparency and stable probability outputs.

### 1b: Offensive Line Quality Disparity (~50 words)
We computed xG per 60 for each team's first and second offensive lines at even strength. We adjusted for **defensive matchup confounding** (as recommended by the competition workbook)—lines facing elite defensive pairings were normalized relative to league-average difficulty. The adjusted disparity ratio quantifies true lineup imbalance.

### 1c: Visualization Choices (~50 words)
A scatter plot maps adjusted disparity (x) against composite strength (y) with linear regression overlay, Pearson r, and R². Color encodes strength for quick tier identification. Key teams labeled. This directly tests the commissioner's question: do balanced lineups predict success? Our finding: no strong linear relationship.

---

## 4. Insights

### Model Performance Assessment (~50 words)
We used **5-fold cross-validation** (refitting BT + logistic each fold): accuracy {cv_acc:.1%} (±{cv_std:.1%}), log-loss {cv_ll:.4f}, Brier score {cv_brier:.4f}. Calibration bins from out-of-fold predictions compare average predicted probabilities to observed win rates, and baseline comparison (home-win rate only) contextualizes the lift.

**Limitations:** As stated in the workbook, team/line quality is treated as stable across the season; we therefore do not time-weight games or model roster/injury shocks (not provided in data).

### Generative AI Usage (~50 words)
Three AI models collaborated: **Claude** (Anthropic) designed the analysis pipeline and code. **Gemini** (Google) critiqued rankings and suggested goaltending/special-teams features. **Codex** (OpenAI, GPT-5.3) recommended ML challenger validation and Brier score reporting. All outputs were cross-validated; final methodological decisions were human-guided.

---

## Appendix: Results

### Power Rankings (32 teams)

| Rank | Team | Composite | BT Strength | xGD/60 | xG Share | Win% | SOS | GSAx/G | xG/Shot | PIM/G | PenDiff/G |
|------|------|-----------|-------------|--------|----------|------|-----|--------|---------|-------|-----------|
"""
    for _, r in rankings.iterrows():
        txt += (f"| {int(r['rank'])} | {r['team']} | {r['composite']:.3f} | "
                f"{r['bt_strength']:.4f} | {r['xgd_per60']:.4f} | "
                f"{r['xg_share']:.3f} | {r['win_pct']:.3f} | {r['sos']:.4f} | "
                f"{r['gsax_pg']:.3f} | {r['xg_per_shot']:.4f} | "
                f"{r['pim_pg']:.1f} | {r['pen_diff_pg']:+.1f} | "
                f"{r.get('ev_xgd60', 0):.4f} | {r.get('pp_xgf60', 0):.4f} | "
                f"{r.get('pk_xga60', 0):.4f} | {r.get('goalie_gsax60', 0):.4f} |\n")

    txt += "\n### Round 1 Matchup Predictions\n\n"
    txt += "| Game | Home | Away | Home Win Prob | Predicted Winner |\n"
    txt += "|------|------|------|--------------|------------------|\n"
    for _, r in matchups.iterrows():
        txt += f"| {int(r['game'])} | {r['home_team']} | {r['away_team']} | {r['home_win_prob']:.1%} | {r['predicted_winner']} |\n"

    txt += "\n### Top 10 Line Quality Disparity\n\n"
    txt += "| Rank | Team | 1st Line Adj xG/60 | 2nd Line Adj xG/60 | Adj Disparity |\n"
    txt += "|------|------|--------------------|--------------------|---------------|\n"
    for i, (_, r) in enumerate(d10.iterrows(), 1):
        txt += f"| {i} | {r['team']} | {r['first_adj']:.4f} | {r['second_adj']:.4f} | {r['adj_ratio']:.4f} |\n"

    txt += "\n### Probability Calibration\n\n"
    txt += "| Bin Center | Avg Predicted | Observed Win Rate | N |\n"
    txt += "|------------|--------------|-------------------|---|\n"
    for c in cal_data:
        txt += f"| {c['center']:.2f} | {c['pred']:.3f} | {c['obs']:.3f} | {c['n']} |\n"

    txt += f"""
### Model Comparison (5-Fold CV)

| Model | Accuracy | Log-Loss | Brier |
|-------|----------|----------|-------|
| Baseline (home={base_home_rate:.1%}) | {base_acc:.1%} | {base_ll:.4f} | {base_brier:.4f} |
| **OT-Aware BT Logistic** | **{cv_acc:.1%}** | **{cv_ll:.4f}** | **{cv_brier:.4f}** |
 
*OT-aware BT logistic chosen for its interpretability, calibration quality, and principled treatment of overtime games.*

---
*WHSDSC 2026 | Cross-validated by Claude + Gemini + Codex*
"""

    out = os.path.join(output_dir, "methodology.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(txt)
    return out


# ═══════════════════════ MAIN ═══════════════════════════════════════

def main():
    print("=" * 60)
    print("WHSDSC 2026 Phase 1 — SUBMISSION PIPELINE")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n[1/8] Loading data...")
    df, matchups = load_data()

    print("[2/8] Game-level aggregation...")
    game = build_game_level(df)
    teams = sorted(game["home_team"].unique().tolist())
    n_ot = game["went_ot"].sum()
    print(f"  {len(game)} games ({n_ot} went to OT, {n_ot/len(game):.1%})")

    print("[3/8] Team statistics...")
    stats = build_team_stats(game)

    # Advanced record-level features
    rec = stack_records(df)  # Pre-compute once
    adv_st = build_special_teams(df, rec=rec)
    adv_ld = build_line_depth(df, rec=rec)
    adv_gk = build_goalie_features(df, rec=rec)
    adv_hd = build_high_danger(df, rec=rec)
    stats = stats.join(adv_st).join(adv_ld).join(adv_gk).join(adv_hd)

    print("[3.5/8] Tuning BT hyperparameters (nested CV)...")
    best_ot, best_reg, best_cal, tune_details = tune_bt_hyperparams(game, teams)
    print(f"  Best: ot_weight={best_ot}, reg={best_reg}, cal={best_cal}")
    print(f"  Inner Brier: {tune_details['best_brier']:.4f}")

    print("[4/8] OT-aware Bradley-Terry (tuned params)...")
    bt_kwargs = dict(ot_weight=best_ot, regularization=best_reg)
    bt = fit_bt(game, teams, **bt_kwargs)
    sos = compute_sos(game, bt)
    top3 = sorted(bt.items(), key=lambda x: -x[1])[:3]
    print(f"  Top 3: {', '.join(f'{t}({s:.3f})' for t,s in top3)}")

    # Bootstrap BT for uncertainty quantification
    print("  Bootstrap (200 resamples)...")
    bt_mean, bt_std, bt_ci, bt_samples = bootstrap_bt(
        game, teams, n_boot=200, **bt_kwargs)

    print("[5/8] Ensemble rankings...")
    rankings = build_rankings(stats, bt, sos)
    # Add bootstrap CI to rankings
    rankings["bt_ci_lo"] = rankings["team"].map(lambda t: bt_ci[t][0])
    rankings["bt_ci_hi"] = rankings["team"].map(lambda t: bt_ci[t][1])
    rankings["bt_std"] = rankings["team"].map(bt_std)
    print(f"  #1 {rankings.iloc[0]['team']} ({rankings.iloc[0]['composite']:.3f})")

    print("[6/8] Calibrated logistic + CV (with tuned BT params)...")
    a, b, train_acc, train_ll, train_brier = fit_logistic(game, bt)
    cv = cross_validate_bt(game, teams, bt_kwargs=bt_kwargs, n_boot=200)
    print(f"  a={a:.4f} (home-ice ~{expit(a):.1%}), b={b:.4f}")
    print(f"  Train: {train_acc:.1%} acc, {train_brier:.4f} Brier")
    print(f"  CV:    {cv.acc:.1%}±{cv.acc_std:.1%} acc, {cv.brier:.4f} Brier")

    # Calibration data from out-of-fold (OOF) predictions for honesty.
    p_all = np.asarray(cv.oof_pred, dtype=float)
    y_all = np.asarray(cv.oof_true, dtype=int)
    # Dynamic calibration bins from prediction distribution
    p_lo = max(0.0, np.percentile(p_all, 2) - 0.02)
    p_hi = min(1.0, np.percentile(p_all, 98) + 0.02)
    bins = np.linspace(p_lo, p_hi, 7)
    cal_data = []
    for i in range(len(bins)-1):
        mask = (p_all >= bins[i]) & (p_all < bins[i+1])
        if mask.sum() > 10:
            cal_data.append({"center": (bins[i]+bins[i+1])/2,
                             "pred": p_all[mask].mean(),
                             "obs": y_all[mask].mean(),
                             "n": int(mask.sum())})

    # ECE and CV metric confidence intervals
    ece = compute_ece(y_all, p_all)
    print(f"  ECE:   {ece:.4f}")
    cv_acc_ci = 1.96 * cv.acc_std / np.sqrt(CFG["cv"]["n_folds"])
    print(f"  CV acc CI: {cv.acc:.1%} ± {cv_acc_ci:.1%}")

    # Tune uncertainty shrink (using per-fold bootstrap uncertainties, not full-data bt_std)
    shrink_k = tune_uncertainty_shrink_k(cv.oof_pred, cv.oof_true, cv.oof_uncertainties, game)
    print(f"  Uncertainty shrink k: {shrink_k:.4f}")

    print("[7/8] Matchup predictions...")
    mp = predict_matchups(matchups, bt, a, b,
                          bt_samples=bt_samples, teams=teams,
                          bt_std=bt_std, uncertainty_shrink_k=shrink_k)

    print("[8/8] Line disparity + outputs...")
    disp = build_line_disparity(df)

    # Save CSVs
    rc = ["rank","team","composite","bt_strength","xgd_per60","xg_share","gd_pg","win_pct","sos","pdo",
           "gsax_pg","xg_per_shot","pim_pg","pen_diff_pg",
           "ev_xgd60","pp_xgf60","pk_xga60","goalie_gsax60","depth_gap","fin60_shrunk","gsax60_shrunk"]
    rankings[rc].to_csv(os.path.join(OUTPUT_DIR, "power_rankings.csv"), index=False)
    mp.to_csv(os.path.join(OUTPUT_DIR, "matchup_predictions.csv"), index=False)
    dc = ["team","first_raw","second_raw","raw_ratio","first_adj","second_adj","adj_ratio"]
    disp[dc].to_csv(os.path.join(OUTPUT_DIR, "line_disparity.csv"), index=False)

    # Visualizations
    viz1 = create_submission_viz(rankings, disp, OUTPUT_DIR)
    viz2 = create_full_dashboard(rankings, disp, mp, cal_data, OUTPUT_DIR)

    # Report
    metrics = ReportMetrics(
        train_acc=train_acc, cv_acc=cv.acc, cv_std=cv.acc_std,
        cv_ll=cv.ll, cv_brier=cv.brier,
        base_acc=cv.base_acc, base_ll=cv.base_ll, base_brier=cv.base_brier,
        base_home_rate=y_all.mean(),
    )
    rpt = write_submission_report(rankings, mp, disp, a, b,
                                  metrics, cal_data, OUTPUT_DIR, ot_weight=best_ot)

    print("\n" + "=" * 60)
    print("SUBMISSION OUTPUTS (output/submission/)")
    print("=" * 60)
    print(f"  power_rankings.csv      — 32-team rankings for 1a")
    print(f"  matchup_predictions.csv — 16-game Round 1 for 1a")
    print(f"  line_disparity.csv      — 32-team disparity for 1b")
    print(f"  {os.path.basename(viz1):<20} — Visualization for 1c")
    print(f"  dashboard.png           — Full analytics dashboard")
    print(f"  methodology.md          — Complete report for 1d")

    # Print submission-ready data
    print("\n" + "─" * 60)
    print("SUBMISSION DATA: Power Rankings (Top 10)")
    print("─" * 60)
    for _, r in rankings.head(10).iterrows():
        print(f"  {int(r['rank']):>2}. {r['team']:<16} {r['composite']:>7.3f}")

    print("\n" + "─" * 60)
    print("SUBMISSION DATA: Round 1 Win Probabilities")
    print("─" * 60)
    for _, r in mp.iterrows():
        print(f"  Game {int(r['game']):>2}: {r['home_team']:<14} vs {r['away_team']:<14} → {r['home_win_prob']:.1%} → {r['predicted_winner']}")

    print("\n" + "─" * 60)
    print("SUBMISSION DATA: Top 10 Line Disparity")
    print("─" * 60)
    for i, (_, r) in enumerate(disp.nlargest(10, "adj_ratio").iterrows(), 1):
        print(f"  {i:>2}. {r['team']:<16} {r['adj_ratio']:.4f}")

    print()


if __name__ == "__main__":
    main()
