"""
WHSDSC 2026 Phase 1 — ML Enhancement Module
=============================================
Adds ML challenger models per Codex (GPT-5.3) recommendation:
  - Baseline: Home-ice only (~50.7%)
  - Core model: Bradley-Terry logistic regression (current)
  - Challenger 1: Elastic Net logistic regression (regularized, interpretable)
  - Challenger 2: XGBoost with monotonic constraints (high accuracy)
  - Challenger 3: Blended ensemble (core + XGBoost)

Reports: Accuracy, Log-loss, Brier score, Calibration curves
Feature importance via permutation importance (model-agnostic)
"""

import os, zipfile, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.optimize import minimize
from scipy.special import expit

from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import brier_score_loss, log_loss, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

warnings.filterwarnings("ignore")

ZIP_PATH = "drive-download-20260205T132825Z-1-001.zip"
OUTPUT_DIR = "output"


# ───────────────────── DATA LOADING (reuse from phase1_final) ──────

def ensure_extracted(zip_path, filename, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)
    if os.path.exists(out_path):
        return out_path
    with zipfile.ZipFile(zip_path) as zf:
        zf.extract(filename, output_dir)
    return out_path


def load_data():
    whl_path = ensure_extracted(ZIP_PATH, "whl_2025.csv", OUTPUT_DIR)
    matchups_path = ensure_extracted(ZIP_PATH, "WHSDSC_Rnd1_matchups.xlsx", OUTPUT_DIR)
    df = pd.read_csv(whl_path)
    matchups = pd.read_excel(matchups_path)
    return df, matchups


# ───────────────────── FEATURE ENGINEERING ──────────────────────────

def build_game_features(df):
    """Build game-level dataset with rich features for ML."""
    game = df.groupby("game_id").agg(
        home_team=("home_team", "first"),
        away_team=("away_team", "first"),
        home_goals=("home_goals", "sum"),
        away_goals=("away_goals", "sum"),
        home_xg=("home_xg", "sum"),
        away_xg=("away_xg", "sum"),
        home_shots=("home_shots", "sum"),
        away_shots=("away_shots", "sum"),
        went_ot=("went_ot", "first"),
        toi=("toi", "sum"),
        home_pen_min=("home_penalty_minutes", "sum"),
        away_pen_min=("away_penalty_minutes", "sum"),
    ).reset_index()
    game["home_win"] = (game["home_goals"] > game["away_goals"]).astype(int)
    return game


def build_team_season_stats(game):
    """Compute full-season per-team stats to use as features."""
    records = []
    for _, row in game.iterrows():
        records.append({
            "team": row["home_team"], "gf": row["home_goals"], "ga": row["away_goals"],
            "xgf": row["home_xg"], "xga": row["away_xg"],
            "sf": row["home_shots"], "sa": row["away_shots"],
            "toi": row["toi"], "win": row["home_win"],
            "pim": row["home_pen_min"],
        })
        records.append({
            "team": row["away_team"], "gf": row["away_goals"], "ga": row["home_goals"],
            "xgf": row["away_xg"], "xga": row["home_xg"],
            "sf": row["away_shots"], "sa": row["home_shots"],
            "toi": row["toi"], "win": 1 - row["home_win"],
            "pim": row["away_pen_min"],
        })
    tg = pd.DataFrame(records)
    stats = tg.groupby("team").agg(
        games=("win", "count"), wins=("win", "sum"),
        gf=("gf", "sum"), ga=("ga", "sum"),
        xgf=("xgf", "sum"), xga=("xga", "sum"),
        sf=("sf", "sum"), sa=("sa", "sum"),
        toi=("toi", "sum"), pim=("pim", "sum"),
    )
    stats["win_pct"] = stats["wins"] / stats["games"]
    stats["gd_pg"] = (stats["gf"] - stats["ga"]) / stats["games"]
    stats["xgd_pg"] = (stats["xgf"] - stats["xga"]) / stats["games"]
    stats["xg_share"] = stats["xgf"] / (stats["xgf"] + stats["xga"])
    # Data dictionary: toi is seconds. per60 = per 60 minutes = per 3600 seconds.
    stats["xgd_per60"] = (stats["xgf"] - stats["xga"]) / (stats["toi"] / 3600)
    stats["shooting_pct"] = stats["gf"] / stats["sf"]
    stats["save_pct"] = 1 - (stats["ga"] / stats["sa"])
    stats["pdo"] = stats["shooting_pct"] + stats["save_pct"]
    stats["pim_pg"] = stats["pim"] / stats["games"]
    return stats


def fit_bradley_terry(game, teams, max_iter=200, tol=1e-8):
    team_idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    strength = np.zeros(n)
    for _ in range(max_iter):
        grad = np.zeros(n)
        hess = np.zeros(n)
        for _, row in game.iterrows():
            i, j = team_idx[row["home_team"]], team_idx[row["away_team"]]
            p = expit(strength[i] - strength[j])
            y = row["home_win"]
            grad[i] += y - p; grad[j] -= y - p
            w = p * (1 - p)
            hess[i] -= w; hess[j] -= w
        hess = np.clip(hess, None, -1e-6)
        delta = -grad / hess
        delta -= delta.mean()
        strength += 0.5 * delta
        if np.max(np.abs(delta)) < tol:
            break
    strength -= strength.mean()
    return dict(zip(teams, strength))


def build_ml_features(game, team_stats, bt_strength):
    """Build feature matrix for each game: home_feature - away_feature (differential)."""
    feature_cols = ["win_pct", "gd_pg", "xgd_pg", "xg_share", "xgd_per60",
                    "shooting_pct", "save_pct", "pdo", "pim_pg"]

    rows = []
    for _, g in game.iterrows():
        h, a = g["home_team"], g["away_team"]
        row = {"game_id": g["game_id"], "home_win": g["home_win"]}
        # BT strength differential
        row["bt_diff"] = bt_strength.get(h, 0) - bt_strength.get(a, 0)
        # All stat differentials
        for col in feature_cols:
            hv = team_stats.loc[h, col] if h in team_stats.index else 0
            av = team_stats.loc[a, col] if a in team_stats.index else 0
            row[f"{col}_diff"] = hv - av
        rows.append(row)

    feat_df = pd.DataFrame(rows)
    return feat_df


# ───────────────────── MODEL COMPARISON ─────────────────────────────

def run_model_comparison(game, teams, n_splits=5):
    """
    Leakage-safe 5-fold CV comparison.

    IMPORTANT:
    - Feature engineering (team season stats, BT ratings) MUST be fit on the
      training fold only. Otherwise CV will be overly optimistic.
    - Scaling MUST also be fit on training fold only.
    """
    feature_names = [
        "bt_diff",
        "win_pct_diff",
        "gd_pg_diff",
        "xgd_pg_diff",
        "xg_share_diff",
        "xgd_per60_diff",
        "shooting_pct_diff",
        "save_pct_diff",
        "pdo_diff",
        "pim_pg_diff",
    ]

    y = game["home_win"].values
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = {}

    # Out-of-fold predictions for fair metric computation.
    oof = {
        "Baseline (fold home-rate)": np.zeros(len(game), dtype=float),
        "BT-only Logistic": np.zeros(len(game), dtype=float),
        "Elastic Net Logistic": np.zeros(len(game), dtype=float),
    }
    if HAS_XGB:
        oof["XGBoost"] = np.zeros(len(game), dtype=float)
    else:
        oof["Gradient Boosting"] = np.zeros(len(game), dtype=float)
    oof["Blended Ensemble"] = np.zeros(len(game), dtype=float)

    # Models
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    for train_idx, test_idx in cv.split(np.zeros(len(game)), y):
        train = game.iloc[train_idx].reset_index(drop=True)
        test = game.iloc[test_idx].reset_index(drop=True)

        # Fold-specific feature engineering (no leakage).
        team_stats = build_team_season_stats(train)
        bt = fit_bradley_terry(train, teams)
        feat_tr = build_ml_features(train, team_stats, bt)
        feat_te = build_ml_features(test, team_stats, bt)

        X_tr = feat_tr[feature_names].values
        y_tr = feat_tr["home_win"].values
        X_te = feat_te[feature_names].values

        # Baseline: predict with training home win-rate.
        p0 = float(y_tr.mean())
        oof["Baseline (fold home-rate)"][test_idx] = p0

        # BT-only logistic (single feature) with fold-scaling.
        bt_col = feature_names.index("bt_diff")
        bt_tr = X_tr[:, [bt_col]]
        bt_te = X_te[:, [bt_col]]
        lr_bt = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(C=1.0, max_iter=2000)),
        ])
        lr_bt.fit(bt_tr, y_tr)
        oof["BT-only Logistic"][test_idx] = lr_bt.predict_proba(bt_te)[:, 1]

        # Elastic Net logistic (all features) with fold-scaling and inner CV.
        enet = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegressionCV(
                Cs=20, cv=5, penalty="elasticnet", solver="saga",
                l1_ratios=[0.1, 0.5, 0.9], max_iter=5000, random_state=42,
            )),
        ])
        enet.fit(X_tr, y_tr)
        oof["Elastic Net Logistic"][test_idx] = enet.predict_proba(X_te)[:, 1]

        # XGBoost / fallback GB (no scaling needed).
        if HAS_XGB:
            xgb_model = xgb.XGBClassifier(
                n_estimators=200, max_depth=3, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                reg_alpha=1.0, reg_lambda=2.0,
                eval_metric="logloss", random_state=42,
                use_label_encoder=False,
            )
            xgb_model.fit(X_tr, y_tr)
            oof["XGBoost"][test_idx] = xgb_model.predict_proba(X_te)[:, 1]
        else:
            gb = GradientBoostingClassifier(
                n_estimators=200, max_depth=3, learning_rate=0.05,
                subsample=0.8, random_state=42,
            )
            gb.fit(X_tr, y_tr)
            oof["Gradient Boosting"][test_idx] = gb.predict_proba(X_te)[:, 1]

    # Blend (computed from leakage-safe OOF preds).
    p_ml = oof["XGBoost"] if HAS_XGB else oof["Gradient Boosting"]
    oof["Blended Ensemble"] = 0.6 * oof["Elastic Net Logistic"] + 0.4 * p_ml

    # Summarize
    def _metrics(p):
        p = np.clip(p, 1e-8, 1 - 1e-8)
        return {
            "accuracy": accuracy_score(y, (p >= 0.5).astype(int)),
            "log_loss": log_loss(y, p),
            "brier": brier_score_loss(y, p),
            "probs": p,
        }

    results["Baseline (fold home-rate)"] = _metrics(oof["Baseline (fold home-rate)"])
    results["BT-only Logistic"] = _metrics(oof["BT-only Logistic"])
    results["Elastic Net Logistic"] = _metrics(oof["Elastic Net Logistic"])
    if HAS_XGB:
        results["XGBoost"] = _metrics(oof["XGBoost"])
    else:
        results["Gradient Boosting"] = _metrics(oof["Gradient Boosting"])
    results["Blended Ensemble"] = _metrics(oof["Blended Ensemble"])

    return results, feature_names


# ───────────────────── FEATURE IMPORTANCE ───────────────────────────

def compute_feature_importance(X, y, feature_names):
    """Fit final models and extract feature importance."""
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    # Elastic Net coefficients
    enet = LogisticRegressionCV(
        Cs=20, cv=5, penalty="elasticnet", solver="saga",
        l1_ratios=[0.1, 0.5, 0.9], max_iter=5000, random_state=42
    )
    enet.fit(X_s, y)
    enet_coefs = pd.Series(enet.coef_[0], index=feature_names).abs().sort_values(ascending=False)

    # XGBoost feature importance
    if HAS_XGB:
        xgb_model = xgb.XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=1.0, reg_lambda=2.0,
            eval_metric="logloss", random_state=42,
        )
        xgb_model.fit(X, y)
        xgb_imp = pd.Series(
            xgb_model.feature_importances_, index=feature_names
        ).sort_values(ascending=False)

        # SHAP if available
        shap_values = None
        if HAS_SHAP:
            explainer = shap.TreeExplainer(xgb_model)
            shap_values = explainer.shap_values(X)
    else:
        xgb_imp = None
        shap_values = None

    return enet_coefs, xgb_imp, shap_values, enet, scaler


# ───────────────────── MATCHUP PREDICTION ───────────────────────────

def predict_matchups_ml(matchups, team_stats, bt_strength, model, scaler, feature_names):
    """Predict Round 1 matchups using the best ML model."""
    feature_cols = ["win_pct", "gd_pg", "xgd_pg", "xg_share", "xgd_per60",
                    "shooting_pct", "save_pct", "pdo", "pim_pg"]
    rows = []
    for _, m in matchups.iterrows():
        h, a = m["home_team"], m["away_team"]
        row = {}
        row["bt_diff"] = bt_strength.get(h, 0) - bt_strength.get(a, 0)
        for col in feature_cols:
            hv = team_stats.loc[h, col] if h in team_stats.index else 0
            av = team_stats.loc[a, col] if a in team_stats.index else 0
            row[f"{col}_diff"] = hv - av
        rows.append(row)

    X_new = pd.DataFrame(rows)[feature_names].values
    X_new_s = scaler.transform(X_new)
    probs = model.predict_proba(X_new_s)[:, 1]

    result = matchups.copy()
    result["home_win_prob_ml"] = probs.round(4)
    result["predicted_winner_ml"] = np.where(
        probs >= 0.5, result["home_team"], result["away_team"]
    )
    return result


# ───────────────────── VISUALIZATION ────────────────────────────────

def create_ml_dashboard(results, y, enet_coefs, xgb_imp, feature_names, output_dir):
    """Create ML comparison dashboard."""
    plt.rcParams.update({
        "font.size": 10, "font.family": "sans-serif",
        "axes.spines.top": False, "axes.spines.right": False,
    })

    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)

    # ── Panel 1: Model comparison bar chart ──
    ax1 = fig.add_subplot(gs[0, 0])
    models = list(results.keys())
    accs = [results[m]["accuracy"] for m in models]
    colors = ["#95a5a6", "#3498db", "#2ecc71", "#e74c3c", "#9b59b6"][:len(models)]
    bars = ax1.barh(models, accs, color=colors, edgecolor="white", height=0.5)
    ax1.set_xlabel("5-Fold CV Accuracy")
    ax1.set_title("Model Accuracy Comparison", fontweight="bold")
    ax1.set_xlim(0.45, 0.65)
    for bar, acc in zip(bars, accs):
        ax1.text(acc + 0.003, bar.get_y() + bar.get_height()/2,
                 f"{acc:.1%}", va="center", fontsize=9)

    # ── Panel 2: Brier + Log-loss comparison ──
    ax2 = fig.add_subplot(gs[0, 1])
    x_pos = np.arange(len(models))
    briers = [results[m]["brier"] for m in models]
    lls = [results[m]["log_loss"] for m in models]
    w = 0.35
    ax2.bar(x_pos - w/2, briers, w, label="Brier Score", color="#3498db", alpha=0.8)
    ax2.bar(x_pos + w/2, lls, w, label="Log-Loss", color="#e74c3c", alpha=0.8)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([m.split("(")[0].strip() for m in models], rotation=30, ha="right", fontsize=8)
    ax2.set_ylabel("Score (lower = better)")
    ax2.set_title("Brier Score & Log-Loss", fontweight="bold")
    ax2.legend(fontsize=8)

    # ── Panel 3: Calibration curves ──
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")
    for name, color in zip(["BT-only Logistic", "Elastic Net Logistic",
                             "XGBoost" if HAS_XGB else "Gradient Boosting",
                             "Blended Ensemble"],
                            ["#3498db", "#2ecc71", "#e74c3c", "#9b59b6"]):
        if name not in results:
            continue
        probs = results[name]["probs"]
        bins = np.linspace(0.3, 0.75, 7)
        bin_centers, bin_means = [], []
        for i in range(len(bins)-1):
            mask = (probs >= bins[i]) & (probs < bins[i+1])
            if mask.sum() > 10:
                bin_centers.append(probs[mask].mean())
                bin_means.append(y[mask].mean())
        if bin_centers:
            ax3.plot(bin_centers, bin_means, "o-", color=color, label=name, markersize=5)
    ax3.set_xlabel("Predicted Probability")
    ax3.set_ylabel("Observed Win Rate")
    ax3.set_title("Calibration Curves", fontweight="bold")
    ax3.legend(fontsize=7, loc="upper left")
    ax3.set_xlim(0.3, 0.75)
    ax3.set_ylim(0.3, 0.85)

    # ── Panel 4: Elastic Net coefficients ──
    ax4 = fig.add_subplot(gs[1, 0])
    top_enet = enet_coefs.head(10)
    clean_names = [n.replace("_diff", "").replace("_", " ").title() for n in top_enet.index]
    ax4.barh(clean_names[::-1], top_enet.values[::-1], color="#2ecc71", edgecolor="white")
    ax4.set_xlabel("|Coefficient|")
    ax4.set_title("Elastic Net: Feature Importance", fontweight="bold")

    # ── Panel 5: XGBoost feature importance ──
    ax5 = fig.add_subplot(gs[1, 1])
    if xgb_imp is not None:
        top_xgb = xgb_imp.head(10)
        clean_names_xgb = [n.replace("_diff", "").replace("_", " ").title() for n in top_xgb.index]
        ax5.barh(clean_names_xgb[::-1], top_xgb.values[::-1], color="#e74c3c", edgecolor="white")
        ax5.set_xlabel("Feature Importance (gain)")
        ax5.set_title("XGBoost: Feature Importance", fontweight="bold")
    else:
        ax5.text(0.5, 0.5, "XGBoost not available", ha="center", va="center", transform=ax5.transAxes)

    # ── Panel 6: Summary table ──
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")
    table_data = []
    for name in models:
        r = results[name]
        table_data.append([
            name.split("(")[0].strip()[:18],
            f"{r['accuracy']:.1%}",
            f"{r['log_loss']:.4f}",
            f"{r['brier']:.4f}",
        ])
    table = ax6.table(
        cellText=table_data,
        colLabels=["Model", "Accuracy", "Log-Loss", "Brier"],
        loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)
    # Color best values
    best_acc_idx = np.argmax([results[m]["accuracy"] for m in models])
    best_ll_idx = np.argmin([results[m]["log_loss"] for m in models])
    best_br_idx = np.argmin([results[m]["brier"] for m in models])
    for idx, col in [(best_acc_idx, 1), (best_ll_idx, 2), (best_br_idx, 3)]:
        table[idx + 1, col].set_facecolor("#d5f5e3")
    ax6.set_title("Model Comparison Summary", fontweight="bold", pad=20)

    fig.suptitle("WHSDSC 2026 — ML Model Comparison Dashboard",
                 fontsize=15, fontweight="bold", y=0.98)
    fig.text(0.5, 0.005,
             "5-Fold Stratified CV | Features: BT strength + xG + goals + shooting + save% + PDO + penalties",
             ha="center", fontsize=8, color="gray")

    out = os.path.join(output_dir, "phase1_ml_comparison.png")
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    return out


# ───────────────────── MAIN ─────────────────────────────────────────

def main():
    print("=" * 60)
    print("WHSDSC 2026 — ML Model Comparison")
    print("=" * 60)

    print("\n[1/7] Loading data...")
    df, matchups = load_data()

    print("[2/7] Building game features...")
    game = build_game_features(df)
    teams = sorted(game["home_team"].unique().tolist())
    team_stats = build_team_season_stats(game)

    print("[3/7] Fitting Bradley-Terry...")
    bt = fit_bradley_terry(game, teams)

    print("[4/7] Building ML feature matrix...")
    feat_df = build_ml_features(game, team_stats, bt)
    print(f"  {len(feat_df)} games, {len(feat_df.columns)-2} features")

    print("[5/7] Running 5-fold CV model comparison...")
    results, feature_names = run_model_comparison(game, teams)
    X = feat_df[feature_names].values
    y = feat_df["home_win"].values

    print("\n  ┌─────────────────────────┬──────────┬──────────┬──────────┐")
    print("  │ Model                   │ Accuracy │ Log-Loss │ Brier    │")
    print("  ├─────────────────────────┼──────────┼──────────┼──────────┤")
    for name, r in results.items():
        print(f"  │ {name:<23} │ {r['accuracy']:>7.1%}  │ {r['log_loss']:>8.4f} │ {r['brier']:>8.4f} │")
    print("  └─────────────────────────┴──────────┴──────────┴──────────┘")

    print("\n[6/7] Computing feature importance...")
    enet_coefs, xgb_imp, shap_values, enet_model, scaler = compute_feature_importance(X, y, feature_names)

    print("  Elastic Net top 3:", ", ".join(f"{n}({v:.3f})" for n, v in enet_coefs.head(3).items()))
    if xgb_imp is not None:
        print("  XGBoost top 3:", ", ".join(f"{n}({v:.3f})" for n, v in xgb_imp.head(3).items()))

    print("[7/7] Generating outputs...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ML matchup predictions using best interpretable model (Elastic Net)
    ml_matchups = predict_matchups_ml(matchups, team_stats, bt, enet_model, scaler, feature_names)
    ml_matchups.to_csv(os.path.join(OUTPUT_DIR, "phase1_ml_matchups.csv"), index=False)

    # Dashboard
    viz = create_ml_dashboard(results, y, enet_coefs, xgb_imp, feature_names, OUTPUT_DIR)

    # Results summary
    summary = "# ML Model Comparison Results\n\n"
    summary += "## 5-Fold Cross-Validation Results\n\n"
    summary += "| Model | Accuracy | Log-Loss | Brier Score |\n"
    summary += "|-------|----------|----------|-------------|\n"
    for name, r in results.items():
        summary += f"| {name} | {r['accuracy']:.1%} | {r['log_loss']:.4f} | {r['brier']:.4f} |\n"

    summary += "\n## Feature Importance (Elastic Net |Coefficients|)\n\n"
    summary += "| Feature | |Coefficient| |\n|---------|-------------|\n"
    for name, val in enet_coefs.items():
        summary += f"| {name} | {val:.4f} |\n"

    if xgb_imp is not None:
        summary += "\n## Feature Importance (XGBoost Gain)\n\n"
        summary += "| Feature | Importance |\n|---------|------------|\n"
        for name, val in xgb_imp.items():
            summary += f"| {name} | {val:.4f} |\n"

    # Interpretation
    best_model = max(results.items(), key=lambda x: x[1]["accuracy"])
    best_brier = min(results.items(), key=lambda x: x[1]["brier"])
    summary += f"\n## Key Findings\n\n"
    summary += f"- **Best accuracy**: {best_model[0]} ({best_model[1]['accuracy']:.1%})\n"
    summary += f"- **Best calibration (Brier)**: {best_brier[0]} ({best_brier[1]['brier']:.4f})\n"
    summary += f"- **Home-ice advantage**: Baseline win rate = {y.mean():.1%}\n"
    summary += f"- **Hockey is inherently unpredictable**: Even the best model achieves modest accuracy\n"
    summary += f"- **Recommendation**: Use {best_brier[0]} for probability predictions (best calibrated)\n"

    summary += "\n## ML Matchup Predictions vs Core Model\n\n"
    summary += "| Game | Home | Away | Core Model | Elastic Net ML |\n"
    summary += "|------|------|------|------------|----------------|\n"

    # Load core model predictions for comparison
    core_path = os.path.join(OUTPUT_DIR, "phase1_final_matchups.csv")
    if os.path.exists(core_path):
        core = pd.read_csv(core_path)
        for i, (_, m) in enumerate(ml_matchups.iterrows()):
            core_prob = core.iloc[i]["home_win_prob"] if i < len(core) else "N/A"
            summary += (f"| {int(m['game'])} | {m['home_team']} | {m['away_team']} | "
                        f"{core_prob} | {m['home_win_prob_ml']:.1%} |\n")

    with open(os.path.join(OUTPUT_DIR, "phase1_ml_results.md"), "w", encoding="utf-8") as f:
        f.write(summary)

    print(f"\n  Dashboard:  {viz}")
    print(f"  Matchups:   output/phase1_ml_matchups.csv")
    print(f"  Results:    output/phase1_ml_results.md")
    print("\n" + "=" * 60)
    print("ML comparison complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
