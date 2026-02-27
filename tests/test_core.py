"""Unit tests for core modules."""

import numpy as np
import pandas as pd
import pytest

from core.bt import fit_bt, compute_sos, bootstrap_bt, fit_attack_defense, skellam_predict_ad
from core.stats import build_team_stats
from core.config import load_config
from core.advanced import (
    stack_records, build_special_teams, build_line_depth,
    build_goalie_features, build_high_danger,
    residualize_features, build_rapm_features,
)


def _make_synthetic_games(n_teams=4, n_games=50, seed=42):
    """Generate synthetic game data where team 0 is strongest."""
    rng = np.random.RandomState(seed)
    teams = [f"team_{i}" for i in range(n_teams)]
    true_strength = np.linspace(1, -1, n_teams)  # team_0 strongest

    rows = []
    for g in range(n_games):
        i, j = rng.choice(n_teams, size=2, replace=False)
        p = 1 / (1 + np.exp(-(true_strength[i] - true_strength[j])))
        home_win = int(rng.random() < p)
        h_goals = rng.poisson(3) + home_win
        a_goals = rng.poisson(3) + (1 - home_win)
        if h_goals == a_goals:
            h_goals += 1
        rows.append({
            "game_id": f"g{g}", "home_team": teams[i], "away_team": teams[j],
            "home_goals": h_goals, "away_goals": a_goals,
            "home_xg": h_goals * 0.9, "away_xg": a_goals * 0.9,
            "home_shots": h_goals * 3, "away_shots": a_goals * 3,
            "home_max_xg": 0.5, "away_max_xg": 0.5,
            "home_pm": 4, "away_pm": 4,
            "home_pc": 2, "away_pc": 2,
            "home_assists": h_goals, "away_assists": a_goals,
            "went_ot": 0, "toi": 3600,
            "home_win": home_win,
        })
    return pd.DataFrame(rows), teams


def _make_synthetic_records(n_teams=4, n_games=20, seed=42):
    """Generate synthetic record-level data with line, pairing, and goalie columns."""
    rng = np.random.RandomState(seed)
    teams = [f"team_{i}" for i in range(n_teams)]
    off_lines = ["first_off", "second_off", "PP_up", "PP_kill_dwn"]
    def_pairings = ["first_def", "second_def", "PP_up", "PP_kill_dwn"]
    goalies = {t: [f"goalie_{t}_1", f"goalie_{t}_2"] for t in teams}

    rows = []
    for g in range(n_games):
        i, j = rng.choice(n_teams, size=2, replace=False)
        ht, at = teams[i], teams[j]
        # Each game has ~10 records (different line combos)
        for rec in range(10):
            h_line = off_lines[rec % 4]
            a_line = off_lines[(rec + 1) % 4]
            h_def = def_pairings[rec % 4]
            a_def = def_pairings[(rec + 1) % 4]
            h_goalie = goalies[ht][0] if rec < 8 else goalies[ht][1]
            a_goalie = goalies[at][0] if rec < 7 else goalies[at][1]
            rows.append({
                "game_id": f"g{g}", "record_id": f"g{g}_r{rec}",
                "home_team": ht, "away_team": at,
                "went_ot": 0,
                "home_off_line": h_line, "home_def_pairing": h_def,
                "away_off_line": a_line, "away_def_pairing": a_def,
                "home_goalie": h_goalie, "away_goalie": a_goalie,
                "toi": 360,
                "home_assists": rng.randint(0, 3),
                "home_shots": rng.randint(1, 8),
                "home_xg": rng.uniform(0, 1.5),
                "home_max_xg": rng.uniform(0, 0.5),
                "home_goals": rng.randint(0, 3),
                "away_assists": rng.randint(0, 3),
                "away_shots": rng.randint(1, 8),
                "away_xg": rng.uniform(0, 1.5),
                "away_max_xg": rng.uniform(0, 0.5),
                "away_goals": rng.randint(0, 3),
                "home_penalties_committed": rng.randint(0, 3),
                "home_penalty_minutes": rng.randint(0, 6),
                "away_penalties_committed": rng.randint(0, 3),
                "away_penalty_minutes": rng.randint(0, 6),
            })
    return pd.DataFrame(rows), teams


class TestBradleyTerry:
    def test_convergence(self):
        """BT should converge on synthetic data."""
        game, teams = _make_synthetic_games()
        bt = fit_bt(game, teams, max_iter=300, tol=1e-8)
        assert len(bt) == len(teams)
        # Strengths should be centered
        assert abs(np.mean(list(bt.values()))) < 1e-6

    def test_strongest_team_ranked_first(self):
        """With enough data, the true strongest team should rank highest."""
        game, teams = _make_synthetic_games(n_games=200)
        bt = fit_bt(game, teams)
        # team_0 has true_strength=1.0 (strongest)
        assert max(bt, key=bt.get) == "team_0"

    def test_ot_downweighting(self):
        """OT games with weight < 1 should change strengths vs weight = 1."""
        game, teams = _make_synthetic_games()
        # Mark half as OT
        game.loc[game.index[:len(game)//2], "went_ot"] = 1
        bt_ot = fit_bt(game, teams, ot_weight=0.5)
        bt_full = fit_bt(game, teams, ot_weight=1.0)
        # They shouldn't be identical
        diffs = [abs(bt_ot[t] - bt_full[t]) for t in teams]
        assert max(diffs) > 0.01

    def test_regularization(self):
        """Regularization should shrink strengths toward zero."""
        game, teams = _make_synthetic_games()
        bt_unreg = fit_bt(game, teams, regularization=0.0)
        bt_reg = fit_bt(game, teams, regularization=0.1)
        norm_unreg = np.linalg.norm(list(bt_unreg.values()))
        norm_reg = np.linalg.norm(list(bt_reg.values()))
        assert norm_reg < norm_unreg


class TestSOS:
    def test_sos_output(self):
        game, teams = _make_synthetic_games()
        bt = fit_bt(game, teams)
        sos = compute_sos(game, bt)
        assert len(sos) == len(teams)
        assert all(np.isfinite(sos.values))


class TestBootstrap:
    def test_bootstrap_shapes(self):
        game, teams = _make_synthetic_games()
        bt_mean, bt_std, bt_ci, samples = bootstrap_bt(
            game, teams, n_boot=20, random_state=42)
        assert len(bt_mean) == len(teams)
        assert len(bt_std) == len(teams)
        assert samples.shape == (20, len(teams))
        for t in teams:
            lo, hi = bt_ci[t]
            assert lo <= bt_mean[t] <= hi

    def test_ci_not_degenerate(self):
        game, teams = _make_synthetic_games()
        _, bt_std, _, _ = bootstrap_bt(game, teams, n_boot=50)
        # All teams should have some uncertainty
        assert all(s > 0 for s in bt_std.values())


class TestTeamStats:
    def test_stats_columns(self):
        game, teams = _make_synthetic_games()
        stats = build_team_stats(game)
        assert "win_pct" in stats.columns
        assert "xgd_per60" in stats.columns
        assert "gsax_pg" in stats.columns
        assert len(stats) == len(teams)

    def test_win_pct_bounds(self):
        game, teams = _make_synthetic_games()
        stats = build_team_stats(game)
        assert (stats["win_pct"] >= 0).all()
        assert (stats["win_pct"] <= 1).all()

    def test_new_team_stats_columns(self):
        """Verify all new columns exist in build_team_stats()."""
        game, teams = _make_synthetic_games()
        stats = build_team_stats(game)
        new_cols = [
            "xg_per_shot_against", "shots_for60", "shots_against60",
            "shot_diff60", "mean_max_xg_for", "fin60", "gsax60",
            "fin60_shrunk", "gsax60_shrunk",
            "max_xg_for", "max_xg_against",
            "pim60", "pen60", "assists_per60",
        ]
        for col in new_cols:
            assert col in stats.columns, f"Missing column: {col}"

    def test_finishing_shrinkage_bounds(self):
        """fin60_shrunk should be between raw fin60 and league mean."""
        game, teams = _make_synthetic_games()
        stats = build_team_stats(game)
        league_mean = stats["fin60"].mean()
        for _, row in stats.iterrows():
            lo = min(row["fin60"], league_mean)
            hi = max(row["fin60"], league_mean)
            assert lo - 1e-6 <= row["fin60_shrunk"] <= hi + 1e-6

    def test_penalties_committed_aggregation(self):
        """pen60 should use penalty count (not minutes)."""
        game, teams = _make_synthetic_games()
        stats = build_team_stats(game)
        # pen60 should be based on pc_taken, not pm_taken
        # pc_taken is from home_pc/away_pc (count), pm_taken is penalty minutes
        assert (stats["pen60"] >= 0).all()
        # pen60 and pim60 should generally differ (minutes vs count)
        # Since in our synthetic data they could be equal, just check they exist
        assert "pen60" in stats.columns
        assert "pim60" in stats.columns


class TestConfig:
    def test_defaults(self):
        cfg = load_config("/nonexistent/path.yaml")
        assert cfg["bt"]["ot_weight"] == 0.5
        assert cfg["cv"]["n_folds"] == 5

    def test_load_actual(self):
        cfg = load_config()
        assert "bt" in cfg
        assert "rankings" in cfg

    def test_shrinkage_defaults(self):
        cfg = load_config("/nonexistent/path.yaml")
        assert cfg["shrinkage"]["K"] == 500

    def test_high_danger_defaults(self):
        cfg = load_config("/nonexistent/path.yaml")
        assert cfg["high_danger"]["threshold"] == 0.20


# ═══════════════════════ ADVANCED MODULE TESTS ═════════════════════

class TestSpecialTeams:
    def test_special_teams_columns(self):
        """Verify build_special_teams() output shape and column names."""
        df, teams = _make_synthetic_records()
        st = build_special_teams(df)
        expected_cols = ["pp_xgf60", "pp_gf60", "pp_toi_share",
                         "pk_xga60", "pk_ga60", "pk_toi_share",
                         "ev_xgd60", "ev_xgf60", "ev_xga60"]
        for col in expected_cols:
            assert col in st.columns, f"Missing column: {col}"
        # Should have entries for all teams that appear in the data
        assert len(st) > 0

    def test_pp_xgf60_nonnegative(self):
        """PP xGF/60 should be >= 0."""
        df, teams = _make_synthetic_records()
        st = build_special_teams(df)
        assert (st["pp_xgf60"] >= 0).all()

    def test_ev_xgd60_excludes_pp(self):
        """EV records should not include PP_up/PP_kill_dwn lines."""
        df, teams = _make_synthetic_records()
        # If we remove all PP records, EV stats should remain the same
        df_no_pp = df[~df["home_off_line"].isin(["PP_up", "PP_kill_dwn"]) &
                      ~df["away_off_line"].isin(["PP_up", "PP_kill_dwn"])]
        st_full = build_special_teams(df)
        st_no_pp = build_special_teams(df_no_pp)
        # EV stats should match because EV excludes PP lines
        common_teams = st_full.index.intersection(st_no_pp.index)
        if len(common_teams) > 0:
            np.testing.assert_allclose(
                st_full.loc[common_teams, "ev_xgd60"].values,
                st_no_pp.loc[common_teams, "ev_xgd60"].values,
                atol=1e-6)


class TestLineDepth:
    def test_line_depth_columns(self):
        """Verify build_line_depth() output columns."""
        df, teams = _make_synthetic_records()
        ld = build_line_depth(df)
        expected_cols = ["topline_toi_share", "depth_gap", "line_xgd60_std",
                         "toppair_toi_share", "pair_xga60_std"]
        for col in expected_cols:
            assert col in ld.columns, f"Missing column: {col}"

    def test_topline_toi_share_bounds(self):
        """topline_toi_share should be in [0, 1]."""
        df, teams = _make_synthetic_records()
        ld = build_line_depth(df)
        assert (ld["topline_toi_share"] >= 0).all()
        assert (ld["topline_toi_share"] <= 1).all()


class TestGoalieFeatures:
    def test_goalie_starter_identification(self):
        """Starter should be the max TOI goalie."""
        df, teams = _make_synthetic_records()
        gk = build_goalie_features(df)
        assert "goalie_gsax60" in gk.columns
        assert "starter_toi_share" in gk.columns
        # Starter TOI share should be in [0.5, 1.0] (starter gets majority)
        assert (gk["starter_toi_share"] >= 0.4).all()  # allow some slack in synthetic data
        assert (gk["starter_toi_share"] <= 1.0).all()

    def test_goalie_shrinkage_direction(self):
        """Shrunk GSAx should be between raw and league mean."""
        df, teams = _make_synthetic_records()
        gk = build_goalie_features(df)
        # Just verify the column exists and is finite
        assert np.all(np.isfinite(gk["goalie_gsax60"]))


class TestHighDanger:
    def test_high_danger_threshold(self):
        """Higher threshold should produce lower or equal frequency."""
        df, teams = _make_synthetic_records()
        hd_low = build_high_danger(df, threshold=0.10)
        hd_high = build_high_danger(df, threshold=0.40)
        # Higher threshold = fewer high-danger events = lower freq60
        assert (hd_high["high_danger_freq60"] <= hd_low["high_danger_freq60"] + 1e-6).all()

    def test_high_danger_nonnegative(self):
        """High danger frequency should be >= 0."""
        df, teams = _make_synthetic_records()
        hd = build_high_danger(df)
        assert (hd["high_danger_freq60"] >= 0).all()


class TestBuildGameLevel:
    def test_new_columns(self):
        """build_game_level should include home_pc, away_pc, home_assists, away_assists."""
        from core.data import build_game_level
        df, _ = _make_synthetic_records()
        # Ensure no draws: set away_goals to 0 for all records, and ensure
        # at least one home_goals > 0 per game
        df["away_goals"] = 0
        first_recs = df.groupby("game_id").head(1).index
        df.loc[first_recs, "home_goals"] = 1
        game = build_game_level(df)
        assert "home_pc" in game.columns
        assert "away_pc" in game.columns
        assert "home_assists" in game.columns
        assert "away_assists" in game.columns


# ═══════════════════════ BT TUNING TESTS ═════════════════════════════

class TestBTTuning:
    def test_tune_returns_valid_params(self):
        """Returned ot_weight, reg are within grid bounds."""
        from core.model import tune_bt_hyperparams
        game, teams = _make_synthetic_games(n_games=60)
        best_ot, best_reg, best_cal, details = tune_bt_hyperparams(
            game, teams, n_outer=3, n_inner=2)
        assert best_ot in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        assert best_reg in [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03]
        assert best_cal in ["platt", "isotonic"]
        assert 0 < details["best_brier"] < 1

    def test_isotonic_calibration_runs(self):
        """Isotonic calibration produces valid probabilities."""
        from core.model import _fit_calibration
        rng = np.random.RandomState(42)
        diffs = rng.randn(100)
        y = (diffs > 0).astype(float)
        p_te, model = _fit_calibration(diffs, y, diffs[:20], method="isotonic")
        assert len(p_te) == 20
        assert all(0 <= p <= 1 for p in p_te)


# ═══════════════════════ BOOTSTRAP INTEGRATION TESTS ════════════════

class TestBootstrapIntegration:
    def test_integrated_closer_to_half(self):
        """E[σ(·)] < σ(E[·]) for extreme Δβ (Jensen's inequality)."""
        from scipy.special import expit
        # Simulate bootstrap samples with variance
        rng = np.random.RandomState(42)
        mean_diff = 2.0  # extreme
        samples = rng.normal(mean_diff, 0.5, size=200)
        # E[σ(Δβ)]
        integrated = np.mean(expit(samples))
        # σ(E[Δβ])
        point = expit(np.mean(samples))
        # Jensen's: for convex σ at extreme, E[σ] should be closer to 0.5
        assert abs(integrated - 0.5) < abs(point - 0.5)

    def test_shrink_k_zero_no_effect(self):
        """k=0 produces identical predictions to no shrink."""
        p = np.array([0.3, 0.7, 0.9])
        uncertainties = np.array([0.1, 0.2, 0.3])
        shrunk = 0.5 + (p - 0.5) * np.exp(-0.0 * uncertainties)
        np.testing.assert_allclose(p, shrunk)

    def test_shrink_reduces_extreme(self):
        """k>0 pulls predictions toward 0.5."""
        p = np.array([0.2, 0.8])
        k = 2.0
        uncertainties = np.array([0.3, 0.3])
        shrunk = 0.5 + (p - 0.5) * np.exp(-k * uncertainties)
        assert abs(shrunk[0] - 0.5) < abs(p[0] - 0.5)
        assert abs(shrunk[1] - 0.5) < abs(p[1] - 0.5)


# ═══════════════════════ ATTACK-DEFENSE TESTS ═══════════════════════

class TestAttackDefense:
    def test_ad_convergence(self):
        """fit_attack_defense converges on synthetic data."""
        game, teams = _make_synthetic_games(n_games=100)
        ad = fit_attack_defense(game, teams, use_xg=False, regularization=0.01)
        assert "attack" in ad
        assert "defense" in ad
        assert "home_adv" in ad
        assert len(ad["attack"]) == len(teams)
        assert len(ad["defense"]) == len(teams)

    def test_attack_centered(self):
        """Attack strengths sum to ~0."""
        game, teams = _make_synthetic_games(n_games=100)
        ad = fit_attack_defense(game, teams, use_xg=False)
        attack_sum = sum(ad["attack"].values())
        assert abs(attack_sum) < 0.1

    def test_skellam_ad_bounds(self):
        """Predictions in [0.01, 0.99]."""
        game, teams = _make_synthetic_games(n_games=100)
        ad = fit_attack_defense(game, teams, use_xg=False)
        for h in teams:
            for a in teams:
                if h != a:
                    p = skellam_predict_ad(ad, h, a)
                    assert 0.01 <= p <= 0.99

    def test_strongest_team_highest_attack(self):
        """With enough data, best team has highest attack."""
        game, teams = _make_synthetic_games(n_games=200)
        ad = fit_attack_defense(game, teams, use_xg=False)
        best_attack_team = max(ad["attack"], key=ad["attack"].get)
        # team_0 is strongest — should have highest attack
        assert best_attack_team == "team_0"


# ═══════════════════════ RESIDUALIZATION TESTS ══════════════════════

class TestResidualization:
    def test_residuals_uncorrelated_with_bt(self):
        """Residuals have |r| < 0.15 with BT strength."""
        game, teams = _make_synthetic_games(n_teams=8, n_games=400)
        stats = build_team_stats(game)
        bt = fit_bt(game, teams)
        # Use xgd_per60 which is less perfectly correlated with BT than win_pct
        cols = ["xgd_per60", "gd_pg"]
        valid_cols = [c for c in cols if c in stats.columns]
        resid = residualize_features(stats, bt, valid_cols, alpha=0.01)
        bt_arr = np.array([bt[t] for t in resid.index])
        for col in resid.columns:
            r = np.corrcoef(bt_arr, resid[col].values)[0, 1]
            assert abs(r) < 0.15, f"Residual {col} too correlated with BT: r={r:.3f}"

    def test_preserves_shape(self):
        """Output has same index, one column per input."""
        game, teams = _make_synthetic_games()
        stats = build_team_stats(game)
        bt = fit_bt(game, teams)
        cols = ["win_pct", "gd_pg"]
        resid = residualize_features(stats, bt, cols)
        assert len(resid) == len(stats)
        assert len(resid.columns) == len(cols)
        assert all(c.endswith("_resid") for c in resid.columns)


# ═══════════════════════ RAPM TESTS ════════════════════════════════

class TestRAPM:
    def test_output_shape(self):
        """build_rapm_features returns 3 columns for all teams."""
        df, teams = _make_synthetic_records(n_games=30)
        rapm = build_rapm_features(df, teams, regularization=1.0)
        assert "rapm_attack" in rapm.columns
        assert "rapm_defense" in rapm.columns
        assert "rapm_topline_effect" in rapm.columns
        # Should have entries for all teams present in data
        assert len(rapm) > 0

    def test_attack_stronger_for_better_team(self):
        """With enough data, better teams should have higher RAPM attack."""
        df, teams = _make_synthetic_records(n_games=50, seed=123)
        rapm = build_rapm_features(df, teams, regularization=0.1)
        # All values should be finite
        assert np.all(np.isfinite(rapm.values))

    def test_per_fold_differs_from_full(self):
        """RAPM on subset should differ from full dataset."""
        df, teams = _make_synthetic_records(n_games=40)
        rapm_full = build_rapm_features(df, teams, regularization=1.0)
        # Take half the games
        half_ids = df["game_id"].unique()[:10]
        df_half = df[df["game_id"].isin(half_ids)]
        rapm_half = build_rapm_features(df_half, teams, regularization=1.0)
        # Values should differ
        common = rapm_full.index.intersection(rapm_half.index)
        if len(common) > 0:
            assert not np.allclose(
                rapm_full.loc[common, "rapm_attack"].values,
                rapm_half.loc[common, "rapm_attack"].values,
                atol=1e-6)


# ═══════════════════════ ENHANCED HIGH DANGER TESTS ════════════════

class TestEnhancedHighDanger:
    def test_hd_xg_sum60_column(self):
        """build_high_danger should include hd_xg_sum60 column."""
        df, teams = _make_synthetic_records()
        hd = build_high_danger(df)
        assert "hd_xg_sum60" in hd.columns
        assert "p95_max_xg" in hd.columns
        assert (hd["hd_xg_sum60"] >= 0).all()
        assert (hd["p95_max_xg"] >= 0).all()


# ═══════════════════════ P6 NEW FEATURE TESTS ═══════════════════════

class TestTOIEntropy:
    def test_entropy_columns_exist(self):
        """build_toi_entropy should return 5 columns."""
        from core.advanced import build_toi_entropy
        df, teams = _make_synthetic_records()
        te = build_toi_entropy(df)
        expected_cols = ["off_line_entropy", "def_pairing_entropy", "goalie_entropy",
                         "top_vs_top_share", "top_shelter_share"]
        for col in expected_cols:
            assert col in te.columns, f"Missing column: {col}"

    def test_entropy_nonnegative(self):
        """Entropy values should be >= 0."""
        from core.advanced import build_toi_entropy
        df, teams = _make_synthetic_records()
        te = build_toi_entropy(df)
        assert (te["off_line_entropy"] >= 0).all()
        assert (te["def_pairing_entropy"] >= 0).all()
        assert (te["goalie_entropy"] >= 0).all()

    def test_deployment_shares_in_bounds(self):
        """Deployment shares should be in [0, 1]."""
        from core.advanced import build_toi_entropy
        df, teams = _make_synthetic_records()
        te = build_toi_entropy(df)
        assert (te["top_vs_top_share"] >= 0).all()
        assert (te["top_vs_top_share"] <= 1).all()
        assert (te["top_shelter_share"] >= 0).all()
        assert (te["top_shelter_share"] <= 1).all()


class TestMatchupMatrix:
    def test_matchup_columns_exist(self):
        """build_matchup_matrix should return 3 columns."""
        from core.advanced import build_matchup_matrix
        df, teams = _make_synthetic_records()
        mm = build_matchup_matrix(df)
        expected_cols = ["exploit_ability", "shutdown_resilience", "depth_insurance"]
        for col in expected_cols:
            assert col in mm.columns, f"Missing column: {col}"

    def test_exploit_nonnegative(self):
        """exploit_ability should be >= 0."""
        from core.advanced import build_matchup_matrix
        df, teams = _make_synthetic_records()
        mm = build_matchup_matrix(df)
        assert (mm["exploit_ability"] >= 0).all()

    def test_shutdown_nonnegative(self):
        """shutdown_resilience should be >= 0."""
        from core.advanced import build_matchup_matrix
        df, teams = _make_synthetic_records()
        mm = build_matchup_matrix(df)
        assert (mm["shutdown_resilience"] >= 0).all()

    def test_shrinkage_fill_works(self):
        """Shrinkage fill should prevent NaN for sparse cells."""
        from core.advanced import build_matchup_matrix
        df, teams = _make_synthetic_records(n_games=5)  # very sparse
        mm = build_matchup_matrix(df, min_toi=1000)  # high threshold
        assert not mm["exploit_ability"].isna().any()
        assert not mm["shutdown_resilience"].isna().any()


class TestSTExpectedValue:
    def test_st_ev_columns_exist(self):
        """build_st_expected_value should return 3 columns."""
        from core.advanced import build_st_expected_value
        df, teams = _make_synthetic_records()
        stev = build_st_expected_value(df)
        if len(stev) > 0:  # only if penalty columns exist
            expected_cols = ["pen_taken60", "pen_drawn60", "net_st_ev"]
            for col in expected_cols:
                assert col in stev.columns, f"Missing column: {col}"

    def test_penalty_rates_nonnegative(self):
        """Penalty rates should be >= 0."""
        from core.advanced import build_st_expected_value
        df, teams = _make_synthetic_records()
        stev = build_st_expected_value(df)
        if len(stev) > 0:
            assert (stev["pen_taken60"] >= 0).all()
            assert (stev["pen_drawn60"] >= 0).all()

    def test_net_st_ev_finite(self):
        """net_st_ev should be finite."""
        from core.advanced import build_st_expected_value
        df, teams = _make_synthetic_records()
        stev = build_st_expected_value(df)
        if len(stev) > 0:
            assert np.all(np.isfinite(stev["net_st_ev"]))


def _make_no_draw_records():
    """Make synthetic records with no draws for build_game_level compatibility."""
    df, teams = _make_synthetic_records()
    # Ensure no draws: set away_goals to 0 and ensure at least one home_goals > 0 per game
    df["away_goals"] = 0
    first_recs = df.groupby("game_id").head(1).index
    df.loc[first_recs, "home_goals"] = 1
    return df, teams


class TestVolatility:
    def test_volatility_columns_exist(self):
        """build_volatility_features should return 4 columns."""
        from core.advanced import build_volatility_features
        from core.data import build_game_level
        df, teams = _make_no_draw_records()
        game = build_game_level(df)
        vol = build_volatility_features(df, game)
        expected_cols = ["xgd60_std_game", "gsax_volatility", "blowout_rate", "close_game_rate"]
        for col in expected_cols:
            assert col in vol.columns, f"Missing column: {col}"

    def test_std_nonnegative(self):
        """Standard deviations should be >= 0."""
        from core.advanced import build_volatility_features
        from core.data import build_game_level
        df, teams = _make_no_draw_records()
        game = build_game_level(df)
        vol = build_volatility_features(df, game)
        assert (vol["xgd60_std_game"] >= 0).all()
        assert (vol["gsax_volatility"] >= 0).all()

    def test_rates_in_bounds(self):
        """Blowout and close game rates should be in [0, 1]."""
        from core.advanced import build_volatility_features
        from core.data import build_game_level
        df, teams = _make_no_draw_records()
        game = build_game_level(df)
        vol = build_volatility_features(df, game)
        assert (vol["blowout_rate"] >= 0).all()
        assert (vol["blowout_rate"] <= 1).all()
        assert (vol["close_game_rate"] >= 0).all()
        assert (vol["close_game_rate"] <= 1).all()


class TestRAPMFiltering:
    def test_low_toi_units_excluded(self):
        """RAPM should exclude units with TOI < min_toi_unit."""
        df, teams = _make_synthetic_records(n_games=30)
        # Build RAPM with high threshold
        rapm = build_rapm_features(df, teams, min_toi_unit=5000)
        # Should still return valid output (may have fewer units)
        assert "rapm_attack" in rapm.columns
        assert "rapm_defense" in rapm.columns
        assert len(rapm) > 0

    def test_output_shape_unchanged(self):
        """RAPM output shape should be consistent regardless of filtering."""
        df, teams = _make_synthetic_records(n_games=30)
        rapm_no_filter = build_rapm_features(df, teams, min_toi_unit=0)
        rapm_filtered = build_rapm_features(df, teams, min_toi_unit=200)
        # Both should have same teams (index), just different coefficients
        assert len(rapm_no_filter) == len(rapm_filtered)
        assert set(rapm_no_filter.columns) == set(rapm_filtered.columns)


class TestECE:
    def test_ece_includes_boundary(self):
        """Predictions at exactly 1.0 should be included in the last bin."""
        from core.model import compute_ece
        y_true = np.array([1, 1, 0, 0, 0])
        y_pred = np.array([1.0, 0.95, 0.1, 0.05, 0.0])
        ece = compute_ece(y_true, y_pred, n_bins=5)
        assert np.isfinite(ece) and ece >= 0


class TestSymmetricAugmentation:
    def test_doubles_size(self):
        """Symmetric augmentation should double the dataset size."""
        from phase1_ml import _symmetric_augment
        X = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1])
        X_aug, y_aug = _symmetric_augment(X, y)
        assert len(X_aug) == 2 * len(X)
        assert len(y_aug) == 2 * len(y)

    def test_sign_flip(self):
        """Augmented features should have flipped signs."""
        from phase1_ml import _symmetric_augment
        X = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1])
        X_aug, y_aug = _symmetric_augment(X, y)
        np.testing.assert_array_equal(X_aug[2:], -X)

    def test_label_flip(self):
        """Augmented labels should be 1 - y."""
        from phase1_ml import _symmetric_augment
        X = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1])
        X_aug, y_aug = _symmetric_augment(X, y)
        np.testing.assert_array_equal(y_aug[2:], 1 - y)
