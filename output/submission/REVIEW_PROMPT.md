# Peer Review Prompt — WHSDSC 2026 Phase 1 Submission

You are reviewing a submission for the **Wharton High School Data Science Competition 2026 (WHSDSC)**, Phase 1: Hockey Analytics. The dataset is the WHL 2025 season (32 teams, 1,312 games, 25,827 line-level records). Your job is to act as a rigorous but fair peer reviewer, checking for correctness, methodology soundness, and presentation quality.

---

## Files to Review

All files are in `output/submission/`:

| File | Purpose |
|------|---------|
| `phase1_submit.py` | Full pipeline source code (in project root) |
| `power_rankings.csv` | Task 1a: 32-team power rankings |
| `matchup_predictions.csv` | Task 1a: 16-game Round 1 win probabilities |
| `line_disparity.csv` | Task 1b: Offensive line quality disparity (32 teams) |
| `TheIsoLab.png` | Task 1c: Single visualization |
| `dashboard.png` | Internal analytics dashboard |
| `methodology.md` | Task 1d: Full methodology report |

---

## Checklist — Please verify each item and flag any issues

### A. Data Pipeline Correctness

1. **Unit consistency**: The raw data has `toi` in **seconds**. Verify that all "per 60 minutes" calculations divide by `toi / 3600` (not `toi / 60`). Check: `build_team_stats()` line computing `xgd_per60`, and `build_line_disparity()` lines computing `xga60`, `raw60`, `adj60`.

2. **Game-level aggregation**: `build_game_level()` sums goals/xg/shots across all line-level rows per `game_id`. Confirm this doesn't double-count (each row should be a unique line matchup within a game). Check: are there exactly 1,312 unique `game_id` values?

3. **Home/away symmetry**: `build_team_stats()` stacks home and away records. Verify that the away `win` column is correctly inverted (`away wins when home_win == 0`).

4. **Line disparity filtering**: We restrict to `first_off`/`second_off` vs `first_def`/`second_def` only. Confirm this correctly excludes PP, PK, third lines, and empty-net. Is this the right interpretation of "even-strength offensive line quality disparity"?

### B. Model Correctness

5. **OT-aware Bradley-Terry**: OT games are downweighted by `ot_weight=0.5`. Verify the gradient update: `grad[i] += w * (y - p)` where `w=0.5` for OT and `w=1.0` for regulation. Is this a valid modification to standard BT?

6. **Logistic regression**: P(home_win) = sigmoid(a + b × BT_diff). Current fit: `a=0.2724`, `b=0.9350`. Does `sigmoid(0.2724) ≈ 56.8%` correctly represent home-ice advantage? Is it reasonable that `b < 1.0` (BT strength differences are somewhat attenuated)?

7. **Cross-validation**: 5-fold CV refits BT from scratch on each training fold. This is honest (no leakage from test fold into strength estimates). The baseline is computed within each fold using only the training home-win rate. Verify that `oof_pred` collects test-fold predictions without overlap.

8. **Probability clipping**: Matchup predictions are clipped to [0.01, 0.99]. Is this appropriate? Could it mask overconfidence issues?

### C. Results Plausibility

9. **Rankings**: Top 5 are Brazil, Thailand, Netherlands, Pakistan, Peru. Brazil dominates (composite 2.073 vs #2 at 1.612). Does this spread seem reasonable for a 32-team league?

10. **Win probabilities**: Range from 54.1% (Vietnam vs Serbia) to 85.2% (Brazil vs Kazakhstan). All 16 predictions favor the home team. Is this plausible given home-ice advantage + seeding?

11. **Calibration table**: Out-of-fold calibration shows predicted vs observed:
    - 0.35 bin: pred 0.351, obs 0.390 (slight underconfidence)
    - 0.59 bin: pred 0.591, obs 0.539 (slight overconfidence)
    - 0.67 bin: pred 0.671, obs 0.672 (excellent)
    - 0.75 bin: pred 0.753, obs 0.699 (overconfident at high end)

    Is this calibration pattern acceptable? Should we be concerned about the 0.75 bin?

12. **Line disparity**: Top 3 are USA (1.370), Guatemala (1.363), Saudi Arabia (1.357). These are all bottom-half teams. The finding "no strong linear relationship between disparity and strength" (r = -0.031) — is this insight well-supported?

### D. Methodology Report Quality

13. **Word counts**: Each section has a target (~50 or ~100 words). Are they approximately within bounds?

14. **Claims accuracy**: Does the report accurately describe what the code does? Any discrepancies between the code logic and the report's description?

15. **Limitations section**: The report states "team/line quality is treated as stable across the season; we do not time-weight games or model roster/injury shocks." Is this sufficient? What other limitations should be mentioned?

16. **AI usage disclosure**: Three AI models listed (Claude, Gemini, Codex). Is the description of each model's contribution accurate and appropriately transparent?

### E. Visualization Quality

17. **TheIsoLab.png** (submission viz): Scatter plot of adjusted disparity vs composite strength. Check: axis labels clear? Trend line and statistics visible? Top/bottom teams labeled? Colorblind-friendly (viridis)?

18. **dashboard.png**: 5-panel internal dashboard. All panels render correctly? Labels readable? No clipping or overlap?

### F. Potential Issues to Flag

19. **Ensemble weight justification**: BT 40%, xGD/60 30%, xG share 10%, GD 10%, win% 10%. These weights were "validated by ML experiment." But the ML experiment used the OLD toi/60 bug. Do the weights still make sense after the fix?

20. **BT convergence**: The Newton-step uses `strength += 0.5 * delta` (damped). Is `max_iter=300` sufficient? Does the algorithm converge for all folds?

21. **Statistical significance**: With 1,312 games and 32 teams (~41 games each), are the per-team BT estimates stable? What's the approximate standard error?

---

## Scoring Criteria (from WHSDSC workbook)

- **Accuracy**: Do predictions match actual outcomes? (Evaluated post-tournament)
- **Methodology**: Is the approach statistically sound, well-justified, and appropriately complex?
- **Communication**: Is the report clear, concise, and well-organized?

Please provide:
1. A summary of issues found (critical / minor / nitpick)
2. An overall quality assessment (1-10 scale)
3. Top 3 suggestions for improvement before submission
