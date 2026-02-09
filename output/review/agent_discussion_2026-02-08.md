# Multi-Agent Review Log (WHSDSC 2026 Phase 1)

Date: 2026-02-08  
Workspace: `C:\Users\13926\Desktop\Wharton`  
Team PNG basename: `TheIsoLab` (outputs `output/submission/TheIsoLab.png`)

## Snapshot (Current Submission Outputs)

Directory: `output/submission/`

- `power_rankings.csv`
- `matchup_predictions.csv`
- `line_disparity.csv`
- `TheIsoLab.png` (Phase 1c upload)
- `dashboard.png` (internal, not for upload)
- `methodology.md`

## Automated Output Sanity Checks (Local)

Status: **OK** (2026-02-08)

Checks performed:
- `power_rankings.csv`: 32 rows, ranks 1..32 unique, composite sorted descending
- `matchup_predictions.csv`: 16 rows, probabilities in [0,1], `predicted_winner` is either home/away
- `line_disparity.csv`: 32 rows, no NaN/Inf in adjusted values
- `TheIsoLab.png`: exists and file size < 5MB

Helper script:
- `check_submission.py` prints a compact pass/fail report and which PNG to upload.

## Known “Stop-Ship” Fixes Already Applied

- Fixed **CV leakage** in ML comparison module (fold-wise recomputation + fold-wise scaling).
- Fixed **per60 unit** errors: `toi` is seconds, per-60 uses `3600`.
- Line disparity now uses only **first/second offensive lines vs first/second defensive pairings** (excludes PP/PK + empty net).
- Calibration bins now use **OOF** predictions (instead of in-sample predictions).
- Phase 1c PNG filename now configurable (and defaults to `TheIsoLab`).

## Decision Points Under Debate

1. **Matchup probabilities model**: BT-only logistic vs Elastic Net logistic (ML challenger)
2. **OT handling**: down-weight OT games to 0.5 vs tie-extension framing
3. **Line disparity metric stability**: keep ratio (as required by workbook) vs smooth/winsorize/log-ratio variants

## Metrics (Leakage-Safe)

Core submission pipeline (BT-only logistic on BT strength diff):
- Train: Accuracy 60.2%, Brier 0.2325
- 5-fold CV: Accuracy 58.1% ± 3.2%, Log-loss 0.6752, Brier 0.2410
- Baseline (fold home-rate): Accuracy ~56.4%, Log-loss ~0.686, Brier ~0.246

ML challengers (from `phase1_ml.py`, leakage-safe 5-fold CV, 10 features = `bt_diff` + team-stat diffs):
- Baseline (fold home-rate): Acc 56.4%, LogLoss 0.6849, Brier 0.2459
- BT-only Logistic: Acc 57.3%, LogLoss 0.6814, Brier 0.2437
- Elastic Net Logistic: Acc 57.0%, LogLoss 0.6774, Brier 0.2421
- XGBoost: Acc 57.5%, LogLoss 0.7154, Brier 0.2575 (poor calibration)
- Blended Ensemble: Acc 56.9%, LogLoss 0.6833, Brier 0.2448

## Gemini Discussion (Concise)

### Gemini Prompt (verbatim)
```text
继续作为 reviewer。补充事实：我们已修复 CV 泄漏与 per60 单位错误。line disparity 现在只用 even-strength 风格的行 (first/second_off 对 first/second_def)，排除 PP/PK 与 empty_net。校准图用 out-of-fold (OOF) 预测分箱。

我们做了 leakage-safe 的 5-fold CV challenger（1312 games, 10 features：bt_diff + team-level diffs）。结果：
- Baseline(折内 home-rate): Acc 56.4%, LogLoss 0.6849, Brier 0.2459
- BT-only Logistic: Acc 57.3%, LogLoss 0.6814, Brier 0.2437
- Elastic Net Logistic: Acc 57.0%, LogLoss 0.6774, Brier 0.2421
- XGBoost: Acc 57.5%, LogLoss 0.7154, Brier 0.2575 (校准差)
- Blended: Acc 56.9%, LogLoss 0.6833, Brier 0.2448

问题（请给可执行建议，别泛泛而谈）：
1) 以竞赛提交为目标，你建议最终的 16 场胜率用哪一个？BT-only 还是 Elastic Net？如果选 Elastic Net，你会如何保持可解释性（展示系数、限制特征数量、单调性等），并在 1d 文本里说得清楚？
2) OT 处理：我们现在在 BT 里对 OT 比赛下权重 0.5。是否更推荐把 OT 当作“平局/半胜半负”（Davidson tie extension 的近似）？在不大改代码的情况下，你建议怎么写成方法论亮点但不被质疑？
3) line disparity 指标：用 ratio(first_adj/second_adj)。有没有更稳健的替代（log-ratio、差值、加上 shrinkage/贝叶斯、Winsorize）？在本数据结构下（只有 first/second lines + PP/PK/empty_net），最推荐哪个？
4) 最后给一个 submission checklist：哪些地方最容易被评委挑错（单位、泄漏、过拟合、叙述不一致、可视化信息密度等），以及你建议的最小修补。
```

### Gemini Response (verbatim)
```text
好的，我们来逐一分析并提供具体建议。

### 1) 最终模型选择与可解释性

**建议：选择 Elastic Net Logistic Regression。**

**理由：**
- **性能更优：** 在预测概率的竞赛中，LogLoss 和 Brier Score 通常比 Accuracy 更重要。Elastic Net 在这两项上都胜过 BT-only，说明它的概率预测更准、校准得更好。XGBoost 校准太差，直接排除。
- **可解释性可控：** 它的核心仍然是逻辑回归，只是加入了正则化。10 个特征的数量非常适合解释。

**如何保持可解释性并在 1 页报告中说清楚：**

1.  **命名：** 在报告中，你可以称之为**“带正则化的逻辑回归 (Regularized Logistic Regression)”**，这比“弹性网络”更易懂。
2.  **核心论点：** “我们选择带正则化的逻辑回归，因为它在惩罚不重要特征、防止过拟合的同时，保留了逻辑回归的核心可解释性。交叉验证结果显示，该模型在 Brier Score (0.2421) 和 LogLoss (0.6774) 上表现最佳，提供了最可靠的胜率预测。”
3.  **展示系数：** 这是关键。在报告中直接放一个简洁的表格或条形图，展示最终模型训练出的 10 个特征的系数 (coefficients)。
    *   **操作建议：** 模型训练完成后，从拟合好的 `LogisticRegressionCV` 或类似对象中提取 `.coef_` 属性。将其与特征名称配对，按系数值从大到小排序，制作成图表。

    **示例图表标题：** “模型核心特征权重 (Feature Coefficients)”

| 特征 (Feature) | 系数 (Coefficient) |
| :--- | :--- |
| bt_diff | 1.25 |
| xGF_per60_diff | 0.85 |
| CorsiFor%_diff | 0.70 |
| ... | ... |
| PDO_diff | -0.15 |

4.  **解读一两个关键系数：** “如图所示，我们模型最强的预测因子是 `bt_diff`（两队 Bradley-Terry 评分之差）。此外，每当主队的 `xGF_per60_diff`（预期场均进球差）增加一个单位，其获胜的对数几率 (log-odds) 就会增加 0.85。” 这句话清晰地连接了模型、特征和业务含义。

通过以上四步，你既选择了性能最好的模型，又以非常专业和清晰的方式解决了可解释性问题。

### 2) Overtime (OT) 处理方法论

**现状：** 在 BT 模型中给 OT 比赛 0.5 的权重。
**问题：** 如何包装这个方法，让它听起来像一个深思熟虑的优点，而不是一个简单的妥协？

**建议：不要改代码，而是优化措辞。**

在你的方法论部分这样描述：

> “为了精确处理加时赛 (OT) 的特殊性，我们对模型进行了调整。常规赛时间内的胜利更能反映球队的稳定实力，而 OT 胜利（包括 3v3 和点球）包含更多偶然性，且比赛规则与常规时间不同。
>
> 因此，在我们的 Bradley-Terry 模型中，我们将 OT 比赛的权重设定为 0.5。这种方法在概念上**近似于将 OT 结果视为一次‘部分胜利’或‘平局’**，类似于公认的 Davidson-Larkin 平局处理模型。这样做既能保留 OT 比赛中的胜负信息，又能降低其在评估球队真实实力时的权重，使我们的最终实力评分对常规时间的表现更为敏感和稳健。”

**关键点：**
- **主动承认特殊性：** 表明你考虑到了 OT 的不同。
- **给出合理解释：** “偶然性”、“规则不同”。
- **高级概念挂钩：** 主动提及 Davidson 模型，表明你了解更复杂的处理方式，但选择了概念相似且更适合当前框架的简化方案。
- **强调优点：** “更为敏感和稳健”。

这样，一个简单的权重调整就变成了一个经过深思熟虑、在理论和实践中取得平衡的建模决策。

### 3) Line Disparity 指标的稳健替代方案

**现状：** `ratio(first_adj / second_adj)`。
**问题：** 分母接近 0 时不稳定，且指标不对称。

**建议：立即更换为对数比 (Log-Ratio)。**

**理由：**
- **稳健性：** `log(A/B) = log(A) - log(B)`。即使 `second_adj` 很小，log 变换也能平滑其影响，避免极端值。
- **对称性：** 如果 A/B = 2，log-ratio 是 `log(2)`。如果 B/A = 2 (即 A/B = 0.5)，log-ratio 是 `log(0.5) = -log(2)`。数值大小相等，方向相反，解释起来非常直观。
- **实现简单：** 只是在现有计算上加一个 `np.log()`，几乎没有代码成本。
- **其他方案对比：**
    - **差值：** `A - B` 忽略了相对大小，不如 log-ratio。
    - **Shrinkage/贝叶斯：** 理论上最好，但实现复杂，需要额外假设（先验分布），在截止日期前风险高。
    - **Winsorize：** 能处理极端异常值，但不能解决比率本身的不稳定和不对称问题。

**具体操作：**
将你的特征工程代码中的 `df['line_disparity'] = df['first_adj'] / df['second_adj']` 修改为：
`df['line_disparity'] = np.log(df['first_adj']) - np.log(df['second_adj'])`
（注意处理 `first_adj` 或 `second_adj` 小于等于 0 的情况，可以给一个很小的正数 `epsilon` 来避免 `log(0)` 错误，例如 `np.log(df['col'] + 1e-6)`）。

在方法论中简单说明：“我们采用对数比来衡量阵容深度差异，这是一个更稳定且对称的指标。”

### 4) Submission Checklist (最小修补建议)

以下是评委最容易挑错的地方，以及你应该做的最后检查和修补：

1.  **叙述一致性 (Narrative Consistency)：**
    - **[必修]** 通读你的 `methodology.md`，然后逐一核对 `power_rankings.csv`, `line_disparity.csv` 和你的图表。报告里写的指标定义、模型名称、特征列表，必须和代码实现、输出文件完全一致。例如，如果你改用了 log-ratio，报告里必须同步更新。

2.  **数据泄漏 (Leakage)：**
    - **[检查]** 确认你的校准图**确实**是用 out-of-fold (OOF) 预测生成的。确保你最终提交的模型系数，是在**整个数据集上重新训练 (retrain)** 得到的，而不是某个 fold 的结果。在报告中可以加一句：“模型超参数通过 5 折交叉验证确定，最终模型则在全部 1312 场比赛数据上进行训练，以确保最准确的参数估计。”

3.  **可视化信息密度 (Visualization Density)：**
    - **[修补]** **校准图：** 确保有一条完美的 `y=x` 虚线作为参考。X 轴标签是“预测胜率 (Predicted Probability)”，Y 轴是“实际胜率 (Actual Frequency)”。标题清晰，例如“模型校准图 (Calibration Plot via 5-Fold CV)”。
    - **[修补]** **系数图：** 确保 Y 轴的特征名称清晰可读，X 轴是“系数值 (Coefficient Value)”。删除所有不必要的网格线、背景色，让信息本身成为焦点。

4.  **单位和定义 (Units & Definitions)：**
    - **[修补]** 在报告的末尾或脚注中，用一两句话快速定义关键术语：“**BT 评分**：基于 Bradley-Terry 模型的球队实力评分，在对数尺度上。**Brier Score**：衡量概率预测准确度的指标，越低越好。**Line Disparity**：衡量一、二线阵容实力差异的对数比指标。” 不要假设评委熟悉所有术语。

5.  **最终产出文件检查 (Final Output Check)：**
    - **[检查]** 打开你所有的 `*.csv` 文件。列名是否干净（没有空格或奇怪字符）？数值格式是否正确？`matchup_predictions.csv` 里的概率加起来是不是 1？有没有 `NaN` 或 `inf`？这是一个低级但致命的错误点。

完成以上检查和修补，你的提交材料将在细节上无懈可击，给评委留下专业、严谨的印象。
```

## Claude Discussion (Token-Optimized)

### How Claude Usage Was Optimized

- Provided only **task constraints + leakage-safe metrics** (no codebase ingestion).
- Asked for **short Markdown** output with a **stop-ship** checklist.

### Claude Prompt (verbatim)
```text
你是资深统计建模 + 竞赛写作 reviewer。请尽量短（<= 3500中文字符），输出 Markdown。

背景：WHSDSC 2026 冰球 Phase1。提交物：32队 power_rankings.csv；16场 home_win_prob；Top10 line disparity ratio；1张 PNG（文件名=队名无空格，本队 TheIsoLab）；简短方法说明。

当前核心：OT-aware Bradley–Terry（OT比赛权重0.5）估计 team strength；然后 logistic: p(home win)=sigmoid(a + b*(BT_diff))。全样本：acc 60.2%，Brier 0.2325。5-fold CV（fold内重算BT+fit logistic）：acc 58.1%±3.2%，logloss 0.6752，Brier 0.2410。Baseline（fold内 home-rate）：acc 56.4%，logloss ~0.686，Brier ~0.246。

我们还做了 leakage-safe 5-fold CV ML challengers（10个特征：bt_diff + team-stats diffs）：
- BT-only Logistic: Acc 57.3%, LogLoss 0.6814, Brier 0.2437
- Elastic Net Logistic: Acc 57.0%, LogLoss 0.6774, Brier 0.2421
- XGBoost: Acc 57.5%, LogLoss 0.7154, Brier 0.2575（校准差）

line disparity：用 even-strength 风格数据（first/second_off vs first/second_def），计算 adjusted xG/60 后做 ratio = first_adj/second_adj。

请回答：
1) 最终16场概率建议用 BT-only 还是 Elastic Net？考虑“概率评分规则未知”场景给出保守决策；如选 Elastic Net，给最小可解释写法（1d如何一句话讲清），以及必须展示的证据（系数表/校准表）。
2) OT 0.5 权重的最强辩护：怎么写成亮点而不被质疑（可提 Davidson/tie 但不要太学术）。
3) line disparity ratio 是否应做稳健化（epsilon, shrinkage, winsorize, log ratio 仅用于回归图）？给最低风险建议。
4) Stop-ship checklist：5-10条，按风险排序。
```

### Claude Response (verbatim)
```text
# Phase1 Review & Recommendations

---

## 1) 最终模型选择：**BT-only Logistic**

**决策理由（评分规则未知场景）：**

| 指标 | BT-only | Elastic Net | 差异 |
|------|---------|-------------|------|
| CV Acc | 57.3% | 57.0% | +0.3pp |
| CV LogLoss | 0.6814 | 0.6774 | +0.6% |
| CV Brier | 0.2437 | 0.2421 | +0.7% |

- Elastic Net 在 Brier/LogLoss 上微弱胜出（~0.7%），但 **差异在CV标准误内**（±3.2%）
- 评分规则未知时，**简单模型更抗过拟合风险**——评委可能用 accuracy、可能用 log-loss、可能用自定义规则
- BT-only 的解释成本为零：「stronger team + home advantage → higher win prob」
- Elastic Net 需解释10个特征的贡献，任一特征若被评委质疑（如 multicollinearity），扣分风险大于收益

**若执意选 Elastic Net 的最低要求：**

> 一句话解释：「We augment BT strength difference with team-level efficiency differentials (goals, shots, xG per 60); elastic-net regularization shrinks redundant features toward zero.」

必须展示：
1. **系数表**：只列 |coef| > 0.01 的特征，标注正负方向
2. **校准图**：10-bin reliability diagram，overlay perfect calibration line
3. **特征重要性**：证明 `bt_diff` 仍是主导项（占比 >60%）

---

## 2) OT 0.5 权重的亮点写法

**推荐表述（简洁版）：**

> Overtime games provide weaker evidence of team dominance—the 5-minute 3v3 format introduces high variance. We down-weight OT outcomes to 0.5 in our Bradley–Terry likelihood, treating them as "partial wins." This is analogous to classical paired-comparison models that handle ties by splitting credit (Davidson 1970), and aligns with NHL standings logic where OT losses still earn 1 point.

**防御要点：**
- **直觉锚点**：NHL 自己给 OT loss 1分，说明联盟认为 OT 结果信息量低于 regulation
- **统计论据**：OT 仅 5 分钟、3v3、shootout 随机性更高——若等权处理，会系统性高估 OT winner 实力
- **敏感性检验**（如被追问）：可补一句「We tested weights in {0.3, 0.5, 0.7}; 0.5 yielded best CV log-loss.」

---

## 3) Line Disparity Ratio 稳健化建议

**最低风险方案：**

```
ratio = (first_adj + ε) / (second_adj + ε),  ε = 0.1 xG/60
```

| 问题 | 解决方案 |
|------|----------|
| 分母接近零导致极端值 | ε 平滑 |
| 异常值影响排名 | Winsorize 至 [5th, 95th] 百分位 |
| 回归图非线性 | **仅在可视化时用 log(ratio)**，提交的 Top10 仍用 raw ratio |

**不建议 shrinkage**：样本量足够（32队），shrinkage 会压缩真实差异，削弱 disparity 信号。

**提交格式确认**：ratio 保留 2-3 位小数，降序排列，确保 Top10 无 NaN/Inf。

---

## 4) Stop-Ship Checklist（按风险排序）

| # | 检查项 | 风险等级 |
|---|--------|----------|
| 1 | **16场比赛是否包含训练集队伍？** 若有新队/新赛季数据，BT 估计可能缺失 | 🔴 Critical |
| 2 | **概率边界检查**：所有 p ∈ [0.01, 0.99]，避免极端概率被 log-loss 惩罚到 ∞ | 🔴 Critical |
| 3 | **CSV 格式**：power_rankings.csv 列名、排序、队名拼写与官方模板一致 | 🔴 Critical |
| 4 | **PNG 文件名**：`TheIsoLab.png`（无空格、大小写正确、扩展名小写） | 🟠 High |
| 5 | **Top10 disparity 无 Inf/NaN**：手动检查分母 | 🟠 High |
| 6 | **主场优势符号方向**：确认 home team 在 logistic 里是正系数 | 🟠 High |
| 7 | **OT 权重实现验证**：打印一场 OT 比赛的 likelihood 贡献，确认是 0.5 | 🟡 Medium |
| 8 | **方法说明字数**：题目若限 200 字，严格遵守 | 🟡 Medium |
| 9 | **Reproducibility**：random seed 固定，重跑一次确认输出一致 | 🟡 Medium |
| 10 | **队名映射表**：确保代码中队名与官方数据完全匹配（大小写、缩写） | 🟡 Medium |

---

**TL;DR 决策：**
- 用 **BT-only Logistic**，更稳健
- OT 0.5 用 NHL 积分逻辑 + 方差论据辩护
- Disparity 加 ε=0.1 + winsorize，log 仅用于画图
- 先过 #1-#3 再提交
```

## Resolution (Codex Recommendation)

Given:
- scoring rule is not explicitly stated in the workbook,
- Elastic Net improves LogLoss/Brier only slightly (and adds narrative/feature-risk),
- BT-only remains easy to audit and already lifts over baseline,

**Recommendation for submission**:
- Keep **BT-only logistic** for the official `matchup_predictions.csv` (most robust, easiest to defend).
- Keep **Elastic Net** as a documented challenger (already in `phase1_ml.py`) to justify that “more complex models add only marginal lift.”

If later you discover the platform uses strict LogLoss/Brier scoring for the 16 probabilities, we can add an optional flag to generate an Elastic-Net-based `matchup_predictions.csv` without changing the rest of the submission artifacts.

---

## Web Evidence Pack (For Model-Choice Debate)

Note: these sources are external and not part of the competition materials; they are here to defend methodological choices.

- **E1 (Accuracy is a bad target for probabilistic forecasting)**: Harrell (2017), *Damage Caused by Classification Accuracy and Other Discontinuous Improper Accuracy Scoring Rules*.  
  URL: https://www.fharrell.com/post/class-damage  
  Key idea: accuracy is discontinuous/improper; prefer proper scoring rules (log-loss, Brier) for probability forecasts.
- **E2 (Why “proper” scores matter)**: Bröcker & Smith (2007), *Scoring probabilistic forecasts: the importance of being proper.*  
  URL: https://eprints.lse.ac.uk/22219/  
  Key idea: only proper scores provide internally consistent evaluations of probability forecasts.
- **E3 (Calibration and proper scoring rules in practice)**: scikit-learn docs on calibration + Brier score loss.  
  URLs: https://sklearn.org/stable/modules/calibration.html and https://scikit-learn.org/dev/modules/model_evaluation.html#brier-score-loss  
  Key idea: log-loss/Brier are proper scoring rules; reliability diagrams diagnose calibration; Brier decomposes (interpret with care).
- **E4 (Brier score origin)**: Brier (1950), *Verification of forecasts expressed in terms of probability.*  
  URL (abstract): https://ui.adsabs.harvard.edu/abs/1950MWRv...78....1B/abstract  
  Key idea: original definition of the Brier score.
- **E5 (Hockey-specific BT with multiple outcomes / OT credit)**: Whelan & Klein (2021), *Bradley-Terry Modeling with Multiple Game Outcomes with Applications to College Hockey* (arXiv).  
  URL: https://arxiv.org/abs/2112.01267  
  Key idea: BT can be extended to regulation/OT/shootout; point systems express that OT carries weaker signal.
- **E6 (Tie-extension reference point)**: Davidson (1970), *On extending the Bradley-Terry model to accommodate ties in paired comparison experiments.*  
  URL (metadata): https://cir.nii.ac.jp/crid/1572543025435752064  
  Key idea: classical tie extension for BT-style paired comparisons.

---

## Tri-Branch Debate (Separation of Powers)

### Executive Branch (Gemini) Prompt (verbatim)
```text
角色：你是“Executive Branch”——优先提升竞赛得分，但必须可解释、可 defend。

Evidence pack（请引用编号支持观点）：
E1 Harrell(2017) 指出 accuracy 等阈值指标是 discontinuous/improper；应优先 proper scoring rules。
E2 Bröcker&Smith(2007) 强调概率预测评估要用 proper scores。
E3 scikit-learn calibration docs：log-loss/Brier 是 proper scoring rules；reliability diagram 用于校准诊断；Brier=calibration+refinement（需谨慎解读）。
E5 Whelan&Klein(2021 arXiv) 多结果 Bradley–Terry 可用于冰球（reg/OT/so）并用点数权重表达“OT信号更弱”。

现状（别复述背景）：核心提交仍用 BT-only logistic；challenger CV(无泄漏) 显示 Elastic Net 在 LogLoss/Brier 略优但差距很小。

任务（请输出 Markdown，<= 2500 中文字符）：
1) 你是否主张把最终16场概率切到 Elastic Net？给出最强论证，明确在“评分规则未知”情况下仍然合理的理由。
2) 给一个最小实现方案：如何在 `phase1_submit.py` 增加 `PRED_MODEL=elasticnet|bt` 开关，且保证 1d 叙事一致（写出要改的 3-6 个点）。
3) 给对 Claude(保守 BT-only) 立场的反驳要点（3-5条）。
```

### Executive Branch (Gemini) Response (verbatim)
```text
好的，作为“Executive Branch”，我的决策将以“在最大化竞赛得分潜力的同时，确保每一步都有坚实、可解释的理论依据”为最高原则。

### 1) 最终决策：切换至 Elastic Net 模型

我**主张**将最终16场比赛的概率预测切换到 **Elastic Net** 模型。

**核心论证：**

我们的首要目标是在竞赛中胜出，这意味着要追求更高的预测分数。证据 E1 和 E2 明确指出，对于概率预测任务，**Proper Scoring Rules**（如 Log-Loss 和 Brier Score）是评估模型优劣的黄金标准，远优于 Accuracy 等不敏感的阈值指标。我们的内部 CV 结果显示，Elastic Net 在这两项关键指标上均**略微但稳定地优于**基础的 BT 模型。

**在“评分规则未知”的情况下，这一决策依然是最合理的，理由如下：**

1.  **方法论的稳健性 (Methodological Robustness):** 选择在业界公认最专业、最严格的评估标准下表现最优的模型，是**最稳健 (robust)** 的策略。这表明我们的方法论遵循了预测科学的最佳实践，而不是为了迎合某个特定、甚至可能不完美的未知评分规则而进行“应试”。这本身就是一种强有力的辩护逻辑：“我们选择了在科学上最站得住脚的模型”。

2.  **避免“更差选择”的风险:** 坚守一个在 LogLoss/Brier 上已知表现更差的 BT 模型，是在主动选择一个**“已知次优”**的方案。如果最终评分规则是 Proper Rule 或其变体（这在任何严肃的预测竞赛中都极有可能），那么这种坚守将直接导致失分。我们不能承担这种“本可避免”的风险。

3.  **模型优势的理论支撑 (E3):** Elastic Net 的优势可能来源于 Brier Score 分解后的**校准 (Calibration) 和精炼 (Refinement)** 两个维度（E3）。作为一种正则化线性模型，它在保留可解释性的同时，有能力通过 L1/L2 惩罚项整合更多特征、处理共线性并防止过拟合，从而捕捉到比单一 BT 评级更复杂的信号，获得更好的“精炼度”。即使优势微弱，也代表了模型能力的真实提升。在高水平竞赛中，微小优势的累积效应可能决定最终排名。

### 2) 最小化实现方案 (`phase1_submit.py`)

为了在 `phase1_submit.py` 中增加模型切换开关并确保叙事一致性，需要修改以下关键点：

1.  **增加入口参数:** 使用 `argparse` 库，在脚本入口处添加一个命令行参数，如 `--pred-model`，允许用户选择 `bt` 或 `elasticnet`，并设置默认值为 `elasticnet`。

2.  **条件化模型加载:** 在主逻辑中，根据 `--pred-model` 的值，使用 `if/elif` 语句加载不同的模型文件（例如 `models/bt_final.pkl` 或 `models/en_final.pkl`）。

3.  **特征矩阵准备:** 确保为 Elastic Net 模型准备正确的特征矩阵。BT 模型可能只需要两队的 ID，而 Elastic Net 需要一个更宽的特征矩阵。此逻辑需要根据所选模型进行切换。

4.  **更新 `methodology.md` 的生成逻辑:** 这是叙事一致性的核心。脚本中生成 `methodology.md` 的部分必须是动态的。
    *   如果选择 `elasticnet`，方法论部分应强调：
        *   我们采用了正则化广义线性模型以提升预测精度。
        *   模型在 LogLoss/Brier Score 等 Proper Scoring Rules 上表现更优。
        *   正则化（L1+L2）有助于特征选择和防止过拟合，增强了模型的泛化能力。
    *   如果选择 `bt`，则保留原有描述。

5.  **`power_rankings.csv` 的生成:** BT 模型可以直接输出各队“实力评级”。Elastic Net 的系数虽然也能解释，但不能直接作为 power ranking。需要提供一种基于模型预测能力的替代性排名方案，或者明确指出在该模式下不生成传统意义的 Power Ranking，以保持逻辑严谨。

6.  **`matchup_predictions.csv` 的输出:** 确保最终的概率预测结果是由所选择的模型 (`loaded_model.predict_proba()`) 生成的。

### 3) 对 Claude (坚守 BT-only) 立场的反驳要点

1.  **“简单”不等于“最优”，尤其当证据指向反面时:** 坚守 BT 的核心理由是其“简单”和“理论纯粹”。但当 CV 证据清晰表明 Elastic Net 在公认的 Proper Scoring Rules 上表现更好时，这种“简单”就成了“简陋”。为了解释性而牺牲预测准确性，违背了竞赛的第一性原理。

2.  **我们当前的 BT 实现本身就是一种“不完备”的简化:** 证据 E5 揭示了现代冰球分析中已采用加权多结果的 Bradley-Terry 模型来处理常规赛、加时赛和点球赛的不同价值。我们当前的基础 BT 模型并未实现这一点。因此，声称它“理论更纯粹”的立场是站不住脚的，因为它本身就忽略了领域内的前沿进展，并非 BT 方法在冰球应用中的最佳实践。

3.  **低估了 Elastic Net 的可解释性和防御力:** Elastic Net 并非无法解释的“黑箱”。它本质上仍是广义线性模型，其系数大小和正负直接反映了特征的影响力，L1 惩罚项还能实现自动特征选择，反而可能**增强**了叙事的洞察力（“模型告诉我们，最近5场比赛的得分效率是关键”）。为“在公认标准下表现更好”的模型辩护，远比为“更简单但表现更差”的模型辩护更有力。

4.  **在竞争中，放弃任何确定的优势都是不可接受的:** 竞赛的本质是追求极致。一个在 Brier Score 上 0.01 的优势，在16场比赛中累积起来可能就是决定性的。主动放弃这种基于数据的确定性优势，是一种策略上的保守和懈怠。
```

### Legislative Branch (Claude) Prompt (verbatim)
```text
角色：你是“Legislative Branch”——优先合规/可审核/可解释，防止评委挑刺；在未知评分规则下以风险最小为目标。

Evidence pack（请引用编号支持观点）：
E1 Harrell(2017) accuracy 是 discontinuous/improper。
E2 Bröcker&Smith(2007) proper scores 对概率评估更一致。
E3 scikit-learn calibration docs：log-loss/Brier 是 proper scoring rules；Brier 需谨慎解读。
E5 Whelan&Klein(2021 arXiv) 冰球 BT 多结果/点数权重。

现状：提交用 BT-only logistic；Elastic Net CV 在 logloss/brier 略好但差距很小；Phase1 还要求强叙事 + 1张 PNG。

任务（请输出 Markdown，<= 2500 中文字符）：
1) 你是否坚持最终16场概率继续用 BT-only？给出最强论证（评分规则未知）。
2) 给出对 Gemini(主张 Elastic Net) 的反驳要点（3-5条）。
3) 提供“最低风险增益”清单（5-8条），限定为：不会改变最终模型，但能降低提交风险/提高可 defend（例如概率裁剪、措辞、图表用语、额外sanity checks）。
```

### Legislative Branch (Claude) Response (verbatim)
```text
# Legislative Branch 分析报告：BT-only 策略的合规性论证

---

## 一、坚持 BT-only 的最强论证

**结论：在评分规则未知的情况下，BT-only 是风险最小的选择。**

### 核心论证链

1. **可解释性即合规护城河**
   - BT 模型产出的是**单一参数（队伍强度 θ）**，评委可直接追溯任何概率预测的来源
   - Elastic Net 涉及多特征权重组合，一旦被问及"为何 Team A 胜率是 62.3%"，需解释多个系数交互——增加被质疑的攻击面

2. **Proper Scoring Rule 的防御逻辑（E1, E2, E3）**
   - 根据 E1，accuracy 是 discontinuous/improper，若评委使用 accuracy 类指标，BT 和 Elastic Net 差异将被噪声掩盖
   - 根据 E2 和 E3，若评委使用 proper scores（log-loss/Brier），**两者差距"略好但很小"意味着统计上不显著**
   - 结论：**无论评委用何种合理指标，性能差异不足以抵消复杂度带来的可审计风险**

3. **BT 的理论血统（E5）**
   - E5 证明 BT 框架可扩展至多结果场景（冰球点数权重），说明该方法论在体育预测领域有学术背书
   - 评委若质疑方法选择，可引用该文献证明 BT 并非"过于简单"，而是"恰当简洁"

4. **未知评分规则下的 Minimax 策略**
   - 最坏情况：评委重叙事/可解释性 → BT 完胜
   - 次坏情况：评委重预测精度 → 差距微小，不足以扣分
   - Elastic Net 的最坏情况：被质疑过拟合/特征工程不透明 → 无法自证清白

---

## 二、对 Gemini（主张 Elastic Net）的反驳要点

| # | Gemini 可能论点 | 反驳 |
|---|----------------|------|
| **R1** | \"Elastic Net 在 CV 上 logloss/Brier 更好\" | 差距\"略好但很小\"→ 在样本量有限时，可能仅是噪声；E3 明确指出 Brier 需谨慎解读 |
| **R2** | \"多特征能捕捉更多信息\" | 16 场样本下，额外特征带来的方差膨胀可能超过偏差减少；Phase1 强调叙事，不鼓励黑箱 |
| **R3** | \"正则化已防止过拟合\" | 正则化选择本身（α, l1_ratio）引入超参数，增加被质疑的自由度；BT 无超参数争议 |
| **R4** | \"可以用 SHAP 解释\" | SHAP 解释需额外假设（特征独立性），且图表复杂度上升，与\"1 张 PNG\"限制冲突 |
| **R5** | \"BT 太简单会显得不专业\" | E5 学术背书 + 历史应用（Elo/国际象棋）证明简洁≠简陋；评委更可能欣赏\"奥卡姆剃刀\"精神 |

---

## 三、最低风险增益清单（不改变模型，降低提交风险）

| # | 措施 | 原理 | 实施成本 |
|---|------|------|----------|
| **1** | **概率裁剪至 [0.05, 0.95]** | 避免极端概率（0/1）触发 log-loss 爆炸；符合 proper scoring 防御（E3） | 1 行代码 |
| **2** | **措辞使用\"estimated probability\"而非\"predicted probability\"** | \"估计\"暗示不确定性，降低被指控过度自信的风险 | 文案修改 |
| **3** | **图表标题加置信区间说明** | 即使不画 CI，标题注明\"point estimates; uncertainty not shown\"可预防质疑 | 文案修改 |
| **4** | **Sanity Check：历史胜率 vs BT 概率散点图** | 证明模型输出与朴素基线一致，无明显异常 | 5 分钟 |
| **5** | **文档中明确声明模型局限** | \"本模型假设队伍强度稳定，未考虑伤病/主客场\"——先发制人封堵质疑 | 2 句话 |
| **6** | **PNG 图表使用中性配色（灰/蓝）** | 避免红绿色盲问题；专业外观降低\"业余\"印象 | 调色板 |
| **7** | **附录提供 Elastic Net 对比结果（但不作为主模型）** | 证明\"我们做过比较\"，主动展示严谨性，同时坚持 BT 决策 | 1 段文字 |
| **8** | **概率表保留 2 位小数** | 过多小数位暗示虚假精度；2 位符合体育博彩惯例 | 格式调整 |

---

## 总结

**BT-only + 上述 8 项低成本措施 = 最小化可审计风险的策略组合。**

在评分规则未知的竞赛中，**可解释性是唯一确定被加分的维度**；而微小的预测精度提升在噪声中不可辨识。Legislative Branch 建议：**守住 BT，强化叙事，打磨细节。**
```

### Judicial Branch (Codex) Ruling

Decision (current): keep **BT-only** as the official `matchup_predictions.csv` generator, because:

- scoring rule for the 16 probabilities is not stated in the workbook (unknown objective),
- Elastic Net lift is real but small, and increases narrative/feature-scrutiny surface area,
- the submission is judged on methodology + communication in addition to accuracy.

Action (implemented): incorporate “lowest risk gains” without changing the model.

- Probability clipping: `home_win_prob` clipped to `[0.01, 0.99]` before rounding.
- Wording precision: removed “no significant linear relationship” → “no strong linear relationship”.
- Colorblind-safe palette: replaced red/green colormaps with `viridis`.
