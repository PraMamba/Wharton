"""
Hockey Analytics Script - Tasks 1a, 1b, 1c, 1d
冰球数据分析脚本：球队实力排名、进攻线差距分析、可视化和方法论总结
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# 配置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 路径配置
DATA_PATH = "output/whl_2025.csv"
OUTPUT_DIR = "output"


def load_data(path: str) -> pd.DataFrame:
    """加载WHL赛季数据"""
    df = pd.read_csv(path)
    print(f"加载数据完成: {len(df)} 条记录")
    return df


def task_1a_power_rankings(df: pd.DataFrame) -> pd.DataFrame:
    """
    任务1a: 球队实力排行榜 - 基于xGD/60指标
    
    计算每支球队的:
    - 总预期进球数 (Total xGF)
    - 总被预期进球数 (Total xGA)
    - 净预期进球差 (xGD = xGF - xGA)
    - 总上场时间 (Total TOI)
    - 每60分钟净预期进球差 (xGD/60)
    """
    # 主队统计
    home_stats = df.groupby('home_team').agg(
        home_xgf=('home_xg', 'sum'),
        home_xga=('away_xg', 'sum'),
        home_toi=('toi', 'sum')
    ).reset_index()
    home_stats = home_stats.rename(columns={'home_team': 'team'})
    
    # 客队统计
    away_stats = df.groupby('away_team').agg(
        away_xgf=('away_xg', 'sum'),
        away_xga=('home_xg', 'sum'),
        away_toi=('toi', 'sum')
    ).reset_index()
    away_stats = away_stats.rename(columns={'away_team': 'team'})
    
    # 合并主客场数据
    team_stats = home_stats.merge(away_stats, on='team', how='outer')
    team_stats = team_stats.fillna(0)
    
    # 计算总数据
    team_stats['total_xgf'] = team_stats['home_xgf'] + team_stats['away_xgf']
    team_stats['total_xga'] = team_stats['home_xga'] + team_stats['away_xga']
    team_stats['total_toi'] = team_stats['home_toi'] + team_stats['away_toi']
    
    # 计算核心指标
    team_stats['xgd'] = team_stats['total_xgf'] - team_stats['total_xga']
    # Data dictionary: toi is seconds. per60 = per 60 minutes = per 3600 seconds.
    team_stats['xgd_per_60'] = team_stats['xgd'] / (team_stats['total_toi'] / 3600)
    
    # 排名
    team_stats = team_stats.sort_values('xgd_per_60', ascending=False)
    team_stats['rank'] = range(1, len(team_stats) + 1)
    
    # 选择输出列
    output_cols = ['rank', 'team', 'total_xgf', 'total_xga', 'xgd', 'total_toi', 'xgd_per_60']
    result = team_stats[output_cols].copy()
    
    # 格式化数值
    result['total_xgf'] = result['total_xgf'].round(2)
    result['total_xga'] = result['total_xga'].round(2)
    result['xgd'] = result['xgd'].round(2)
    result['total_toi'] = result['total_toi'].round(0).astype(int)
    result['xgd_per_60'] = result['xgd_per_60'].round(4)
    
    return result


def task_1b_line_disparity(df: pd.DataFrame) -> tuple:
    """
    任务1b: 进攻线实力差距分析
    
    计算每支球队第一和第二进攻线的xGF/60，并计算差距比率
    Disparity Ratio = (第一进攻线 xGF/60) / (第二进攻线 xGF/60)
    """
    # 准备主队进攻线数据
    home_lines = df[['home_team', 'home_off_line', 'home_xg', 'toi']].copy()
    home_lines.columns = ['team', 'line', 'xg', 'toi']
    
    # 准备客队进攻线数据
    away_lines = df[['away_team', 'away_off_line', 'away_xg', 'toi']].copy()
    away_lines.columns = ['team', 'line', 'xg', 'toi']
    
    # 合并数据
    all_lines = pd.concat([home_lines, away_lines], ignore_index=True)
    
    # 只保留第一和第二进攻线
    all_lines = all_lines[all_lines['line'].isin(['first_off', 'second_off'])]
    
    # 按球队和进攻线分组统计
    line_stats = all_lines.groupby(['team', 'line']).agg(
        total_xg=('xg', 'sum'),
        total_toi=('toi', 'sum')
    ).reset_index()
    
    # 计算每条进攻线的xGF/60
    # Data dictionary: toi is seconds. per60 = per 60 minutes = per 3600 seconds.
    line_stats['xgf_per_60'] = np.where(
        line_stats['total_toi'] > 0,
        line_stats['total_xg'] / (line_stats['total_toi'] / 3600),
        0
    )
    
    # 透视表：每支球队的第一和第二进攻线效率
    pivot = line_stats.pivot(index='team', columns='line', values='xgf_per_60')
    pivot = pivot.reset_index()
    pivot.columns = ['team', 'first_line_xgf60', 'second_line_xgf60']
    
    # 计算差距比率
    pivot['disparity_ratio'] = pivot['first_line_xgf60'] / pivot['second_line_xgf60']
    
    # 按差距比率排序
    pivot = pivot.sort_values('disparity_ratio', ascending=False)
    
    # 格式化
    pivot['first_line_xgf60'] = pivot['first_line_xgf60'].round(4)
    pivot['second_line_xgf60'] = pivot['second_line_xgf60'].round(4)
    pivot['disparity_ratio'] = pivot['disparity_ratio'].round(4)
    
    # 返回所有球队数据和前10名
    top10 = pivot.head(10).copy()
    
    return pivot, top10


def task_1c_visualization(
    power_rankings: pd.DataFrame,
    line_disparity: pd.DataFrame,
    output_path: str
) -> str:
    """
    任务1c: 数据可视化
    
    创建散点图显示球队整体实力(xGD/60)与进攻线实力差距(Disparity Ratio)的关系
    """
    # 合并数据
    plot_data = power_rankings.merge(
        line_disparity[['team', 'disparity_ratio']], 
        on='team', 
        how='inner'
    )
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 8), dpi=150)
    
    # 绘制散点
    scatter = ax.scatter(
        plot_data['disparity_ratio'],
        plot_data['xgd_per_60'],
        c=plot_data['xgd_per_60'],
        cmap='RdYlGn',
        s=100,
        alpha=0.7,
        edgecolors='black',
        linewidth=0.5
    )
    
    # 添加回归线
    x = plot_data['disparity_ratio'].values
    y = plot_data['xgd_per_60'].values
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = slope * x_line + intercept
    
    ax.plot(x_line, y_line, 'b--', linewidth=2, alpha=0.8, 
            label=f'回归线 (R² = {r_value**2:.3f})')
    
    # 标注前5强和后5弱球队
    top_teams = plot_data.nsmallest(5, 'rank')
    bottom_teams = plot_data.nlargest(5, 'rank')
    
    for _, row in top_teams.iterrows():
        ax.annotate(
            row['team'],
            (row['disparity_ratio'], row['xgd_per_60']),
            textcoords='offset points',
            xytext=(5, 5),
            fontsize=8,
            fontweight='bold',
            color='darkgreen'
        )
    
    for _, row in bottom_teams.iterrows():
        ax.annotate(
            row['team'],
            (row['disparity_ratio'], row['xgd_per_60']),
            textcoords='offset points',
            xytext=(5, -10),
            fontsize=8,
            fontweight='bold',
            color='darkred'
        )
    
    # 添加水平参考线 (xGD/60 = 0)
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    
    # 设置标题和标签
    ax.set_title('球队实力 vs. 进攻线依赖度分析\nTeam Strength vs. Offensive Line Disparity', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('进攻线差距比率 (Disparity Ratio = 第一线xGF/60 ÷ 第二线xGF/60)', fontsize=11)
    ax.set_ylabel('球队整体实力 (xGD/60 = 净预期进球差/60分钟)', fontsize=11)
    
    # 添加图例
    ax.legend(loc='upper right', fontsize=10)
    
    # 添加颜色条
    cbar = plt.colorbar(scatter)
    cbar.set_label('xGD/60', fontsize=10)
    
    # 确定回归线趋势的描述
    if slope < -0.1:
        trend_desc = "数据显示，拥有更均衡进攻线的球队（较低的差距比）倾向于展现出更强的整体实力（较高的xGD/60）。"
    elif slope > 0.1:
        trend_desc = "数据显示，更依赖第一进攻线的球队（较高的差距比）可能具有更强的整体实力。"
    else:
        trend_desc = "数据显示，进攻线均衡度与球队整体实力之间没有显著的相关性。"
    
    # 添加底部说明文字
    fig.text(0.5, 0.02, trend_desc, ha='center', fontsize=10, style='italic',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 调整布局
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    
    # 保存图像
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"可视化图表已保存: {output_path}")
    return output_path


def task_1d_methodology(
    power_rankings: pd.DataFrame,
    line_disparity_top10: pd.DataFrame,
    output_path: str
) -> str:
    """
    任务1d: 方法论总结
    
    生成Markdown格式的方法论文档
    """
    # 计算相关系数用于报告
    markdown_content = """# 冰球数据分析方法论总结

## 一、引言

本分析采用**预期进球(Expected Goals, xG)**模型来评估球队和进攻线的真实表现能力。与传统的进球统计相比，xG模型通过量化每次射门的进球概率，能够更准确地反映球队的进攻创造力和防守稳固性，有效减少运气因素对评估结果的干扰。

预期进球(xG)是一个基于历史数据的统计模型，它考虑了射门位置、射门角度、比赛情境等多种因素，为每次射门赋予一个0到1之间的进球概率值。累计这些概率值，我们可以得到一支球队在一定时期内"应该"获得的进球数，从而更客观地评价其攻防表现。

---

## 二、任务1a：球队实力排行榜方法

### 2.1 核心指标：xGD/60 (每60分钟净预期进球差)

我们选择**xGD/60**作为衡量球队综合实力的核心指标，其计算方法如下：

1. **汇总全赛季数据**
   - 对于每支球队，分别统计其作为主队和客队时的预期进球(xGF)和被预期进球(xGA)
   - 计算总预期进球数：`Total xGF = Σ(主场xG) + Σ(客场xG)`
   - 计算总被预期进球数：`Total xGA = Σ(对手主场xG) + Σ(对手客场xG)`

2. **计算净预期进球差**
   ```
   xGD = Total xGF - Total xGA
   ```
   正值表示该队进攻创造优于防守漏洞，负值则相反。

3. **时间标准化**
   - 统计球队总上场时间(Total TOI，单位：秒)
   - 计算标准化指标：
   ```
   xGD/60 = xGD / (Total TOI / 60)
   ```
   该指标表示每60分钟（约一场比赛时长）的净预期进球差。

### 2.2 指标优越性

- **消除比赛场次差异**：通过时间标准化，可以公平比较不同上场时间的球队
- **反映真实实力**：xG模型过滤了运气因素，更能体现球队的系统性攻防能力
- **综合性强**：同时考虑了进攻和防守两个维度

---

## 三、任务1b：进攻线实力差距分析方法

### 3.1 核心指标：Disparity Ratio (差距比率)

我们使用**差距比率**来量化球队对第一进攻线的依赖程度：

1. **按进攻线分组**
   - 识别数据中的进攻线标识：`first_off`(第一进攻线) 和 `second_off`(第二进攻线)
   - 将每条记录按球队和进攻线进行分组

2. **计算各线效率**
   对于每支球队的每条进攻线：
   ```
   Line xGF/60 = (该线总xGF) / (该线总上场时间 / 60)
   ```
   这是每条进攻线每60分钟创造的预期进球数。

3. **计算差距比率**
   ```
   Disparity Ratio = (第一进攻线 xGF/60) / (第二进攻线 xGF/60)
   ```
   
### 3.2 指标解读

- **比率 > 1**：第一进攻线效率高于第二进攻线（正常情况）
- **比率越高**：球队越依赖第一进攻线的表现
- **比率接近1**：两条进攻线效率接近，阵容深度更均衡

---

## 四、任务1c：可视化分析方法

### 4.1 图表设计

采用**散点图 + 回归线**的组合来揭示两个核心指标之间的关系：

- **Y轴（纵轴）**：球队整体实力 (xGD/60)
- **X轴（横轴）**：进攻线实力差距 (Disparity Ratio)
- **每个点**：代表一支球队
- **回归线**：显示两个变量之间的线性关系趋势

### 4.2 洞察价值

回归线的斜率揭示了关键问题的答案：
- **负斜率**：拥有更均衡进攻线的球队往往具有更强的整体实力
- **正斜率**：更依赖第一进攻线的球队可能整体实力更强
- **接近水平**：两者之间无显著相关性

---

## 五、主要发现

### 5.1 球队实力排名（前5名）

| 排名 | 球队 | xGD/60 |
|------|------|--------|
"""
    
    # 添加前5名球队
    for _, row in power_rankings.head(5).iterrows():
        markdown_content += f"| {row['rank']} | {row['team']} | {row['xgd_per_60']:.4f} |\n"
    
    markdown_content += """
### 5.2 进攻线依赖度最高的球队（前5名）

| 球队 | 第一线xGF/60 | 第二线xGF/60 | 差距比率 |
|------|--------------|--------------|----------|
"""
    
    # 添加进攻线差距前5名
    for _, row in line_disparity_top10.head(5).iterrows():
        markdown_content += f"| {row['team']} | {row['first_line_xgf60']:.4f} | {row['second_line_xgf60']:.4f} | {row['disparity_ratio']:.4f} |\n"
    
    markdown_content += """
---

## 六、结论

本分析方法具有以下特点：

1. **科学严谨**：基于xG模型的量化分析，减少主观判断和运气因素的影响
2. **标准化处理**：通过时间标准化(per 60)，确保不同比赛时间的球队可以公平比较
3. **多维度评估**：同时考察整体实力和阵容深度，提供更全面的球队画像
4. **可复现性强**：明确的计算公式和数据处理流程，便于验证和重现

通过散点图可视化，我们能够直观地观察到球队整体实力与进攻线配置均衡度之间的关系，为深入理解冰球比赛中的战术选择和阵容构建提供数据支撑。

---

*本报告基于WHL 2025赛季完整数据生成*
"""
    
    # 保存文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    
    print(f"方法论文档已保存: {output_path}")
    return output_path


def main():
    """主函数：执行所有分析任务"""
    print("=" * 60)
    print("冰球数据分析 - Tasks 1a, 1b, 1c, 1d")
    print("=" * 60)
    
    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 加载数据
    df = load_data(DATA_PATH)
    
    # 任务1a: 球队实力排行榜
    print("\n[任务1a] 生成球队实力排行榜...")
    power_rankings = task_1a_power_rankings(df)
    rankings_path = os.path.join(OUTPUT_DIR, "task1a_power_rankings.csv")
    power_rankings.to_csv(rankings_path, index=False, encoding='utf-8-sig')
    print(f"  -> 已保存: {rankings_path}")
    print(f"  -> 共 {len(power_rankings)} 支球队")
    print(f"  -> 第1名: {power_rankings.iloc[0]['team']} (xGD/60: {power_rankings.iloc[0]['xgd_per_60']:.4f})")
    print(f"  -> 最后1名: {power_rankings.iloc[-1]['team']} (xGD/60: {power_rankings.iloc[-1]['xgd_per_60']:.4f})")
    
    # 任务1b: 进攻线实力差距分析
    print("\n[任务1b] 分析进攻线实力差距...")
    line_disparity_all, line_disparity_top10 = task_1b_line_disparity(df)
    disparity_path = os.path.join(OUTPUT_DIR, "task1b_line_disparity_top10.csv")
    line_disparity_top10.to_csv(disparity_path, index=False, encoding='utf-8-sig')
    print(f"  -> 已保存: {disparity_path}")
    print(f"  -> 差距比率最高: {line_disparity_top10.iloc[0]['team']} ({line_disparity_top10.iloc[0]['disparity_ratio']:.4f})")
    
    # 任务1c: 数据可视化
    print("\n[任务1c] 生成可视化图表...")
    viz_path = os.path.join(OUTPUT_DIR, "task1c_visualization.png")
    task_1c_visualization(power_rankings, line_disparity_all, viz_path)
    
    # 任务1d: 方法论总结
    print("\n[任务1d] 撰写方法论文档...")
    methodology_path = os.path.join(OUTPUT_DIR, "task1d_methodology.md")
    task_1d_methodology(power_rankings, line_disparity_top10, methodology_path)
    
    # 完成
    print("\n" + "=" * 60)
    print("所有任务完成！输出文件：")
    print(f"  1. {rankings_path}")
    print(f"  2. {disparity_path}")
    print(f"  3. {viz_path}")
    print(f"  4. {methodology_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
