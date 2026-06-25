# 因子有效性分析报告

- 生成时间: 2026-06-25T21:00:04+08:00
- 因子配置: `config/factor_config.yaml`
- 收益列: `future_return_20`（位于 `output/archive/stock_pool/`）
- 分析窗口: 43 个交易日 （可算收益 43 日）
- 收益截止日: 20260527
- 收益 horizon: 20 个交易日
- 面板样本: 41980 行

## 结论

**当前打分模型（基于 future_return_20）: 有效**
- 规则通过: 4/5 （至少 4 项通过）

## 核心指标（final_score）

| 指标 | 值 |
|------|-----|
| IC 均值 | 0.1918 |
| IC 标准差 | 0.0679 |
| IC_IR | 2.826 |
| IC 胜率 | 1.0 |
| IC 有效日数 | 42 |
| 五分位价差 (Q5-Q1) | 6.2935% |
| 五分位单调 | False |
| Top20% 超额 | 5.7087% |

## 操盘参考（不参与失效判定）

模拟每日 `final_score` **前 3 只**，以 `future_return_20 > 0` 视为单笔获胜（到期盈利；未模拟 ±10% 止盈止损路径）。

- Top3 单笔胜率: **67.46%**（85/126 笔）
- Top3 平均 future_return_20: 12.5154%
- Top3 二十日最大跌幅: **-25.9944%**（全部 126 笔中单笔最差）
- Top3 每日最差一只平均 future_return_20: -1.9059%
- 信号日数: 42 日
- 当日 Top3 全部为正的比例: 40.48%

### 五分位平均 future_return_20 (%)

- Q1: 0.7815%
- Q2: -0.2776%
- Q3: -0.6947%
- Q4: -0.0529%
- Q5: 7.075%

## 判定规则明细

- [通过] ic_mean: IC均值=0.1918（阈值>=0.02）
- [通过] ic_ir: IC_IR=2.826（阈值>=0.3）
- [通过] ic_positive_ratio: IC胜率=1.0（阈值>=0.55）
- [通过] quintile_spread: 五分位价差=6.2935%（阈值>=1.5%）
- [未通过] monotonic: 五分位单调递增=False

## 各类因子 IC 汇总

| 分数列 | IC均值 | IC_IR | IC胜率 |
|--------|--------|-------|--------|
| final_score | 0.1918 | 2.826 | 1.0 |
| value_score | -0.2789 | -2.2198 | 0.0 |
| growth_score | 0.0893 | 1.2542 | 0.8571 |
| capital_score | 0.3243 | 3.5886 | 1.0 |
| sector_score | 0.0835 | 0.8809 | 0.75 |

## 成分因子平均 IC

- `pb_percentile_5y`: 0.4424
- `pe_percentile_5y`: 0.3349
- `turnover_rate`: 0.3279
- `sector_money_flow_5d`: 0.1849
- `sector_new_high_ratio`: 0.1518
- `net_profit_yoy`: 0.1292
- `sector_rank_score`: 0.0959
- `sector_momentum_score`: 0.0959
- `roe`: 0.093
- `ma_trend_score`: 0.0717
- `leader_stock_strength`: 0.0614
- `price_structure_score`: 0.0611
- `trend_strength_score`: 0.0527
- `northbound_flow`: 0.0196
- `main_net_inflow_10d`: -0.0229
- `debt_ratio`: -0.0525
