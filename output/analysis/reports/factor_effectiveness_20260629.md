# 因子有效性分析报告

- 生成时间: 2026-06-29T21:29:06+08:00
- 因子配置: `config/factor_config.yaml`
- 收益列: `future_return_20`（位于 `output/archive/stock_pool/`）
- 分析窗口: 45 个交易日 （可算收益 45 日）
- 收益截止日: 20260529
- 收益 horizon: 20 个交易日
- 面板样本: 43978 行

## 结论

**当前打分模型（基于 future_return_20）: 有效**
- 规则通过: 4/5 （至少 4 项通过）

## 核心指标（final_score）

| 指标 | 值 |
|------|-----|
| IC 均值 | 0.1952 |
| IC 标准差 | 0.0684 |
| IC_IR | 2.8531 |
| IC 胜率 | 1.0 |
| IC 有效日数 | 44 |
| 五分位价差 (Q5-Q1) | 6.9652% |
| 五分位单调 | False |
| Top20% 超额 | 5.9999% |

## 操盘参考（不参与失效判定）

模拟每日 `final_score` **第 1 名**与**前 3 只**，以 `future_return_20 > 0` 视为单笔获胜（到期盈利；未模拟 ±10% 止盈止损路径）。

- Top1 单笔胜率: **72.73%**（32/44 笔）

- Top3 单笔胜率: **68.18%**（90/132 笔）
- Top3 平均 future_return_20: 13.1407%
- Top3 二十日最大跌幅: **-25.9944%**（全部 132 笔中单笔最差）
- Top3 每日最差一只平均 future_return_20: -1.3125%
- 信号日数: 44 日
- 当日 Top3 全部为正的比例: 40.91%

### 五分位平均 future_return_20 (%)

- Q1: 0.1701%
- Q2: -0.6474%
- Q3: -0.9487%
- Q4: -0.033%
- Q5: 7.1353%

## 判定规则明细

- [通过] ic_mean: IC均值=0.1952（阈值>=0.02）
- [通过] ic_ir: IC_IR=2.8531（阈值>=0.3）
- [通过] ic_positive_ratio: IC胜率=1.0（阈值>=0.55）
- [通过] quintile_spread: 五分位价差=6.9652%（阈值>=1.5%）
- [未通过] monotonic: 五分位单调递增=False

## 各类因子 IC 汇总

| 分数列 | IC均值 | IC_IR | IC胜率 |
|--------|--------|-------|--------|
| final_score | 0.1952 | 2.8531 | 1.0 |
| value_score | -0.286 | -2.2494 | 0.0 |
| growth_score | 0.0917 | 1.2801 | 0.8636 |
| capital_score | 0.3268 | 3.6613 | 1.0 |
| sector_score | 0.0862 | 0.9352 | 0.7727 |

## 成分因子平均 IC

- `pb_percentile_5y`: 0.4443
- `pe_percentile_5y`: 0.3361
- `turnover_rate`: 0.3303
- `sector_money_flow_5d`: 0.1794
- `sector_new_high_ratio`: 0.1315
- `net_profit_yoy`: 0.1154
- `sector_rank_score`: 0.0957
- `sector_momentum_score`: 0.0957
- `roe`: 0.0938
- `ma_trend_score`: 0.0755
- `leader_stock_strength`: 0.071
- `price_structure_score`: 0.0633
- `trend_strength_score`: 0.057
- `northbound_flow`: 0.022
- `main_net_inflow_10d`: -0.0096
- `debt_ratio`: -0.0592
