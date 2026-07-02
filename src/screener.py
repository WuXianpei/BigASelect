"""股票筛选模块：六步顺序筛选"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .fund_flow_provider import get_sector_money_flow


def _apply_remove_mask(df: pd.DataFrame, remove: Any) -> pd.DataFrame:
    """统一剔除掩码：兼容单行时 numpy 标量布尔值"""
    if isinstance(remove, pd.Series):
        mask = remove.fillna(False)
    else:
        mask = pd.Series(bool(remove), index=df.index)
    return df[~mask]


def _series_from(df: pd.DataFrame, column: str) -> pd.Series:
    """从 DataFrame 取列并转为 Series，缺列时返回全 NaN"""
    if column in df.columns:
        return pd.to_numeric(df[column], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype=float)


def _is_step_enabled(rules_config: dict[str, Any], step_id: str) -> bool:
    """读取 steps 中某步是否启用，缺省为 true"""
    for step in rules_config.get("steps", []):
        if step.get("id") == step_id:
            return step.get("enabled", True)
    return True


def _should_truncate_pool(rules_config: dict[str, Any]) -> bool:
    """target_count 为 null 或 <=0 时不截断"""
    target = rules_config.get("target_count")
    if target is None:
        return False
    try:
        return int(target) > 0
    except (TypeError, ValueError):
        return False


def format_target_count_label(rules_config: dict[str, Any]) -> str:
    """日志用：描述目标池规模"""
    if not _should_truncate_pool(rules_config):
        return "不截断"
    return str(int(rules_config["target_count"]))


def apply_screening_phase1(
    df: pd.DataFrame,
    rules_config: dict[str, Any],
) -> pd.DataFrame:
    """第一阶段的筛选（步骤 1-2），在轻量 enrichment 之后执行"""
    if not rules_config.get("enabled", True):
        return df

    params = rules_config.get("params", {})
    result = df.copy()
    before = len(result)

    if _is_step_enabled(rules_config, "step_1"):
        result = _step1_risk_filter(result)
        print(f"      步骤1 风险剔除: {before} → {len(result)}")
    else:
        print(f"      步骤1 风险剔除: 已禁用，保留 {len(result)} 只")

    before = len(result)
    if _is_step_enabled(rules_config, "step_2"):
        result = _step2_liquidity_filter(result, params)
        print(f"      步骤2 流动性过滤: {before} → {len(result)}")
    else:
        print(f"      步骤2 流动性过滤: 已禁用，保留 {len(result)} 只")

    return result.reset_index(drop=True)


def apply_screening_phase2(
    df: pd.DataFrame,
    rules_config: dict[str, Any],
    market_snapshot: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """第二阶段的筛选（步骤 3-6），在完整 enrichment 之后执行"""
    result = apply_screening_trend_capital(df, rules_config)
    return apply_screening_fundamental_market(result, rules_config, market_snapshot)


def apply_screening_trend_capital(
    df: pd.DataFrame,
    rules_config: dict[str, Any],
) -> pd.DataFrame:
    """筛选步骤 3-4：趋势结构 + 资金持续性（技术 enrichment 之后）"""
    if not rules_config.get("enabled", True):
        return df

    params = rules_config.get("params", {})
    step3_on = _is_step_enabled(rules_config, "step_3")
    step4_on = _is_step_enabled(rules_config, "step_4")
    result = df.copy()
    if step4_on:
        result = _attach_sector_money_flow_3d(result)

    before = len(result)
    if step3_on:
        result = _step3_trend_structure_filter(result, params)
        print(f"      步骤3 趋势结构过滤: {before} → {len(result)}")
    else:
        print(f"      步骤3 趋势结构过滤: 已禁用，保留 {len(result)} 只")

    before = len(result)
    if step4_on:
        result = _step4_capital_persistence_filter(result)
        print(f"      步骤4 资金持续性过滤: {before} → {len(result)}")
    else:
        print(f"      步骤4 资金持续性过滤: 已禁用，保留 {len(result)} 只")

    return result.reset_index(drop=True)


def apply_screening_fundamental_market(
    df: pd.DataFrame,
    rules_config: dict[str, Any],
    market_snapshot: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """筛选步骤 5（估值财务）；步骤 6 市场环境仅流程末尾告警"""
    _ = market_snapshot
    if not rules_config.get("enabled", True):
        return _truncate_pool(df, rules_config)

    params = rules_config.get("params", {})
    result = df.copy()

    before = len(result)
    if _is_step_enabled(rules_config, "step_5"):
        result = _step5_fundamental_filter(result, params)
        print(f"      步骤5 估值财务过滤: {before} → {len(result)}")
    else:
        print(f"      步骤5 估值财务过滤: 已禁用，保留 {len(result)} 只")

    result = result.drop(columns=["_sector_money_flow_3d"], errors="ignore")
    print(f"      步骤6 市场环境: 不参与筛选（流程末尾输出告警）")

    if _should_truncate_pool(rules_config):
        result = _apply_sort(result, rules_config)
    return _truncate_pool(result, rules_config)


def apply_screening_phase2_legacy(
    df: pd.DataFrame,
    rules_config: dict[str, Any],
    market_snapshot: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """一次性执行步骤 3-6（不分阶段 enrichment 时使用）"""
    if not rules_config.get("enabled", True):
        return _truncate_pool(df, rules_config)

    params = rules_config.get("params", {})
    step3_on = _is_step_enabled(rules_config, "step_3")
    step4_on = _is_step_enabled(rules_config, "step_4")
    result = df.copy()
    if step4_on:
        result = _attach_sector_money_flow_3d(result)

    before = len(result)
    if step3_on:
        result = _step3_trend_structure_filter(result, params)
        print(f"      步骤3 趋势结构过滤: {before} → {len(result)}")
    else:
        print(f"      步骤3 趋势结构过滤: 已禁用，保留 {len(result)} 只")

    before = len(result)
    if step4_on:
        result = _step4_capital_persistence_filter(result)
        print(f"      步骤4 资金持续性过滤: {before} → {len(result)}")
    else:
        print(f"      步骤4 资金持续性过滤: 已禁用，保留 {len(result)} 只")

    before = len(result)
    if _is_step_enabled(rules_config, "step_5"):
        result = _step5_fundamental_filter(result, params)
        print(f"      步骤5 估值财务过滤: {before} → {len(result)}")
    else:
        print(f"      步骤5 估值财务过滤: 已禁用，保留 {len(result)} 只")

    result = result.drop(columns=["_sector_money_flow_3d"], errors="ignore")
    print(f"      步骤6 市场环境: 不参与筛选（流程末尾输出告警）")

    if _should_truncate_pool(rules_config):
        result = _apply_sort(result, rules_config)
    return _truncate_pool(result, rules_config)


def apply_screening(
    df: pd.DataFrame,
    rules_config: dict[str, Any],
) -> pd.DataFrame:
    """兼容旧接口：仅执行阶段一"""
    return apply_screening_phase1(df, rules_config)


def _step1_risk_filter(df: pd.DataFrame) -> pd.DataFrame:
    """剔除 is_st=1、is_suspended=1 或 risk_flag=1"""
    if df.empty:
        return df

    for col in ("is_st", "is_suspended", "risk_flag"):
        if col not in df.columns:
            df[col] = 0

    mask = (
        (pd.to_numeric(df["is_st"], errors="coerce").fillna(0) != 1)
        & (pd.to_numeric(df["is_suspended"], errors="coerce").fillna(0) != 1)
        & (pd.to_numeric(df["risk_flag"], errors="coerce").fillna(0) != 1)
    )
    return df[mask]


def _step2_liquidity_filter(df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    """剔除成交额或换手率不达标的股票（无有效行情数据时不剔除）"""
    if df.empty:
        return df

    min_amount = params.get("min_amount", 30_000_000)
    min_turnover = params.get("min_turnover_rate", 0.3)

    amount = _series_from(df, "amount")
    turnover = _series_from(df, "turnover_rate")

    remove = pd.Series(False, index=df.index)
    valid_amount = amount.notna() & (amount > 0)
    valid_turnover = turnover.notna() & (turnover > 0)

    if valid_amount.any():
        remove = remove | (valid_amount & (amount < min_amount))
    if valid_turnover.any():
        remove = remove | (valid_turnover & (turnover < min_turnover))

    return _apply_remove_mask(df, remove)


def _step3_trend_structure_filter(df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    """剔除长期弱势：ma60<ma120，或 return_90d<-20% 且 price_percentile_1y 低位"""
    if df.empty:
        return df

    ret_threshold = params.get("return_90d_threshold", -20)
    pct_max = params.get("price_percentile_1y_max", 20)

    ma60 = _series_from(df, "ma60")
    ma120 = _series_from(df, "ma120")
    ret_90d = _series_from(df, "return_90d")
    price_pct = _series_from(df, "price_percentile_1y")

    remove = pd.Series(False, index=df.index)

    valid_ma = ma60.notna() & ma120.notna()
    remove = remove | (valid_ma & (ma60 < ma120))

    valid_weak = ret_90d.notna() & price_pct.notna()
    remove = remove | (valid_weak & (ret_90d < ret_threshold) & (price_pct < pct_max))

    return _apply_remove_mask(df, remove)


def _step4_capital_persistence_filter(df: pd.DataFrame) -> pd.DataFrame:
    """剔除个股与板块资金三项同时为负的标的"""
    if df.empty:
        return df

    flow_3d = _series_from(df, "main_net_inflow_3d")
    flow_5d = _series_from(df, "main_net_inflow_5d")
    sector_3d = _series_from(df, "_sector_money_flow_3d")

    valid = flow_3d.notna() & flow_5d.notna() & sector_3d.notna()
    remove = valid & (flow_3d < 0) & (flow_5d < 0) & (sector_3d < 0)
    return _apply_remove_mask(df, remove)


def _step5_fundamental_filter(df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    """剔除估值与财务极端异常的股票"""
    if df.empty:
        return df

    pe_max = params.get("pe_ttm_max", 300)
    pe_min = params.get("pe_ttm_min", 0)
    pb_max = params.get("pb_max", 15)
    yoy_min = params.get("net_profit_yoy_min", -80)
    debt_max = params.get("debt_ratio_max", 80)

    pe = _series_from(df, "pe_ttm")
    pb = _series_from(df, "pb")
    yoy = _series_from(df, "net_profit_yoy")
    debt = _series_from(df, "debt_ratio")

    remove = pd.Series(False, index=df.index)

    pe_valid = pe.notna()
    remove = remove | (pe_valid & ((pe > pe_max) | (pe < pe_min)))

    pb_valid = pb.notna()
    remove = remove | (pb_valid & (pb > pb_max))

    yoy_valid = yoy.notna()
    remove = remove | (yoy_valid & (yoy < yoy_min))

    debt_valid = debt.notna()
    remove = remove | (debt_valid & (debt > debt_max))

    return _apply_remove_mask(df, remove)


def evaluate_market_environment_alert(
    params: dict[str, Any],
    market_snapshot: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    评估市场环境是否偏离建议区间（仅告警，不参与筛选）。
    基准：上证指数 market_risk_index、index_ma60_position。
    """
    risk_max = params.get("market_risk_index_max", 70)
    ma60_min = params.get("index_ma60_position_min", 0)

    info: dict[str, Any] = {
        "alert": False,
        "market_risk_index": None,
        "index_ma60_position": None,
        "market_risk_index_max": risk_max,
        "index_ma60_position_min": ma60_min,
        "message": "",
    }

    if not market_snapshot:
        info["message"] = "无市场环境数据，跳过环境评估"
        return info

    risk = pd.to_numeric(market_snapshot.get("market_risk_index"), errors="coerce")
    ma60_pos = pd.to_numeric(market_snapshot.get("index_ma60_position"), errors="coerce")

    if pd.isna(risk) or pd.isna(ma60_pos):
        info["message"] = "市场风险指数或 MA60 位置缺失，跳过环境评估"
        return info

    risk_f = float(risk)
    ma60_f = float(ma60_pos)
    info["market_risk_index"] = risk_f
    info["index_ma60_position"] = ma60_f

    issues: list[str] = []
    if risk_f >= risk_max:
        issues.append(f"market_risk_index={risk_f:.2f}（建议<{risk_max}）")
    if ma60_f < ma60_min:
        issues.append(f"index_ma60_position={ma60_f:.4f}（建议>={ma60_min}）")

    if issues:
        info["alert"] = True
        info["message"] = "当前大盘环境偏离建议区间：" + "；".join(issues)
    else:
        info["message"] = (
            f"市场环境正常：market_risk_index={risk_f:.2f}，"
            f"index_ma60_position={ma60_f:.4f}"
        )

    return info


def print_market_environment_alert(
    params: dict[str, Any],
    market_snapshot: dict[str, Any] | None,
) -> dict[str, Any]:
    """在流程末尾打印市场环境告警，并返回评估结果"""
    info = evaluate_market_environment_alert(params, market_snapshot)
    if info.get("alert"):
        print(f"\n      [市场环境告警] {info['message']}")
        print("        （步骤6 已改为仅告警，不影响股票池数量）")
    else:
        print(f"\n      [市场环境] {info['message']}")
    return info


def _attach_sector_money_flow_3d(df: pd.DataFrame) -> pd.DataFrame:
    """按行业附加近3日板块主力净流入（亿元，筛选内部用）"""
    if df.empty or "industry" not in df.columns:
        df["_sector_money_flow_3d"] = np.nan
        return df

    industries = df["industry"].dropna().astype(str).str.strip()
    unique_inds = [i for i in industries.unique().tolist() if i]
    flow_by_industry: dict[str, float | None] = {
        ind: get_sector_money_flow(ind, 3) for ind in unique_inds
    }

    df["_sector_money_flow_3d"] = df["industry"].apply(
        lambda x: flow_by_industry.get(str(x).strip()) if pd.notna(x) and str(x).strip() else None
    )
    return df


def _apply_sort(df: pd.DataFrame, rules_config: dict[str, Any]) -> pd.DataFrame:
    """按配置排序"""
    sort_rules = rules_config.get("sort_by", [])
    if not sort_rules or df.empty:
        return df

    by: list[str] = []
    ascending: list[bool] = []
    for rule in sort_rules:
        field = rule.get("field")
        if field and field in df.columns:
            by.append(field)
            ascending.append(rule.get("order", "asc") == "asc")

    if by:
        return df.sort_values(by=by, ascending=ascending, na_position="last").reset_index(
            drop=True
        )
    return df


def _truncate_pool(df: pd.DataFrame, rules_config: dict[str, Any]) -> pd.DataFrame:
    """截取目标股票池数量；未配置 target_count 时原样返回"""
    if not _should_truncate_pool(rules_config):
        return df.copy()
    target = int(rules_config["target_count"])
    return df.head(target).copy()
