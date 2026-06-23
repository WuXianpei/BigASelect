"""股票池多因子打分模型（配置见 config/factor_config.yaml）"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.config_loader import load_factor_config
from src.market_enricher import get_primary_market_snapshot


def _get_config(config: dict[str, Any] | None) -> dict[str, Any]:
    return config if config is not None else load_factor_config()


def resolve_market_weights(
    market_risk_index: float | None,
    config: dict[str, Any] | None = None,
) -> tuple[str, dict[str, float]]:
    """根据 market_risk_index 返回档位名称与四类因子权重"""
    cfg = _get_config(config)
    regime_cfg = cfg.get("market_regime", {})
    default_regime = regime_cfg.get("default_regime", "neutral")
    thresholds = regime_cfg.get("thresholds", {})
    weight_table = regime_cfg.get("weights", {})

    if market_risk_index is None or pd.isna(market_risk_index):
        return default_regime, dict(weight_table.get(default_regime, {}))

    risk = float(market_risk_index)
    low_max = thresholds.get("low_max", 40)
    high_min = thresholds.get("high_min", 70)

    if risk < low_max:
        regime = "low"
    elif risk > high_min:
        regime = "high"
    else:
        regime = "neutral"

    return regime, dict(weight_table.get(regime, weight_table.get(default_regime, {})))


def _apply_tier_rule(value: float, tiers: list[dict[str, Any]]) -> float:
    """按 min 阈值从高到低匹配，取首个满足 value >= min 的档位"""
    for tier in sorted(tiers, key=lambda t: t.get("min", float("-inf")), reverse=True):
        if value >= tier.get("min", float("-inf")):
            return float(tier.get("points", 0))
    return 0.0


def compute_ma_trend_score(
    row: pd.Series,
    config: dict[str, Any] | None = None,
) -> float | None:
    """均线趋势评分（0-100），规则来自 factor_config.yaml"""
    cfg = _get_config(config)
    score_cfg = cfg.get("derived_scores", {}).get("ma_trend_score", {})
    max_score = float(score_cfg.get("max_score", 100))
    require_field = score_cfg.get("require_field", "close")

    if pd.isna(row.get(require_field)):
        return None

    close = row.get(require_field)
    score = 0.0

    for rule in score_cfg.get("rules", []):
        rule_type = rule.get("type")
        if rule_type == "close_above_ma":
            ma_val = row.get(rule.get("ma_field"))
            if pd.notna(ma_val) and close > ma_val:
                score += float(rule.get("points", 0))
        elif rule_type == "field_positive":
            val = row.get(rule.get("field"))
            if pd.notna(val) and val > 0:
                score += float(rule.get("points", 0))
        elif rule_type == "capped_field":
            val = row.get(rule.get("field"))
            if pd.notna(val):
                score += min(float(rule.get("cap", 0)), float(val))

    return round(min(max_score, score), 4)


def compute_price_structure_score(
    row: pd.Series,
    config: dict[str, Any] | None = None,
) -> float | None:
    """价格结构评分（0-100），规则来自 factor_config.yaml"""
    cfg = _get_config(config)
    score_cfg = cfg.get("derived_scores", {}).get("price_structure_score", {})
    max_score = float(score_cfg.get("max_score", 100))

    score = 0.0
    has_signal = False

    for rule in score_cfg.get("rules", []):
        if rule.get("type") != "tier":
            continue
        val = row.get(rule.get("field"))
        if pd.isna(val):
            continue
        has_signal = True
        score += _apply_tier_rule(float(val), rule.get("tiers", []))

    if not has_signal:
        return None
    return round(min(max_score, score), 4)


def compute_sector_rank_score(rank: float | None, max_rank: int) -> float | None:
    """行业强度排名转为 0-100 分（1=最强 → 100）"""
    if rank is None or pd.isna(rank) or max_rank <= 1:
        return None
    return round((1.0 - (float(rank) - 1.0) / (max_rank - 1.0)) * 100.0, 4)


def _percentile_rank(series: pd.Series, missing_fill: float = 50.0) -> pd.Series:
    """截面分位排名（0-100），缺失值填 missing_fill"""
    valid = series.dropna()
    if valid.empty:
        return pd.Series(missing_fill, index=series.index, dtype=float)

    ranks = valid.rank(method="average", pct=True) * 100.0
    out = pd.Series(missing_fill, index=series.index, dtype=float)
    out.loc[ranks.index] = ranks
    return out


def _prepare_factor_value(
    df: pd.DataFrame,
    column: str,
    *,
    use_percentile_norm: bool,
    config: dict[str, Any],
) -> pd.Series:
    """取打分用数值列，并按需要做截面归一化"""
    norm_cfg = config.get("normalization", {})
    missing_fill = float(norm_cfg.get("missing_fill", 50))
    scale_fields = set(norm_cfg.get("scale_0_100_fields", []))
    ratio_fields = set(norm_cfg.get("ratio_0_1_fields", []))

    if column not in df.columns:
        return pd.Series(missing_fill, index=df.index, dtype=float)

    raw = df[column]
    if column in ratio_fields:
        raw = raw * 100.0

    if not use_percentile_norm or column in scale_fields:
        return raw.fillna(missing_fill).astype(float)

    return _percentile_rank(raw.astype(float), missing_fill=missing_fill)


def _weighted_components(
    df: pd.DataFrame,
    components: list[dict[str, Any]],
    *,
    use_percentile_norm: bool,
    config: dict[str, Any],
) -> pd.Series:
    """按配置中的 field/weight 计算加权和"""
    total = pd.Series(0.0, index=df.index, dtype=float)
    for comp in components:
        field = comp.get("field")
        weight = float(comp.get("weight", 0))
        if not field:
            continue
        values = _prepare_factor_value(
            df,
            field,
            use_percentile_norm=use_percentile_norm,
            config=config,
        )
        total = total + weight * values
    return total.round(4)


def _attach_sector_features(
    pool_df: pd.DataFrame,
    sector_df: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """合并行业强度字段到股票池"""
    if sector_df is None or sector_df.empty:
        return pool_df.copy()

    merge_cfg = config.get("sector_merge", {})
    join_key = merge_cfg.get("join_key", "industry")
    if join_key not in pool_df.columns:
        return pool_df.copy()

    merge_cols = merge_cfg.get("columns", [])
    sector_cols = [join_key] + [c for c in merge_cols if c in sector_df.columns]
    sector_part = sector_df[sector_cols].drop_duplicates(subset=[join_key], keep="first")

    merged = pool_df.merge(sector_part, on=join_key, how="left", suffixes=("", "_sector"))

    rank_col = "sector_strength_rank"
    if rank_col in sector_part.columns:
        max_rank = int(sector_part[rank_col].max())
        if max_rank > 0:
            merged["sector_rank_score"] = merged[rank_col].apply(
                lambda r: compute_sector_rank_score(r, max_rank)
            )
        else:
            merged["sector_rank_score"] = np.nan
    else:
        merged["sector_rank_score"] = np.nan

    return merged


def _compute_derived_scores(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """计算配置中定义的衍生评分列"""
    out = df.copy()
    derived = config.get("derived_scores", {})
    if "ma_trend_score" in derived and "ma_trend_score" not in out.columns:
        out["ma_trend_score"] = out.apply(
            lambda row: compute_ma_trend_score(row, config),
            axis=1,
        )
    if "price_structure_score" in derived and "price_structure_score" not in out.columns:
        out["price_structure_score"] = out.apply(
            lambda row: compute_price_structure_score(row, config),
            axis=1,
        )
    return out


def _compute_factor_scores(
    df: pd.DataFrame,
    *,
    use_percentile_norm: bool,
    config: dict[str, Any],
) -> pd.DataFrame:
    """计算四类因子分"""
    out = _compute_derived_scores(df, config)

    for _name, factor_cfg in config.get("factors", {}).items():
        output_col = factor_cfg.get("output_column")
        components = factor_cfg.get("components", [])
        if not output_col or not components:
            continue
        out[output_col] = _weighted_components(
            out,
            components,
            use_percentile_norm=use_percentile_norm,
            config=config,
        )

    return out


def score_stock_pool(
    pool_df: pd.DataFrame,
    sector_df: pd.DataFrame,
    market_df: pd.DataFrame,
    *,
    use_percentile_norm: bool | None = None,
    factor_config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    对股票池打分并写入因子分、权重与总分。

    因子公式与市场权重见 config/factor_config.yaml。
    """
    if pool_df is None or pool_df.empty:
        return pool_df.copy() if pool_df is not None else pd.DataFrame()

    config = _get_config(factor_config)
    norm_cfg = config.get("normalization", {})
    if use_percentile_norm is None:
        use_percentile_norm = bool(norm_cfg.get("use_percentile_norm", True))

    snapshot = get_primary_market_snapshot(market_df)
    regime, weights = resolve_market_weights(snapshot.get("market_risk_index"), config)

    scored = _attach_sector_features(pool_df, sector_df, config)
    scored = _compute_factor_scores(
        scored,
        use_percentile_norm=use_percentile_norm,
        config=config,
    )

    final_cfg = config.get("final_score", {})
    regime_col = final_cfg.get("regime_output_column", "score_market_regime")
    weight_cols = final_cfg.get("weight_output_columns", {})
    final_col = final_cfg.get("output_column", "final_score")

    scored[regime_col] = regime
    for factor_key, col_name in weight_cols.items():
        scored[col_name] = weights.get(factor_key)

    factor_outputs = {
        cfg.get("output_column"): factor_key
        for factor_key, cfg in config.get("factors", {}).items()
        if cfg.get("output_column")
    }
    final_expr = pd.Series(0.0, index=scored.index, dtype=float)
    for output_col, factor_key in factor_outputs.items():
        if output_col in scored.columns and factor_key in weights:
            final_expr = final_expr + scored[output_col] * weights[factor_key]
    scored[final_col] = final_expr.round(4)

    sort_asc = bool(final_cfg.get("sort_ascending", False))
    scored = scored.sort_values(final_col, ascending=sort_asc, na_position="last").reset_index(
        drop=True
    )

    for col in config.get("future_return_columns", []):
        scored[col] = np.nan

    return scored


def summarize_scores(df: pd.DataFrame) -> dict[str, Any]:
    """汇总打分结果，供日志/测试使用"""
    if df is None or df.empty:
        return {
            "count": 0,
            "market_regime": None,
            "final_score_min": None,
            "final_score_max": None,
            "final_score_mean": None,
            "top5": [],
        }

    config = load_factor_config()
    final_col = config.get("final_score", {}).get("output_column", "final_score")
    regime_col = config.get("final_score", {}).get("regime_output_column", "score_market_regime")

    top = df.nlargest(5, final_col)[["ts_code", "name", final_col]].rename(
        columns={final_col: "final_score"}
    )
    return {
        "count": len(df),
        "market_regime": df[regime_col].iloc[0] if regime_col in df.columns else None,
        "final_score_min": float(df[final_col].min()),
        "final_score_max": float(df[final_col].max()),
        "final_score_mean": round(float(df[final_col].mean()), 4),
        "top5": top.to_dict(orient="records"),
    }
