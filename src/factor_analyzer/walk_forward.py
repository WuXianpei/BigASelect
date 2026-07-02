"""Walk-forward 样本外评估（C 方案）"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.factor_analyzer.metrics import (
    compute_component_ic_stats,
    evaluate_score_on_panel,
    slice_panel_by_dates,
    split_walk_forward_dates,
)
from src.factor_analyzer.factor_pool import (
    apply_factor_pool_changes,
    run_factor_pool_screening,
)
from src.factor_analyzer.optimizer import propose_factor_config
from src.factor_analyzer.ridge_optimizer import propose_factor_config_ridge_regime
from src.factor_analyzer.rescorer import rescore_archived_day


def _metric_value(eval_result: dict[str, Any], metric: str) -> float | None:
    """从评估结果提取主指标数值"""
    if metric == "quintile_spread":
        return eval_result.get("quintile", {}).get("quintile_spread")
    ic = eval_result.get("ic", {})
    if metric == "ic_mean":
        return ic.get("ic_mean")
    return ic.get("ic_ir")


def build_panel_for_dates(
    dates: list[str],
    *,
    factor_config: dict[str, Any],
    return_col: str,
    archive_root,
) -> pd.DataFrame:
    """用指定 factor_config 重算若干交易日的面板"""
    parts: list[pd.DataFrame] = []
    for td in dates:
        scored = rescore_archived_day(td, factor_config=factor_config, root=archive_root)
        if scored is None or scored.empty or return_col not in scored.columns:
            continue
        day = scored.dropna(subset=[return_col]).copy()
        if day.empty:
            continue
        day["trade_date"] = td
        parts.append(day)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def run_walk_forward_optimization(
    panel: pd.DataFrame,
    analysis_dates: list[str],
    *,
    factor_config: dict[str, Any],
    analysis_cfg: dict[str, Any],
    primary_score: str,
    return_col: str,
    archive_root,
) -> dict[str, Any]:
    """
    C 方案：训练集定权 → 验证/测试集样本外评估。
    返回 baseline vs proposed 在各分段上的指标及是否推荐替换。
    """
    wf_cfg = analysis_cfg.get("walk_forward", {})
    proposed_cfg = analysis_cfg.get("proposed_config", {})
    if not wf_cfg.get("enabled", True):
        return {"enabled": False, "reason": "walk_forward 未启用"}

    split = split_walk_forward_dates(
        analysis_dates,
        train_days=int(wf_cfg.get("train_days", 90)),
        validate_days=int(wf_cfg.get("validate_days", 30)),
        test_days=int(wf_cfg.get("test_days", 40)),
        min_total_days=int(wf_cfg.get("min_total_days", 160)),
    )
    if not split.get("enabled"):
        return {"enabled": False, "reason": split.get("reason", "无法切分")}

    train_dates = split["train_dates"]
    val_dates = split["validate_dates"]
    test_dates = split["test_dates"]

    train_panel = slice_panel_by_dates(panel, train_dates)
    time_decay = proposed_cfg.get("time_decay", {})
    component_stats = compute_component_ic_stats(
        train_panel,
        factor_config,
        return_col=return_col,
        time_decay=time_decay if proposed_cfg.get("optimization_method") == "ic_heuristic" else None,
    )

    pool_screening = run_factor_pool_screening(
        factor_config,
        train_panel,
        component_stats,
        analysis_cfg,
        return_col=return_col,
        analysis_day_count=len(analysis_dates),
    )

    opt_method = str(proposed_cfg.get("optimization_method", "ridge_regime"))

    if opt_method == "ridge_regime":
        proposed = propose_factor_config_ridge_regime(
            factor_config,
            train_panel,
            proposed_cfg,
            return_col=return_col,
            tune_mode="walk_forward",
        )
    else:
        proposed = propose_factor_config(
            factor_config,
            component_stats,
            proposed_cfg,
            tune_mode="walk_forward",
        )

    pool_applied: list[str] = []
    if pool_screening.get("enabled"):
        pool_cfg = analysis_cfg.get("factor_pool", {})
        proposed, pool_applied = apply_factor_pool_changes(
            proposed,
            removals=pool_screening.get("removal_candidates", []),
            additions=pool_screening.get("addition_candidates", []),
            add_cfg=pool_cfg.get("addition", {}),
        )
        pool_screening["applied_changes"] = pool_applied
        meta = proposed.setdefault("_proposed_meta", {})
        if pool_applied:
            meta.setdefault("changes", []).extend(pool_applied)
            meta["pool_changes"] = pool_applied

    oos_metric = str(wf_cfg.get("oos_primary_metric", "ic_ir"))
    require_improve = bool(wf_cfg.get("require_test_improvement", True))

    baseline_eval: dict[str, dict[str, Any]] = {}
    proposed_eval: dict[str, dict[str, Any]] = {}

    for label, dates in (
        ("train", train_dates),
        ("validate", val_dates),
        ("test", test_dates),
    ):
        base_slice = slice_panel_by_dates(panel, dates)
        baseline_eval[label] = evaluate_score_on_panel(
            base_slice, primary_score, return_col=return_col
        )
        prop_panel = build_panel_for_dates(
            dates,
            factor_config=proposed,
            return_col=return_col,
            archive_root=archive_root,
        )
        proposed_eval[label] = evaluate_score_on_panel(
            prop_panel, primary_score, return_col=return_col
        )

    base_test = _metric_value(baseline_eval["test"], oos_metric)
    prop_test = _metric_value(proposed_eval["test"], oos_metric)
    base_val = _metric_value(baseline_eval["validate"], oos_metric)
    prop_val = _metric_value(proposed_eval["validate"], oos_metric)

    test_improved = None
    if base_test is not None and prop_test is not None:
        test_improved = prop_test > base_test
    elif not require_improve:
        test_improved = True

    recommend = bool(test_improved) if require_improve else True

    return {
        "enabled": True,
        "split": split,
        "oos_primary_metric": oos_metric,
        "component_stats_train": component_stats,
        "baseline": baseline_eval,
        "proposed": proposed_eval,
        "test_improvement": {
            "baseline": base_test,
            "proposed": prop_test,
            "improved": test_improved,
            "metric": oos_metric,
        },
        "validate_improvement": {
            "baseline": base_val,
            "proposed": prop_val,
            "improved": prop_val > base_val if base_val is not None and prop_val is not None else None,
            "metric": oos_metric,
        },
        "recommend_replace": recommend,
        "proposed_config": proposed,
        "proposed_meta": proposed.get("_proposed_meta", {}),
        "optimization_method": opt_method,
        "factor_pool": pool_screening,
    }
