"""因子池增删筛选（保守：训练集提名，Walk-forward 测试门禁）"""

from __future__ import annotations

import copy
from typing import Any

import pandas as pd

from src.factor_analyzer.metrics import compute_component_ic_stats


def _collect_fields(factor_config: dict[str, Any]) -> dict[str, tuple[str, dict]]:
    """field -> (factor_name, component dict)"""
    mapping: dict[str, tuple[str, dict]] = {}
    for fname, block in factor_config.get("factors", {}).items():
        for comp in block.get("components", []):
            field = comp.get("field")
            if field:
                mapping[field] = (fname, comp)
    return mapping


def _renormalize_factor_group(components: list[dict]) -> None:
    """类内按绝对值归一化权重"""
    total = sum(abs(float(c.get("weight", 0))) for c in components) or 1.0
    for comp in components:
        w = float(comp.get("weight", 0))
        sign = 1.0 if w >= 0 else -1.0
        comp["weight"] = round(sign * abs(w) / total, 4)


def screen_removal_candidates(
    component_stats: dict[str, dict[str, Any]],
    pool_cfg: dict[str, Any],
    factor_config: dict[str, Any],
) -> list[dict[str, Any]]:
    """训练集上 IC_IR 偏弱的成分因子，建议剔除（每类至少保留 1 个）"""
    rem_cfg = pool_cfg.get("removal", {})
    if not pool_cfg.get("enabled", True) or not rem_cfg.get("enabled", True):
        return []

    ic_ir_max = float(rem_cfg.get("ic_ir_max", 0.0))
    min_days = int(rem_cfg.get("min_ic_days", 15))
    max_n = int(rem_cfg.get("max_per_run", 2))

    group_counts: dict[str, int] = {}
    for fname, block in factor_config.get("factors", {}).items():
        group_counts[fname] = len(block.get("components", []))

    candidates: list[dict[str, Any]] = []
    for field, stats in component_stats.items():
        ic_ir = float(stats.get("ic_ir", 0.0))
        ic_days = int(stats.get("ic_days", 0))
        if ic_days < min_days or ic_ir >= ic_ir_max:
            continue
        factor_name, _ = _collect_fields(factor_config).get(field, (None, {}))
        if not factor_name or group_counts.get(factor_name, 0) <= 1:
            continue
        candidates.append(
            {
                "field": field,
                "factor": factor_name,
                "ic_ir": ic_ir,
                "ic_mean": stats.get("ic_mean"),
                "ic_days": ic_days,
                "action": "remove",
            }
        )

    candidates.sort(key=lambda x: x["ic_ir"])
    return candidates[:max_n]


def screen_addition_candidates(
    train_panel: pd.DataFrame,
    factor_config: dict[str, Any],
    pool_cfg: dict[str, Any],
    *,
    return_col: str,
    analysis_day_count: int,
) -> list[dict[str, Any]]:
    """白名单候选字段中 IC 较好且尚未纳入配置的因子"""
    add_cfg = pool_cfg.get("addition", {})
    if not pool_cfg.get("enabled", True) or not add_cfg.get("enabled", True):
        return []

    min_days = int(add_cfg.get("min_analysis_days", 200))
    if analysis_day_count < min_days:
        return [
            {
                "action": "add_skipped",
                "reason": f"分析窗口 {analysis_day_count} 日 < {min_days}，暂不自动纳入",
            }
        ]

    raw_candidates = add_cfg.get("candidates", {})
    if not raw_candidates:
        return []

    existing = set(_collect_fields(factor_config).keys())
    pending_fields = [f for f in raw_candidates if f not in existing and f in train_panel.columns]
    if not pending_fields:
        return []

    # 仅对候选字段算 IC
    pseudo_cfg = {
        "factors": {
            "candidates": {
                "components": [{"field": f} for f in pending_fields],
            }
        }
    }
    stats = compute_component_ic_stats(train_panel, pseudo_cfg, return_col=return_col)
    ic_ir_min = float(add_cfg.get("ic_ir_min", 0.05))
    min_ic_days = int(add_cfg.get("min_ic_days", 15))
    max_n = int(add_cfg.get("max_per_run", 1))

    candidates: list[dict[str, Any]] = []
    for field, st in stats.items():
        ic_ir = float(st.get("ic_ir", 0.0))
        ic_days = int(st.get("ic_days", 0))
        if ic_days < min_ic_days or ic_ir < ic_ir_min:
            continue
        candidates.append(
            {
                "field": field,
                "factor": raw_candidates[field],
                "ic_ir": ic_ir,
                "ic_mean": st.get("ic_mean"),
                "ic_days": ic_days,
                "action": "add",
            }
        )

    candidates.sort(key=lambda x: x["ic_ir"], reverse=True)
    return candidates[:max_n]


def apply_factor_pool_changes(
    factor_config: dict[str, Any],
    *,
    removals: list[dict[str, Any]],
    additions: list[dict[str, Any]],
    add_cfg: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """将增删应用到配置副本，返回 (新配置, 变更说明)"""
    new_cfg = copy.deepcopy(factor_config)
    changes: list[str] = []

    for item in removals:
        if item.get("action") != "remove":
            continue
        field = item["field"]
        factor_name = item["factor"]
        block = new_cfg.get("factors", {}).get(factor_name, {})
        comps = block.get("components", [])
        new_comps = [c for c in comps if c.get("field") != field]
        if len(new_comps) == len(comps) or not new_comps:
            continue
        block["components"] = new_comps
        _renormalize_factor_group(new_comps)
        changes.append(
            f"删除 {factor_name}.{field}（训练 IC_IR={item.get('ic_ir')}）"
        )

    init_w = float(add_cfg.get("initial_weight", 0.15))
    for item in additions:
        if item.get("action") != "add":
            continue
        field = item["field"]
        factor_name = item["factor"]
        block = new_cfg.get("factors", {}).get(factor_name)
        if not block:
            continue
        comps = block.setdefault("components", [])
        if any(c.get("field") == field for c in comps):
            continue
        sign = 1.0 if float(item.get("ic_mean", 0) or 0) >= 0 else -1.0
        comps.append({"field": field, "weight": round(sign * init_w, 4)})
        _renormalize_factor_group(comps)
        changes.append(
            f"新增 {factor_name}.{field}（训练 IC_IR={item.get('ic_ir')}）"
        )

    return new_cfg, changes


def run_factor_pool_screening(
    factor_config: dict[str, Any],
    train_panel: pd.DataFrame,
    component_stats: dict[str, dict[str, Any]],
    analysis_cfg: dict[str, Any],
    *,
    return_col: str,
    analysis_day_count: int,
) -> dict[str, Any]:
    """训练集筛选增删建议（是否写入 proposed 由 Walk-forward 测试门禁决定）"""
    pool_cfg = analysis_cfg.get("factor_pool", {})
    if not pool_cfg.get("enabled", True):
        return {
            "enabled": False,
            "removal_candidates": [],
            "addition_candidates": [],
            "addition_skipped_reason": None,
        }

    removals = screen_removal_candidates(component_stats, pool_cfg, factor_config)
    additions_raw = screen_addition_candidates(
        train_panel,
        factor_config,
        pool_cfg,
        return_col=return_col,
        analysis_day_count=analysis_day_count,
    )
    additions = [a for a in additions_raw if a.get("action") == "add"]
    add_skipped_note = next(
        (a.get("reason") for a in additions_raw if a.get("action") == "add_skipped"),
        None,
    )

    return {
        "enabled": True,
        "removal_candidates": removals,
        "addition_candidates": additions,
        "addition_skipped_reason": add_skipped_note,
    }
