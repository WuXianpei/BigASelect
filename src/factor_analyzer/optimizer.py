"""根据成分 IC 统计生成 factor_config.proposed.yaml（A 方案）"""

from __future__ import annotations

import copy
from typing import Any

import yaml


def _normalize_component_stats(
    component_input: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """兼容旧版 flat IC 字典与新 stats 结构"""
    if not component_input:
        return {}
    sample = next(iter(component_input.values()))
    if isinstance(sample, dict):
        return component_input  # type: ignore[return-value]
    return {
        field: {"ic_mean": float(val), "ic_ir": float(val), "ic_mean_weighted": float(val)}
        for field, val in component_input.items()
    }


def _component_signal(
    stats: dict[str, Any],
    *,
    use_ic_ir: bool,
) -> float:
    """调权信号：优先 IC_IR，否则时间衰减加权 IC 均值"""
    if use_ic_ir:
        return float(stats.get("ic_ir", 0.0))
    return float(stats.get("ic_mean_weighted", stats.get("ic_mean", 0.0)))


def _clamp_magnitude(orig: float, new_mag: float, cap_ratio: float, min_w: float) -> float:
    """单次调权幅度上限（相对原权重绝对值 ±cap_ratio）"""
    if abs(orig) < 1e-9:
        return max(new_mag, min_w)
    lo = abs(orig) * (1.0 - cap_ratio)
    hi = abs(orig) * (1.0 + cap_ratio)
    return max(min(new_mag, hi), lo, min_w)


def propose_factor_config(
    factor_config: dict[str, Any],
    component_stats: dict[str, Any],
    proposed_cfg: dict[str, Any],
    *,
    tune_mode: str = "auto",
) -> dict[str, Any]:
    """
    A 方案：时间衰减 IC / IC_IR 驱动调权，保留符号，类内归一化，单次变动有上限。
    tune_mode: walk_forward | ineffective | soft_tune
    """
    stats_map = _normalize_component_stats(component_stats)
    boost = float(proposed_cfg.get("ic_boost_factor", 3.0))
    min_w = float(proposed_cfg.get("min_component_weight", 0.05))
    cap_ratio = float(proposed_cfg.get("max_weight_change_ratio", 0.30))
    use_ic_ir = bool(proposed_cfg.get("use_ic_ir", True))
    neg_threshold = float(proposed_cfg.get("negative_signal_threshold", 0.0))
    opposite_penalty = float(proposed_cfg.get("opposite_direction_penalty", 0.5))
    signal_clip = float(proposed_cfg.get("signal_clip", 2.0))

    mode_notes = {
        "walk_forward": "Walk-forward 训练集 IC 定权（C 方案验证）",
        "ineffective": "模型判定失效，全样本 IC 定权",
        "soft_tune": "模型仍有效，微调建议（需结合样本外结论）",
    }
    note = proposed_cfg.get(
        "source_note",
        "由因子有效性分析自动生成，请人工审阅后替换 config/factor_config.yaml",
    )
    mode_label = mode_notes.get(tune_mode, tune_mode)

    new_cfg = copy.deepcopy(factor_config)
    changes: list[str] = []

    for factor_name, factor_block in new_cfg.get("factors", {}).items():
        components = factor_block.get("components", [])
        if not components:
            continue

        magnitudes: list[float] = []
        for comp in components:
            field = comp.get("field", "")
            stats = stats_map.get(field, {})
            signal = _component_signal(stats, use_ic_ir=use_ic_ir)
            signal = max(-signal_clip, min(signal_clip, signal))
            orig = float(comp.get("weight", 0))

            # 信号与权重方向相反，或信号低于负阈值 → 降权
            if (orig >= 0 and signal < neg_threshold) or (orig < 0 and signal > -neg_threshold):
                mag = max(abs(orig) * opposite_penalty, min_w)
            elif orig >= 0 and signal >= neg_threshold:
                mag = abs(orig) * (1.0 + signal * boost)
            elif orig < 0 and signal <= -neg_threshold:
                mag = abs(orig) * (1.0 + abs(signal) * boost)
            else:
                mag = max(abs(orig) * opposite_penalty, min_w)

            mag = _clamp_magnitude(orig, mag, cap_ratio, min_w)
            magnitudes.append(mag)

        total = sum(magnitudes) or 1.0
        for comp, mag in zip(components, magnitudes):
            orig = float(comp.get("weight", 0))
            sign = 1.0 if orig >= 0 else -1.0
            new_w = round(sign * mag / total, 4)
            # 归一化后再做一次幅度上限
            capped_mag = _clamp_magnitude(orig, abs(new_w), cap_ratio, min_w)
            new_w = round(sign * capped_mag, 4)
            if abs(new_w - orig) > 1e-4:
                st = stats_map.get(comp.get("field", ""), {})
                sig_label = "IC_IR" if use_ic_ir else "IC_w"
                sig_val = _component_signal(st, use_ic_ir=use_ic_ir)
                changes.append(
                    f"{factor_name}.{comp.get('field')}: {orig} -> {new_w} "
                    f"({sig_label}={sig_val})"
                )
            comp["weight"] = new_w

        # 二次归一化（cap 后绝对值之和可能偏离 1）
        abs_sum = sum(abs(float(c.get("weight", 0))) for c in components) or 1.0
        for comp in components:
            orig = float(comp.get("weight", 0))
            sign = 1.0 if orig >= 0 else -1.0
            comp["weight"] = round(sign * abs(orig) / abs_sum, 4)

    header = (
        f"# {note}\n"
        f"# 调权模式: {mode_label}\n"
        f"# 变更摘要（{len(changes)} 项）:\n"
        + "".join(f"#   - {line}\n" for line in changes[:25])
    )
    if len(changes) > 25:
        header += f"#   ... 另有 {len(changes) - 25} 项\n"

    new_cfg["_proposed_meta"] = {
        "changes": changes,
        "component_stats": stats_map,
        "tune_mode": tune_mode,
        "use_ic_ir": use_ic_ir,
    }
    new_cfg["_yaml_header_comment"] = header
    return new_cfg


def write_proposed_config(cfg: dict[str, Any], path, *, recommend: bool | None = None) -> None:
    """写入 proposed yaml（带顶部注释）"""
    meta = cfg.pop("_proposed_meta", None)
    header = cfg.pop("_yaml_header_comment", "")
    if recommend is True:
        header += "# 结论: 样本外测试集改善，建议审阅后替换 config/factor_config.yaml\n"
    elif recommend is False:
        header += "# 结论: 样本外测试集未改善，仅供参考，暂不推荐直接替换\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    body = yaml.dump(cfg, allow_unicode=True, sort_keys=False, default_flow_style=False)
    path.write_text(header + "\n" + body, encoding="utf-8")
    if meta:
        cfg["_proposed_meta"] = meta
