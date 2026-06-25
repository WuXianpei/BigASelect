"""根据成分 IC 生成 factor_config.proposed.yaml"""

from __future__ import annotations

import copy
from typing import Any

import yaml


def propose_factor_config(
    factor_config: dict[str, Any],
    component_ic: dict[str, float],
    proposed_cfg: dict[str, Any],
) -> dict[str, Any]:
    """
    按成分 IC 调整各类因子内权重，保留权重符号，不修改 market_regime。
    """
    boost = float(proposed_cfg.get("ic_boost_factor", 5.0))
    min_w = float(proposed_cfg.get("min_component_weight", 0.05))
    note = proposed_cfg.get(
        "source_note",
        "由因子有效性分析自动生成，请人工审阅后替换 config/factor_config.yaml",
    )

    new_cfg = copy.deepcopy(factor_config)
    changes: list[str] = []

    for factor_name, factor_block in new_cfg.get("factors", {}).items():
        components = factor_block.get("components", [])
        if not components:
            continue

        magnitudes: list[float] = []
        for comp in components:
            field = comp.get("field", "")
            ic = component_ic.get(field, 0.0)
            orig = float(comp.get("weight", 0))
            sign = 1.0 if orig >= 0 else -1.0
            # IC 与原始符号一致则放大，相反则缩小
            if orig >= 0 and ic >= 0:
                mag = abs(orig) * (1.0 + ic * boost)
            elif orig < 0 and ic <= 0:
                mag = abs(orig) * (1.0 + abs(ic) * boost)
            else:
                mag = max(abs(orig) * 0.5, min_w)
            magnitudes.append(max(mag, min_w))

        total = sum(magnitudes) or 1.0
        for comp, mag in zip(components, magnitudes):
            orig = float(comp.get("weight", 0))
            sign = 1.0 if orig >= 0 else -1.0
            new_w = round(sign * mag / total, 4)
            if abs(new_w - orig) > 1e-4:
                changes.append(
                    f"{factor_name}.{comp.get('field')}: {orig} -> {new_w} (IC={component_ic.get(comp.get('field'), 0)})"
                )
            comp["weight"] = new_w

    header = (
        f"# {note}\n"
        f"# 变更摘要（{len(changes)} 项）:\n"
        + "".join(f"#   - {line}\n" for line in changes[:20])
    )
    if len(changes) > 20:
        header += f"#   ... 另有 {len(changes) - 20} 项\n"

    new_cfg["_proposed_meta"] = {
        "changes": changes,
        "component_ic": component_ic,
    }
    new_cfg["_yaml_header_comment"] = header
    return new_cfg


def write_proposed_config(cfg: dict[str, Any], path) -> None:
    """写入 proposed yaml（带顶部注释）"""
    meta = cfg.pop("_proposed_meta", None)
    header = cfg.pop("_yaml_header_comment", "")
    path.parent.mkdir(parents=True, exist_ok=True)
    body = yaml.dump(cfg, allow_unicode=True, sort_keys=False, default_flow_style=False)
    path.write_text(header + "\n" + body, encoding="utf-8")
    if meta:
        cfg["_proposed_meta"] = meta
