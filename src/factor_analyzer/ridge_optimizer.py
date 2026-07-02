"""Ridge 截面回归 + Regime 大类权重优化（保守方案）"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pandas as pd

from src.factor_analyzer.optimizer import _clamp_magnitude, write_proposed_config

# 导出 write_proposed_config 供外部使用
__all__ = [
    "propose_factor_config_ridge_regime",
    "write_proposed_config",
    "fit_ridge",
]


def fit_ridge(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """Ridge 回归系数（含截距列时截距不惩罚）"""
    n_features = X.shape[1]
    penalty = np.eye(n_features) * alpha
    # 第一列为截距时不惩罚
    if n_features > 0:
        penalty[0, 0] = 0.0
    xtx = X.T @ X + penalty
    xty = X.T @ y
    try:
        return np.linalg.solve(xtx, xty)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(xtx, xty, rcond=None)[0]


def _standardize_matrix(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回标准化后的 X（保留 intercept 列）、均值、标准差"""
    mu = X.mean(axis=0)
    sigma = X.std(axis=0, ddof=0)
    sigma = np.where(sigma < 1e-9, 1.0, sigma)
    Xs = (X - mu) / sigma
    return Xs, mu, sigma


def _select_ridge_alpha(
    X: np.ndarray,
    y: np.ndarray,
    alphas: list[float],
) -> float:
    """按训练集 RSS 选择最小正则化强度（保守：略偏大优先）"""
    best_alpha = alphas[0]
    best_rss = float("inf")
    for alpha in sorted(alphas):
        beta = fit_ridge(X, y, alpha)
        resid = y - X @ beta
        rss = float(np.sum(resid**2))
        if rss < best_rss * 0.999:
            best_rss = rss
            best_alpha = alpha
    return best_alpha


def _collect_component_fields(factor_config: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """factor_name -> components 列表"""
    out: dict[str, list[dict[str, Any]]] = {}
    for name, block in factor_config.get("factors", {}).items():
        comps = block.get("components", [])
        if comps:
            out[name] = comps
    return out


def _prepare_ridge_frame(
    df: pd.DataFrame,
    fields: list[str],
    return_col: str,
    *,
    min_coverage: float = 0.5,
) -> tuple[pd.DataFrame, list[str]]:
    """筛选覆盖率足够的字段，缺失值用列中位数填充"""
    usable: list[str] = []
    for f in fields:
        if f not in df.columns:
            continue
        cov = df[f].notna().mean()
        if cov >= min_coverage:
            usable.append(f)
    if not usable:
        return pd.DataFrame(), []

    sub = df[usable + [return_col]].copy()
    sub = sub.dropna(subset=[return_col])
    for f in usable:
        med = sub[f].median()
        sub[f] = sub[f].fillna(med)
    sub = sub.dropna()
    return sub, usable


def _fit_component_ridge_weights(
    train_panel: pd.DataFrame,
    factor_config: dict[str, Any],
    *,
    return_col: str,
    ridge_cfg: dict[str, Any],
    proposed_cfg: dict[str, Any],
) -> tuple[dict[str, float], float]:
    """全局 Ridge：成分字段 -> future_return，映射各类内权重"""
    groups = _collect_component_fields(factor_config)
    all_fields: list[str] = []
    for comps in groups.values():
        for c in comps:
            f = c.get("field")
            if f and f not in all_fields:
                all_fields.append(f)

    df = train_panel.dropna(subset=[return_col]).copy()
    min_cov = float(ridge_cfg.get("min_field_coverage", 0.5))
    sub, usable = _prepare_ridge_frame(df, all_fields, return_col, min_coverage=min_cov)
    if len(usable) < 2 or len(sub) < 100:
        return {}, float(ridge_cfg.get("fixed_alpha", 1.0))

    X_raw = sub[usable].astype(float).values
    y = sub[return_col].astype(float).values

    if ridge_cfg.get("standardize", True):
        X_body, _, _ = _standardize_matrix(X_raw)
    else:
        X_body = X_raw

    intercept = np.ones((len(y), 1))
    X = np.hstack([intercept, X_body])

    alphas = [float(a) for a in ridge_cfg.get("alphas", [0.1, 1.0, 10.0])]
    if ridge_cfg.get("alpha_select") == "fixed":
        alpha = float(ridge_cfg.get("fixed_alpha", 1.0))
    else:
        alpha = _select_ridge_alpha(X, y, alphas)

    beta = fit_ridge(X, y, alpha)
    coef_map = {usable[i]: float(beta[i + 1]) for i in range(len(usable))}

    min_w = float(proposed_cfg.get("min_component_weight", 0.05))
    cap = float(proposed_cfg.get("max_weight_change_ratio", 0.30))

    weight_map: dict[str, float] = {}
    for _fname, comps in groups.items():
        mags: list[float] = []
        signs: list[float] = []
        origs: list[float] = []
        fields: list[str] = []
        for comp in comps:
            field = comp.get("field", "")
            orig = float(comp.get("weight", 0))
            beta_v = coef_map.get(field, 0.0)
            if abs(beta_v) < 1e-8:
                sign = 1.0 if orig >= 0 else -1.0
                mag = max(abs(orig) * 0.5, min_w)
            else:
                sign = 1.0 if beta_v >= 0 else -1.0
                mag = max(abs(beta_v), min_w)
            mag = _clamp_magnitude(orig, mag, cap, min_w)
            mags.append(mag)
            signs.append(sign)
            origs.append(orig)
            fields.append(field)

        total = sum(mags) or 1.0
        for field, sign, mag, orig in zip(fields, signs, mags, origs):
            weight_map[field] = round(sign * mag / total, 4)

    return weight_map, alpha


def _fit_regime_category_weights(
    train_panel: pd.DataFrame,
    factor_config: dict[str, Any],
    *,
    return_col: str,
    regime_cfg: dict[str, Any],
    ridge_cfg: dict[str, Any],
) -> tuple[dict[str, dict[str, float]], list[str]]:
    """
    按 score_market_regime 分档，Ridge 定 value/growth/capital/sector 大类权重。
    样本不足时保留原权重。
    """
    regime_col = factor_config.get("final_score", {}).get(
        "regime_output_column", "score_market_regime"
    )
    score_cols = {
        key: block.get("output_column")
        for key, block in factor_config.get("factors", {}).items()
        if block.get("output_column")
    }
    category_keys = ["value", "growth", "capital", "sector"]
    orig_table = copy.deepcopy(
        factor_config.get("market_regime", {}).get("weights", {})
    )
    new_table: dict[str, dict[str, float]] = copy.deepcopy(orig_table)
    notes: list[str] = []

    min_days = int(regime_cfg.get("min_days", 8))
    min_rows = int(regime_cfg.get("min_rows", 500))
    cap = float(regime_cfg.get("max_category_weight_change", 0.15))
    alpha = float(ridge_cfg.get("fixed_alpha", 1.0))
    if ridge_cfg.get("alpha_select") != "fixed":
        alpha = float(ridge_cfg.get("regime_alpha", 10.0))

    if regime_col not in train_panel.columns:
        notes.append("训练面板无 score_market_regime，跳过大类分档优化")
        return new_table, notes

    for regime in ("low", "neutral", "high"):
        if regime not in orig_table:
            continue
        orig_w = orig_table[regime]
        sub = train_panel[train_panel[regime_col].astype(str) == regime].copy()
        n_days = sub["trade_date"].nunique() if "trade_date" in sub.columns else 0
        if len(sub) < min_rows or n_days < min_days:
            notes.append(f"{regime}: 样本不足({len(sub)}行/{n_days}日)，保留原权重")
            continue

        feats = [score_cols[k] for k in category_keys if score_cols.get(k) in sub.columns]
        if len(feats) < 2:
            continue

        mat = sub[feats + [return_col]].dropna()
        if len(mat) < min_rows:
            notes.append(f"{regime}: 有效样本不足，保留原权重")
            continue

        X_raw = mat[feats].astype(float).values
        y = mat[return_col].astype(float).values
        if ridge_cfg.get("standardize", True):
            X_body, _, _ = _standardize_matrix(X_raw)
        else:
            X_body = X_raw
        X = np.hstack([np.ones((len(y), 1)), X_body])
        beta = fit_ridge(X, y, alpha)

        raw: dict[str, float] = {}
        for i, key in enumerate(category_keys):
            col = score_cols.get(key)
            if col not in feats:
                raw[key] = float(orig_w.get(key, 0.25))
                continue
            idx = feats.index(col) + 1
            b = float(beta[idx])
            # 得分越高越好；负系数降至 min 权重
            raw[key] = max(b, 0.01)

        total = sum(raw.values()) or 1.0
        proposed_w: dict[str, float] = {}
        for key in category_keys:
            orig = float(orig_w.get(key, 0.25))
            target = raw.get(key, orig) / total
            lo = max(orig * (1 - cap), 0.05)
            hi = min(orig * (1 + cap), 0.85)
            proposed_w[key] = round(max(min(target, hi), lo), 4)

        wsum = sum(proposed_w.values()) or 1.0
        new_table[regime] = {k: round(v / wsum, 4) for k, v in proposed_w.items()}
        notes.append(
            f"{regime}: "
            + ", ".join(
                f"{k} {orig_w.get(k)}->{new_table[regime][k]}"
                for k in category_keys
            )
        )

    return new_table, notes


def propose_factor_config_ridge_regime(
    factor_config: dict[str, Any],
    train_panel: pd.DataFrame,
    proposed_cfg: dict[str, Any],
    *,
    return_col: str = "future_return_20",
    tune_mode: str = "walk_forward",
) -> dict[str, Any]:
    """Ridge 定成分权重 + Regime 定大类权重"""
    ridge_cfg = proposed_cfg.get("ridge", {})
    regime_cfg = proposed_cfg.get("regime", {})
    note = proposed_cfg.get(
        "source_note",
        "由因子有效性分析自动生成，请人工审阅后替换 config/factor_config.yaml",
    )

    new_cfg = copy.deepcopy(factor_config)
    changes: list[str] = []

    weight_map, alpha = _fit_component_ridge_weights(
        train_panel,
        factor_config,
        return_col=return_col,
        ridge_cfg=ridge_cfg,
        proposed_cfg=proposed_cfg,
    )

    if weight_map:
        for fname, block in new_cfg.get("factors", {}).items():
            for comp in block.get("components", []):
                field = comp.get("field", "")
                if field not in weight_map:
                    continue
                orig = float(comp.get("weight", 0))
                new_w = weight_map[field]
                if abs(new_w - orig) > 1e-4:
                    changes.append(f"{fname}.{field}: {orig} -> {new_w} (Ridge)")
                comp["weight"] = new_w
    else:
        changes.append("成分 Ridge 样本不足，保留原成分权重")

    regime_table, regime_notes = _fit_regime_category_weights(
        train_panel,
        factor_config,
        return_col=return_col,
        regime_cfg=regime_cfg,
        ridge_cfg=ridge_cfg,
    )
    if "market_regime" not in new_cfg:
        new_cfg["market_regime"] = {}
    new_cfg["market_regime"]["weights"] = regime_table
    for line in regime_notes:
        if "->" in line:
            changes.append(f"regime.{line}")

    mode_label = {
        "walk_forward": "Ridge+Regime Walk-forward 训练集定权",
        "ineffective": "Ridge+Regime（模型失效）",
        "soft_tune": "Ridge+Regime 微调",
    }.get(tune_mode, tune_mode)

    header = (
        f"# {note}\n"
        f"# 调权模式: {mode_label}\n"
        f"# Ridge alpha: {alpha}\n"
        f"# 变更摘要（{len(changes)} 项）:\n"
        + "".join(f"#   - {line}\n" for line in changes[:25])
    )
    if len(changes) > 25:
        header += f"#   ... 另有 {len(changes) - 25} 项\n"

    new_cfg["_proposed_meta"] = {
        "changes": changes,
        "tune_mode": tune_mode,
        "optimization_method": "ridge_regime",
        "ridge_alpha": alpha,
        "regime_notes": regime_notes,
    }
    new_cfg["_yaml_header_comment"] = header
    return new_cfg
