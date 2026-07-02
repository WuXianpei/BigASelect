"""IC、五分位等有效性指标"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _daily_spearman(score: pd.Series, ret: pd.Series) -> float | None:
    """单日 Spearman IC（秩相关，无需 scipy）"""
    valid = score.notna() & ret.notna()
    if valid.sum() < 10:
        return None
    s = score[valid].astype(float)
    r = ret[valid].astype(float)
    if s.nunique() < 2 or r.nunique() < 2:
        return None
    corr = s.rank(method="average").corr(r.rank(method="average"))
    if corr is None or np.isnan(corr):
        return None
    return float(corr)


def compute_daily_ic_panel(
    panel: pd.DataFrame,
    score_col: str,
    return_col: str = "future_return_20",
) -> pd.DataFrame:
    """按 trade_date 计算每日 IC"""
    rows: list[dict[str, Any]] = []
    for td, grp in panel.groupby("trade_date"):
        ic = _daily_spearman(grp[score_col], grp[return_col])
        if ic is not None:
            rows.append({"trade_date": td, "ic": ic, "score_column": score_col, "n": len(grp)})
    return pd.DataFrame(rows)


def summarize_ic(daily_ic: pd.DataFrame) -> dict[str, Any]:
    """汇总 IC 序列"""
    if daily_ic.empty:
        return {
            "ic_mean": None,
            "ic_std": None,
            "ic_ir": None,
            "ic_positive_ratio": None,
            "ic_days": 0,
        }
    ic = daily_ic["ic"].astype(float)
    mean = float(ic.mean())
    std = float(ic.std(ddof=1)) if len(ic) > 1 else 0.0
    ir = mean / std if std > 1e-9 else None
    return {
        "ic_mean": round(mean, 4),
        "ic_std": round(std, 4) if std else 0.0,
        "ic_ir": round(ir, 4) if ir is not None else None,
        "ic_positive_ratio": round(float((ic > 0).mean()), 4),
        "ic_days": int(len(ic)),
    }


def compute_quintile_stats(
    panel: pd.DataFrame,
    score_col: str,
    return_col: str = "future_return_20",
) -> dict[str, Any]:
    """全样本五分位分组平均收益与单调性"""
    df = panel[[score_col, return_col]].dropna()
    if len(df) < 50:
        return {
            "quintile_means": {},
            "quintile_spread": None,
            "monotonic": None,
            "top20_excess": None,
        }

    try:
        df = df.copy()
        df["quintile"] = pd.qcut(df[score_col].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    except ValueError:
        return {
            "quintile_means": {},
            "quintile_spread": None,
            "monotonic": None,
            "top20_excess": None,
        }

    means = df.groupby("quintile", observed=True)[return_col].mean()
    quintile_means = {int(k): round(float(v), 4) for k, v in means.items()}
    spread = None
    if 1 in quintile_means and 5 in quintile_means:
        spread = round(quintile_means[5] - quintile_means[1], 4)

    ordered = [quintile_means.get(i) for i in range(1, 6) if i in quintile_means]
    monotonic = None
    if len(ordered) >= 2:
        mono_steps = sum(ordered[i] <= ordered[i + 1] for i in range(len(ordered) - 1))
        monotonic = mono_steps >= len(ordered) - 1

    top20 = df[df[score_col] >= df[score_col].quantile(0.8)][return_col].mean()
    overall = df[return_col].mean()
    top20_excess = round(float(top20 - overall), 4) if pd.notna(top20) and pd.notna(overall) else None

    return {
        "quintile_means": quintile_means,
        "quintile_spread": spread,
        "monotonic": monotonic,
        "top20_excess": top20_excess,
    }


def evaluate_verdict(
    ic_summary: dict[str, Any],
    quintile: dict[str, Any],
    verdict_cfg: dict[str, Any],
    *,
    sample_sufficient: bool,
) -> dict[str, Any]:
    """根据阈值判定有效/失效/样本不足"""
    rules: list[dict[str, Any]] = []

    def _rule(name: str, passed: bool | None, detail: str) -> None:
        rules.append({"name": name, "passed": passed, "detail": detail})

    if not sample_sufficient:
        return {
            "status": "insufficient_sample",
            "status_label": "样本不足",
            "effective": None,
            "rules": rules,
            "passed_count": 0,
        }

    ic_mean = ic_summary.get("ic_mean")
    ic_ir = ic_summary.get("ic_ir")
    ic_pos = ic_summary.get("ic_positive_ratio")
    spread = quintile.get("quintile_spread")
    monotonic = quintile.get("monotonic")

    _rule(
        "ic_mean",
        ic_mean is not None and ic_mean >= verdict_cfg.get("ic_mean_min", 0.02),
        f"IC均值={ic_mean}（阈值>={verdict_cfg.get('ic_mean_min', 0.02)}）",
    )
    _rule(
        "ic_ir",
        ic_ir is not None and ic_ir >= verdict_cfg.get("ic_ir_min", 0.3),
        f"IC_IR={ic_ir}（阈值>={verdict_cfg.get('ic_ir_min', 0.3)}）",
    )
    _rule(
        "ic_positive_ratio",
        ic_pos is not None and ic_pos >= verdict_cfg.get("ic_positive_ratio_min", 0.55),
        f"IC胜率={ic_pos}（阈值>={verdict_cfg.get('ic_positive_ratio_min', 0.55)}）",
    )
    _rule(
        "quintile_spread",
        spread is not None and spread >= verdict_cfg.get("quintile_spread_min", 1.5),
        f"五分位价差={spread}%（阈值>={verdict_cfg.get('quintile_spread_min', 1.5)}%）",
    )
    if verdict_cfg.get("require_monotonic", True):
        _rule("monotonic", monotonic is True, f"五分位单调递增={monotonic}")
    else:
        _rule("monotonic", True, "未要求单调性")

    passed_count = sum(1 for r in rules if r["passed"] is True)
    pass_min = int(verdict_cfg.get("pass_min_rules", 4))
    effective = passed_count >= pass_min

    return {
        "status": "effective" if effective else "ineffective",
        "status_label": "有效" if effective else "失效",
        "effective": effective,
        "rules": rules,
        "passed_count": passed_count,
        "pass_min_rules": pass_min,
    }


def compute_component_ic(
    panel: pd.DataFrame,
    factor_config: dict[str, Any],
    return_col: str = "future_return_20",
) -> dict[str, float]:
    """各成分字段跨日平均 IC（兼容旧接口）"""
    stats = compute_component_ic_stats(panel, factor_config, return_col=return_col)
    return {field: s.get("ic_mean", 0.0) for field, s in stats.items()}


def compute_component_ic_stats(
    panel: pd.DataFrame,
    factor_config: dict[str, Any],
    return_col: str = "future_return_20",
    *,
    time_decay: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """
    各成分字段 IC 统计（含时间衰减加权均值与 IC_IR），供 A 方案调权使用。
    """
    fields: set[str] = set()
    for factor_cfg in factor_config.get("factors", {}).values():
        for comp in factor_cfg.get("components", []):
            field = comp.get("field")
            if field:
                fields.add(field)

    series: dict[str, list[tuple[str, float]]] = {f: [] for f in fields}
    for td, grp in panel.groupby("trade_date"):
        td_str = str(td)
        for field in fields:
            if field not in grp.columns:
                continue
            ic = _daily_spearman(grp[field], grp[return_col])
            if ic is not None:
                series[field].append((td_str, ic))

    decay_cfg = time_decay or {}
    use_decay = bool(decay_cfg.get("enabled", False))
    recent_days = int(decay_cfg.get("recent_days", 20))
    recent_w = float(decay_cfg.get("recent_weight", 2.0))
    older_w = float(decay_cfg.get("older_weight", 0.5))

    result: dict[str, dict[str, Any]] = {}
    for field, points in series.items():
        if not points:
            result[field] = {
                "ic_mean": 0.0,
                "ic_mean_weighted": 0.0,
                "ic_std": 0.0,
                "ic_ir": 0.0,
                "ic_days": 0,
            }
            continue

        points.sort(key=lambda x: x[0])
        dates = [p[0] for p in points]
        ics = [p[1] for p in points]
        ic_arr = np.array(ics, dtype=float)

        if use_decay and len(ics) > 0:
            weights = _time_decay_weights(len(ics), recent_days, recent_w, older_w)
            wsum = float(np.sum(weights))
            wmean = float(np.sum(ic_arr * weights) / wsum) if wsum > 0 else float(ic_arr.mean())
        else:
            wmean = float(ic_arr.mean())

        std = float(ic_arr.std(ddof=1)) if len(ic_arr) > 1 else 0.0
        ir = wmean / std if std > 1e-9 else 0.0
        result[field] = {
            "ic_mean": round(float(ic_arr.mean()), 4),
            "ic_mean_weighted": round(wmean, 4),
            "ic_std": round(std, 4),
            "ic_ir": round(ir, 4),
            "ic_days": len(ics),
        }

    return result


def _time_decay_weights(
    n: int,
    recent_days: int,
    recent_weight: float,
    older_weight: float,
) -> np.ndarray:
    """按距窗口末尾的天数分配衰减权重（末尾 recent_days 日权重更高）"""
    weights = np.empty(n, dtype=float)
    for i in range(n):
        age_from_end = n - 1 - i
        weights[i] = recent_weight if age_from_end < recent_days else older_weight
    return weights


def split_walk_forward_dates(
    dates: list[str],
    *,
    train_days: int = 60,
    validate_days: int = 20,
    test_days: int = 20,
    min_total_days: int = 80,
) -> dict[str, Any]:
    """
    按时间顺序切分 walk-forward 窗口：训练 | 验证 | 测试（C 方案）。
    日期不足时按比例 60/20/20 切分；总样本低于 min_total_days 则 disabled。
    """
    n = len(dates)
    if n < min_total_days:
        return {
            "enabled": False,
            "reason": f"样本 {n} 日低于 walk-forward 最低要求 {min_total_days}",
            "train_dates": [],
            "validate_dates": [],
            "test_dates": [],
        }

    need = train_days + validate_days + test_days
    if n >= need:
        train_end = train_days
        val_end = train_days + validate_days
        return {
            "enabled": True,
            "mode": "fixed",
            "train_dates": dates[:train_end],
            "validate_dates": dates[train_end:val_end],
            "test_dates": dates[val_end:val_end + test_days],
        }

    # 按比例切分
    t_n = max(int(n * 0.6), 1)
    v_n = max(int(n * 0.2), 1)
    test_n = n - t_n - v_n
    if test_n < 1:
        test_n = 1
        v_n = max(n - t_n - test_n, 1)
    return {
        "enabled": True,
        "mode": "proportional",
        "train_dates": dates[:t_n],
        "validate_dates": dates[t_n : t_n + v_n],
        "test_dates": dates[t_n + v_n :],
    }


def slice_panel_by_dates(panel: pd.DataFrame, dates: list[str]) -> pd.DataFrame:
    """按 trade_date 子集截取面板"""
    if not dates or panel.empty or "trade_date" not in panel.columns:
        return panel.iloc[0:0].copy()
    date_set = set(dates)
    return panel[panel["trade_date"].astype(str).isin(date_set)].copy()


def evaluate_score_on_panel(
    panel: pd.DataFrame,
    score_col: str,
    return_col: str = "future_return_20",
) -> dict[str, Any]:
    """在指定面板上评估单一分数列的 IC 与五分位"""
    daily_ic = compute_daily_ic_panel(panel, score_col, return_col=return_col)
    ic_summary = summarize_ic(daily_ic)
    quintile = compute_quintile_stats(panel, score_col, return_col=return_col)
    return {
        "ic": ic_summary,
        "quintile": quintile,
        "panel_rows": len(panel),
        "trade_days": panel["trade_date"].nunique() if not panel.empty else 0,
    }


def compute_top3_return_stats(
    panel: pd.DataFrame,
    score_col: str,
    return_col: str = "future_return_20",
    *,
    top_k: int = 3,
) -> dict[str, Any]:
    """
    每日按分数取 Top-K，统计 future_return_20 胜率（收益>0 视为赢）。
    仅供参考，不参与失效判定。
    """
    if panel.empty or score_col not in panel.columns or return_col not in panel.columns:
        return {
            "top_k": top_k,
            "pick_count": 0,
            "win_count": 0,
            "win_rate": None,
            "avg_return_pct": None,
            "max_loss_pct": None,
            "avg_day_worst_return_pct": None,
            "signal_days": 0,
            "daily_all_win_rate": None,
        }

    picks: list[float] = []
    daily_worst: list[float] = []
    daily_all_win = 0
    signal_days = 0

    for _, grp in panel.groupby("trade_date"):
        day = grp.dropna(subset=[score_col, return_col])
        if len(day) < top_k:
            continue
        top = day.nlargest(top_k, score_col)
        rets = top[return_col].astype(float).tolist()
        picks.extend(rets)
        daily_worst.append(min(rets))
        signal_days += 1
        if all(r > 0 for r in rets):
            daily_all_win += 1

    if not picks:
        return {
            "top_k": top_k,
            "pick_count": 0,
            "win_count": 0,
            "win_rate": None,
            "avg_return_pct": None,
            "max_loss_pct": None,
            "avg_day_worst_return_pct": None,
            "signal_days": 0,
            "daily_all_win_rate": None,
        }

    win_count = sum(1 for r in picks if r > 0)
    min_ret = float(min(picks))
    return {
        "top_k": top_k,
        "pick_count": len(picks),
        "win_count": win_count,
        "win_rate": round(win_count / len(picks), 4),
        "avg_return_pct": round(float(np.mean(picks)), 4),
        "max_loss_pct": round(min_ret, 4),
        "avg_day_worst_return_pct": round(float(np.mean(daily_worst)), 4),
        "signal_days": signal_days,
        "daily_all_win_rate": round(daily_all_win / signal_days, 4) if signal_days else None,
        "note": f"每日 final_score 前 {top_k} 只，{return_col}>0 视为赢；不参与失效判定",
    }
