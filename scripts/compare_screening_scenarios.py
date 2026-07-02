"""筛选规则对照实验：基于 archive 历史池离线重放不同截断/过滤方案"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.archive_manager import get_archive_root  # noqa: E402
from src.config_loader import (  # noqa: E402
    load_factor_analysis_config,
    load_factor_config,
    load_screening_rules,
    load_settings,
)
from src.factor_analyzer.archive_loader import (  # noqa: E402
    list_analysis_archive_dates,
    resolve_analysis_window,
)
from src.factor_analyzer.metrics import (  # noqa: E402
    compute_daily_ic_panel,
    compute_top3_return_stats,
    slice_panel_by_dates,
    split_walk_forward_dates,
    summarize_ic,
)
from src.factor_analyzer.rescorer import rescore_archived_day  # noqa: E402
from src.factor_analyzer.return_backfill import get_return_column  # noqa: E402
from src.screener import (  # noqa: E402
    _step2_liquidity_filter,
    _step3_trend_structure_filter,
    _step5_fundamental_filter,
)


def _load_panel(
    analysis_dates: list[str],
    *,
    return_col: str,
    archive_root: Path,
    factor_config: dict[str, Any],
) -> pd.DataFrame:
    """加载带 future_return 与 final_score 的历史面板"""
    parts: list[pd.DataFrame] = []
    for td in analysis_dates:
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


def _top_by_amount(df: pd.DataFrame, n: int) -> pd.DataFrame:
  if df.empty or "amount" not in df.columns:
      return df
  return df.nlargest(n, "amount")


def _apply_step2(df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    return _step2_liquidity_filter(df, params)


def _apply_step3(df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    return _step3_trend_structure_filter(df, params)


def _apply_step5(df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    return _step5_fundamental_filter(df, params)


def _apply_step2_step5(df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    out = _apply_step2(df, params)
    return _apply_step5(out, params)


def _stricter_liquidity(df: pd.DataFrame, _: dict[str, Any]) -> pd.DataFrame:
    """更严流动性：5000万 + 换手率 0.5%"""
    params = {
        "min_amount": 50_000_000,
        "min_turnover_rate": 0.5,
    }
    return _step2_liquidity_filter(df, params)


def _stricter_step5(df: pd.DataFrame, _: dict[str, Any]) -> pd.DataFrame:
    """更严估值财务：pe<=150, pb<=10, 利润同比>=-50%, 负债率<=70%"""
    params = {
        "pe_ttm_max": 150,
        "pe_ttm_min": 0,
        "pb_max": 10,
        "net_profit_yoy_min": -50,
        "debt_ratio_max": 70,
    }
    return _step5_fundamental_filter(df, params)


def _liquidity_step5_stricter(df: pd.DataFrame, base_params: dict[str, Any]) -> pd.DataFrame:
    out = _apply_step2(df, base_params)
    return _stricter_step5(out, base_params)


def _no_step3_proxy(df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    """
    近似「仅 1+2+5」：在归档池上无法再放宽 step3，仅重验 step2+step5。
    与 current 差异应极小，用于确认归档约束。
    """
    return _apply_step2_step5(df, params)


SCENARIOS: dict[str, Callable[[pd.DataFrame, dict[str, Any]], pd.DataFrame]] = {
    "A_current_1000": lambda d, p: d,
    "B_amount_top800": lambda d, p: _top_by_amount(d, 800),
    "C_amount_top500": lambda d, p: _top_by_amount(d, 500),
    "D_amount_top300": lambda d, p: _top_by_amount(d, 300),
    "E_liquidity_step5_recheck": _no_step3_proxy,
    "F_stricter_liquidity": _stricter_liquidity,
    "G_stricter_step5": _stricter_step5,
    "H_liquidity_plus_stricter_step5": _liquidity_step5_stricter,
    "I_skip_step3_keep_weak_trend": lambda d, p: d,  # 归档无法模拟，占位与 A 相同
}


def _filter_panel(
    panel: pd.DataFrame,
    scenario_fn: Callable[[pd.DataFrame, dict[str, Any]], pd.DataFrame],
    params: dict[str, Any],
) -> pd.DataFrame:
    """按交易日应用场景过滤"""
    parts: list[pd.DataFrame] = []
    for _, grp in panel.groupby("trade_date", sort=True):
        filtered = scenario_fn(grp.copy(), params)
        if filtered is None or filtered.empty:
            continue
        parts.append(filtered)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def _top3_overlap_vs_baseline(
    baseline: pd.DataFrame,
    variant: pd.DataFrame,
    *,
    score_col: str,
    top_k: int = 3,
) -> dict[str, Any]:
    """统计相对 baseline 的 Top3 重合度"""
    same = 0
    total = 0
    for td in baseline["trade_date"].unique():
        b = baseline[baseline["trade_date"] == td]
        v = variant[variant["trade_date"] == td]
        if len(b) < top_k or len(v) < top_k:
            continue
        b_codes = set(b.nlargest(top_k, score_col)["ts_code"].astype(str))
        v_codes = set(v.nlargest(top_k, score_col)["ts_code"].astype(str))
        total += 1
        if b_codes == v_codes:
            same += 1
    return {
        "overlap_days": same,
        "comparable_days": total,
        "overlap_rate": (same / total) if total else None,
    }


def _evaluate_scenario(
    panel: pd.DataFrame,
    scenario_fn: Callable[[pd.DataFrame, dict[str, Any]], pd.DataFrame],
    *,
    name: str,
    params: dict[str, Any],
    score_col: str,
    return_col: str,
    baseline_panel: pd.DataFrame | None = None,
) -> dict[str, Any]:
    filtered = _filter_panel(panel, scenario_fn, params)
    top3 = compute_top3_return_stats(filtered, score_col, return_col, top_k=3)
    top1 = compute_top3_return_stats(filtered, score_col, return_col, top_k=1)
    daily_ic = compute_daily_ic_panel(filtered, score_col, return_col=return_col)
    ic = summarize_ic(daily_ic)

    avg_pool = (
        filtered.groupby("trade_date").size().mean() if not filtered.empty else 0.0
    )
    out: dict[str, Any] = {
        "scenario": name,
        "avg_pool_size": round(float(avg_pool), 1),
        "signal_days_top3": top3.get("signal_days", 0),
        "top3_win_rate": top3.get("win_rate"),
        "top3_avg_return_pct": top3.get("avg_return_pct"),
        "top3_max_loss_pct": top3.get("max_loss_pct"),
        "top1_win_rate": top1.get("win_rate"),
        "ic_mean": ic.get("ic_mean"),
        "ic_ir": ic.get("ic_ir"),
    }
    if baseline_panel is not None and name != "A_current_1000":
        out["top3_overlap_with_baseline"] = _top3_overlap_vs_baseline(
            baseline_panel, filtered, score_col=score_col
        )
    return out


def run_compare(args: argparse.Namespace) -> int:
    settings = load_settings()
    archive_root = get_archive_root(settings)
    factor_config = load_factor_config()
    analysis_cfg = load_factor_analysis_config()
    rules = load_screening_rules()
    params = rules.get("params", {})
    return_col = get_return_column(analysis_cfg)

    archived = list_analysis_archive_dates(archive_root)
    window = resolve_analysis_window(
        archived,
        min_days=args.min_days,
        max_days=args.max_days,
        return_horizon=20,
    )
    analysis_dates = window["analysis_dates"]
    if not analysis_dates:
        print("[错误] 无可用分析日期")
        return 1

    print(f"加载面板：{len(analysis_dates)} 个交易日（{analysis_dates[0]} ~ {analysis_dates[-1]}）")
    panel = _load_panel(
        analysis_dates,
        return_col=return_col,
        archive_root=archive_root,
        factor_config=factor_config,
    )
    if panel.empty:
        print("[错误] 面板为空")
        return 1

    score_col = "final_score"
    if score_col not in panel.columns:
        print(f"[错误] 缺少 {score_col}")
        return 1

    wf = split_walk_forward_dates(
        analysis_dates,
        train_days=90,
        validate_days=30,
        test_days=40,
        min_total_days=80,
    )
    test_dates = wf.get("test_dates", []) if wf.get("enabled") else analysis_dates[-40:]
    test_panel = slice_panel_by_dates(panel, test_dates)

    print(f"全样本 {len(analysis_dates)} 日 / 测试集 {len(test_dates)} 日 / 面板 {len(panel)} 行")

    baseline_full = _filter_panel(panel, SCENARIOS["A_current_1000"], params)
    baseline_test = _filter_panel(test_panel, SCENARIOS["A_current_1000"], params)

    results: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "return_col": return_col,
        "analysis_dates_count": len(analysis_dates),
        "test_dates_count": len(test_dates),
        "test_date_range": [test_dates[0], test_dates[-1]] if test_dates else [],
        "limitation": (
            "归档 stock_pool 为筛选+按 amount 截 1000 后的结果，"
            "无法在归档上模拟「取消 step3」等放宽筛选；step3/step5 再过滤对当前归档几乎无影响。"
        ),
        "full_sample": [],
        "test_sample": [],
    }

    for name, fn in SCENARIOS.items():
        if name == "I_skip_step3_keep_weak_trend":
            continue  # 与 A 等价，仅作说明
        full_m = _evaluate_scenario(
            panel, fn, name=name, params=params,
            score_col=score_col, return_col=return_col,
            baseline_panel=baseline_full,
        )
        test_m = _evaluate_scenario(
            test_panel, fn, name=name, params=params,
            score_col=score_col, return_col=return_col,
            baseline_panel=baseline_test,
        )
        results["full_sample"].append(full_m)
        results["test_sample"].append(test_m)
        print(
            f"  {name}: 测试集 Top3胜率={test_m['top3_win_rate']:.2%} "
            f"Top1={test_m['top1_win_rate']:.2%} "
            f"IC={test_m['ic_mean']:.4f} "
            f"池均={test_m['avg_pool_size']:.0f}"
            if test_m["top3_win_rate"] is not None
            else f"  {name}: 无有效信号"
        )

    out_dir = PROJECT_ROOT / "output" / "analysis" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d")
    json_path = out_dir / f"screening_compare_{stamp}.json"
    md_path = out_dir / f"screening_compare_{stamp}.md"

    json_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# 筛选规则对照实验",
        "",
        f"- 生成时间：{results['generated_at']}",
        f"- 全样本：{results['analysis_dates_count']} 个交易日",
        f"- **测试集**：{results['test_dates_count']} 日（{results['test_date_range']}）",
        "",
        "## 实验限制",
        "",
        results["limitation"],
        "",
        "## 测试集结果（决策主依据）",
        "",
        "| 方案 | 均池规模 | Top3胜率 | Top1胜率 | 均收益% | 最大亏损% | IC均值 | IC_IR | Top3与现行重合 |",
        "|------|---------|---------|---------|--------|----------|--------|-------|---------------|",
    ]
    for row in results["test_sample"]:
        ov = row.get("top3_overlap_with_baseline") or {}
        ov_txt = (
            f"{ov.get('overlap_rate', 0):.1%}"
            if ov.get("overlap_rate") is not None
            else "-"
        )
        lines.append(
            f"| {row['scenario']} | {row['avg_pool_size']:.0f} | "
            f"{row['top3_win_rate']:.2%} | {row['top1_win_rate']:.2%} | "
            f"{row['top3_avg_return_pct']:.2f} | {row['top3_max_loss_pct']:.2f} | "
            f"{row['ic_mean']:.4f} | {row['ic_ir']:.4f} | {ov_txt} |"
        )

    lines.extend(["", "## 全样本结果", ""])
    lines.append(
        "| 方案 | 均池规模 | Top3胜率 | Top1胜率 | IC均值 | IC_IR |"
    )
    lines.append("|------|---------|---------|---------|--------|-------|")
    for row in results["full_sample"]:
        lines.append(
            f"| {row['scenario']} | {row['avg_pool_size']:.0f} | "
            f"{row['top3_win_rate']:.2%} | {row['top1_win_rate']:.2%} | "
            f"{row['ic_mean']:.4f} | {row['ic_ir']:.4f} |"
        )

    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n报告已写入：\n  {md_path}\n  {json_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="筛选规则对照实验")
    parser.add_argument("--min-days", type=int, default=60)
    parser.add_argument("--max-days", type=int, default=300)
    args = parser.parse_args()
    return run_compare(args)


if __name__ == "__main__":
    raise SystemExit(main())
