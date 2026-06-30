"""因子有效性分析：基于 archive 与 future_return_20 评估打分模型"""



from __future__ import annotations



import argparse

import sys

import time

from pathlib import Path



import pandas as pd



PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(0, str(PROJECT_ROOT))



from src.archive_manager import get_archive_root  # noqa: E402

from src.config_loader import (  # noqa: E402

    PROJECT_ROOT as CFG_ROOT,

    load_factor_analysis_config,

    load_factor_config,

    load_settings,

)

from src.factor_analyzer.archive_loader import (  # noqa: E402

    get_trading_calendar,

    list_analysis_archive_dates,

    resolve_analysis_window,

)

from src.factor_analyzer.metrics import (  # noqa: E402

    compute_component_ic,

    compute_daily_ic_panel,

    compute_quintile_stats,

    compute_top3_return_stats,

    evaluate_verdict,

    summarize_ic,

)

from src.factor_analyzer.optimizer import propose_factor_config, write_proposed_config  # noqa: E402

from src.factor_analyzer.report import build_report_payload, write_reports  # noqa: E402

from src.factor_analyzer.rescorer import rescore_archived_day  # noqa: E402

from src.factor_analyzer.return_backfill import (  # noqa: E402

    backfill_future_return,

    get_return_column,

)

from src.network_setup import setup_network  # noqa: E402





def run_analysis(args: argparse.Namespace) -> int:

    settings = load_settings()

    setup_network(settings)

    analysis_cfg = load_factor_analysis_config()

    factor_config = load_factor_config()

    archive_root = get_archive_root(settings)



    window_cfg = analysis_cfg.get("window", {})

    min_days = args.min_days if args.min_days is not None else window_cfg.get("min_trading_days", 60)

    max_days = args.days if args.days is not None else window_cfg.get("max_trading_days", 120)

    return_horizon = window_cfg.get("return_horizon", 20)



    output_cfg = analysis_cfg.get("output", {})

    reports_dir = CFG_ROOT / settings.get("output_dir", "output") / output_cfg.get(

        "reports_dir", "analysis/reports"

    )

    proposed_dir = CFG_ROOT / settings.get("output_dir", "output") / output_cfg.get(

        "proposed_dir", "analysis/proposed"

    )

    report_prefix = output_cfg.get("report_prefix", "factor_effectiveness")



    score_cfg = analysis_cfg.get("score_columns", {})

    primary_score = score_cfg.get("primary", "final_score")

    factor_scores = list(score_cfg.get("factor_scores", []))

    return_col = get_return_column(analysis_cfg)



    print("=" * 50)

    print("BigASelect - 因子有效性分析")

    print("=" * 50)

    print(f"归档目录: {archive_root}")

    print(f"收益 horizon: {return_horizon} 个交易日（列 {return_col}）")

    print(f"分析窗口: 最少 {min_days}，最多 {max_days} 个交易日")

    print()



    t0 = time.perf_counter()

    archived = list_analysis_archive_dates(archive_root)

    if not archived:

        print("[错误] archive 中无完整归档数据，请先运行历史批跑或每日归档。")

        return 1



    calendar = get_trading_calendar()

    window_info = resolve_analysis_window(

        archived,

        min_days=min_days,

        max_days=max_days,

        return_horizon=return_horizon,

        calendar=calendar,

    )

    analysis_dates = window_info["analysis_dates"]

    window_info["primary_score"] = primary_score



    print(f"archive 完整归档: {len(archived)} 日")

    print(f"可算 {return_col} 的交易日: {len(window_info['return_ready_dates'])} 日")

    print(f"本次分析窗口: {len(analysis_dates)} 日")

    if analysis_dates:

        print(f"  范围: {analysis_dates[0]} … {analysis_dates[-1]}")

    print()



    if not analysis_dates:

        print(f"[错误] 无可用分析日期（archive 太新，尚无 T+{return_horizon} 收益）。")

        return 1



    if args.dry_run:

        print("[dry-run] 分析日期:")

        for d in analysis_dates:

            print(f"  {d}")

        return 0



    print(f"[1/4] 回填 archive stock_pool 列 {return_col}...")

    return_count = backfill_future_return(

        analysis_dates,

        return_horizon=return_horizon,

        analysis_cfg=analysis_cfg,

        calendar=calendar,

        rebuild=args.rebuild_returns,

        root=archive_root,

    )

    print(f"      窗口内有效 {return_col} 共 {return_count} 条")

    if return_count == 0:

        print("[错误] 无有效 future_return_20，无法分析。")

        return 1



    print("[2/4] 用当前 factor_config 重算历史分数...")

    panel_parts: list[pd.DataFrame] = []

    for idx, td in enumerate(analysis_dates, start=1):

        scored = rescore_archived_day(td, factor_config=factor_config, root=archive_root)

        if scored is None or scored.empty or return_col not in scored.columns:

            continue

        day = scored.dropna(subset=[return_col]).copy()

        if day.empty:

            continue

        day["trade_date"] = td

        panel_parts.append(day)

        if idx == len(analysis_dates) or idx % max(1, len(analysis_dates) // 5) == 0:

            print(f"      进度 {idx}/{len(analysis_dates)}")



    if not panel_parts:

        print("[错误] 无有效面板数据（分数与 future_return_20 无法对齐）。")

        return 1



    panel = pd.concat(panel_parts, ignore_index=True)

    print(f"      面板 {len(panel)} 行 × {len(analysis_dates)} 日")



    print("[3/4] 计算 IC / 五分位指标...")

    ic_summaries: dict = {}

    quintile_by_score: dict = {}

    score_cols = [primary_score] + [c for c in factor_scores if c != primary_score]



    for col in score_cols:

        if col not in panel.columns:

            continue

        daily_ic = compute_daily_ic_panel(panel, col, return_col=return_col)

        ic_summaries[col] = summarize_ic(daily_ic)

        quintile_by_score[col] = compute_quintile_stats(panel, col, return_col=return_col)



    component_ic = compute_component_ic(panel, factor_config, return_col=return_col)

    ref_cfg = analysis_cfg.get("strategy_reference", {})
    top_k = int(ref_cfg.get("top_k", 3))
    top1_stats = compute_top3_return_stats(
        panel, primary_score, return_col=return_col, top_k=1
    )
    top3_stats = compute_top3_return_stats(
        panel, primary_score, return_col=return_col, top_k=top_k
    )

    verdict = evaluate_verdict(

        ic_summaries.get(primary_score, {}),

        quintile_by_score.get(primary_score, {}),

        analysis_cfg.get("verdict", {}),

        sample_sufficient=window_info.get("sample_sufficient", False),

    )



    print("[4/4] 生成报告...")

    payload = build_report_payload(

        window_info=window_info,

        ic_summaries=ic_summaries,

        quintile_by_score=quintile_by_score,

        verdict=verdict,

        component_ic=component_ic,

        factor_config_path="config/factor_config.yaml",

        return_column=return_col,

        return_valid_count=return_count,

        panel_rows=len(panel),

        top3_stats=top3_stats,

        top1_stats=top1_stats,

    )

    md_path, json_path = write_reports(

        payload, reports_dir=reports_dir, prefix=report_prefix

    )



    proposed_path = None

    if (

        verdict.get("status") == "ineffective"

        and analysis_cfg.get("proposed_config", {}).get("enabled", True)

    ):

        proposed = propose_factor_config(

            factor_config,

            component_ic,

            analysis_cfg.get("proposed_config", {}),

        )

        proposed_path = proposed_dir / "factor_config.proposed.yaml"

        write_proposed_config(proposed, proposed_path)

        print(f"      建议配置: {proposed_path}")



    dur = time.perf_counter() - t0

    print()

    print("=" * 50)

    print(f"结论: {verdict.get('status_label', verdict.get('status'))}")

    primary_ic = ic_summaries.get(primary_score, {})

    print(

        f"final_score IC均值={primary_ic.get('ic_mean')}  "

        f"IC_IR={primary_ic.get('ic_ir')}  "

        f"五分位价差={quintile_by_score.get(primary_score, {}).get('quintile_spread')}%"

    )

    if top1_stats.get("win_rate") is not None:
        print(
            f"Top1 二十日收益胜率={top1_stats['win_rate'] * 100:.2f}%"
            f"（{top1_stats['win_count']}/{top1_stats['pick_count']} 笔，仅供参考）"
        )
    if top3_stats.get("win_rate") is not None:

        print(

            f"Top{top_k} 二十日收益胜率={top3_stats['win_rate'] * 100:.2f}%"

            f"（{top3_stats['win_count']}/{top3_stats['pick_count']} 笔，仅供参考）"

        )

        if top3_stats.get("max_loss_pct") is not None:

            print(f"Top{top_k} 二十日最大跌幅={top3_stats['max_loss_pct']}%")

    print(f"报告: {md_path}")

    print(f"JSON: {json_path}")

    if proposed_path:

        print(f"请审阅并手动替换: config/factor_config.yaml ← {proposed_path.name}")

    print(f"耗时 {dur:.1f}s")

    print("=" * 50)

    return 0





def main() -> None:

    parser = argparse.ArgumentParser(description="因子有效性分析（future_return_20）")

    parser.add_argument("--days", type=int, default=None, help="最大分析交易日数（默认 120）")

    parser.add_argument("--min-days", type=int, default=None, help="最低样本日数（默认 60）")

    parser.add_argument(

        "--rebuild-returns",

        action="store_true",

        help="全量重建 archive stock_pool 中的 future_return_20 列",

    )

    parser.add_argument("--dry-run", action="store_true", help="仅列出分析日期")

    args = parser.parse_args()

    sys.exit(run_analysis(args))





if __name__ == "__main__":

    main()

