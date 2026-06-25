"""主程序入口"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

import pandas as pd

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import (  # noqa: E402
    get_validation_field_names,
    get_output_path,
    load_field_schema,
    load_screening_rules,
    load_settings,
)
from src.network_setup import setup_network  # noqa: E402
from src.data_fetcher import (  # noqa: E402
    fetch_a_share_list,
    fetch_market_indices,
    fetch_sector_data,
    get_trade_date,
)
from src.market_enricher import get_primary_market_snapshot  # noqa: E402
from src.exporter import export_csv  # noqa: E402
from src.fund_flow_provider import attach_northbound_flows, prefetch_all_money_flow  # noqa: E402
from src.output_utils import get_today_date_str  # noqa: E402
from src.stock_scorer import score_stock_pool, summarize_scores  # noqa: E402
from src.run_logger import RunLogger  # noqa: E402
from concurrent.futures import ThreadPoolExecutor

from src.screener import (
    apply_screening_fundamental_market,
    apply_screening_phase1,
    apply_screening_trend_capital,
    print_market_environment_alert,
)  # noqa: E402
from src.stock_enricher import enrich_financial_pool, enrich_light, enrich_technical_pool  # noqa: E402


def run(*, production: bool = False) -> None:
    """执行筛选并输出三个 CSV 文件（当日）"""
    settings = load_settings()
    setup_network(settings)
    if production:
        settings = {**settings, "test_mode": False}

    rules = load_screening_rules()
    output_date = get_today_date_str()

    logger = RunLogger(settings, PROJECT_ROOT)
    logger.start()
    t0 = time.perf_counter()

    try:
        outputs = run_for_trade_date(
            str(settings.get("trade_date") or output_date).replace("-", ""),
            production=production,
            settings=settings,
            rules=rules,
            logger=logger,
        )
        if _should_auto_archive(settings):
            from src.archive_manager import archive_pipeline_outputs, get_archive_root

            dur = time.perf_counter() - t0
            result = archive_pipeline_outputs(
                outputs,
                settings=settings,
                duration_sec=dur,
            )
            root = get_archive_root(settings)
            pool_file = result["stock_pool"]
            stock_count = result["stock_count"]
            print(
                f"[归档] {result['trade_date']} → {root}，"
                f"{pool_file.name}，{stock_count} 只"
            )
            logger.set_extra(
                archive_root=str(root),
                archive_stock_pool=str(pool_file),
                archive_stock_count=stock_count,
            )
        report_path = logger.finalize(status="success")
        print(f"[运行日志] 报告已保存: {report_path}")
    except Exception as exc:
        logger.set_extra(error_trace=traceback.format_exc())
        report_path = logger.finalize(status="failed", error=str(exc))
        print(f"[运行日志] 失败报告: {report_path}")
        raise
    finally:
        logger.stop()


def _should_auto_archive(settings: dict) -> bool:
    """正式每日跑批时自动归档；测试模式默认不归档"""
    if not settings.get("auto_archive", True):
        return False
    return not settings.get("test_mode", False)


def run_for_trade_date(
    trade_date: str,
    *,
    production: bool = True,
    settings: dict | None = None,
    rules: dict | None = None,
    logger: RunLogger | None = None,
    quiet: bool = False,
) -> dict[str, Path]:
    """
    对指定交易日执行完整 pipeline（筛选 + 打分）。

    trade_date: YYYYMMDD，写入文件名后缀并作为行情截止日。
    返回三份 CSV 路径。
    """
    trade_date = str(trade_date).replace("-", "")
    run_ts = trade_date

    if settings is None:
        settings = load_settings()
        setup_network(settings)
    if production:
        settings = {**settings, "test_mode": False}
    settings = {**settings, "trade_date": trade_date}

    if rules is None:
        rules = load_screening_rules()

    own_logger = logger is None
    if own_logger:
        logger = RunLogger(settings, PROJECT_ROOT)
        logger.start()

    from src.run_cache import clear_run_caches

    clear_run_caches()

    try:
        outputs = _run_pipeline(
            settings,
            rules,
            logger,
            run_ts=run_ts,
            quiet=quiet,
        )
        if own_logger:
            logger.finalize(status="success")
        return outputs
    except Exception as exc:
        if own_logger:
            logger.set_extra(error_trace=traceback.format_exc())
            logger.finalize(status="failed", error=str(exc))
        raise exc
    finally:
        if own_logger:
            logger.stop()


def _run_pipeline(
    settings: dict,
    rules: dict,
    logger: RunLogger,
    *,
    run_ts: str,
    quiet: bool = False,
) -> dict[str, Path]:
    """主流程（可被日志记录器包装），返回三份 CSV 路径"""
    print("=" * 50)
    print("BigASelect - A股股票筛选程序")
    print("=" * 50)

    trade_date = get_trade_date(settings)
    t0 = time.perf_counter()
    is_test = settings.get("test_mode", False)
    mode_label = "测试模式" if is_test else "正式模式（全市场）"

    print(f"运行模式: {mode_label}")
    print(f"行情模式: {settings.get('quote_mode', 'last_close')}")
    print(f"交易日: {trade_date}")
    print(f"目标股票池数量: {rules.get('target_count', 1000)}")
    print(f"筛选规则已启用: {rules.get('enabled', False)}")
    if is_test:
        print(f"测试样本: {len(settings.get('test_symbols', []))} 只")
    print()

    logger.set_extra(
        mode=mode_label,
        trade_date=trade_date,
        target_count=rules.get("target_count", 1000),
        screening_enabled=rules.get("enabled", False),
    )

    stock_pool_schema = load_field_schema("stock_pool_fields.yaml")
    market_schema = load_field_schema("market_context_fields.yaml")
    sector_schema = load_field_schema("sector_strength_fields.yaml")

    # --- 1. 股票列表 ---
    t_step = time.perf_counter()
    if is_test and settings.get("test_symbols"):
        stock_df = pd.DataFrame(settings["test_symbols"])
        print(f"[1/3] [测试模式] 使用指定样本 {len(stock_df)} 只")
        for _, r in stock_df.iterrows():
            print(f"        {r['ts_code']} {r.get('name', '')}")
    else:
        print("[1/3] 获取 A 股股票列表...")
        stock_df = fetch_a_share_list(settings)
        print(f"      共获取 {len(stock_df)} 只股票")
    dur = time.perf_counter() - t_step
    print(f"      耗时 {dur:.1f}s")
    logger.record_step("获取股票列表", dur, stock_count=len(stock_df))

    # --- 轻量 enrichment ---
    t_step = time.perf_counter()
    print("[1/3] 轻量指标 enrichment（风险标识 + 流动性字段）...")
    stock_df = enrich_light(stock_df, settings)
    dur = time.perf_counter() - t_step
    print(f"      耗时 {dur:.1f}s")
    logger.record_step("轻量enrichment", dur, stock_count=len(stock_df))

    # --- 筛选 1-2 ---
    t_step = time.perf_counter()
    print("[1/3] 筛选步骤 1-2（风险剔除 + 流动性）...")
    count_before_p1 = len(stock_df)
    pool_df = apply_screening_phase1(stock_df, rules)
    dur = time.perf_counter() - t_step
    print(f"      耗时 {dur:.1f}s")

    if is_test and not settings.get("test_symbols"):
        test_size = settings.get("test_pool_size", 5)
        pool_df = pool_df.head(test_size)
        print(f"      [测试模式] 仅处理 {len(pool_df)} 只股票")

    print(f"      步骤 1-2 后剩余 {len(pool_df)} 只股票")
    logger.record_step(
        "筛选步骤1-2",
        dur,
        input_count=count_before_p1,
        output_count=len(pool_df),
    )

    # --- 资金流预拉 ---
    t_step = time.perf_counter()
    prefetch_all_money_flow(settings)
    dur = time.perf_counter() - t_step
    print(f"      资金流预拉耗时 {dur:.1f}s")
    logger.record_step("资金流预拉", dur)

    # --- 分阶段 enrichment：先技术（全量）→ 筛选 3-4 → 财务（幸存者）→ 筛选 5-6 ---
    staggered = settings.get("staggered_enrichment", True) and rules.get("enabled", True)

    t_step = time.perf_counter()
    if staggered:
        print("[1/3] 技术 enrichment（日K + 技术指标 + 同花顺资金流）...")
        pool_df = enrich_technical_pool(pool_df, settings, skip_spot=True)
    else:
        print("[1/3] 完整指标 enrichment（技术 + 估值 + 财务）...")
        from src.stock_enricher import enrich_stock_pool

        pool_df = enrich_stock_pool(pool_df, settings, skip_spot=True)
    dur = time.perf_counter() - t_step
    print(f"      耗时 {dur:.1f}s")
    logger.record_step(
        "技术enrichment" if staggered else "完整enrichment",
        dur,
        stock_count=len(pool_df),
    )

    if staggered:
        t_step = time.perf_counter()
        print("[1/3] 筛选步骤 3-4（趋势 + 资金持续性）...")
        count_before = len(pool_df)
        pool_df = apply_screening_trend_capital(pool_df, rules)
        dur = time.perf_counter() - t_step
        print(f"      耗时 {dur:.1f}s")
        print(f"      步骤 3-4 后剩余 {len(pool_df)} 只股票")
        logger.record_step(
            "筛选步骤3-4",
            dur,
            input_count=count_before,
            output_count=len(pool_df),
        )

        t_step = time.perf_counter()
        print("[1/3] 财务 enrichment（估值 + 财务，仅幸存者）...")
        with ThreadPoolExecutor(max_workers=1) as bg:
            market_future = bg.submit(fetch_market_indices, settings)
            pool_df = enrich_financial_pool(pool_df, settings)
            market_df = market_future.result()
        dur = time.perf_counter() - t_step
        print(f"      耗时 {dur:.1f}s")
        logger.record_step("财务enrichment", dur, stock_count=len(pool_df))
        market_snapshot = get_primary_market_snapshot(market_df)
        logger.record_step("市场环境指标", 0, index_count=len(market_df))

        t_step = time.perf_counter()
        print("[1/3] 筛选步骤 5（估值财务）...")
        count_before_p2 = len(pool_df)
        pool_df = apply_screening_fundamental_market(pool_df, rules, market_snapshot)
        dur = time.perf_counter() - t_step
        print(f"      耗时 {dur:.1f}s")
        print(f"      最终股票池 {len(pool_df)} 只股票")
        logger.record_step(
            "筛选步骤5",
            dur,
            input_count=count_before_p2,
            output_count=len(pool_df),
        )
    else:
        t_step = time.perf_counter()
        print("[1/3] 获取市场环境指标（筛选步骤 6 使用）...")
        market_df = fetch_market_indices(settings)
        market_snapshot = get_primary_market_snapshot(market_df)
        dur = time.perf_counter() - t_step
        print(f"      已计算市场宏观快照，耗时 {dur:.1f}s")
        logger.record_step("市场环境指标", dur, index_count=len(market_df))

        t_step = time.perf_counter()
        print("[1/3] 筛选步骤 3-6（趋势 + 资金 + 基本面 + 市场环境）...")
        from src.screener import apply_screening_phase2

        count_before_p2 = len(pool_df)
        pool_df = apply_screening_phase2(pool_df, rules, market_snapshot=market_snapshot)
        dur = time.perf_counter() - t_step
        print(f"      耗时 {dur:.1f}s")
        print(f"      最终股票池 {len(pool_df)} 只股票")
        logger.record_step(
            "筛选步骤3-6",
            dur,
            input_count=count_before_p2,
            output_count=len(pool_df),
        )

    if settings.get("fetch_northbound_flow", True) and not pool_df.empty:
        t_step = time.perf_counter()
        print("[1/3] 附加北向个股 northbound_flow（与行业强度并行）...")
        sector_filter = None
        if is_test and settings.get("test_skip_full_sector", True):
            sector_filter = pool_df["industry"].dropna().unique().tolist()
            print(f"      [测试模式] 仅计算 {len(sector_filter)} 个相关行业")

        with ThreadPoolExecutor(max_workers=2) as executor:
            north_future = executor.submit(attach_northbound_flows, pool_df, settings)
            sector_future = executor.submit(
                fetch_sector_data,
                settings,
                industries_filter=sector_filter,
            )
            pool_df = north_future.result()
            sector_df = sector_future.result()

        dur = time.perf_counter() - t_step
        print(f"      北向+行业并行耗时 {dur:.1f}s")
        logger.record_step("北向与行业并行", dur, stock_count=len(pool_df), industry_count=len(sector_df))
    else:
        t_step = time.perf_counter()
        print("[3/3] 获取行业强度数据...")
        sector_filter = None
        if is_test and settings.get("test_skip_full_sector", True):
            sector_filter = pool_df["industry"].dropna().unique().tolist()
            print(f"      [测试模式] 仅计算 {len(sector_filter)} 个相关行业")
        sector_df = fetch_sector_data(settings, industries_filter=sector_filter)
        dur = time.perf_counter() - t_step
        print(f"      共计算 {len(sector_df)} 个行业，耗时 {dur:.1f}s")
        logger.record_step(
            "行业强度",
            dur,
            industry_count=len(sector_df),
            sector_filtered=sector_filter is not None,
        )

    print()

    # --- 2. 输出市场环境 CSV（已在筛选前获取）---
    market_path = get_output_path(settings, market_schema, run_ts=run_ts)
    export_csv(market_df, market_schema, market_path)
    print(f"[2/3] 市场环境指标已输出: {market_path}")
    logger.set_extra(market_context_output=str(market_path))
    print()

    sector_path = get_output_path(settings, sector_schema, run_ts=run_ts)
    export_csv(sector_df, sector_schema, sector_path)
    print(f"[3/3] 行业强度已输出: {sector_path}（共 {len(sector_df)} 个行业）")
    logger.set_extra(sector_strength_output=str(sector_path))
    print()

    # --- 打分并输出股票池 ---
    t_step = time.perf_counter()
    print("[1/3] 多因子打分（Value/Growth/Capital/Sector）...")
    pool_df = score_stock_pool(pool_df, sector_df, market_df)
    score_summary = summarize_scores(pool_df)
    dur = time.perf_counter() - t_step
    score_min = score_summary.get("final_score_min")
    score_max = score_summary.get("final_score_max")
    if score_min is not None and score_max is not None:
        score_range = f"[{score_min:.2f}, {score_max:.2f}]"
    else:
        score_range = "无（股票池为空或未打分）"
    print(f"      市场档位: {score_summary.get('market_regime')}，"
          f"总分区间 {score_range}，耗时 {dur:.1f}s")
    if score_summary.get("top5") and not quiet:
        print("      总分 Top5:")
        for item in score_summary["top5"]:
            print(f"        {item['ts_code']} {item.get('name', '')} "
                  f"final={item['final_score']:.2f}")
    logger.record_step("多因子打分", dur, **{k: v for k, v in score_summary.items() if k != "top5"})

    pool_path = get_output_path(settings, stock_pool_schema, run_ts=run_ts)
    export_csv(pool_df, stock_pool_schema, pool_path)
    print(f"      已输出: {pool_path}")

    if quiet:
        field_stats = {
            "total_stocks": len(pool_df),
            "fully_filled_stocks": 0,
            "missing_by_field": {},
        }
        print(f"      股票池 {len(pool_df)} 只（批跑精简日志）")
    else:
        field_stats = _print_field_report(pool_df, stock_pool_schema)
    logger.set_extra(
        stock_pool_output=str(pool_path),
        stock_pool_count=len(pool_df),
        stock_pool_field_stats=field_stats,
        stock_pool_score_summary=score_summary,
    )
    print()

    env_alert = print_market_environment_alert(rules.get("params", {}), market_snapshot)
    logger.set_extra(market_environment_alert=env_alert)

    print("=" * 50)
    print(f"完成！总耗时 {time.perf_counter() - t0:.1f}s，请检查 output/ 目录。")
    print("=" * 50)

    return {
        "trade_date": trade_date,
        "stock_pool": pool_path,
        "market_context": market_path,
        "sector_strength": sector_path,
    }


def _print_field_report(df: pd.DataFrame, schema: dict) -> dict:
    """打印字段填充情况，并返回汇总统计（不含 backfill 回填列）"""
    fields = get_validation_field_names(schema)
    print("\n      --- 股票池字段填充报告 ---")

    missing_by_field: dict[str, int] = {f: 0 for f in fields}
    full_count = 0

    for idx, (_, row) in enumerate(df.iterrows()):
        code = row.get("ts_code", "?")
        missing = [f for f in fields if f not in df.columns or pd.isna(row.get(f))]
        for f in missing:
            missing_by_field[f] += 1
        ok = len(fields) - len(missing)
        if ok == len(fields):
            full_count += 1
        # 正式模式股票多，仅打印前 20 条明细
        if len(df) <= 20 or idx < 20:
            print(f"      {code}: {ok}/{len(fields)} 字段有效", end="")
            if missing:
                print(f"  缺失: {', '.join(missing)}")
            else:
                print()

    if len(df) > 20:
        print(f"      ... 其余 {len(df) - 20} 只详见日志报告 stock_pool_field_stats")

    stats = {
        "total_stocks": len(df),
        "fully_filled_stocks": full_count,
        "missing_by_field": {k: v for k, v in missing_by_field.items() if v > 0},
    }
    print(
        f"\n      汇总: {full_count}/{len(df)} 只股票字段全部有效；"
        f"缺失最多的字段: {_top_missing(missing_by_field)}"
    )
    return stats


def _top_missing(missing_by_field: dict[str, int], top_n: int = 5) -> str:
    """取缺失最多的字段描述"""
    items = sorted(missing_by_field.items(), key=lambda x: x[1], reverse=True)
    items = [(k, v) for k, v in items if v > 0][:top_n]
    if not items:
        return "无"
    return ", ".join(f"{k}({v})" for k, v in items)


def main() -> None:
    parser = argparse.ArgumentParser(description="BigASelect A股股票筛选")
    parser.add_argument(
        "--production",
        action="store_true",
        help="正式模式：处理全市场（忽略 settings.yaml 中的 test_mode）",
    )
    args = parser.parse_args()
    run(production=args.production)


if __name__ == "__main__":
    main()
