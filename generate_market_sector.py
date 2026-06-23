"""单独生成 market_context.csv 与 sector_strength.csv"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import (  # noqa: E402
    get_output_path,
    load_field_schema,
    load_settings,
)
from src.data_fetcher import fetch_market_indices, fetch_sector_data, get_trade_date  # noqa: E402
from src.exporter import export_csv  # noqa: E402
from src.fund_flow_provider import prefetch_industry_money_flow  # noqa: E402
from src.network_setup import setup_network  # noqa: E402
from src.output_utils import get_today_date_str  # noqa: E402


def run() -> None:
    """拉取最新市场/行业数据并输出两份 CSV"""
    settings = load_settings()
    setup_network(settings)
    # 独立任务始终输出全行业，不受 test_mode 限制
    settings = {**settings, "test_mode": False}
    run_ts = get_today_date_str()

    trade_date = get_trade_date(settings)
    quote_mode = settings.get("quote_mode", "last_close")
    close_time = settings.get("market_close_time", "15:00")

    market_schema = load_field_schema("market_context_fields.yaml")
    sector_schema = load_field_schema("sector_strength_fields.yaml")

    print("=" * 50)
    print("BigASelect - 市场环境 & 行业强度")
    print("=" * 50)
    print(f"行情模式: {quote_mode}")
    if quote_mode == "last_close":
        print(
            f"数据截止: {trade_date}（非交易时段或收盘前取最近可用收盘，"
            f"收盘时间 {close_time} 北京时间）"
        )
    else:
        print(f"数据日期: {trade_date}")
    print()

    t0 = time.perf_counter()

    t_step = time.perf_counter()
    prefetch_industry_money_flow()
    print(f"      行业资金流预拉耗时 {time.perf_counter() - t_step:.1f}s")
    print()

    t_step = time.perf_counter()
    print("[1/2] 生成 market_context.csv ...")
    market_df = fetch_market_indices(settings)
    market_path = get_output_path(settings, market_schema, run_ts=run_ts)
    export_csv(market_df, market_schema, market_path)
    print(f"      行数 {len(market_df)}，耗时 {time.perf_counter() - t_step:.1f}s")
    print(f"      已输出: {market_path}")
    print()

    t_step = time.perf_counter()
    print("[2/2] 生成 sector_strength.csv（全部行业）...")
    sector_df = fetch_sector_data(settings, industries_filter=None)
    sector_path = get_output_path(settings, sector_schema, run_ts=run_ts)
    export_csv(sector_df, sector_schema, sector_path)
    print(f"      行数 {len(sector_df)}，耗时 {time.perf_counter() - t_step:.1f}s")
    print(f"      已输出: {sector_path}")

    print()
    print("=" * 50)
    print(f"完成！总耗时 {time.perf_counter() - t0:.1f}s")
    print("=" * 50)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="单独生成 market_context.csv 与 sector_strength.csv",
    )
    args = parser.parse_args()
    _ = args
    try:
        run()
    except Exception:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
