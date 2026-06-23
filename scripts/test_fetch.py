"""字段获取测试脚本：验证股票池与市场指数字段能否正常拉取"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.config_loader import get_field_names, load_field_schema, load_settings
from src.market_enricher import fetch_market_context
from src.sector_enricher import fetch_sector_strength
from src.stock_enricher import enrich_stock_pool

# 测试样本股票
SAMPLE_STOCKS = pd.DataFrame({
    "ts_code": ["000001.SZ", "600519.SH", "300750.SZ", "688981.SH", "832982.BJ"],
    "name": ["平安银行", "贵州茅台", "宁德时代", "中芯国际", "锦波生物"],
})


def _field_coverage(df: pd.DataFrame, fields: list[str]) -> None:
    """打印字段填充率"""
    print(f"\n{'字段':<20} {'非空数':>6} {'填充率':>8}")
    print("-" * 38)
    for field in fields:
        if field not in df.columns:
            print(f"{field:<20} {'缺失列':>6}")
            continue
        filled = df[field].notna().sum()
        rate = filled / len(df) * 100 if len(df) else 0
        print(f"{field:<20} {filled:>6} {rate:>7.1f}%")


def main() -> None:
    settings = load_settings()
    settings["enrich_workers"] = 4

    pool_schema = load_field_schema("stock_pool_fields.yaml")
    market_schema = load_field_schema("market_context_fields.yaml")
    pool_fields = get_field_names(pool_schema)
    market_fields = get_field_names(market_schema)

    print("=" * 50)
    print("股票池字段获取测试（5 只样本）")
    print("=" * 50)
    pool_df = enrich_stock_pool(SAMPLE_STOCKS.copy(), settings)
    _field_coverage(pool_df, pool_fields)
    print("\n样本数据预览：")
    preview_cols = [c for c in pool_fields if c in pool_df.columns]
    print(pool_df[preview_cols].to_string(index=False))
    missing = [c for c in pool_fields if c not in pool_df.columns]
    if missing:
        print(f"\n未获取到列: {', '.join(missing)}")

    print("\n" + "=" * 50)
    print("市场指数字段获取测试（5 个指数）")
    print("=" * 50)
    market_df = fetch_market_context(settings)
    _field_coverage(market_df, market_fields)
    print("\n指数数据预览：")
    preview_cols = [c for c in market_fields if c in market_df.columns]
    print(market_df[preview_cols].to_string(index=False))

    sector_schema = load_field_schema("sector_strength_fields.yaml")
    sector_fields = get_field_names(sector_schema)
    print("\n" + "=" * 50)
    print("行业强度字段获取测试")
    print("=" * 50)
    sector_df = fetch_sector_strength(settings)
    _field_coverage(sector_df, sector_fields)
    print("\n行业数据预览（前10行）：")
    preview_cols = [c for c in sector_fields if c in sector_df.columns]
    print(sector_df[preview_cols].head(10).to_string(index=False))

    out_dir = PROJECT_ROOT / "output"
    out_dir.mkdir(exist_ok=True)
    from src.exporter import align_to_schema
    align_to_schema(pool_df, pool_schema).to_csv(
        out_dir / "test_stock_pool.csv", index=False, encoding="utf-8-sig"
    )
    align_to_schema(market_df, market_schema).to_csv(
        out_dir / "test_market_context.csv", index=False, encoding="utf-8-sig"
    )
    align_to_schema(sector_df, sector_schema).to_csv(
        out_dir / "test_sector_strength.csv", index=False, encoding="utf-8-sig"
    )
    print(f"\n测试文件已输出到 {out_dir}")


if __name__ == "__main__":
    main()
