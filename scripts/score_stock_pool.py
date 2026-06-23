"""对已有 stock_pool CSV 补打多因子分并写回（无需重新拉取股票数据）"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import load_field_schema, load_settings  # noqa: E402
from src.exporter import export_csv  # noqa: E402
from src.stock_scorer import score_stock_pool, summarize_scores  # noqa: E402

_TS_PATTERN = re.compile(r"stock_pool_(\d{8}(?:_\d{6})?)\.csv$")


def _parse_pool_suffix(name: str) -> str | None:
    match = _TS_PATTERN.match(name)
    return match.group(1) if match else None


def _resolve_triplet(pool_path: Path) -> tuple[Path, Path, Path]:
    """根据 stock_pool 路径推断同日期后缀的 market / sector 文件"""
    suffix = _parse_pool_suffix(pool_path.name)
    if not suffix:
        raise ValueError(f"文件名需为 stock_pool_YYYYMMDD.csv 或 stock_pool_YYYYMMDD_HHMMSS.csv: {pool_path.name}")
    output_dir = pool_path.parent
    market_path = output_dir / f"market_context_{suffix}.csv"
    sector_path = output_dir / f"sector_strength_{suffix}.csv"
    for path, label in ((market_path, "market_context"), (sector_path, "sector_strength")):
        if not path.is_file():
            raise FileNotFoundError(f"缺少同时间戳 {label} 文件: {path}")
    return pool_path, market_path, sector_path


def _latest_pool(output_dir: Path) -> Path:
    """取 output 下日期后缀最新的 stock_pool CSV"""
    candidates: list[tuple[str, Path]] = []
    for path in output_dir.glob("stock_pool_*.csv"):
        suffix = _parse_pool_suffix(path.name)
        if suffix:
            candidates.append((suffix, path))
    if not candidates:
        raise FileNotFoundError(f"未找到 stock_pool_*.csv: {output_dir}")
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def run(pool_path: Path | None = None, *, in_place: bool = True) -> Path:
    settings = load_settings()
    schema = load_field_schema("stock_pool_fields.yaml")
    output_dir = PROJECT_ROOT / settings.get("output_dir", "output")

    if pool_path is None:
        pool_path = _latest_pool(output_dir)

    pool_path, market_path, sector_path = _resolve_triplet(pool_path.resolve())
    pool_df = pd.read_csv(pool_path)
    market_df = pd.read_csv(market_path)
    sector_df = pd.read_csv(sector_path)

    scored = score_stock_pool(pool_df, sector_df, market_df)
    summary = summarize_scores(scored)

    out_path = pool_path if in_place else pool_path.with_name(
        pool_path.stem + "_scored.csv"
    )
    export_csv(scored, schema, out_path)

    print(f"已打分: {len(scored)} 只")
    print(f"市场档位: {summary.get('market_regime')}")
    print(f"总分区间: [{summary.get('final_score_min'):.2f}, {summary.get('final_score_max'):.2f}]")
    print(f"已写入: {out_path}")
    if summary.get("top5"):
        print("Top5:")
        for item in summary["top5"]:
            print(f"  {item['ts_code']} {item.get('name', '')} final={item['final_score']:.2f}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="对已有 stock_pool CSV 补打多因子分")
    parser.add_argument(
        "pool_csv",
        nargs="?",
        help="stock_pool CSV 路径，省略则使用 output 下最新一份",
    )
    parser.add_argument(
        "--no-in-place",
        action="store_true",
        help="写入新文件 *_scored.csv，不覆盖原文件",
    )
    args = parser.parse_args()
    pool = Path(args.pool_csv) if args.pool_csv else None
    run(pool, in_place=not args.no_in_place)


if __name__ == "__main__":
    main()
