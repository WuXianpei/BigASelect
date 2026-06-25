"""三份输出 CSV 字段填充诊断"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import (  # noqa: E402
    get_field_names,
    get_validation_field_names,
    load_field_schema,
    load_settings,
)
from src.network_setup import setup_network  # noqa: E402


def _analyze_df(df: pd.DataFrame, fields: list[str], label: str) -> dict:
    """统计字段缺失"""
    total_rows = len(df)
    missing_by_field: dict[str, int] = {}
    for f in fields:
        if f not in df.columns:
            missing_by_field[f] = total_rows
        else:
            na = df[f].isna().sum()
            empty = (df[f].astype(str).str.strip() == "").sum() if df[f].dtype == object else 0
            missing_by_field[f] = int(na + empty)

    full_rows = 0
    if total_rows > 0:
        for _, row in df.iterrows():
            ok = all(
                f in df.columns and pd.notna(row.get(f)) and str(row.get(f)).strip() != ""
                for f in fields
            )
            if ok:
                full_rows += 1

    return {
        "label": label,
        "rows": total_rows,
        "fields": len(fields),
        "full_rows": full_rows,
        "missing_by_field": {k: v for k, v in missing_by_field.items() if v > 0},
    }


def main() -> None:
    setup_network(load_settings())
    output_dir = PROJECT_ROOT / "output"

    schemas = {
        "stock_pool.csv": load_field_schema("stock_pool_fields.yaml"),
        "market_context.csv": load_field_schema("market_context_fields.yaml"),
        "sector_strength.csv": load_field_schema("sector_strength_fields.yaml"),
    }

    print("=" * 60)
    print("三份输出文件字段缺失诊断")
    print("=" * 60)

    all_stats = []
    for filename, schema in schemas.items():
        path = output_dir / filename
        all_fields = get_field_names(schema)
        fields = (
            get_validation_field_names(schema)
            if filename == "stock_pool.csv"
            else all_fields
        )
        if not path.exists():
            print(f"\n[{filename}] 文件不存在: {path}")
            continue
        df = pd.read_csv(path, encoding="utf-8-sig")
        stats = _analyze_df(df, fields, filename)
        all_stats.append(stats)

        print(f"\n## {filename}（{stats['rows']} 行，{stats['fields']} 字段）")
        print(f"   全字段有效行: {stats['full_rows']}/{stats['rows']}")
        if not stats["missing_by_field"]:
            print("   全部字段均有值")
        else:
            items = sorted(stats["missing_by_field"].items(), key=lambda x: -x[1])
            for field, cnt in items:
                pct = cnt / stats["rows"] * 100 if stats["rows"] else 0
                print(f"   - {field}: 缺 {cnt}/{stats['rows']} ({pct:.0f}%)")

    print("\n" + "=" * 60)
    print("汇总：按缺失率排序的字段")
    print("=" * 60)
    combined: list[tuple[str, str, int, int]] = []
    for s in all_stats:
        for field, cnt in s["missing_by_field"].items():
            combined.append((s["label"], field, cnt, s["rows"]))
    combined.sort(key=lambda x: (-x[2] / max(x[3], 1), x[0], x[1]))
    for file, field, cnt, rows in combined:
        print(f"  {file:22s} {field:30s} 缺 {cnt}/{rows}")


if __name__ == "__main__":
    main()
