"""future_return_20 回填至 archive/stock_pool 列"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.archive_manager import get_archive_root, resolve_archive_csv
from src.factor_analyzer.archive_loader import (
    build_archive_close_index,
    shift_trading_date,
)
from src.factor_analyzer.price_lookup import fetch_close_map_for_stocks


def get_return_column(analysis_cfg: dict[str, Any]) -> str:
    """收益列名（默认 future_return_20）"""
    fr = analysis_cfg.get("future_return", {})
    return fr.get("column", "future_return_20")


def _lookup_close(
    trade_date: str,
    ts_code: str,
    close_index: dict[tuple[str, str], float],
) -> float | None:
    return close_index.get((trade_date, ts_code))


def _save_stock_pool(path: Path, df: pd.DataFrame) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")


def backfill_future_return(
    analysis_dates: list[str],
    *,
    return_horizon: int,
    analysis_cfg: dict[str, Any],
    calendar: list[str],
    close_index: dict[tuple[str, str], float] | None = None,
    rebuild: bool = False,
    root: Path | None = None,
) -> int:
    """
    为分析日期计算 future_return_20，写回 archive/stock_pool CSV。
    返回窗口内 newly available 的有效收益条数（含已有值）。
    """
    archive_root = root or get_archive_root()
    col = get_return_column(analysis_cfg)

    if close_index is None:
        all_dates = set(analysis_dates)
        for td in analysis_dates:
            exit_d = shift_trading_date(td, return_horizon, calendar)
            if exit_d:
                all_dates.add(exit_d)
        close_index = build_archive_close_index(sorted(all_dates), archive_root)

    fr_cfg = analysis_cfg.get("future_return", {})
    allow_fetch = bool(fr_cfg.get("fetch_missing_exit_close", True))
    workers = int(fr_cfg.get("fetch_workers", 8))

    fetch_tasks: list[tuple[str, str, str, str, float, Path, int]] = []
    pending_updates: dict[Path, pd.DataFrame] = {}
    valid_count = 0

    for td in analysis_dates:
        pool_path = resolve_archive_csv(archive_root, "stock_pool", td)
        if pool_path is None or not pool_path.is_file():
            continue

        pool_df = pd.read_csv(pool_path, dtype={"ts_code": str})
        if pool_df.empty:
            continue
        if col not in pool_df.columns:
            pool_df[col] = np.nan

        exit_d = shift_trading_date(td, return_horizon, calendar)
        if exit_d is None:
            continue

        day_updated = False
        for idx, row in pool_df.iterrows():
            ts_code = str(row.get("ts_code", "")).strip()
            if not ts_code:
                continue

            existing = pd.to_numeric(row.get(col), errors="coerce")
            if not rebuild and pd.notna(existing):
                valid_count += 1
                continue

            close_t = pd.to_numeric(row.get("close"), errors="coerce")
            if pd.isna(close_t) or close_t <= 0:
                continue

            exit_close = _lookup_close(exit_d, ts_code, close_index)
            if exit_close is not None:
                pool_df.at[idx, col] = round((exit_close / float(close_t) - 1.0) * 100.0, 4)
                day_updated = True
                valid_count += 1
            elif allow_fetch:
                name = str(row.get("name", "") or "")
                fetch_tasks.append(
                    (td, ts_code, name, exit_d, float(close_t), pool_path, idx)
                )

        if day_updated:
            pending_updates[pool_path] = pool_df

    if fetch_tasks:
        print(
            f"      补拉 T+{return_horizon} 收盘价: {len(fetch_tasks)} 条（按股票去重并发 {workers}）",
            flush=True,
        )
        unique_tasks = [(t[3], t[1], t[2]) for t in fetch_tasks]
        fetched = fetch_close_map_for_stocks(unique_tasks, workers=workers)

        for td, ts_code, _name, exit_d, close_t, pool_path, idx in fetch_tasks:
            exit_close = close_index.get((exit_d, ts_code)) or fetched.get((exit_d, ts_code))
            if exit_close is None:
                continue

            pool_df = pending_updates.get(pool_path)
            if pool_df is None:
                pool_df = pd.read_csv(pool_path, dtype={"ts_code": str})
                if col not in pool_df.columns:
                    pool_df[col] = np.nan
                pending_updates[pool_path] = pool_df

            if not rebuild:
                cur = pd.to_numeric(pool_df.at[idx, col], errors="coerce")
                if pd.notna(cur):
                    valid_count += 1
                    continue

            pool_df.at[idx, col] = round((exit_close / close_t - 1.0) * 100.0, 4)
            valid_count += 1

    saved = 0
    for path, df in pending_updates.items():
        _save_stock_pool(path, df)
        saved += 1
    if saved:
        print(f"      已写回 archive stock_pool: {saved} 个文件（列 {col}）", flush=True)

    return valid_count
