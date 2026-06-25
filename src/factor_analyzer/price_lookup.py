"""收盘价查询：archive 优先，按股票缓存日 K 补拉"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import pandas as pd

from src.bj_resolver import resolve_api_symbol
from src.data_utils import find_column


def _api_symbol_from_ts_code(ts_code: str, name: str | None = None) -> str:
    """ts_code（如 600519.SH）→ 6 位 API 代码"""
    code = str(ts_code).split(".")[0].zfill(6)
    return resolve_api_symbol(code, name)


def close_from_daily(daily: pd.DataFrame | None, trade_date: str) -> float | None:
    """从日 K 取指定交易日（精确匹配）收盘价"""
    if daily is None or daily.empty:
        return None
    td = str(trade_date).replace("-", "")
    date_col = find_column(daily.columns, "date", "日期")
    close_col = find_column(daily.columns, "close", "收盘", "收盘价")
    if not date_col or not close_col:
        return None
    dates = pd.to_datetime(daily[date_col], errors="coerce")
    mask = dates.dt.strftime("%Y%m%d") == td
    if not mask.any():
        return None
    val = pd.to_numeric(daily.loc[mask, close_col].iloc[-1], errors="coerce")
    if pd.isna(val) or val <= 0:
        return None
    return float(val)


def fetch_close_map_for_stocks(
    tasks: list[tuple[str, str, str]],
    *,
    workers: int = 8,
) -> dict[tuple[str, str], float]:
    """
    批量获取 (exit_date, ts_code) 收盘价。
    tasks: (exit_date, ts_code, name)
    按 ts_code 只拉一次日 K，再查多个 exit_date。
    """
    from src.stock_enricher import _fetch_stock_daily  # noqa: PLC2701

    by_code: dict[str, list[tuple[str, str]]] = {}
    for exit_d, ts_code, name in tasks:
        by_code.setdefault(ts_code, []).append((exit_d, name))

    result: dict[tuple[str, str], float] = {}

    def _one(ts_code: str, entries: list[tuple[str, str]]) -> list[tuple[str, str, float]]:
        name = entries[0][1] if entries else None
        api_sym = _api_symbol_from_ts_code(ts_code, name)
        daily = _fetch_stock_daily(api_sym)
        rows: list[tuple[str, str, float]] = []
        for exit_d, _ in entries:
            c = close_from_daily(daily, exit_d)
            if c is not None:
                rows.append((exit_d, ts_code, c))
        return rows

    total = len(by_code)
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_one, code, ents): code for code, ents in by_code.items()
        }
        for fut in as_completed(futures):
            done += 1
            if done == total or done % max(50, total // 10) == 0:
                print(f"        exit 收盘价补拉进度 {done}/{total}", flush=True)
            try:
                for exit_d, ts_code, close in fut.result():
                    result[(exit_d, ts_code)] = close
            except Exception:
                continue
    return result
