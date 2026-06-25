"""归档数据加载与交易日窗口"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.archive_manager import (
    get_archive_root,
    list_archived_dates,
    resolve_archive_csv,
)
from src.data_fetcher import get_calendar_last_trading_date
from src.data_fetcher import _get_trading_days_up_to  # noqa: PLC2701
from datetime import datetime
from zoneinfo import ZoneInfo


def get_trading_calendar() -> list[str]:
    """升序 A 股交易日 YYYYMMDD（截至日历最近交易日）"""
    today = pd.Timestamp(datetime.now(ZoneInfo("Asia/Shanghai")).date())
    days = _get_trading_days_up_to(today)
    return [d.strftime("%Y%m%d") for d in days]


def shift_trading_date(
    trade_date: str,
    offset: int,
    calendar: list[str] | None = None,
) -> str | None:
    """沿交易日历偏移，越界返回 None"""
    cal = calendar or get_trading_calendar()
    td = str(trade_date).replace("-", "")
    if td not in cal:
        return None
    idx = cal.index(td) + offset
    if idx < 0 or idx >= len(cal):
        return None
    return cal[idx]


def list_analysis_archive_dates(root: Path | None = None) -> list[str]:
    """已完整归档（pool+market+sector）的交易日"""
    root = root or get_archive_root()
    return list_archived_dates(root)


def load_archived_triplet(
    trade_date: str,
    root: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    """读取某日归档三件套，缺失返回 None"""
    root = root or get_archive_root()
    paths = {
        key: resolve_archive_csv(root, key, trade_date)
        for key in ("stock_pool", "market_context", "sector_strength")
    }
    if any(p is None for p in paths.values()):
        return None
    return (
        pd.read_csv(paths["stock_pool"]),
        pd.read_csv(paths["market_context"]),
        pd.read_csv(paths["sector_strength"]),
    )


def resolve_analysis_window(
    archived_dates: list[str],
    *,
    min_days: int,
    max_days: int,
    return_horizon: int,
    calendar: list[str] | None = None,
) -> dict[str, Any]:
    """
    确定可计算 future_return 的分析日期窗口。
    交易日 T 要求 T+return_horizon 在日历内且 <= 最近交易日。
    """
    cal = calendar or get_trading_calendar()
    cal_set = set(cal)
    latest_trading = get_calendar_last_trading_date()

    return_ready: list[str] = []
    for td in archived_dates:
        exit_d = shift_trading_date(td, return_horizon, cal)
        if exit_d is None:
            continue
        if exit_d > latest_trading:
            continue
        if td not in cal_set:
            continue
        return_ready.append(td)

    if not return_ready:
        return {
            "analysis_dates": [],
            "return_ready_dates": [],
            "return_end": None,
            "sample_sufficient": False,
            "requested_min": min_days,
            "requested_max": max_days,
        }

    return_end = return_ready[-1]
    window = return_ready[-max_days:]
    sample_sufficient = len(window) >= min_days

    return {
        "analysis_dates": window,
        "return_ready_dates": return_ready,
        "return_end": return_end,
        "sample_sufficient": sample_sufficient,
        "requested_min": min_days,
        "requested_max": max_days,
        "return_horizon": return_horizon,
        "latest_trading": latest_trading,
    }


def build_archive_close_index(
    archived_dates: list[str],
    root: Path | None = None,
) -> dict[tuple[str, str], float]:
    """(trade_date, ts_code) -> close，来自 archive stock_pool"""
    root = root or get_archive_root()
    index: dict[tuple[str, str], float] = {}
    for td in archived_dates:
        path = resolve_archive_csv(root, "stock_pool", td)
        if path is None or not path.is_file():
            continue
        df = pd.read_csv(path, usecols=["ts_code", "close"])
        for row in df.itertuples(index=False):
            code = str(getattr(row, "ts_code", ""))
            close = pd.to_numeric(getattr(row, "close", None), errors="coerce")
            if code and pd.notna(close) and close > 0:
                index[(td, code)] = float(close)
    return index
