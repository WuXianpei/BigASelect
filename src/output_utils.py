"""输出文件日期后缀（北京时间）"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

_MARKET_TZ = ZoneInfo("Asia/Shanghai")


def get_today_date_str() -> str:
    """北京时间当日 YYYYMMDD，用作 CSV 文件名后缀"""
    return datetime.now(_MARKET_TZ).strftime("%Y%m%d")


def get_run_timestamp() -> str:
    """与 get_today_date_str 相同，供 CSV 输出路径使用"""
    return get_today_date_str()
