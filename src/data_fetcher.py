"""数据获取模块"""

from __future__ import annotations

from datetime import datetime, time
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

# A 股收盘时间（北京时间）
_MARKET_TZ = ZoneInfo("Asia/Shanghai")
_MARKET_CLOSE = time(15, 0)

# 交易日历内存缓存
_TRADING_DAYS_CACHE: list[pd.Timestamp] | None = None


def get_trade_date(settings: dict[str, Any]) -> str:
    """
    获取有效行情日期（YYYYMMDD）。
    last_close 模式：未收盘用上一交易日，已收盘用当天（均为「最新可用收盘日」）。
    """
    trade_date = settings.get("trade_date")
    if trade_date:
        return str(trade_date).replace("-", "")

    if settings.get("quote_mode", "last_close") == "last_close":
        return get_latest_close_date(settings)

    return _now_cn().strftime("%Y%m%d")


def get_latest_close_date(settings: dict[str, Any]) -> str:
    """
    最新可用收盘日：
    - 非交易日 → 最近一个交易日
    - 交易日未收盘（<15:00）→ 上一交易日
    - 交易日已收盘 → 当天
    """
    now = _now_cn()
    today = pd.Timestamp(now.date())
    days = _get_trading_days_up_to(today)
    if not days:
        return now.strftime("%Y%m%d")

    close_time = _parse_close_time(settings.get("market_close_time", "15:00"))
    today_str = today.strftime("%Y%m%d")
    today_is_trading = today_str in {d.strftime("%Y%m%d") for d in days}

    if today_is_trading and now.time() < close_time:
        if len(days) >= 2:
            return days[-2].strftime("%Y%m%d")
        return days[-1].strftime("%Y%m%d")

    return days[-1].strftime("%Y%m%d")


def get_last_trading_date() -> str:
    """兼容旧调用：等同于最新可用收盘日（无 settings 时按当前时间推断）"""
    return get_latest_close_date({})


def get_calendar_last_trading_date() -> str:
    """日历上最近一个交易日（YYYYMMDD），不含「未收盘则用昨日」逻辑"""
    now = _now_cn()
    today = pd.Timestamp(now.date())
    days = _get_trading_days_up_to(today)
    if not days:
        return now.strftime("%Y%m%d")
    return days[-1].strftime("%Y%m%d")


def _now_cn() -> datetime:
    """北京时间当前时刻"""
    return datetime.now(_MARKET_TZ)


def _parse_close_time(text: str) -> time:
    """解析 HH:MM 收盘时间配置"""
    parts = str(text).strip().split(":")
    hour = int(parts[0])
    minute = int(parts[1]) if len(parts) > 1 else 0
    return time(hour, minute)


def _get_trading_days_up_to(end_date: pd.Timestamp) -> list[pd.Timestamp]:
    """获取 <= end_date 的全部交易日（升序）"""
    global _TRADING_DAYS_CACHE
    if _TRADING_DAYS_CACHE is None:
        _TRADING_DAYS_CACHE = _load_trading_calendar()

    end = pd.Timestamp(end_date.date())
    return [d for d in _TRADING_DAYS_CACHE if d <= end]


def _load_trading_calendar() -> list[pd.Timestamp]:
    """加载 A 股交易日历"""
    import akshare as ak

    from .data_utils import retry_call

    def _fetch() -> list[pd.Timestamp]:
        cal = ak.tool_trade_date_hist_sina()
        if cal is None or cal.empty:
            raise ValueError("交易日历为空")
        date_col = "trade_date" if "trade_date" in cal.columns else cal.columns[0]
        dates = pd.to_datetime(cal[date_col], errors="coerce").dropna().sort_values()
        return [pd.Timestamp(d.date()) for d in dates]

    result = retry_call(_fetch, retries=2, label="交易日历")
    return result if result is not None else []


def get_backfill_end_date(settings: dict[str, Any] | None = None) -> str:
    """
    历史批跑截止日：与 quote_mode=last_close 一致的最新可用收盘日。
    交易日已收盘（默认 15:00 后）则含当天；未收盘则仍为上一交易日。
    """
    settings = settings or {}
    return get_latest_close_date(settings)


def get_trading_days_window(end_date: str, count: int) -> list[str]:
    """
    取截止 end_date（含）向前 count 个交易日，升序返回 YYYYMMDD 列表。
    """
    end = pd.Timestamp(str(end_date).replace("-", ""))
    days = _get_trading_days_up_to(end)
    eligible = [d for d in days if d <= end]
    if not eligible:
        return []
    window = eligible[-count:]
    return [d.strftime("%Y%m%d") for d in window]


def fetch_a_share_list(settings: dict[str, Any]) -> pd.DataFrame:
    """
    获取 A 股股票列表（基础数据）
    后续可根据 stock_pool_fields.yaml 中的 source 字段扩展更多指标
    """
    provider = settings.get("data_provider", "akshare")

    if provider == "akshare":
        return _fetch_from_akshare()
    if provider == "tushare":
        return _fetch_from_tushare(settings)
    raise ValueError(f"不支持的数据提供方: {provider}")


def _fetch_from_akshare() -> pd.DataFrame:
  """通过 akshare 获取 A 股代码列表"""
  import akshare as ak

  df = ak.stock_info_a_code_name()
  # 统一列名
  df = df.rename(columns={"code": "symbol", "name": "name"})
  df["symbol"] = df["symbol"].astype(str).str.zfill(6)
  df["ts_code"] = df.apply(_symbol_to_ts_code, axis=1)
  return df


def _fetch_from_tushare(settings: dict[str, Any]) -> pd.DataFrame:
  """通过 tushare 获取 A 股列表"""
  token = settings.get("tushare", {}).get("token", "")
  if not token:
    raise ValueError("使用 tushare 需在 config/settings.yaml 中配置 token")

  import tushare as ts

  pro = ts.pro_api(token)
  df = pro.stock_basic(
    exchange="",
    list_status="L",
    fields="ts_code,symbol,name,industry,market,list_date",
  )
  return df


def _symbol_to_ts_code(row: pd.Series) -> str:
  """根据股票代码推断 ts_code 后缀"""
  symbol = str(row["symbol"]).zfill(6)
  if symbol.startswith("92") or symbol.startswith(("4", "8")):
    return f"{symbol}.BJ"
  if symbol.startswith(("6", "9")):
    return f"{symbol}.SH"
  if symbol.startswith(("0", "2", "3")):
    return f"{symbol}.SZ"
  return f"{symbol}.SZ"


def fetch_market_indices(settings: dict[str, Any]) -> pd.DataFrame:
    """
    获取市场整体环境数据（兼容旧接口名）
    实际逻辑见 market_enricher.fetch_market_context
    """
    from .market_enricher import fetch_market_context

    return fetch_market_context(settings)


def fetch_sector_data(
    settings: dict[str, Any],
    industries_filter: list[str] | None = None,
) -> pd.DataFrame:
    """获取行业强度数据"""
    from .sector_enricher import fetch_sector_strength

    return fetch_sector_strength(settings, industries_filter=industries_filter)
