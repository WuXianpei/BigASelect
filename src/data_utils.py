"""数据源工具：重试、代码转换、多源回退"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, TypeVar

import pandas as pd

T = TypeVar("T")


def retry_call(
    fn: Callable[[], T],
    retries: int = 3,
    delay: float = 1.5,
    label: str = "",
    silent: bool = False,
) -> T | None:
    """带重试的函数调用，失败返回 None"""
    last_err: Exception | None = None
    for i in range(retries):
        try:
            return fn()
        except Exception as exc:
            last_err = exc
            if i < retries - 1:
                time.sleep(delay * (i + 1))
    if label and not silent:
        print(f"        [警告] {label} 获取失败: {last_err}")
    return None


def symbol_to_ts_code(symbol: str) -> str:
    """6位股票代码转 ts_code"""
    symbol = str(symbol).zfill(6)
    if symbol.startswith("92") or symbol.startswith(("4", "8")):
        return f"{symbol}.BJ"
    if symbol.startswith(("6", "9")):
        return f"{symbol}.SH"
    return f"{symbol}.SZ"


def infer_market(symbol: str) -> str:
    """根据代码推断市场板块"""
    s = str(symbol).zfill(6)
    if s.startswith(("4", "8", "92")):
        return "北交所"
    if s.startswith("688"):
        return "科创板"
    if s.startswith("300"):
        return "创业板"
    if s.startswith("6"):
        return "上交所"
    return "深交所"


def symbol_to_sina_prefix(symbol: str) -> str:
    """6位代码转新浪行情前缀格式，如 sz000001"""
    symbol = str(symbol).zfill(6)
    if symbol.startswith("92") or symbol.startswith(("4", "8")):
        return f"bj{symbol}"
    if symbol.startswith(("6", "9")):
        return f"sh{symbol}"
    return f"sz{symbol}"


def parse_sina_code(code: str) -> str:
    """解析新浪代码字段 sz000001 → 000001"""
    code = str(code).lower()
    for prefix in ("sh", "sz", "bj"):
        if code.startswith(prefix):
            return code[len(prefix):].zfill(6)
    return code.zfill(6)


def normalize_hist_df(df: pd.DataFrame) -> pd.Series | None:
    """统一日K数据为收盘价 Series"""
    if df is None or df.empty:
        return None

    close_col = None
    for col in ("close", "收盘", "收盘价"):
        if col in df.columns:
            close_col = col
            break
    if close_col is None:
        return None

    return pd.to_numeric(df[close_col], errors="coerce").dropna()


def trim_daily_to_quote_date(daily: pd.DataFrame, quote_date: str) -> pd.DataFrame:
    """截取日K至指定收盘日（含），用于统一「最新可用收盘」口径"""
    if daily is None or daily.empty:
        return daily

    date_col = find_column(daily.columns, "date", "日期")
    if not date_col:
        return daily

    dates = pd.to_datetime(daily[date_col], errors="coerce")
    cutoff = pd.Timestamp(quote_date)
    mask = dates <= cutoff
    trimmed = daily.loc[mask].copy()
    return trimmed if not trimmed.empty else daily


def select_quote_bars(
    daily: pd.DataFrame,
    quote_date: str,
) -> tuple[pd.Series | None, pd.Series | None]:
    """
    选取目标收盘日 K 线及前一交易日 K 线。
    若目标日数据尚未入库（如刚收盘），自动回退到 <= quote_date 的最后一根。
    """
    if daily is None or daily.empty:
        return None, None

    trimmed = trim_daily_to_quote_date(daily, quote_date)
    if trimmed.empty:
        return None, None

    current = trimmed.iloc[-1]
    previous = trimmed.iloc[-2] if len(trimmed) >= 2 else None
    return current, previous


def find_column(columns: pd.Index, *candidates: str) -> str | None:
    """在列名中模糊匹配目标列"""
    col_list = [str(c) for c in columns]
    for cand in candidates:
        for col in col_list:
            if cand in col:
                return col
    return None


# 东财/同花顺行业名称差异映射
INDUSTRY_ALIASES: dict[str, str] = {
    "银行Ⅱ": "银行",
    "银行Ⅲ": "银行",
    "白酒Ⅱ": "白酒",
    "白酒Ⅲ": "白酒",
    "医疗美容": "医疗美容",
    "医疗美容业": "医疗美容",
    "医药制造业": "化学制药",
}


def normalize_industry_name(name: str | None) -> str | None:
    """统一行业名称，便于与同花顺板块匹配"""
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return None
    text = str(name).strip()
    if text in INDUSTRY_ALIASES:
        return INDUSTRY_ALIASES[text]
    for suffix in ("Ⅲ", "Ⅱ"):
        if text.endswith(suffix):
            return text[: -len(suffix)]
    return text


def expand_industry_names(names: list[str]) -> set[str]:
    """扩展行业名集合，包含别名"""
    expanded: set[str] = set()
    for raw in names:
        if not raw or (isinstance(raw, float) and pd.isna(raw)):
            continue
        text = str(raw).strip()
        expanded.add(text)
        normalized = normalize_industry_name(text)
        if normalized:
            expanded.add(normalized)
        for alias, target in INDUSTRY_ALIASES.items():
            if text in (alias, target):
                expanded.add(alias)
                expanded.add(target)
    return expanded


def parse_chinese_amount(value: Any) -> float | None:
    """
    解析中文金额字符串为元，如 4.88亿、-8501.28万、12.3
  """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None

    text = str(value).strip().replace(",", "")
    if not text or text in ("--", "-", "nan"):
        return None

    multiplier = 1.0
    if text.endswith("亿"):
        multiplier = 1e8
        text = text[:-1]
    elif text.endswith("万"):
        multiplier = 1e4
        text = text[:-1]

    try:
        return float(text) * multiplier
    except ValueError:
        num = pd.to_numeric(text, errors="coerce")
        return float(num) if pd.notna(num) else None
