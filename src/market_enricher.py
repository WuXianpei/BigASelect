"""市场环境指标 enrichment（数据丰富）模块"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np
import pandas as pd

from .data_fetcher import get_trade_date
from .data_utils import find_column, retry_call, trim_daily_to_quote_date
from .fund_flow_provider import get_market_money_flow_index

# 筛选步骤 6 默认基准：上证指数
PRIMARY_INDEX_CODE = "000001"
PRIMARY_INDEX_SYMBOL = "sh000001"

_DEFAULT_INDICES: list[dict[str, str]] = [
    {"code": "000001", "symbol": "sh000001", "name": "上证指数"},
    {"code": "399006", "symbol": "sz399006", "name": "创业板指"},
    {"code": "000688", "symbol": "sh000688", "name": "科创50"},
    {"code": "899050", "symbol": "bj899050", "name": "北证50"},
    {"code": "000300", "symbol": "sh000300", "name": "沪深300"},
]


def fetch_market_context(settings: dict[str, Any]) -> pd.DataFrame:
    """获取 A 股市场宏观环境（多指数 + 全市场指标，每个指数一行）"""
    quote_date = get_trade_date(settings)
    indices = settings.get("market_indices") or _DEFAULT_INDICES
    macro = _fetch_macro_fields(quote_date)
    rows = _fetch_index_rows(indices, quote_date, macro)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def get_primary_market_snapshot(df: pd.DataFrame) -> dict[str, Any]:
    """取基准指数（默认上证指数）所在行，供筛选步骤 6 使用"""
    if df is None or df.empty:
        return {}
    if "index_code" in df.columns:
        code = df["index_code"].astype(str).str.zfill(6)
        match = df[code == PRIMARY_INDEX_CODE.zfill(6)]
        if not match.empty:
            return match.iloc[0].to_dict()
    return df.iloc[0].to_dict()


def _fetch_macro_fields(quote_date: str) -> dict[str, Any]:
    """拉取全市场宏观字段（各行共用）"""
    macro: dict[str, Any] = {
        "market_risk_index": None,
        "market_breadth_up": None,
        "market_breadth_down": None,
        "market_money_flow": get_market_money_flow_index(),
        "northbound_total_flow": _fetch_northbound_total_flow(),
        "vix_proxy": _fetch_vix_proxy(),
        "sector_rotation_state": _calc_sector_rotation_state(),
    }

    primary_close = _fetch_index_close(PRIMARY_INDEX_SYMBOL, quote_date)
    if primary_close is not None and len(primary_close) >= 2:
        latest = float(primary_close.iloc[-1])
        prev = float(primary_close.iloc[-2])
        change_pct = round((latest / prev - 1) * 100, 4) if prev != 0 else None
        ma20 = _calc_ma_position(primary_close, 20)
        vol20 = _calc_volatility_20d(primary_close)
        macro["market_risk_index"] = _calc_market_risk_index(
            vol20,
            ma20,
            change_pct,
            macro.get("vix_proxy"),
        )

    macro.update(_fetch_market_breadth())
    return macro


def _fetch_index_rows(
    indices: list[dict[str, str]],
    quote_date: str,
    macro: dict[str, Any],
) -> list[dict[str, Any]]:
    """并发拉取各指数行情行"""
    rows: list[dict[str, Any]] = []
    max_workers = min(8, max(len(indices), 1))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_build_index_row, item, quote_date, macro): item
            for item in indices
        }
        for future in as_completed(futures):
            try:
                row = future.result()
                if row:
                    rows.append(row)
            except Exception:
                continue

    # 按 settings 中指数顺序排序
    order = {str(it.get("code", "")).zfill(6): i for i, it in enumerate(indices)}
    rows.sort(key=lambda r: order.get(str(r.get("index_code", "")).zfill(6), 999))
    return rows


def _build_index_row(
    item: dict[str, str],
    quote_date: str,
    macro: dict[str, Any],
) -> dict[str, Any]:
    """组装单指数一行"""
    code = str(item.get("code", "")).zfill(6)
    symbol = item.get("symbol", "")
    name = item.get("name", code)

    row: dict[str, Any] = {
        "index_code": code,
        "index_name": name,
        "index_close": None,
        "index_change_pct": None,
        "index_ma20_position": None,
        "index_ma60_position": None,
        "index_ma120_position": None,
        "market_volatility_20d": None,
        **macro,
    }

    if not symbol:
        return row

    close = _fetch_index_close(symbol, quote_date)
    if close is not None and len(close) >= 2:
        latest = float(close.iloc[-1])
        prev = float(close.iloc[-2])
        row["index_close"] = round(latest, 4)
        if prev != 0:
            row["index_change_pct"] = round((latest / prev - 1) * 100, 4)
        row["index_ma20_position"] = _calc_ma_position(close, 20)
        row["index_ma60_position"] = _calc_ma_position(close, 60)
        row["index_ma120_position"] = _calc_ma_position(close, 120)
        row["market_volatility_20d"] = _calc_volatility_20d(close)

    return row


def _fetch_index_close(symbol: str, quote_date: str) -> pd.Series | None:
    """获取指数收盘价序列（截至最新可用收盘日）"""
    import akshare as ak

    def _fetch() -> pd.Series:
        df = ak.stock_zh_index_daily(symbol=symbol)
        if df is None or df.empty or "close" not in df.columns:
            raise ValueError(f"指数 {symbol} 无数据")
        if "date" in df.columns:
            df = trim_daily_to_quote_date(df, quote_date)
        return pd.to_numeric(df["close"], errors="coerce").dropna()

    return retry_call(_fetch, label=f"指数{symbol}", silent=True)


def _calc_ma_position(close: pd.Series, period: int) -> float | None:
    """收盘价相对均线的偏离（%）"""
    if len(close) < period:
        return None
    ma = close.rolling(period).mean().iloc[-1]
    latest = close.iloc[-1]
    if pd.isna(ma) or ma == 0:
        return None
    return round(float((latest / ma - 1) * 100), 4)


def _calc_volatility_20d(close: pd.Series, window: int = 20) -> float | None:
    """20 日年化实现波动率（%）"""
    if len(close) < window + 1:
        return None
    returns = close.pct_change().dropna()
    if len(returns) < window:
        return None
    vol = returns.tail(window).std() * np.sqrt(252) * 100
    return round(float(vol), 4) if pd.notna(vol) else None


def _calc_market_risk_index(
    volatility: float | None,
    ma20_position: float | None,
    change_pct: float | None,
    vix_proxy: float | None,
) -> float | None:
    """市场风险指数（0-100，基于上证指数）"""
    score = 50.0

    if volatility is not None:
        if volatility >= 25:
            score += 25
        elif volatility >= 18:
            score += 15
        elif volatility <= 12:
            score -= 10

    if ma20_position is not None:
        if ma20_position <= -5:
            score += 15
        elif ma20_position >= 5:
            score -= 10

    if change_pct is not None:
        if change_pct <= -2:
            score += 10
        elif change_pct >= 2:
            score -= 5

    if vix_proxy is not None:
        if vix_proxy >= 25:
            score += 15
        elif vix_proxy >= 20:
            score += 8
        elif vix_proxy <= 15:
            score -= 8

    return round(float(np.clip(score, 0, 100)), 2)


def _fetch_market_breadth() -> dict[str, int | None]:
    """全市场涨跌家数"""
    import akshare as ak

    def _fetch() -> dict[str, int]:
        raw = ak.stock_zh_a_spot()
        pct_col = find_column(raw.columns, "涨跌幅")
        if not pct_col:
            raise ValueError("行情缺少涨跌幅列")
        pct = pd.to_numeric(raw[pct_col], errors="coerce")
        return {
            "market_breadth_up": int((pct > 0).sum()),
            "market_breadth_down": int((pct < 0).sum()),
        }

    result = retry_call(_fetch, retries=2, label="全市场涨跌家数", silent=True)
    return result if result is not None else {"market_breadth_up": None, "market_breadth_down": None}


def _fetch_northbound_total_flow() -> float | None:
    """北向资金当日净流入（亿元）"""
    import akshare as ak

    def _fetch_from_summary() -> float:
        df = ak.stock_hsgt_fund_flow_summary_em()
        if df is None or df.empty:
            raise ValueError("北向汇总为空")
        net_col = find_column(df.columns, "成交净买额", "资金流入")
        type_col = find_column(df.columns, "类型", "板块")
        if not net_col or not type_col:
            raise ValueError("北向汇总缺少必要列")
        north = df[df[type_col].astype(str).str.contains("股通", na=False)]
        north = north[~north[type_col].astype(str).str.contains("港股", na=False)]
        total = pd.to_numeric(north[net_col], errors="coerce").sum()
        if pd.isna(total):
            raise ValueError("北向净买额无效")
        return round(float(total), 4)

    def _fetch_from_hist() -> float:
        df = ak.stock_hsgt_hist_em(symbol="北向资金")
        if df is None or df.empty:
            raise ValueError("北向历史为空")
        flow_col = find_column(df.columns, "当日资金流入")
        if not flow_col:
            raise ValueError("北向历史缺少资金流入列")
        series = pd.to_numeric(df[flow_col], errors="coerce").dropna()
        if series.empty:
            raise ValueError("北向历史无有效资金流入")
        return round(float(series.iloc[-1]), 4)

    result = retry_call(_fetch_from_summary, retries=1, label="北向当日汇总", silent=True)
    if result is not None:
        return result
    return retry_call(_fetch_from_hist, retries=1, label="北向历史资金", silent=True)


def _fetch_vix_proxy() -> float | None:
    """50ETF 期权 QVIX（恐慌指数代理）"""
    import akshare as ak

    def _fetch() -> float:
        df = ak.index_option_50etf_qvix()
        if df is None or df.empty or "close" not in df.columns:
            raise ValueError("QVIX 数据为空")
        val = pd.to_numeric(df["close"].iloc[-1], errors="coerce")
        if pd.isna(val):
            raise ValueError("QVIX 收盘无效")
        return round(float(val), 4)

    return retry_call(_fetch, retries=2, label="50ETF QVIX", silent=True)


def _calc_sector_rotation_state() -> str | None:
    """行业轮动状态判定（同花顺行业汇总）"""
    import akshare as ak

    def _fetch() -> str:
        boards = ak.stock_board_industry_summary_ths()
        pct_col = find_column(boards.columns, "涨跌幅")
        if not pct_col or boards.empty:
            raise ValueError("行业板块涨跌幅为空")
        pct = pd.to_numeric(boards[pct_col], errors="coerce").dropna()
        if pct.empty:
            raise ValueError("行业涨跌幅无有效值")

        positive = int((pct > 0).sum())
        negative = int((pct < 0).sum())
        total = len(pct)
        spread = float(pct.max() - pct.min())

        if positive >= total * 0.7:
            return "普涨"
        if negative >= total * 0.7:
            return "普跌"
        if spread >= 3:
            return "强轮动"
        return "结构性"

    return retry_call(_fetch, retries=3, delay=1.0, label="行业轮动状态", silent=True)
