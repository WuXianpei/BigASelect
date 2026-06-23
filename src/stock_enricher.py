"""股票池指标 enrichment（数据丰富）模块"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from .bj_resolver import get_bj_industry, merge_bj_industry_to_map, is_bj_symbol, resolve_api_symbol
from .data_utils import (
    find_column,
    infer_market,
    normalize_hist_df,
    parse_sina_code,
    retry_call,
    select_quote_bars,
    symbol_to_sina_prefix,
    symbol_to_ts_code,
    trim_daily_to_quote_date,
)
from .concurrency import resolve_worker_count
from .data_fetcher import get_calendar_last_trading_date, get_trade_date
from .fund_flow_provider import get_main_net_inflow

# 行业映射内存缓存，避免重复拉取业绩报表
_INDUSTRY_MAP_CACHE: dict[str, str] | None = None

# 日K数据缓存，同一次运行内避免重复请求
_DAILY_CACHE: dict[str, pd.DataFrame] = {}

# 小批量阈值：低于此数量时逐只拉取行情，避免全市场扫描
_SMALL_BATCH_THRESHOLD = 20


def clear_enrichment_caches() -> None:
    """清空单次运行内缓存（历史批跑每个交易日之间需调用）"""
    global _INDUSTRY_MAP_CACHE
    _INDUSTRY_MAP_CACHE = None
    _DAILY_CACHE.clear()


def enrich_light(df: pd.DataFrame, settings: dict[str, Any]) -> pd.DataFrame:
    """轻量 enrichment：风险标识 + 行情快照（用于筛选步骤 1-2）"""
    if df.empty:
        return df

    result = df.copy()
    names = result.get("name", pd.Series(dtype=str))

    result["is_st"] = names.apply(_detect_is_st).astype(int)
    result["risk_flag"] = names.apply(_detect_risk_flag).astype(int)
    result["is_suspended"] = 0

    quote_mode = settings.get("quote_mode", "last_close")
    trade_date = get_trade_date(settings)
    if quote_mode == "last_close":
        print(f"      获取最新收盘行情（数据截止 {trade_date}，筛选步骤 1-2）...")
    else:
        print("      获取实时行情快照（筛选步骤 1-2）...")

    symbols = result["ts_code"].str.split(".").str[0].tolist()
    name_series = result.get("name")
    name_by_sym: dict[str, str | None] = {}
    if name_series is not None:
        for sym, nm in zip(symbols, name_series, strict=False):
            name_by_sym[sym] = None if pd.isna(nm) else str(nm)

    quote_df = _fetch_quote_batch(symbols, settings, name_by_sym)
    if not quote_df.empty:
        result = _merge_columns(result, quote_df, on="ts_code")

    close = (
        pd.to_numeric(result["close"], errors="coerce")
        if "close" in result.columns
        else pd.Series(np.nan, index=result.index)
    )
    amount = (
        pd.to_numeric(result["amount"], errors="coerce")
        if "amount" in result.columns
        else pd.Series(np.nan, index=result.index)
    )
    result["is_suspended"] = _detect_suspended_batch(
        close, amount, quote_df, quote_mode=quote_mode
    )

    return result


def _detect_suspended_batch(
    close: pd.Series,
    amount: pd.Series,
    quote_df: pd.DataFrame,
    *,
    quote_mode: str = "last_close",
) -> pd.Series:
    """判定停牌标识"""
    if quote_df.empty:
        return pd.Series(0, index=close.index, dtype=int)

    if quote_mode == "last_close":
        suspended = pd.Series(0, index=close.index, dtype=int)
        has_price = close.notna() & (close > 0)
        no_trade = has_price & amount.notna() & (amount <= 0)
        suspended.loc[no_trade] = 1
        return suspended

    # 实时快照：非交易时段数据可能全为 0
    valid_rate = (
        (pd.to_numeric(quote_df.get("amount"), errors="coerce") > 0).mean()
        if "amount" in quote_df.columns
        else 0.0
    )
    if valid_rate < 0.3:
        print(
            f"        [警告] 实时行情有效率低 ({valid_rate:.1%})，"
            "可能为非交易时段，跳过停牌判定"
        )
        return pd.Series(0, index=close.index, dtype=int)

    suspended = pd.Series(0, index=close.index, dtype=int)
    has_quote = close.notna() & amount.notna()
    suspended.loc[has_quote & (close <= 0) & (amount <= 0)] = 1
    return suspended


def enrich_stock_pool(
    df: pd.DataFrame,
    settings: dict[str, Any],
    *,
    skip_spot: bool = False,
) -> pd.DataFrame:
    """为股票池补充行情、技术、基本面等字段（技术 + 财务全量，兼容旧流程）"""
    result = enrich_technical_pool(df, settings, skip_spot=skip_spot)
    return enrich_financial_pool(result, settings)


def enrich_technical_pool(
    df: pd.DataFrame,
    settings: dict[str, Any],
    *,
    skip_spot: bool = False,
) -> pd.DataFrame:
    """技术 enrichment：行业 + 日K技术指标 + 同花顺资金流（筛选步骤 3-4 前）"""
    if df.empty:
        return df

    result = _attach_industry_and_market(df, settings, skip_spot=skip_spot)
    symbols = result["ts_code"].str.split(".").str[0].tolist()
    name_by_sym = _build_name_map(result, symbols)
    quote_date = get_trade_date(settings)

    print(f"      计算技术指标与资金流（共 {len(symbols)} 只）...")
    max_workers = resolve_worker_count(
        settings, "enrich_workers", default=20, item_count=len(symbols)
    )
    enrich_rows = _fetch_technical_batch(
        symbols,
        max_workers=max_workers,
        name_by_sym=name_by_sym,
        quote_date=quote_date,
        progress_label="技术 enrichment",
    )
    if enrich_rows:
        enrich_df = pd.DataFrame(enrich_rows)
        result = _merge_columns(result, enrich_df, on="ts_code")

    return result


def enrich_financial_pool(df: pd.DataFrame, settings: dict[str, Any]) -> pd.DataFrame:
    """财务 enrichment：估值 + 财务（仅对当前 DataFrame 中的股票）"""
    if df.empty:
        return df

    result = df.copy()
    symbols = result["ts_code"].str.split(".").str[0].tolist()
    name_by_sym = _build_name_map(result, symbols)

    print(f"      计算估值与财务（共 {len(symbols)} 只）...")
    max_workers = resolve_worker_count(
        settings, "enrich_workers", default=20, item_count=len(symbols)
    )
    enrich_rows = _fetch_financial_batch(
        symbols,
        max_workers=max_workers,
        name_by_sym=name_by_sym,
        progress_label="财务 enrichment",
    )
    if enrich_rows:
        enrich_df = pd.DataFrame(enrich_rows)
        result = _merge_columns(result, enrich_df, on="ts_code")

    return result


def _build_name_map(df: pd.DataFrame, symbols: list[str]) -> dict[str, str | None]:
    """构建 代码→名称 映射"""
    name_by_sym: dict[str, str | None] = {}
    name_series = df.get("name")
    if name_series is not None:
        for sym, nm in zip(symbols, name_series, strict=False):
            name_by_sym[sym] = None if pd.isna(nm) else str(nm)
    return name_by_sym


def _attach_industry_and_market(
    df: pd.DataFrame,
    settings: dict[str, Any],
    *,
    skip_spot: bool,
) -> pd.DataFrame:
    """附加行业、市场板块，可选刷新行情快照"""
    result = df.copy()
    symbols = result["ts_code"].str.split(".").str[0].tolist()

    if not skip_spot:
        quote_mode = settings.get("quote_mode", "last_close")
        trade_date = get_trade_date(settings)
        if quote_mode == "last_close":
            print(f"      获取最新收盘行情（数据截止 {trade_date}）...")
        else:
            print("      获取实时行情快照...")
        name_by_sym = _build_name_map(result, symbols)
        quote_df = _fetch_quote_batch(symbols, settings, name_by_sym)
        if not quote_df.empty:
            result = _merge_columns(result, quote_df, on="ts_code")

    print("      构建行业映射...")
    industry_map = _build_industry_map()
    if industry_map:
        result["industry"] = result.apply(
            lambda r: industry_map.get(
                str(r["ts_code"]).split(".")[0],
                get_bj_industry(
                    str(r["ts_code"]).split(".")[0],
                    None if pd.isna(r.get("name")) else str(r.get("name")),
                ),
            ),
            axis=1,
        )

    result["market"] = result["ts_code"].str.split(".").str[0].map(infer_market)
    return result


def _merge_columns(left: pd.DataFrame, right: pd.DataFrame, on: str) -> pd.DataFrame:
    """合并数据帧，右侧非空值覆盖左侧同名列"""
    overlap = [c for c in right.columns if c in left.columns and c != on]
    merged = left.drop(columns=overlap, errors="ignore").merge(right, on=on, how="left")
    return merged


def _detect_is_st(name: str) -> int:
    """识别 ST 股票"""
    if pd.isna(name):
        return 0
    return 1 if "ST" in str(name).upper() else 0


def _detect_risk_flag(name: str) -> int:
    """识别退市/风险警示等特殊标的"""
    if pd.isna(name):
        return 0
    text = str(name)
    for keyword in ("退", "警示", "PT"):
        if keyword in text:
            return 1
    return 0


def _fetch_quote_batch(
    symbols: list[str],
    settings: dict[str, Any],
    name_by_sym: dict[str, str | None] | None = None,
) -> pd.DataFrame:
    """按配置获取行情：最近收盘或实时快照"""
    quote_mode = settings.get("quote_mode", "last_close")
    if quote_mode == "last_close":
        return _fetch_last_close_batch(symbols, settings, name_by_sym)
    if len(symbols) <= _SMALL_BATCH_THRESHOLD:
        return _fetch_spot_for_symbols(symbols, settings, name_by_sym)
    spot_df = _fetch_spot_snapshot()
    if spot_df.empty:
        return spot_df
    if "turnover_rate" not in spot_df.columns or spot_df["turnover_rate"].isna().mean() > 0.5:
        em_spot = _fetch_spot_em()
        if not em_spot.empty and "turnover_rate" in em_spot.columns:
            spot_df = spot_df.merge(
                em_spot[["ts_code", "turnover_rate"]], on="ts_code", how="left", suffixes=("", "_em")
            )
            if "turnover_rate_em" in spot_df.columns:
                spot_df["turnover_rate"] = spot_df["turnover_rate"].fillna(spot_df["turnover_rate_em"])
                spot_df = spot_df.drop(columns=["turnover_rate_em"])
    return spot_df


def _fetch_last_close_batch(
    symbols: list[str],
    settings: dict[str, Any],
    name_by_sym: dict[str, str | None] | None = None,
) -> pd.DataFrame:
    """批量获取最新收盘行情：优先腾讯批量接口，失败则逐只日K"""
    quote_date = get_trade_date(settings)
    batch_df = _fetch_tencent_batch_quotes(symbols, quote_date, settings)
    if not batch_df.empty:
        print(f"        腾讯批量行情 {len(batch_df)}/{len(symbols)} 只")
        return batch_df

    print("        腾讯批量失败，回退逐只日K...")
    return _fetch_last_close_daily_fallback(symbols, settings, name_by_sym, quote_date)


def _fetch_last_close_daily_fallback(
    symbols: list[str],
    settings: dict[str, Any],
    name_by_sym: dict[str, str | None] | None,
    quote_date: str,
) -> pd.DataFrame:
    """逐只拉日K获取收盘（批量接口不可用时的回退）"""
    max_workers = resolve_worker_count(
        settings, "enrich_workers", default=20, item_count=len(symbols)
    )
    rows: list[dict[str, Any]] = []
    total = len(symbols)
    done = 0
    log_every = max(100, total // 20)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _fetch_last_close_one,
                sym,
                (name_by_sym or {}).get(sym),
                quote_date,
            ): sym
            for sym in symbols
        }
        for future in as_completed(futures):
            done += 1
            if done == total or done % log_every == 0:
                print(f"        最近收盘进度 {done}/{total}")
            try:
                row = future.result()
                if row:
                    rows.append(row)
            except Exception:
                continue
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _fetch_tencent_batch_quotes(
    symbols: list[str],
    quote_date: str,
    settings: dict[str, Any],
) -> pd.DataFrame:
    """腾讯 qt.gtimg.cn 批量行情（全市场约数秒）"""
    import requests

    batch_size = settings.get("quote_batch_size", 80)
    use_fresh_amount = quote_date == get_calendar_last_trading_date()
    rows: list[dict[str, Any]] = []

    tencent_codes = [symbol_to_sina_prefix(s) for s in symbols]

    def _fetch_batch(batch: list[str]) -> list[dict[str, Any]]:
        url = "http://qt.gtimg.cn/q=" + ",".join(batch)
        resp = requests.get(
            url,
            timeout=20,
            proxies={"http": None, "https": None},
        )
        resp.encoding = "gbk"
        batch_rows: list[dict[str, Any]] = []
        for line in resp.text.split(";"):
            row = _parse_tencent_quote_line(line, quote_date, use_fresh_amount)
            if row:
                batch_rows.append(row)
        return batch_rows

    for i in range(0, len(tencent_codes), batch_size):
        batch = tencent_codes[i : i + batch_size]
        try:
            rows.extend(retry_call(lambda b=batch: _fetch_batch(b), retries=2, delay=0.5) or [])
        except Exception:
            continue

    return pd.DataFrame(rows).drop_duplicates("ts_code") if rows else pd.DataFrame()


def _parse_tencent_quote_line(
    line: str,
    quote_date: str,
    use_fresh_amount: bool,
) -> dict[str, Any] | None:
    """解析腾讯行情单行"""
    line = line.strip()
    if not line or "=" not in line:
        return None

    _, val = line.split("=", 1)
    val = val.strip().strip('"')
    parts = val.split("~")
    if len(parts) < 39 or not parts[2]:
        return None

    symbol = str(parts[2]).zfill(6)
    ts_code = symbol_to_ts_code(symbol)
    current = _safe_float(parts[3])
    prev_close = _safe_float(parts[4])

    if use_fresh_amount:
        close = current if current else prev_close
    else:
        close = prev_close if prev_close else current

    if close is None or close <= 0:
        return None

    row: dict[str, Any] = {
        "ts_code": ts_code,
        "symbol": symbol,
        "close": close,
        "quote_date": quote_date,
    }

    if parts[30]:
        row["bar_date"] = parts[30][:8]

    if use_fresh_amount:
        pct = _safe_float(parts[32])
        if pct is not None:
            row["pct_chg"] = round(pct, 4)
        open_p = _safe_float(parts[5])
        high_p = _safe_float(parts[33])
        low_p = _safe_float(parts[34])
        if open_p is not None:
            row["open"] = open_p
        if high_p is not None:
            row["high"] = high_p
        if low_p is not None:
            row["low"] = low_p
        amount_wan = _safe_float(parts[37])
        if amount_wan is not None:
            row["amount"] = amount_wan * 10_000
        turnover = _safe_float(parts[38])
        if turnover is not None:
            row["turnover_rate"] = round(turnover, 4)
        vol = _safe_float(parts[36])
        if vol is not None:
            row["volume"] = vol * 100

    return row


def _safe_float(text: str) -> float | None:
    """安全解析浮点数"""
    try:
        if text is None or str(text).strip() == "":
            return None
        return float(text)
    except (TypeError, ValueError):
        return None


def _fetch_last_close_one(
    symbol: str,
    name: str | None = None,
    quote_date: str | None = None,
) -> dict[str, Any]:
    """单只股票指定收盘日：收盘价、涨跌幅、成交额、换手率"""
    api_symbol = resolve_api_symbol(symbol, name)
    ts_code = symbol_to_ts_code(symbol)
    row: dict[str, Any] = {"ts_code": ts_code, "symbol": symbol}

    daily = _fetch_stock_daily(api_symbol)
    if daily is None or daily.empty or not quote_date:
        return row

    current, previous = select_quote_bars(daily, quote_date)
    if current is None:
        return row

    close_col = find_column(daily.columns, "close", "收盘", "收盘价")
    if not close_col:
        return row

    close_val = pd.to_numeric(current[close_col], errors="coerce")
    if pd.isna(close_val):
        return row

    row["close"] = float(close_val)
    row["quote_date"] = quote_date

    date_col = find_column(daily.columns, "date", "日期")
    if date_col:
        row["bar_date"] = str(current[date_col])[:10]

    vol_col = find_column(daily.columns, "volume", "成交量")
    amt_col = find_column(daily.columns, "amount", "成交额")
    if vol_col:
        vol = pd.to_numeric(current[vol_col], errors="coerce")
        if pd.notna(vol):
            row["volume"] = float(vol)
    if amt_col:
        amt = pd.to_numeric(current[amt_col], errors="coerce")
        if pd.notna(amt):
            row["amount"] = float(amt)
    elif row.get("close") and row.get("volume"):
        row["amount"] = row["close"] * row["volume"]

    if "turnover" in daily.columns:
        tr = pd.to_numeric(current["turnover"], errors="coerce")
        if pd.notna(tr):
            row["turnover_rate"] = round(float(tr) * 100, 4)

    if previous is not None and close_col:
        prev_close = pd.to_numeric(previous[close_col], errors="coerce")
        if pd.notna(prev_close) and prev_close != 0:
            row["pct_chg"] = round((row["close"] / float(prev_close) - 1) * 100, 4)

    return row


def _extract_quote_fields_from_daily(
    daily: pd.DataFrame,
    quote_date: str,
) -> dict[str, Any]:
    """从日K提取指定收盘日的涨跌幅、成交量、成交额"""
    row: dict[str, Any] = {}
    if daily is None or daily.empty:
        return row

    current, previous = select_quote_bars(daily, quote_date)
    if current is None:
        current = daily.iloc[-1]
        previous = daily.iloc[-2] if len(daily) >= 2 else None

    close_col = find_column(daily.columns, "close", "收盘", "收盘价")
    if not close_col:
        return row

    close_val = pd.to_numeric(current[close_col], errors="coerce")
    if pd.isna(close_val):
        return row
    row["close"] = float(close_val)

    vol_col = find_column(daily.columns, "volume", "成交量")
    amt_col = find_column(daily.columns, "amount", "成交额")
    if vol_col:
        vol = pd.to_numeric(current[vol_col], errors="coerce")
        if pd.notna(vol):
            row["volume"] = float(vol)
    if amt_col:
        amt = pd.to_numeric(current[amt_col], errors="coerce")
        if pd.notna(amt):
            row["amount"] = float(amt)
    elif row.get("close") and row.get("volume"):
        row["amount"] = row["close"] * row["volume"]

    pct_col = find_column(daily.columns, "pct_chg", "涨跌幅")
    if pct_col:
        pct = pd.to_numeric(current[pct_col], errors="coerce")
        if pd.notna(pct):
            row["pct_chg"] = round(float(pct), 4)

    if "pct_chg" not in row and previous is not None and close_col:
        prev_close = pd.to_numeric(previous[close_col], errors="coerce")
        if pd.notna(prev_close) and prev_close != 0:
            row["pct_chg"] = round((row["close"] / float(prev_close) - 1) * 100, 4)

    return row


def _fetch_spot_for_symbols(
    symbols: list[str],
    settings: dict[str, Any],
    name_by_sym: dict[str, str | None] | None = None,
) -> pd.DataFrame:
    """小批量：并发逐只获取行情（避免全市场扫描）"""
    max_workers = resolve_worker_count(
        settings, "enrich_workers", default=20, item_count=len(symbols)
    )
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_fetch_spot_one, sym, (name_by_sym or {}).get(sym)): sym
            for sym in symbols
        }
        for future in as_completed(futures):
            try:
                row = future.result()
                if row:
                    rows.append(row)
            except Exception:
                continue
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _fetch_spot_one(symbol: str, name: str | None = None) -> dict[str, Any]:
    """单只股票行情快照（含收盘价、成交额、换手率）"""
    import akshare as ak

    api_symbol = resolve_api_symbol(symbol, name)
    ts_code = symbol_to_ts_code(symbol)
    row: dict[str, Any] = {"ts_code": ts_code, "symbol": symbol}

    daily = _fetch_stock_daily(api_symbol)
    close = normalize_hist_df(daily) if daily is not None and not daily.empty else None
    if daily is not None and not daily.empty:
        if close is not None and not close.empty:
            row["close"] = float(close.iloc[-1])
        vol_col = find_column(daily.columns, "volume", "成交量")
        amt_col = find_column(daily.columns, "amount", "成交额")
        if vol_col:
            vol = pd.to_numeric(daily.iloc[-1][vol_col], errors="coerce")
            if pd.notna(vol):
                row["volume"] = float(vol)
        if amt_col:
            amt = pd.to_numeric(daily.iloc[-1][amt_col], errors="coerce")
            if pd.notna(amt):
                row["amount"] = float(amt)
        elif row.get("close") and row.get("volume"):
            row["amount"] = row["close"] * row["volume"]

    def _valuation() -> dict[str, Any]:
        val = ak.stock_value_em(symbol=api_symbol)
        if val is None or val.empty:
            raise ValueError("估值数据为空")
        latest = val.iloc[-1]
        out: dict[str, Any] = {}
        col_close = find_column(val.columns, "当日收盘价", "收盘价")
        col_pe = find_column(val.columns, "PE(TTM)")
        col_pb = find_column(val.columns, "市净率", "PB")
        col_float = find_column(val.columns, "流通股本")
        if col_close and "close" not in row:
            out["close"] = pd.to_numeric(latest[col_close], errors="coerce")
        if col_pe:
            out["pe_ttm"] = pd.to_numeric(latest[col_pe], errors="coerce")
        if col_pb:
            out["pb"] = pd.to_numeric(latest[col_pb], errors="coerce")
        if col_float and row.get("volume"):
            float_shares = pd.to_numeric(latest[col_float], errors="coerce")
            if pd.notna(float_shares) and float_shares > 0:
                out["turnover_rate"] = round(float(row["volume"]) / float_shares * 100, 4)
        return out

    val_data = retry_call(_valuation, retries=2, delay=0.5)
    if val_data:
        row.update({k: v for k, v in val_data.items() if k not in row or pd.isna(row.get(k))})

    # 北交所：新浪日K 含 turnover 字段时补充换手率
    if is_bj_symbol(symbol) and "turnover_rate" not in row and daily is not None:
        if "turnover" in daily.columns:
            tr = pd.to_numeric(daily.iloc[-1]["turnover"], errors="coerce")
            if pd.notna(tr):
                row["turnover_rate"] = round(float(tr) * 100, 4)

    if daily is not None and not daily.empty and close is not None and len(close) >= 2 and "close" in row:
        prev = float(close.iloc[-2])
        if prev != 0:
            row["pct_chg"] = round((row["close"] / prev - 1) * 100, 4)

    return row


def _fetch_spot_snapshot() -> pd.DataFrame:
    """批量获取 A 股行情：优先新浪，失败则东财"""
    df = _fetch_spot_sina()
    if not df.empty:
        return df
    return _fetch_spot_em()


def _fetch_spot_sina() -> pd.DataFrame:
    """新浪全市场行情"""
    import akshare as ak

    def _fetch() -> pd.DataFrame:
        raw = ak.stock_zh_a_spot()
        col_map = {
            "代码": "raw_code",
            "名称": "name",
            "最新价": "close",
            "涨跌幅": "pct_chg",
            "成交额": "amount",
        }
        out = raw.rename(columns={k: v for k, v in col_map.items() if k in raw.columns})
        if "raw_code" not in out.columns:
            raise ValueError("新浪行情缺少代码列")
        out["symbol"] = out["raw_code"].map(parse_sina_code)
        out["ts_code"] = out["symbol"].map(symbol_to_ts_code)
        keep = ["ts_code", "name", "close", "pct_chg", "amount"]
        return out[[c for c in keep if c in out.columns]].drop_duplicates("ts_code")

    result = retry_call(_fetch, label="新浪行情")
    return result if result is not None else pd.DataFrame()


def _fetch_spot_em() -> pd.DataFrame:
    """东财全市场行情（备用）"""
    import akshare as ak

    def _fetch() -> pd.DataFrame:
        raw = ak.stock_zh_a_spot_em()
        col_map = {
            "代码": "symbol",
            "名称": "name",
            "最新价": "close",
            "涨跌幅": "pct_chg",
            "成交额": "amount",
            "换手率": "turnover_rate",
            "市盈率-动态": "pe_ttm",
            "市净率": "pb",
        }
        out = raw.rename(columns={k: v for k, v in col_map.items() if k in raw.columns})
        out["symbol"] = out["symbol"].astype(str).str.zfill(6)
        out["ts_code"] = out["symbol"].map(symbol_to_ts_code)
        keep = [
            "ts_code", "name", "close", "pct_chg", "amount",
            "turnover_rate", "pe_ttm", "pb",
        ]
        return out[[c for c in keep if c in out.columns]].drop_duplicates("ts_code")

    result = retry_call(_fetch, retries=2, label="东财行情")
    return result if result is not None else pd.DataFrame()


def _build_industry_map() -> dict[str, str]:
    """构建 代码→行业 映射：业绩报表 + 北交所列表"""
    mapping = _fetch_industry_from_yjbb()
    if not mapping:
        mapping = _fetch_industry_from_boards()

    merge_bj_industry_to_map(mapping)

    return mapping


def _fetch_industry_from_yjbb() -> dict[str, str]:
    """从业绩报表批量获取所属行业"""
    global _INDUSTRY_MAP_CACHE
    if _INDUSTRY_MAP_CACHE is not None:
        return _INDUSTRY_MAP_CACHE

    import akshare as ak

    def _fetch() -> dict[str, str]:
        for date in ("20250331", "20241231", "20240930", "20240630"):
            df = ak.stock_yjbb_em(date=date)
            if df is None or df.empty:
                continue
            industry_col = find_column(df.columns, "所属行业", "行业")
            code_col = find_column(df.columns, "股票代码", "代码")
            if not industry_col or not code_col:
                continue
            codes = df[code_col].astype(str).str.zfill(6)
            return dict(zip(codes, df[industry_col], strict=False))
        raise ValueError("业绩报表无行业数据")

    result = retry_call(_fetch, retries=2, label="业绩报表行业")
    if result is not None:
        _INDUSTRY_MAP_CACHE = result
    return result if result is not None else {}


def _fetch_industry_from_boards() -> dict[str, str]:
    """从行业板块成分股构建映射（备用）"""
    import akshare as ak

    def _fetch() -> dict[str, str]:
        boards = ak.stock_board_industry_name_em()
        if boards.empty or "板块名称" not in boards.columns:
            raise ValueError("行业板块列表为空")

        mapping: dict[str, str] = {}
        for _, row in boards.iterrows():
            industry = row["板块名称"]
            cons = ak.stock_board_industry_cons_em(symbol=industry)
            if cons.empty or "代码" not in cons.columns:
                continue
            for code in cons["代码"].astype(str).str.zfill(6):
                if code not in mapping:
                    mapping[code] = industry
        return mapping

    result = retry_call(_fetch, retries=2, label="行业板块映射")
    return result if result is not None else {}


def _fetch_technical_batch(
    symbols: list[str],
    max_workers: int = 8,
    name_by_sym: dict[str, str | None] | None = None,
    quote_date: str | None = None,
    progress_label: str = "技术 enrichment",
) -> list[dict[str, Any]]:
    """并发获取技术指标与资金流"""
    return _run_symbol_batch(
        symbols,
        max_workers,
        lambda sym: _calc_technical_one(
            sym, name=(name_by_sym or {}).get(sym), quote_date=quote_date
        ),
        progress_label=progress_label,
    )


def _fetch_financial_batch(
    symbols: list[str],
    max_workers: int = 8,
    name_by_sym: dict[str, str | None] | None = None,
    progress_label: str = "财务 enrichment",
) -> list[dict[str, Any]]:
    """并发获取估值与财务指标"""
    return _run_symbol_batch(
        symbols,
        max_workers,
        lambda sym: _fetch_valuation_financial_one(sym, name=(name_by_sym or {}).get(sym)),
        progress_label=progress_label,
    )


def _run_symbol_batch(
    symbols: list[str],
    max_workers: int,
    worker,
    *,
    progress_label: str,
) -> list[dict[str, Any]]:
    """通用并发批处理（带进度日志）"""
    rows: list[dict[str, Any]] = []
    total = len(symbols)
    done = 0
    log_every = max(50, total // 20)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(worker, sym): sym for sym in symbols}
        for future in as_completed(futures):
            done += 1
            if done == total or done % log_every == 0:
                print(f"        {progress_label} 进度 {done}/{total}")
            try:
                row = future.result()
                if row:
                    rows.append(row)
            except Exception:
                continue
    return rows


def _fetch_enrich_batch(
    symbols: list[str],
    max_workers: int = 8,
    name_by_sym: dict[str, str | None] | None = None,
    quote_date: str | None = None,
) -> list[dict[str, Any]]:
    """并发：技术指标 + 估值财务（单股合并，减少重复请求）"""
    rows: list[dict[str, Any]] = []
    total = len(symbols)
    done = 0
    log_every = max(50, total // 20)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _enrich_one_symbol,
                sym,
                (name_by_sym or {}).get(sym),
                quote_date,
            ): sym
            for sym in symbols
        }
        for future in as_completed(futures):
            done += 1
            if done == total or done % log_every == 0:
                print(f"        完整 enrichment 进度 {done}/{total}")
            try:
                row = future.result()
                if row:
                    rows.append(row)
            except Exception:
                continue
    return rows


def _enrich_one_symbol(
    symbol: str,
    name: str | None = None,
    quote_date: str | None = None,
) -> dict[str, Any]:
    """单只股票完整 enrichment（技术 + 估值 + 财务，技术/财务并行）"""
    with ThreadPoolExecutor(max_workers=2) as executor:
        tech_future = executor.submit(
            _calc_technical_one, symbol, name=name, quote_date=quote_date
        )
        fin_future = executor.submit(_fetch_valuation_financial_one, symbol, name=name)
        tech = tech_future.result()
        fin = fin_future.result()

    for key, val in fin.items():
        if key != "ts_code" and (key not in tech or pd.isna(tech.get(key))):
            tech[key] = val
    return tech


def _fetch_stock_daily(symbol: str) -> pd.DataFrame | None:
    """获取日K数据（含收盘价与成交量）"""
    if symbol in _DAILY_CACHE:
        return _DAILY_CACHE[symbol]

    import akshare as ak

    sina_symbol = symbol_to_sina_prefix(symbol)

    def _sina() -> pd.DataFrame:
        df = ak.stock_zh_a_daily(symbol=sina_symbol)
        if df is None or df.empty:
            raise ValueError("新浪日K为空")
        return df

    daily = retry_call(_sina, retries=1, delay=0.3, silent=True)
    if daily is not None:
        _DAILY_CACHE[symbol] = daily
        return daily

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=600)).strftime("%Y-%m-%d")

    def _tx() -> pd.DataFrame:
        df = ak.stock_zh_a_hist_tx(
            symbol=symbol_to_ts_code(symbol).lower(),
            start_date=start_date,
            end_date=end_date,
        )
        if df is None or df.empty:
            raise ValueError("腾讯日K为空")
        return df

    daily = retry_call(_tx, retries=1, delay=0.3, silent=True)
    if daily is not None:
        _DAILY_CACHE[symbol] = daily
    return daily


def _fetch_stock_history(symbol: str) -> pd.Series | None:
    """获取日K收盘价序列（兼容旧调用）"""
    daily = _fetch_stock_daily(symbol)
    if daily is None:
        return None
    return normalize_hist_df(daily)


def _calc_technical_one(
    symbol: str,
    name: str | None = None,
    quote_date: str | None = None,
) -> dict[str, Any]:
    """单只股票：日K技术指标 + 5日资金流"""
    api_symbol = resolve_api_symbol(symbol, name)
    ts_code = symbol_to_ts_code(symbol)
    daily = _fetch_stock_daily(api_symbol)
    if daily is None or daily.empty:
        return {"ts_code": ts_code}

    if quote_date:
        daily = trim_daily_to_quote_date(daily, quote_date)
    if daily is None or daily.empty:
        return {"ts_code": ts_code}

    close = normalize_hist_df(daily)
    if close is None or close.empty:
        return {"ts_code": ts_code}

    volume = None
    vol_col = find_column(daily.columns, "volume", "成交量")
    if vol_col:
        volume = pd.to_numeric(daily[vol_col], errors="coerce")

    metrics = _calc_technical_metrics(close, volume, daily=daily)
    metrics["ts_code"] = ts_code

    if quote_date:
        bar_fields = _extract_quote_fields_from_daily(daily, quote_date)
        for key, val in bar_fields.items():
            if val is not None and (key not in metrics or pd.isna(metrics.get(key))):
                metrics[key] = val

    for days, field in (
        (1, "main_net_inflow_1d"),
        (3, "main_net_inflow_3d"),
        (5, "main_net_inflow_5d"),
        (10, "main_net_inflow_10d"),
    ):
        flow_val = get_main_net_inflow(symbol, days, api_symbol, daily=daily)
        if flow_val is not None:
            metrics[field] = flow_val

    return metrics


def _calc_technical_metrics(
    close: pd.Series,
    volume: pd.Series | None = None,
    *,
    daily: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """由日K收盘价序列计算技术指标"""
    result: dict[str, Any] = {}
    if len(close) < 5:
        return result

    ma5 = close.rolling(5).mean()
    ma10 = close.rolling(10).mean()
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()
    ma120 = close.rolling(120).mean()
    ma250 = close.rolling(250).mean()

    latest_close = float(close.iloc[-1])
    result["close"] = latest_close

    if daily is not None and not daily.empty:
        last = daily.iloc[-1]
        for src_cols, key in (
            (("open", "开盘", "开盘价"), "open"),
            (("high", "最高", "最高价"), "high"),
            (("low", "最低", "最低价"), "low"),
        ):
            col = find_column(daily.columns, *src_cols)
            if col:
                val = pd.to_numeric(last[col], errors="coerce")
                if pd.notna(val):
                    result[key] = round(float(val), 4)

    for period, ma_series, key in (
        (5, ma5, "ma5"),
        (10, ma10, "ma10"),
        (20, ma20, "ma20"),
        (60, ma60, "ma60"),
        (120, ma120, "ma120"),
        (250, ma250, "ma250"),
    ):
        if pd.notna(ma_series.iloc[-1]):
            result[key] = round(float(ma_series.iloc[-1]), 4)

    if len(ma20) >= 6 and pd.notna(ma20.iloc[-1]) and pd.notna(ma20.iloc[-6]) and ma20.iloc[-6] != 0:
        result["ma20_slope"] = round(
            (ma20.iloc[-1] - ma20.iloc[-6]) / ma20.iloc[-6] * 100, 4
        )
    if len(ma60) >= 6 and pd.notna(ma60.iloc[-1]) and pd.notna(ma60.iloc[-6]) and ma60.iloc[-6] != 0:
        result["ma60_slope"] = round(
            (ma60.iloc[-1] - ma60.iloc[-6]) / ma60.iloc[-6] * 100, 4
        )

    if len(close) >= 20:
        high_20 = float(close.tail(20).max())
        low_20 = float(close.tail(20).min())
        result["high_20d"] = round(high_20, 4)
        result["low_20d"] = round(low_20, 4)
        if high_20 != 0:
            result["close_vs_20d_high"] = round((latest_close / high_20 - 1) * 100, 4)

    if len(close) >= 60:
        high_60 = float(close.tail(60).max())
        low_60 = float(close.tail(60).min())
        result["high_60d"] = round(high_60, 4)
        result["low_60d"] = round(low_60, 4)
        if high_60 != 0:
            result["close_vs_60d_high"] = round((latest_close / high_60 - 1) * 100, 4)

    year_window = close.tail(252)
    if len(year_window) >= 20:
        result["price_percentile_1y"] = round(
            float((year_window < latest_close).mean() * 100), 4
        )

    if len(close) >= 21:
        daily_ret = close.pct_change().tail(20).dropna()
        if len(daily_ret) >= 5:
            result["volatility_20d"] = round(float(daily_ret.std() * (252**0.5) * 100), 4)

    if volume is not None and len(volume) >= 20:
        vol_tail = volume.dropna()
        if len(vol_tail) >= 20:
            vol_mean = vol_tail.tail(20).mean()
            if vol_mean > 0:
                result["volume_ratio"] = round(float(vol_tail.iloc[-1] / vol_mean), 4)

    if len(close) > 60 and close.iloc[-61] != 0:
        result["ret_60d"] = round((latest_close / float(close.iloc[-61]) - 1) * 100, 4)

    if len(close) > 90 and close.iloc[-91] != 0:
        result["return_90d"] = round((latest_close / float(close.iloc[-91]) - 1) * 100, 4)

    result["price_above_ma20_days"] = _count_consecutive_above(close, ma20)
    result["price_above_ma60_days"] = _count_consecutive_above(close, ma60)
    result["trend_strength_score"] = _calc_trend_strength_score(
        latest_close, ma20.iloc[-1], ma60.iloc[-1], result
    )

    rsi = _calc_rsi(close, 14)
    macd = _calc_macd_hist(close)
    if rsi is not None:
        result["rsi_14"] = round(rsi, 4)
    if macd is not None:
        result["macd_hist"] = round(macd, 4)

    return result


def _count_consecutive_above(close: pd.Series, ma: pd.Series) -> int:
    """从最近交易日向前统计收盘价连续站上均线的天数"""
    if len(close) < 2 or ma.isna().all():
        return 0
    count = 0
    for i in range(len(close) - 1, -1, -1):
        c = close.iloc[i]
        m = ma.iloc[i]
        if pd.isna(c) or pd.isna(m) or c <= m:
            break
        count += 1
    return count


def _calc_trend_strength_score(
    close: float,
    ma20: float,
    ma60: float,
    metrics: dict[str, Any],
) -> float:
    """趋势强度评分（0-100）"""
    score = 0.0
    if pd.notna(ma20) and close > ma20:
        score += 25
    if pd.notna(ma60) and close > ma60:
        score += 25
    ma20_slope = metrics.get("ma20_slope")
    if ma20_slope is not None and ma20_slope > 0:
        score += 25
    close_vs_20d = metrics.get("close_vs_20d_high")
    if close_vs_20d is not None and close_vs_20d > -10:
        score += 25
    return round(score, 2)


def _calc_rsi(close: pd.Series, period: int = 14) -> float | None:
    """计算 RSI"""
    if len(close) < period + 1:
        return None
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    val = rsi.iloc[-1]
    return float(val) if pd.notna(val) else None


def _calc_macd_hist(close: pd.Series) -> float | None:
    """计算 MACD 柱状图"""
    if len(close) < 35:
        return None
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    hist = (dif - dea) * 2
    val = hist.iloc[-1]
    return float(val) if pd.notna(val) else None


def _fetch_valuation_financial_batch(
    symbols: list[str],
    max_workers: int = 8,
) -> list[dict[str, Any]]:
    """并发获取 PE/PB/换手率、ROE、净利润同比"""
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_fetch_valuation_financial_one, sym): sym for sym in symbols}
        for future in as_completed(futures):
            try:
                row = future.result()
                if row:
                    rows.append(row)
            except Exception:
                continue
    return rows


def _fetch_valuation_financial_one(symbol: str, name: str | None = None) -> dict[str, Any]:
    """单只股票估值与财务指标（估值/财务 API 并行请求）"""
    api_symbol = resolve_api_symbol(symbol, name)
    ts_code = symbol_to_ts_code(symbol)
    row: dict[str, Any] = {"ts_code": ts_code}

    def _valuation() -> dict[str, Any]:
        import akshare as ak

        val = ak.stock_value_em(symbol=api_symbol)
        if val is None or val.empty:
            raise ValueError("估值数据为空")
        latest = val.iloc[-1]
        out: dict[str, Any] = {}
        col_pe = find_column(val.columns, "PE(TTM)")
        col_pb = find_column(val.columns, "市净率", "PB")
        col_float = find_column(val.columns, "流通股本")
        date_col = find_column(val.columns, "数据日期", "日期", "trade_date")

        if col_pe:
            out["pe_ttm"] = pd.to_numeric(latest[col_pe], errors="coerce")
        if col_pb:
            out["pb"] = pd.to_numeric(latest[col_pb], errors="coerce")

        hist = val.copy()
        if date_col:
            hist["_dt"] = pd.to_datetime(hist[date_col], errors="coerce")
            cutoff = hist["_dt"].max() - pd.Timedelta(days=365 * 5)
            hist = hist[hist["_dt"] >= cutoff]

        if col_pe and out.get("pe_ttm") is not None:
            pe_hist = pd.to_numeric(hist[col_pe], errors="coerce").dropna()
            pe_hist = pe_hist[pe_hist > 0]
            if len(pe_hist) >= 20:
                out["pe_percentile_5y"] = round(
                    float((pe_hist < out["pe_ttm"]).mean() * 100), 4
                )

        if col_pb and out.get("pb") is not None:
            pb_hist = pd.to_numeric(hist[col_pb], errors="coerce").dropna()
            pb_hist = pb_hist[pb_hist > 0]
            if len(pb_hist) >= 20:
                out["pb_percentile_5y"] = round(
                    float((pb_hist < out["pb"]).mean() * 100), 4
                )

        if col_float:
            float_shares = pd.to_numeric(latest[col_float], errors="coerce")
            if pd.notna(float_shares) and float_shares > 0:
                daily = _fetch_stock_daily(api_symbol)
                if daily is not None and not daily.empty:
                    vol_col = find_column(daily.columns, "volume", "成交量")
                    if vol_col:
                        vol = pd.to_numeric(daily.iloc[-1][vol_col], errors="coerce")
                        if pd.notna(vol):
                            out["turnover_rate"] = round(float(vol) / float_shares * 100, 4)
        return out

    def _financial() -> dict[str, Any]:
        import akshare as ak

        fin = ak.stock_financial_abstract_ths(symbol=api_symbol, indicator="按报告期")
        if fin is None or fin.empty:
            raise ValueError("财务数据为空")
        latest = fin.iloc[-1]
        out: dict[str, Any] = {}
        roe_col = find_column(fin.columns, "净资产收益率")
        yoy_col = find_column(fin.columns, "净利润同比增长率")
        debt_col = find_column(fin.columns, "资产负债率")
        if roe_col:
            raw_roe = str(latest[roe_col]).replace("%", "")
            out["roe"] = pd.to_numeric(raw_roe, errors="coerce")
        if yoy_col and latest[yoy_col] not in (False, None, ""):
            raw_yoy = str(latest[yoy_col]).replace("%", "")
            out["net_profit_yoy"] = pd.to_numeric(raw_yoy, errors="coerce")
        if debt_col:
            raw_debt = str(latest[debt_col]).replace("%", "")
            out["debt_ratio"] = pd.to_numeric(raw_debt, errors="coerce")
        return out

    with ThreadPoolExecutor(max_workers=2) as executor:
        val_future = executor.submit(retry_call, _valuation, 2, 0.3)
        fin_future = executor.submit(retry_call, _financial, 2, 0.3)
        val_data = val_future.result()
        fin_data = fin_future.result()

    if val_data:
        row.update(val_data)
    if fin_data:
        row.update(fin_data)

    return row
