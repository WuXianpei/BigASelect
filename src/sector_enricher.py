"""行业强度 enrichment（数据丰富）模块"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from .concurrency import resolve_worker_count
from .data_fetcher import get_trade_date
from .data_utils import (
    expand_industry_names,
    find_column,
    normalize_hist_df,
    normalize_industry_name,
    retry_call,
)
from .fund_flow_provider import get_sector_money_flow, lookup_symbol_by_name
from .stock_enricher import _fetch_stock_daily


def fetch_sector_strength(
    settings: dict[str, Any],
    industries_filter: list[str] | None = None,
) -> pd.DataFrame:
    """获取各行业强度指标，每个行业一行"""
    if industries_filter is None:
        industries_filter = settings.get("sector_industries_filter")

    industry_filter_set = (
        expand_industry_names(industries_filter) if industries_filter else None
    )

    summary = _fetch_industry_summary()
    if summary is None or summary.empty:
        return pd.DataFrame()

    industry_col = find_column(summary.columns, "板块", "行业")
    leader_name_col = find_column(summary.columns, "领涨股")
    leader_pct_col = find_column(summary.columns, "领涨股-涨跌幅")
    up_col = find_column(summary.columns, "上涨家数")
    down_col = find_column(summary.columns, "下跌家数")
    if not industry_col:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for _, row in summary.iterrows():
        industry = str(row[industry_col])
        if industry_filter_set and not (
            industry in industry_filter_set
            or normalize_industry_name(industry) in industry_filter_set
        ):
            continue

        record: dict[str, Any] = {
            "industry": industry,
            "sector_money_flow_1d": get_sector_money_flow(industry, 1),
            "sector_money_flow_3d": get_sector_money_flow(industry, 3),
            "sector_money_flow_5d": get_sector_money_flow(industry, 5),
            "sector_money_flow_10d": get_sector_money_flow(industry, 10),
            "sector_momentum_score": None,
            "sector_strength_rank": None,
            "leader_stock_strength": None,
            "leader_stock_return_20d": None,
            "sector_high_stock_ratio": _calc_high_stock_ratio(row, up_col, down_col),
            "sector_new_high_ratio": None,
            "_pct_chg_5d": None,
            "_leader_name": str(row[leader_name_col]) if leader_name_col else None,
        }
        if leader_pct_col:
            record["leader_stock_strength"] = pd.to_numeric(
                row[leader_pct_col], errors="coerce"
            )
        rows.append(record)

    if not rows:
        return pd.DataFrame()

    max_workers = resolve_worker_count(
        settings, "sector_enrich_workers", default=8, item_count=len(rows)
    )
    quote_date = get_trade_date(settings)
    industries = [r["industry"] for r in rows]

    print(f"      计算行业指数指标与领涨股20日收益（共 {len(rows)} 个行业，并发 {max_workers}）...")
    with ThreadPoolExecutor(max_workers=2) as executor:
        index_future = executor.submit(
            _fetch_index_metrics_batch,
            industries,
            max_workers=max_workers,
            quote_date=quote_date,
        )
        leader_future = executor.submit(
            _fetch_leader_returns_batch,
            rows,
            max_workers=max_workers,
        )
        index_metrics = index_future.result()
        leader_returns = leader_future.result()

    for record in rows:
        industry = record["industry"]
        metrics = index_metrics.get(industry, {})
        record["sector_new_high_ratio"] = metrics.get("sector_new_high_ratio")
        record["_pct_chg_5d"] = metrics.get("pct_chg_5d")
        record["leader_stock_return_20d"] = leader_returns.get(industry)
        record["sector_momentum_score"] = _calc_momentum_score(record)

    df = pd.DataFrame(rows)
    df = df.drop(columns=["_pct_chg_5d", "_leader_name"], errors="ignore")
    df = _assign_strength_rank(df)
    return df


def _calc_high_stock_ratio(
    row: pd.Series,
    up_col: str | None,
    down_col: str | None,
) -> float | None:
    """行业内上涨个股占比"""
    if not up_col or not down_col:
        return None
    up = pd.to_numeric(row.get(up_col), errors="coerce")
    down = pd.to_numeric(row.get(down_col), errors="coerce")
    if pd.isna(up) or pd.isna(down):
        return None
    total = float(up) + float(down)
    if total <= 0:
        return None
    return round(float(up) / total, 4)


def _calc_momentum_score(record: dict[str, Any]) -> float | None:
    """板块动量评分（0-100）"""
    score = 50.0
    has_signal = False

    pct5 = record.get("_pct_chg_5d")
    if pct5 is not None and not pd.isna(pct5):
        score += float(np.clip(pct5 * 3, -20, 20))
        has_signal = True

    flow5 = record.get("sector_money_flow_5d")
    if flow5 is not None and not pd.isna(flow5):
        score += float(np.clip(flow5 / 5, -15, 15))
        has_signal = True

    nhr = record.get("sector_new_high_ratio")
    if nhr is not None and not pd.isna(nhr):
        score += float((nhr - 0.5) * 30)
        has_signal = True

    leader = record.get("leader_stock_strength")
    if leader is not None and not pd.isna(leader):
        score += float(np.clip(leader / 2, -10, 10))
        has_signal = True

    high_ratio = record.get("sector_high_stock_ratio")
    if high_ratio is not None and not pd.isna(high_ratio):
        score += float((high_ratio - 0.5) * 20)
        has_signal = True

    if not has_signal:
        return None
    return round(float(np.clip(score, 0, 100)), 2)


def _assign_strength_rank(df: pd.DataFrame) -> pd.DataFrame:
    """按动量评分降序排名（1=最强）"""
    if df.empty or "sector_momentum_score" not in df.columns:
        return df
    result = df.copy()
    result["sector_strength_rank"] = (
        result["sector_momentum_score"]
        .rank(method="min", ascending=False, na_option="bottom")
        .astype("Int64")
    )
    return result.sort_values(
        "sector_strength_rank", ascending=True, na_position="last"
    ).reset_index(drop=True)


def _fetch_industry_summary() -> pd.DataFrame | None:
    """同花顺行业板块汇总"""
    import akshare as ak

    def _fetch() -> pd.DataFrame:
        df = ak.stock_board_industry_summary_ths()
        if df is None or df.empty:
            raise ValueError("行业汇总为空")
        return df

    return retry_call(_fetch, label="行业汇总")


def _fetch_index_metrics_batch(
    industries: list[str],
    max_workers: int = 6,
    quote_date: str | None = None,
) -> dict[str, dict[str, float | None]]:
    """并发计算各行业指数指标"""
    result: dict[str, dict[str, float | None]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_calc_industry_index_metrics, name, quote_date): name
            for name in industries
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                result[name] = future.result()
            except Exception:
                result[name] = {}
    return result


def _fetch_leader_returns_batch(
    rows: list[dict[str, Any]],
    max_workers: int = 6,
) -> dict[str, float | None]:
    """并发计算领涨股20日收益率"""
    result: dict[str, float | None] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _fetch_leader_return_20d,
                row.get("_leader_name"),
                row["industry"],
            ): row["industry"]
            for row in rows
            if row.get("_leader_name")
        }
        for future in as_completed(futures):
            industry = futures[future]
            try:
                result[industry] = future.result()
            except Exception:
                result[industry] = None
    return result


def _calc_industry_index_metrics(
    industry: str,
    quote_date: str | None = None,
) -> dict[str, float | None]:
    """计算单行业指数：5日涨跌幅、20日新高比例"""
    import akshare as ak

    candidates = [industry, normalize_industry_name(industry)]
    candidates = [c for c in candidates if c]

    end_date = quote_date or datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=90)).strftime("%Y%m%d")

    hist = None
    for name in candidates:

        def _fetch(sym: str = name) -> pd.DataFrame:
            df = ak.stock_board_industry_index_ths(
                symbol=sym,
                start_date=start_date,
                end_date=end_date,
            )
            if df is None or df.empty:
                raise ValueError(f"{sym} 指数无数据")
            return df

        hist = retry_call(_fetch, retries=2, delay=0.5)
        if hist is not None:
            break

    if hist is None:
        return {}

    close = normalize_hist_df(hist)
    if close is None or len(close) < 6:
        return {}

    latest = float(close.iloc[-1])
    metrics: dict[str, float | None] = {}

    if close.iloc[-6] != 0:
        metrics["pct_chg_5d"] = round((latest / float(close.iloc[-6]) - 1) * 100, 4)

    if len(close) >= 20:
        high_20 = float(close.tail(20).max())
        if high_20 > 0:
            metrics["sector_new_high_ratio"] = round(latest / high_20, 4)

    return metrics


def _fetch_leader_return_20d(
    leader_name: str | None,
    industry: str,
) -> float | None:
    """领涨股近20日收益率（%）"""
    if not leader_name:
        return None

    code = _resolve_leader_code(leader_name, industry)
    if not code:
        return None

    daily = retry_call(
        lambda: _fetch_stock_daily(code),
        retries=2,
        delay=1.0,
        label=f"领涨股日K {leader_name}",
        silent=True,
    )
    if daily is None or daily.empty:
        return None

    close = normalize_hist_df(daily)
    if close is None or len(close) < 21:
        return None

    prev = float(close.iloc[-21])
    if prev == 0:
        return None
    return round((float(close.iloc[-1]) / prev - 1) * 100, 4)


def _resolve_leader_code(leader_name: str, industry: str) -> str | None:
    """由领涨股名称解析代码（优先同花顺，备用东财成分股）"""
    code = lookup_symbol_by_name(leader_name)
    if code:
        return code

    import akshare as ak

    candidates = [industry, normalize_industry_name(industry)]
    candidates = [c for c in candidates if c]

    for name in candidates:

        def _fetch(sym: str = name) -> str:
            cons = ak.stock_board_industry_cons_em(symbol=sym)
            if cons is None or cons.empty:
                raise ValueError(f"{sym} 成分股为空")
            name_col = find_column(cons.columns, "名称")
            code_col = find_column(cons.columns, "代码")
            if not name_col or not code_col:
                raise ValueError("成分股缺少名称/代码列")
            match = cons[cons[name_col].astype(str) == leader_name]
            if match.empty:
                raise ValueError(f"未找到领涨股 {leader_name}")
            return str(match.iloc[0][code_col]).zfill(6)

        result = retry_call(_fetch, retries=2, delay=0.5, label=f"领涨股{leader_name}")
        if result:
            return result
    return None
