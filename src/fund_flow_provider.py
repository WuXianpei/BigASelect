"""资金流数据源：个股 / 行业 / 市场汇总均来自同花顺（THS）排行"""

from __future__ import annotations

import pandas as pd

from .data_utils import (
    find_column,
    normalize_industry_name,
    parse_chinese_amount,
    retry_call,
)

# 统一数据源标识（个股主力净流入、行业资金流、市场汇总均来自同花顺）
MONEY_FLOW_SOURCE = "同花顺"

# 同花顺个股排行周期 → 内部 key
_INDIVIDUAL_PERIOD_MAP: dict[int, str] = {
    1: "即时",
    3: "3日排行",
    5: "5日排行",
    10: "10日排行",
}

# 个股排行缓存：周期名 → {6位代码: 净流入(元)}
_INDIVIDUAL_CACHE: dict[str, dict[str, float]] = {}

# 行业排行缓存：周期 → {行业名: 净流入(亿元)}
_INDUSTRY_CACHE: dict[str, dict[str, float]] = {}

# 北向个股当日净流入缓存（沪股通/深股通，数据源：东财）
_NORTHBOUND_CACHE: dict[str, float] = {}
# 已知无北向数据或非沪股通标的（跳过逐只请求）
_NORTHBOUND_SKIP: set[str] = set()
# 接口整体不可用（探测失败后不再请求）
_NORTHBOUND_API_AVAILABLE: bool | None = None

# 股票简称 → 6位代码（来自同花顺即时排行）
_NAME_TO_CODE_CACHE: dict[str, str] = {}


def prefetch_all_money_flow(settings: dict | None = None) -> None:
    """预拉全部资金流数据（同花顺个股 + 行业，各周期并发拉取）"""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    settings = settings or {}
    print(f"      预拉资金流（统一数据源: {MONEY_FLOW_SOURCE}）...")
    periods = ("即时", "3日排行", "5日排行", "10日排行")
    tasks: list[tuple[str, str]] = [
        *((("individual", p) for p in periods)),
        *((("industry", p) for p in periods)),
    ]

    from .concurrency import resolve_worker_count

    max_workers = resolve_worker_count(settings, "prefetch_workers", default=4, item_count=len(tasks))

    def _run(task: tuple[str, str]) -> None:
        kind, period = task
        if kind == "individual":
            prefetch_individual_flow(period)
        else:
            prefetch_industry_flow(period)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_run, task) for task in tasks]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception:
                continue


def prefetch_industry_money_flow() -> None:
    """预拉行业资金流（market_context / sector_strength 专用，跳过个股排行）"""
    print(f"      预拉行业资金流（统一数据源: {MONEY_FLOW_SOURCE}）...")
    for period_name in ("即时", "3日排行", "5日排行", "10日排行"):
        prefetch_industry_flow(period_name)


def prefetch_individual_flow(period: str = "5日排行") -> dict[str, float]:
    """预拉个股资金流排行，返回 代码→净流入(元)"""
    if period in _INDIVIDUAL_CACHE:
        return _INDIVIDUAL_CACHE[period]

    import akshare as ak

    def _fetch() -> dict[str, float]:
        df = ak.stock_fund_flow_individual(symbol=period)
        if df is None or df.empty:
            raise ValueError(f"个股{period}资金流为空")

        code_col = find_column(df.columns, "股票代码", "代码")
        flow_col = find_column(
            df.columns,
            "资金流入净额",
            "净额",
            "主力净流入",
        )
        if not code_col or not flow_col:
            raise ValueError(f"个股{period}资金流缺少必要列")

        result: dict[str, float] = {}
        for _, row in df.iterrows():
            code = str(row[code_col]).zfill(6)
            val = parse_chinese_amount(row[flow_col])
            if val is not None:
                result[code] = val
            if period == "即时":
                name_col = find_column(df.columns, "股票简称", "名称")
                if name_col:
                    _NAME_TO_CODE_CACHE[str(row[name_col]).strip()] = code
        return result

    data = retry_call(
        _fetch,
        retries=3,
        delay=1.5,
        label=f"{MONEY_FLOW_SOURCE}个股{period}",
    )
    _INDIVIDUAL_CACHE[period] = data if data is not None else {}
    return _INDIVIDUAL_CACHE[period]


def get_main_net_inflow(
    symbol: str,
    days: int,
    api_symbol: str | None = None,
    *,
    daily: pd.DataFrame | None = None,
) -> float | None:
    """获取个股近 N 日主力净流入（元），仅来自同花顺排行"""
    _ = daily  # 保留参数以兼容旧调用，不再用日K推算
    period_name = _INDIVIDUAL_PERIOD_MAP.get(days)
    if not period_name:
        return None

    mapping = prefetch_individual_flow(period_name)
    for code in (api_symbol, symbol):
        if code:
            val = mapping.get(str(code).zfill(6))
            if val is not None:
                return val
    return None


def get_money_flow_5d(
    symbol: str,
    api_symbol: str | None = None,
    *,
    daily: pd.DataFrame | None = None,
) -> float | None:
    """兼容旧调用：近5日主力净流入"""
    return get_main_net_inflow(symbol, 5, api_symbol, daily=daily)


def get_market_money_flow_index() -> float | None:
    """近5日全市场主力净流入合计（亿元），由同花顺行业5日排行汇总"""
    mapping = prefetch_industry_flow("5日排行")
    if not mapping:
        return None
    return round(sum(mapping.values()), 4)


def get_industry_money_flow(period: str, industry: str) -> float | None:
    """获取行业资金流（亿元），支持行业别名匹配"""
    mapping = prefetch_industry_flow(period)
    if not mapping:
        return None

    for name in _industry_lookup_candidates(industry):
        if name in mapping:
            return mapping[name]

    candidates = _industry_lookup_candidates(industry)
    if candidates:
        base = candidates[0]
        for key, val in mapping.items():
            if key in base or base in key:
                return val
    return None


def get_sector_money_flow(industry: str, days: int) -> float | None:
    """获取行业近 N 日主力净流入（亿元）"""
    period = _INDIVIDUAL_PERIOD_MAP.get(days)
    if not period:
        return None
    return get_industry_money_flow(period, industry)


def get_northbound_flow(symbol: str, api_symbol: str | None = None) -> float | None:
    """北向资金当日净流入（元），仅沪深港通标的；非标的或失败静默跳过"""
    global _NORTHBOUND_API_AVAILABLE
    from .bj_resolver import is_bj_symbol

    if _NORTHBOUND_API_AVAILABLE is False:
        return None

    if is_bj_symbol(symbol) or (api_symbol and is_bj_symbol(api_symbol)):
        return None

    code = str(api_symbol or symbol).zfill(6)
    if code in _NORTHBOUND_SKIP:
        return None
    if code in _NORTHBOUND_CACHE:
        return _NORTHBOUND_CACHE[code]

    import akshare as ak

    try:
        df = ak.stock_hsgt_individual_em(symbol=code)
    except Exception:
        _NORTHBOUND_SKIP.add(code)
        return None

    if df is None or df.empty:
        _NORTHBOUND_SKIP.add(code)
        return None

    flow_col = find_column(df.columns, "当日增持资金", "增持资金", "今日增持资金")
    if not flow_col:
        _NORTHBOUND_SKIP.add(code)
        return None

    val = pd.to_numeric(df.iloc[-1][flow_col], errors="coerce")
    if pd.isna(val):
        _NORTHBOUND_SKIP.add(code)
        return None

    result = round(float(val), 2)
    _NORTHBOUND_CACHE[code] = result
    _NORTHBOUND_API_AVAILABLE = True
    return result


def probe_northbound_api() -> bool:
    """探测北向个股接口是否可用（用一只常见沪股通标的试一次）"""
    global _NORTHBOUND_API_AVAILABLE
    if _NORTHBOUND_API_AVAILABLE is not None:
        return _NORTHBOUND_API_AVAILABLE

    val = get_northbound_flow("600519", "600519")
    if val is not None or "600519" in _NORTHBOUND_CACHE:
        _NORTHBOUND_API_AVAILABLE = True
        return True

    # 600519 在 skip 中说明接口返回空/异常，视为不可用
    _NORTHBOUND_API_AVAILABLE = False
    return False


def attach_northbound_flows(
    df: pd.DataFrame,
    settings: dict | None = None,
) -> pd.DataFrame:
    """为筛选后股票池附加 northbound_flow（失败静默跳过，汇总一行日志）"""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if df.empty or "ts_code" not in df.columns:
        return df

    settings = settings or {}
    if not settings.get("fetch_northbound_flow", True):
        return df

    if not probe_northbound_api():
        print("      北向个股接口不可用，跳过 northbound_flow 字段")
        return df

    result = df.copy()
    codes = [str(c).split(".")[0] for c in result["ts_code"]]
    flow_map: dict[str, float | None] = {}
    from .concurrency import resolve_worker_count

    max_workers = resolve_worker_count(settings, "northbound_workers", default=12, item_count=len(codes))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(get_northbound_flow, code, code): code for code in codes
        }
        for future in as_completed(futures):
            code = futures[future]
            try:
                flow_map[code] = future.result()
            except Exception:
                flow_map[code] = None

    flows = [flow_map.get(code) for code in codes]
    ok = sum(1 for v in flows if v is not None)
    skipped = len(flows) - ok
    result["northbound_flow"] = flows
    print(
        f"      北向个股 northbound_flow: 成功 {ok}，跳过 {skipped} "
        f"（非沪股通或暂无数据）"
    )
    return result


def prefetch_industry_flow(period: str) -> dict[str, float]:
    """预拉行业资金流排行，返回 行业→净流入(亿元)"""
    if period in _INDUSTRY_CACHE:
        return _INDUSTRY_CACHE[period]

    import akshare as ak

    def _fetch() -> dict[str, float]:
        df = ak.stock_fund_flow_industry(symbol=period)
        if df is None or df.empty:
            raise ValueError(f"行业{period}资金流为空")

        industry_col = find_column(df.columns, "行业")
        flow_col = find_column(df.columns, "净额", "资金流入净额")
        if not industry_col or not flow_col:
            raise ValueError(f"行业{period}资金流缺少必要列")

        result: dict[str, float] = {}
        for _, row in df.iterrows():
            name = str(row[industry_col])
            val = _parse_industry_flow_yi(row[flow_col])
            if val is not None:
                result[name] = val
        return result

    data = retry_call(
        _fetch,
        retries=3,
        delay=1.5,
        label=f"{MONEY_FLOW_SOURCE}行业{period}",
    )
    _INDUSTRY_CACHE[period] = data if data is not None else {}
    return _INDUSTRY_CACHE[period]


def lookup_symbol_by_name(name: str) -> str | None:
    """由股票简称查 6 位代码（同花顺即时排行缓存）"""
    if not name:
        return None
    if not _NAME_TO_CODE_CACHE:
        prefetch_individual_flow("即时")
    return _NAME_TO_CODE_CACHE.get(str(name).strip())


def clear_money_flow_cache() -> None:
    """清空缓存（测试用）"""
    global _NORTHBOUND_API_AVAILABLE
    _INDIVIDUAL_CACHE.clear()
    _INDUSTRY_CACHE.clear()
    _NORTHBOUND_CACHE.clear()
    _NORTHBOUND_SKIP.clear()
    _NORTHBOUND_API_AVAILABLE = None
    _NAME_TO_CODE_CACHE.clear()


def _industry_lookup_candidates(industry: str) -> list[str]:
    """生成用于同花顺行业匹配的名称候选列表"""
    candidates: list[str] = []
    text = str(industry).strip()
    if not text:
        return candidates

    normalized = normalize_industry_name(text)
    for name in (text, normalized):
        if name and name not in candidates:
            candidates.append(name)

    for name in list(candidates):
        for suffix in ("Ⅲ", "Ⅱ"):
            if name.endswith(suffix):
                short = name[: -len(suffix)]
                if short and short not in candidates:
                    candidates.append(short)
        if name.endswith("业") and len(name) > 2:
            short = name[:-1]
            if short not in candidates:
                candidates.append(short)

    return candidates


def _parse_industry_flow_yi(value: object) -> float | None:
    """解析同花顺行业净额为亿元（列值为纯数字时已是亿元）"""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None

    text = str(value).strip()
    if not text or text in ("--", "-"):
        return None

    if text.endswith(("亿", "万")):
        yuan = parse_chinese_amount(text)
        if yuan is None:
            return None
        return round(yuan / 1e8, 4)

    num = pd.to_numeric(text, errors="coerce")
    return round(float(num), 4) if pd.notna(num) else None
