"""北交所代码解析：旧代码 83xxxx → 新代码 92xxxx"""

from __future__ import annotations

from typing import Any

import pandas as pd

from .data_utils import retry_call

# 已知旧→新代码映射（2024 年北交所代码切换）
_LEGACY_TO_NEW: dict[str, str] = {
    "832982": "920982",
}

_BJ_INFO_CACHE: dict[str, Any] | None = None


def _load_bj_info() -> dict[str, Any]:
    """加载北交所证券列表（代码、简称、行业）"""
    global _BJ_INFO_CACHE
    if _BJ_INFO_CACHE is not None:
        return _BJ_INFO_CACHE

    import akshare as ak

    def _fetch() -> dict[str, Any]:
        df = ak.stock_info_bj_name_code()
        if df is None or df.empty:
            raise ValueError("北交所证券列表为空")

        code_col = None
        name_col = None
        industry_col = None
        for col in df.columns:
            cs = str(col)
            if "代码" in cs and code_col is None:
                code_col = col
            elif "简称" in cs or "名称" in cs:
                name_col = col
            elif "行业" in cs:
                industry_col = col

        by_code: dict[str, dict[str, str]] = {}
        by_name: dict[str, str] = {}
        industry_map: dict[str, str] = {}

        for _, row in df.iterrows():
            code = str(row[code_col]).zfill(6) if code_col else ""
            if not code:
                continue
            name = str(row[name_col]) if name_col else ""
            industry = str(row[industry_col]) if industry_col and pd.notna(row[industry_col]) else ""

            by_code[code] = {"name": name, "industry": industry}
            if name:
                by_name[name] = code
            if industry:
                industry_map[code] = industry

        return {"by_code": by_code, "by_name": by_name, "industry_map": industry_map}

    result = retry_call(_fetch, retries=2, label="北交所证券列表")
    _BJ_INFO_CACHE = result if result is not None else {
        "by_code": {},
        "by_name": {},
        "industry_map": {},
    }
    return _BJ_INFO_CACHE


def is_bj_symbol(symbol: str) -> bool:
    """是否为北交所股票代码"""
    s = str(symbol).zfill(6)
    return s.startswith(("4", "8", "92"))


def resolve_api_symbol(symbol: str, name: str | None = None) -> str:
    """
    解析用于 API 请求的 6 位代码。
    北交所旧代码 83xxxx 会映射到 92xxxx。
    """
    s = str(symbol).zfill(6)
    if not is_bj_symbol(s):
        return s

    if s.startswith("92"):
        return s

    if s in _LEGACY_TO_NEW:
        return _LEGACY_TO_NEW[s]

    if name:
        info = _load_bj_info()
        resolved = info["by_name"].get(name)
        if resolved:
            return resolved

    return s


def get_bj_industry_map() -> dict[str, str]:
    """北交所 代码→行业 映射（使用新 92 代码）"""
    info = _load_bj_info()
    return dict(info.get("industry_map", {}))


def merge_bj_industry_to_map(mapping: dict[str, str]) -> dict[str, str]:
    """将北交所行业映射合并入主映射，含旧代码别名"""
    bj_map = get_bj_industry_map()
    if bj_map:
        mapping.update(bj_map)
        for legacy, new_code in _LEGACY_TO_NEW.items():
            if new_code in bj_map:
                mapping[legacy] = bj_map[new_code]
    return mapping


def get_bj_industry(symbol: str, name: str | None = None) -> str | None:
    """获取北交所股票所属行业"""
    api_sym = resolve_api_symbol(symbol, name)
    info = _load_bj_info()
    row = info["by_code"].get(api_sym)
    if row and row.get("industry"):
        return row["industry"]
    return info.get("industry_map", {}).get(api_sym)
