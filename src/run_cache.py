"""批跑前清理进程内缓存"""

from __future__ import annotations

from src.fund_flow_provider import clear_money_flow_cache
from src.stock_enricher import clear_enrichment_caches


def clear_run_caches() -> None:
    """每个历史交易日运行前调用，避免沿用上日缓存"""
    clear_enrichment_caches()
    clear_money_flow_cache()
