"""用当前 factor_config 对归档数据重算打分"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.factor_analyzer.archive_loader import load_archived_triplet
from src.stock_scorer import score_stock_pool


def rescore_archived_day(
    trade_date: str,
    *,
    factor_config: dict[str, Any],
    root=None,
) -> pd.DataFrame | None:
    """读取 archive 并用当前配置重算分数"""
    triplet = load_archived_triplet(trade_date, root)
    if triplet is None:
        return None
    pool_df, market_df, sector_df = triplet
    if pool_df.empty:
        return None
    return score_stock_pool(
        pool_df,
        sector_df,
        market_df,
        factor_config=factor_config,
    )
