"""筛选模块：步骤开关与截断逻辑"""

from __future__ import annotations

import pandas as pd

from src.screener import (
    _should_truncate_pool,
    _truncate_pool,
    apply_screening_fundamental_market,
    apply_screening_phase1,
    apply_screening_trend_capital,
    format_target_count_label,
)


def _rules(**overrides) -> dict:
    base = {
        "enabled": True,
        "target_count": None,
        "params": {
            "min_amount": 30_000_000,
            "min_turnover_rate": 0.3,
            "pe_ttm_max": 300,
            "pe_ttm_min": 0,
            "pb_max": 15,
            "net_profit_yoy_min": -80,
            "debt_ratio_max": 80,
        },
        "steps": [
            {"id": "step_1", "enabled": True},
            {"id": "step_2", "enabled": True},
            {"id": "step_3", "enabled": False},
            {"id": "step_4", "enabled": False},
            {"id": "step_5", "enabled": True},
        ],
    }
    base.update(overrides)
    return base


def test_no_truncate_when_target_count_null():
    rules = _rules(target_count=None)
    df = pd.DataFrame({"ts_code": [f"{i:06d}.SZ" for i in range(5)]})
    out = _truncate_pool(df, rules)
    assert len(out) == 5
    assert format_target_count_label(rules) == "不截断"
    assert not _should_truncate_pool(rules)


def test_truncate_when_target_count_positive():
    rules = _rules(target_count=2)
    df = pd.DataFrame({"ts_code": ["A", "B", "C"]})
    out = _truncate_pool(df, rules)
    assert len(out) == 2
    assert format_target_count_label(rules) == "2"


def test_phase1_skips_step2_when_disabled():
    rules = _rules(steps=[
        {"id": "step_1", "enabled": True},
        {"id": "step_2", "enabled": False},
    ])
    df = pd.DataFrame(
        {
            "is_st": [0, 0],
            "is_suspended": [0, 0],
            "risk_flag": [0, 0],
            "amount": [1_000_000, 50_000_000],
            "turnover_rate": [0.1, 1.0],
        }
    )
    out = apply_screening_phase1(df, rules)
    assert len(out) == 2


def test_trend_capital_skips_step3_when_disabled():
    rules = _rules()
    df = pd.DataFrame(
        {
            "ma60": [10.0, 5.0],
            "ma120": [20.0, 10.0],
        }
    )
    out = apply_screening_trend_capital(df, rules)
    assert len(out) == 2


def test_fundamental_skips_step5_when_disabled():
    rules = _rules(steps=[
        {"id": "step_5", "enabled": False},
    ])
    df = pd.DataFrame({"pe_ttm": [9999.0]})
    out = apply_screening_fundamental_market(df, rules)
    assert len(out) == 1
