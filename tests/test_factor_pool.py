"""因子池增删筛选单元测试"""

from __future__ import annotations

import pandas as pd

from src.factor_analyzer.factor_pool import (
    apply_factor_pool_changes,
    screen_removal_candidates,
)


def _cfg() -> dict:
    return {
        "factors": {
            "value": {
                "components": [
                    {"field": "good", "weight": 0.6},
                    {"field": "bad", "weight": 0.4},
                ]
            },
            "growth": {
                "components": [{"field": "solo", "weight": 1.0}],
            },
        }
    }


def test_screen_removal_negative_ic_ir():
    stats = {
        "good": {"ic_ir": 0.5, "ic_mean": 0.1, "ic_days": 20},
        "bad": {"ic_ir": -0.3, "ic_mean": -0.05, "ic_days": 20},
        "solo": {"ic_ir": -0.5, "ic_mean": -0.1, "ic_days": 20},
    }
    pool_cfg = {
        "enabled": True,
        "removal": {"enabled": True, "ic_ir_max": 0.0, "min_ic_days": 15, "max_per_run": 2},
    }
    rem = screen_removal_candidates(stats, pool_cfg, _cfg())
    fields = {r["field"] for r in rem}
    assert "bad" in fields
    assert "solo" not in fields  # 每类至少保留 1 个


def test_apply_removal_renormalizes():
    cfg = _cfg()
    rem = [{"action": "remove", "field": "bad", "factor": "value", "ic_ir": -0.2}]
    new_cfg, changes = apply_factor_pool_changes(
        cfg, removals=rem, additions=[], add_cfg={}
    )
    assert len(changes) == 1
    comps = new_cfg["factors"]["value"]["components"]
    assert len(comps) == 1
    assert comps[0]["field"] == "good"
    assert abs(comps[0]["weight"]) == 1.0
