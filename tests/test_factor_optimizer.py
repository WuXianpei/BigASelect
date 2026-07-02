"""因子优化 A+C 方案单元测试"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from src.factor_analyzer.metrics import (
    compute_component_ic_stats,
    split_walk_forward_dates,
)
from src.factor_analyzer.optimizer import propose_factor_config


def _sample_factor_config() -> dict:
    return {
        "factors": {
            "value": {
                "components": [
                    {"field": "good_field", "weight": 0.6},
                    {"field": "bad_field", "weight": 0.4},
                ]
            },
        }
    }


def test_split_walk_forward_fixed():
    dates = [f"202501{i:02d}" for i in range(1, 32)] + [f"202502{i:02d}" for i in range(1, 29)]
    dates += [f"202503{i:02d}" for i in range(1, 32)] + [f"202504{i:02d}" for i in range(1, 11)]
    assert len(dates) >= 100
    split = split_walk_forward_dates(dates, train_days=60, validate_days=20, test_days=20, min_total_days=80)
    assert split["enabled"] is True
    assert len(split["train_dates"]) == 60
    assert len(split["validate_dates"]) == 20
    assert len(split["test_dates"]) == 20


def test_split_walk_forward_insufficient():
    dates = [f"202601{i:02d}" for i in range(1, 11)]
    split = split_walk_forward_dates(dates, min_total_days=80)
    assert split["enabled"] is False


def test_time_decay_ic_stats():
    rows = []
    for d in range(30):
        td = f"202601{d+1:02d}"
        for j in range(20):
            rows.append(
                {
                    "trade_date": td,
                    "strong_field": float(d + j * 0.01),
                    "weak_field": float(30 - d - j * 0.01),
                    "future_return_20": float(d + j * 0.02),
                }
            )
    panel = pd.DataFrame(rows)
    cfg = {
        "factors": {
            "x": {
                "components": [
                    {"field": "strong_field", "weight": 1.0},
                    {"field": "weak_field", "weight": 1.0},
                ]
            }
        }
    }
    stats = compute_component_ic_stats(
        panel,
        cfg,
        time_decay={"enabled": True, "recent_days": 5, "recent_weight": 2.0, "older_weight": 0.5},
    )
    assert stats["strong_field"]["ic_mean"] > stats["weak_field"]["ic_mean"]
    assert stats["strong_field"]["ic_days"] == 30


def test_propose_weight_cap_and_negative_ic():
    cfg = _sample_factor_config()
    stats = {
        "good_field": {"ic_ir": 1.5, "ic_mean": 0.2},
        "bad_field": {"ic_ir": -0.8, "ic_mean": -0.1},
    }
    proposed_cfg = {
        "use_ic_ir": True,
        "ic_boost_factor": 3.0,
        "max_weight_change_ratio": 0.30,
        "min_component_weight": 0.05,
        "negative_signal_threshold": 0.0,
        "opposite_direction_penalty": 0.5,
        "signal_clip": 2.0,
    }
    orig = copy.deepcopy(cfg)
    new = propose_factor_config(cfg, stats, proposed_cfg, tune_mode="walk_forward")
    good_orig = orig["factors"]["value"]["components"][0]["weight"]
    good_new = new["factors"]["value"]["components"][0]["weight"]
    bad_new = new["factors"]["value"]["components"][1]["weight"]
    assert abs(good_new) <= abs(good_orig) * 1.31 + 1e-6
    assert abs(bad_new) < abs(good_new)


def test_propose_preserves_sign():
    cfg = {
        "factors": {
            "value": {
                "components": [{"field": "pe_pct", "weight": -0.5}],
            }
        }
    }
    stats = {"pe_pct": {"ic_ir": -0.5, "ic_mean": -0.2}}
    new = propose_factor_config(
        cfg,
        stats,
        {"use_ic_ir": True, "ic_boost_factor": 2.0, "max_weight_change_ratio": 0.5, "min_component_weight": 0.05},
        tune_mode="soft_tune",
    )
    assert new["factors"]["value"]["components"][0]["weight"] < 0
