"""Ridge + Regime 优化器单元测试"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.factor_analyzer.ridge_optimizer import fit_ridge, propose_factor_config_ridge_regime


def _sample_config() -> dict:
    return {
        "market_regime": {
            "weights": {
                "low": {"value": 0.2, "growth": 0.4, "capital": 0.25, "sector": 0.15},
                "neutral": {"value": 0.25, "growth": 0.35, "capital": 0.25, "sector": 0.15},
                "high": {"value": 0.4, "growth": 0.25, "capital": 0.2, "sector": 0.15},
            }
        },
        "final_score": {"regime_output_column": "score_market_regime"},
        "factors": {
            "value": {
                "output_column": "value_score",
                "components": [
                    {"field": "f_a", "weight": 0.6},
                    {"field": "f_b", "weight": 0.4},
                ],
            },
            "growth": {
                "output_column": "growth_score",
                "components": [{"field": "f_c", "weight": 1.0}],
            },
            "capital": {
                "output_column": "capital_score",
                "components": [{"field": "f_d", "weight": 1.0}],
            },
            "sector": {
                "output_column": "sector_score",
                "components": [{"field": "f_e", "weight": 1.0}],
            },
        },
    }


def _build_panel(n_days: int = 30, per_day: int = 40) -> pd.DataFrame:
    rows = []
    for d in range(n_days):
        td = f"202601{d+1:02d}"
        regime = "neutral" if d % 3 else ("low" if d % 2 else "high")
        for i in range(per_day):
            fa = float(d + i * 0.01)
            rows.append(
                {
                    "trade_date": td,
                    "score_market_regime": regime,
                    "f_a": fa,
                    "f_b": fa * 0.5,
                    "f_c": fa * 0.3,
                    "f_d": fa * 0.8,
                    "f_e": fa * 0.2,
                    "value_score": fa * 0.4,
                    "growth_score": fa * 0.3,
                    "capital_score": fa * 0.9,
                    "sector_score": fa * 0.1,
                    "future_return_20": fa * 0.05 + np.random.default_rng(i + d).normal(0, 0.01),
                }
            )
    return pd.DataFrame(rows)


def test_fit_ridge_basic():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 3))
    y = 2 * x[:, 0] - x[:, 1] + rng.normal(scale=0.1, size=200)
    X = np.hstack([np.ones((200, 1)), x])
    beta = fit_ridge(X, y, alpha=1.0)
    assert beta[1] > 0
    assert beta[2] < 0


def test_propose_ridge_regime_returns_config():
    cfg = _sample_config()
    panel = _build_panel(n_days=25, per_day=50)
    proposed_cfg = {
        "ridge": {"alphas": [1.0], "alpha_select": "fixed", "fixed_alpha": 1.0, "standardize": True},
        "regime": {"min_days": 5, "min_rows": 200, "max_category_weight_change": 0.2},
        "min_component_weight": 0.05,
        "max_weight_change_ratio": 0.5,
    }
    out = propose_factor_config_ridge_regime(cfg, panel, proposed_cfg, tune_mode="walk_forward")
    assert out["_proposed_meta"]["optimization_method"] == "ridge_regime"
    w_sum = sum(abs(c["weight"]) for c in out["factors"]["value"]["components"])
    assert abs(w_sum - 1.0) < 0.01
