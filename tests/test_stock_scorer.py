"""股票池打分模型单元测试（使用 output/ 下最新 CSV，无需重新生成数据）"""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.stock_scorer import (  # noqa: E402
    compute_ma_trend_score,
    compute_price_structure_score,
    resolve_market_weights,
    score_stock_pool,
    summarize_scores,
)

OUTPUT_DIR = PROJECT_ROOT / "output"
_TS_PATTERN = re.compile(r"stock_pool_(\d{8}(?:_\d{6})?)\.csv$")


def _latest_triplet() -> tuple[Path, Path, Path]:
    """取 output/ 下日期后缀一致且最新的三份 CSV"""
    groups: dict[str, dict[str, Path]] = {}
    for path in OUTPUT_DIR.glob("*.csv"):
        if path.name.startswith("stock_pool_"):
            match = _TS_PATTERN.match(path.name)
            if not match:
                continue
            ts = match.group(1)
            groups.setdefault(ts, {})["pool"] = path
        elif path.name.startswith("market_context_"):
            suffix = path.stem.removeprefix("market_context_")
            groups.setdefault(suffix, {})["market"] = path
        elif path.name.startswith("sector_strength_"):
            suffix = path.stem.removeprefix("sector_strength_")
            groups.setdefault(suffix, {})["sector"] = path

    complete = [(ts, files) for ts, files in groups.items() if len(files) == 3]
    if not complete:
        raise FileNotFoundError(
            "output/ 下未找到完整的三份 CSV（stock_pool/market_context/sector_strength 同一时间戳）"
        )
    complete.sort(key=lambda item: item[0], reverse=True)
    _ts, files = complete[0]
    return files["pool"], files["market"], files["sector"]


class TestStockScorer(unittest.TestCase):
    """基于最新输出数据的打分测试"""

    @classmethod
    def setUpClass(cls) -> None:
        pool_path, market_path, sector_path = _latest_triplet()
        cls.pool_path = pool_path
        cls.market_path = market_path
        cls.sector_path = sector_path
        cls.pool_df = pd.read_csv(pool_path)
        cls.market_df = pd.read_csv(market_path)
        cls.sector_df = pd.read_csv(sector_path)
        cls.scored_df = score_stock_pool(cls.pool_df, cls.sector_df, cls.market_df)

    def test_data_files_loaded(self) -> None:
        self.assertGreater(len(self.pool_df), 0, "股票池应非空")
        self.assertGreater(len(self.market_df), 0, "市场环境应非空")
        self.assertGreater(len(self.sector_df), 0, "行业强度应非空")

    def test_score_columns_present(self) -> None:
        required = [
            "value_score",
            "growth_score",
            "capital_score",
            "sector_score",
            "final_score",
            "score_market_regime",
            "score_weight_value",
            "score_weight_growth",
            "score_weight_capital",
            "score_weight_sector",
            "future_return_10",
            "future_return_30",
            "future_return_60",
        ]
        for col in required:
            self.assertIn(col, self.scored_df.columns, f"缺少列 {col}")

    def test_final_score_computed_for_all(self) -> None:
        self.assertEqual(
            self.scored_df["final_score"].notna().sum(),
            len(self.scored_df),
            "每只股票都应有 final_score",
        )

    def test_market_weights_sum_to_one(self) -> None:
        weight_sum = (
            self.scored_df["score_weight_value"].iloc[0]
            + self.scored_df["score_weight_growth"].iloc[0]
            + self.scored_df["score_weight_capital"].iloc[0]
            + self.scored_df["score_weight_sector"].iloc[0]
        )
        self.assertAlmostEqual(float(weight_sum), 1.0, places=4)

    def test_final_score_matches_weighted_formula(self) -> None:
        row = self.scored_df.iloc[0]
        expected = (
            row["value_score"] * row["score_weight_value"]
            + row["growth_score"] * row["score_weight_growth"]
            + row["capital_score"] * row["score_weight_capital"]
            + row["sector_score"] * row["score_weight_sector"]
        )
        self.assertAlmostEqual(float(row["final_score"]), float(expected), places=3)

    def test_regime_matches_market_risk_index(self) -> None:
        primary = self.market_df[self.market_df["index_code"].astype(str).str.zfill(6) == "000001"]
        risk = float(primary.iloc[0]["market_risk_index"])
        regime, weights = resolve_market_weights(risk)
        self.assertEqual(self.scored_df["score_market_regime"].iloc[0], regime)
        self.assertAlmostEqual(
            float(self.scored_df["score_weight_value"].iloc[0]),
            weights["value"],
        )

    def test_summarize_scores(self) -> None:
        summary = summarize_scores(self.scored_df)
        self.assertEqual(summary["count"], len(self.scored_df))
        self.assertIn("top5", summary)
        self.assertEqual(len(summary["top5"]), 5)

    def test_derived_ma_trend_score_range(self) -> None:
        row = self.pool_df.iloc[0]
        score = compute_ma_trend_score(row)
        if score is not None:
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 100.0)

    def test_derived_price_structure_score_range(self) -> None:
        row = self.pool_df.iloc[0]
        score = compute_price_structure_score(row)
        if score is not None:
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 100.0)

    def test_final_score_ranking_differs(self) -> None:
        unique_scores = self.scored_df["final_score"].nunique()
        self.assertGreater(unique_scores, 1, "final_score 应存在差异")


class TestMarketWeights(unittest.TestCase):
    def test_low_neutral_high_regimes(self) -> None:
        self.assertEqual(resolve_market_weights(30)[0], "low")
        self.assertEqual(resolve_market_weights(50)[0], "neutral")
        self.assertEqual(resolve_market_weights(80)[0], "high")


if __name__ == "__main__":
    unittest.main(verbosity=2)
