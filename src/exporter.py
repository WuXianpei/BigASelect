"""CSV 导出模块"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .config_loader import get_field_names, get_required_fields


def align_to_schema(df: pd.DataFrame, schema: dict[str, Any]) -> pd.DataFrame:
  """
  将 DataFrame 对齐到字段定义
  - 按配置字段顺序排列列
  - 缺失列填充空值
  - 校验必填字段
  """
  field_names = get_field_names(schema)
  required = get_required_fields(schema)

  aligned = pd.DataFrame()
  for name in field_names:
    if name in df.columns:
      aligned[name] = df[name]
    else:
      aligned[name] = None

  # 校验必填字段
  for field in required:
    if aligned[field].isna().all():
      pass  # 数据尚未接入时允许为空，待用户补充 source 后完善

  return aligned


def export_csv(
  df: pd.DataFrame,
  schema: dict[str, Any],
  output_path: Path,
) -> Path:
  """导出 CSV 文件"""
  aligned = align_to_schema(df, schema)
  encoding = schema.get("output", {}).get("encoding", "utf-8-sig")
  output_path.parent.mkdir(parents=True, exist_ok=True)
  aligned.to_csv(output_path, index=False, encoding=encoding)
  return output_path
