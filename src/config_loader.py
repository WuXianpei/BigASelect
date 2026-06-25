"""配置加载模块"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"


def load_yaml(filename: str) -> dict[str, Any]:
    """加载 YAML 配置文件"""
    path = CONFIG_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def load_settings() -> dict[str, Any]:
    """加载全局设置"""
    return load_yaml("settings.yaml")


def load_screening_rules() -> dict[str, Any]:
    """加载筛选规则"""
    return load_yaml("screening_rules.yaml")


def load_factor_config() -> dict[str, Any]:
    """加载多因子打分模型配置"""
    return load_yaml("factor_config.yaml")


def load_factor_analysis_config() -> dict[str, Any]:
    """加载因子有效性分析配置"""
    return load_yaml("factor_analysis.yaml")


def load_field_schema(filename: str) -> dict[str, Any]:
    """加载输出字段定义"""
    return load_yaml(filename)


def get_output_path(
    settings: dict[str, Any],
    schema: dict[str, Any],
    *,
    run_ts: str | None = None,
) -> Path:
    """根据配置获取输出文件路径；run_ts 为 YYYYMMDD 时追加到文件名（同日覆盖）"""
    output_dir = PROJECT_ROOT / settings.get("output_dir", "output")
    output_dir.mkdir(parents=True, exist_ok=True)
    base_filename = schema["output"]["filename"]
    base_path = Path(base_filename)
    if run_ts:
        filename = f"{base_path.stem}_{run_ts}{base_path.suffix}"
    else:
        filename = base_filename
    return output_dir / filename


def get_field_names(schema: dict[str, Any]) -> list[str]:
    """从字段定义中提取列名列表"""
    return [field["name"] for field in schema.get("fields", [])]


def get_required_fields(schema: dict[str, Any]) -> list[str]:
    """获取必填字段列表"""
    return [
        field["name"]
        for field in schema.get("fields", [])
        if field.get("required", False)
    ]
