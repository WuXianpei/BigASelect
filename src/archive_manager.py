"""历史归档目录约定与批跑状态管理"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from src.config_loader import PROJECT_ROOT, load_yaml

_MARKET_TZ = ZoneInfo("Asia/Shanghai")

# 归档子目录名（相对 archive root）
STOCK_POOL_DIR = "stock_pool"
MARKET_CONTEXT_DIR = "market_context"
SECTOR_STRENGTH_DIR = "sector_strength"
META_DIR = "meta"
INDEX_FILENAME = "backfill_index.json"

FILE_KEYS = ("stock_pool", "market_context", "sector_strength")
DIR_BY_KEY = {
    "stock_pool": STOCK_POOL_DIR,
    "market_context": MARKET_CONTEXT_DIR,
    "sector_strength": SECTOR_STRENGTH_DIR,
}
PREFIX_BY_KEY = {
    "stock_pool": "stock_pool",
    "market_context": "market_context",
    "sector_strength": "sector_strength",
}


def load_backfill_config() -> dict[str, Any]:
    """加载历史批跑配置"""
    return load_yaml("backfill_history.yaml")


def get_archive_root(settings: dict[str, Any] | None = None) -> Path:
    """归档根目录绝对路径"""
    cfg = load_backfill_config()
    archive_cfg = cfg.get("archive", {})
    if settings and settings.get("archive_root"):
        rel = settings["archive_root"]
    else:
        rel = archive_cfg.get("root", "output/archive")
    root = Path(rel)
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    return root.resolve()


def ensure_archive_dirs(root: Path | None = None) -> Path:
    """创建归档目录结构"""
    root = root or get_archive_root()
    for sub in (STOCK_POOL_DIR, MARKET_CONTEXT_DIR, SECTOR_STRENGTH_DIR, META_DIR):
        (root / sub).mkdir(parents=True, exist_ok=True)
    return root


def archive_csv_path(root: Path, key: str, trade_date: str) -> Path:
    """归档 CSV 路径：archive/{type}/{YYYYMMDD}.csv"""
    prefix = PREFIX_BY_KEY[key]
    return root / DIR_BY_KEY[key] / f"{prefix}_{trade_date}.csv"


def legacy_archive_csv_path(root: Path, key: str, trade_date: str) -> Path:
    """兼容仅 YYYYMMDD.csv 的旧命名"""
    return root / DIR_BY_KEY[key] / f"{trade_date}.csv"


def resolve_archive_csv(root: Path, key: str, trade_date: str) -> Path | None:
    """查找已存在的归档文件（新/旧命名）"""
    for path in (
        archive_csv_path(root, key, trade_date),
        legacy_archive_csv_path(root, key, trade_date),
    ):
        if path.is_file():
            return path
    return None


def is_date_archived(root: Path, trade_date: str) -> bool:
    """三个 CSV 均已归档且 stock_pool 非空"""
    pool_path = resolve_archive_csv(root, "stock_pool", trade_date)
    if pool_path is None or pool_path.stat().st_size < 50:
        return False
    for key in ("market_context", "sector_strength"):
        if resolve_archive_csv(root, key, trade_date) is None:
            return False
    return True


def copy_outputs_to_archive(
    outputs: dict[str, Path],
    trade_date: str,
    root: Path | None = None,
) -> dict[str, Path]:
    """将 pipeline 产出的 CSV 复制到 archive"""
    root = ensure_archive_dirs(root)
    archived: dict[str, Path] = {}
    for key in FILE_KEYS:
        src = outputs.get(key)
        if src is None or not Path(src).is_file():
            raise FileNotFoundError(f"缺少输出文件 {key}: {src}")
        dest = archive_csv_path(root, key, trade_date)
        shutil.copy2(src, dest)
        archived[key] = dest
    return archived


def archive_pipeline_outputs(
    outputs: dict[str, Path | str],
    *,
    root: Path | None = None,
    settings: dict[str, Any] | None = None,
    duration_sec: float | None = None,
) -> dict[str, Any]:
    """复制三份 CSV 到 archive 并更新 backfill_index（每日收盘与批跑共用）"""
    import pandas as pd

    trade_date = str(outputs["trade_date"]).replace("-", "")
    if root is None:
        root = ensure_archive_dirs(get_archive_root(settings))
    archived = copy_outputs_to_archive(outputs, trade_date, root)
    stock_count = 0
    pool_file = archived["stock_pool"]
    if pool_file.is_file():
        stock_count = len(pd.read_csv(pool_file, usecols=["ts_code"]))
    mark_date_completed(
        trade_date,
        root=root,
        stock_count=stock_count,
        duration_sec=duration_sec,
    )
    return {**archived, "stock_count": stock_count, "trade_date": trade_date}


def load_backfill_index(root: Path | None = None) -> dict[str, Any]:
    """读取批跑索引"""
    root = root or get_archive_root()
    path = root / META_DIR / INDEX_FILENAME
    if not path.is_file():
        return {
            "completed": [],
            "failed": {},
        }
    data = json.loads(path.read_text(encoding="utf-8"))
    data.setdefault("completed", [])
    data.setdefault("failed", {})
    return data


def save_backfill_index(index: dict[str, Any], root: Path | None = None) -> Path:
    """写入批跑索引"""
    root = ensure_archive_dirs(root)
    path = root / META_DIR / INDEX_FILENAME
    index["updated_at"] = datetime.now(_MARKET_TZ).isoformat(timespec="seconds")
    path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def mark_date_completed(
    trade_date: str,
    *,
    root: Path | None = None,
    stock_count: int | None = None,
    duration_sec: float | None = None,
) -> None:
    """记录某日批跑成功"""
    root = root or get_archive_root()
    index = load_backfill_index(root)
    completed = set(index.get("completed", []))
    completed.add(trade_date)
    index["completed"] = sorted(completed)
    index["failed"] = {k: v for k, v in index.get("failed", {}).items() if k != trade_date}
    details = index.setdefault("details", {})
    details[trade_date] = {
        "status": "completed",
        "stock_count": stock_count,
        "duration_sec": round(duration_sec, 2) if duration_sec is not None else None,
    }
    save_backfill_index(index, root)


def mark_date_failed(
    trade_date: str,
    error: str,
    *,
    root: Path | None = None,
) -> None:
    """记录某日批跑失败（断点续跑时可重试）"""
    root = root or get_archive_root()
    index = load_backfill_index(root)
    failed = index.setdefault("failed", {})
    failed[trade_date] = error
    details = index.setdefault("details", {})
    details[trade_date] = {"status": "failed", "error": error}
    save_backfill_index(index, root)


def list_archived_dates(root: Path | None = None) -> list[str]:
    """已完整归档的交易日列表（升序）"""
    root = root or get_archive_root()
    pool_dir = root / STOCK_POOL_DIR
    if not pool_dir.is_dir():
        return []
    dates: set[str] = set()
    for path in pool_dir.glob("*.csv"):
        name = path.stem
        if name.startswith("stock_pool_"):
            dates.add(name.removeprefix("stock_pool_"))
        elif len(name) == 8 and name.isdigit():
            dates.add(name)
    return sorted(d for d in dates if is_date_archived(root, d))
