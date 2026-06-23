"""并发数解析：按 CPU 核数做安全上限，避免过度并发拖垮内存或触发限流"""

from __future__ import annotations

import os
from typing import Any


def resolve_worker_count(
    settings: dict[str, Any],
    key: str,
    default: int,
    *,
    item_count: int | None = None,
) -> int:
    """
    解析 settings 中的并发数。

    - 数值：直接使用，但不超过 CPU×3（网络 I/O 任务的安全上限）
    - null / "auto"：min(default, CPU×2+4)
    """
    cpu = os.cpu_count() or 4
    configured = settings.get(key)

    if configured is None or configured == "auto":
        workers = min(default, cpu * 2 + 4)
    else:
        workers = int(configured)

    workers = min(workers, cpu * 3)
    if item_count is not None:
        workers = min(workers, max(item_count, 1))
    return max(workers, 1)
