"""运行日志：控制台输出同时写入文件，并生成 JSON 摘要供后续优化分析"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO


class _TeeWriter:
    """同时写入控制台与日志文件"""

    def __init__(self, *streams: TextIO) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()

    def isatty(self) -> bool:
        return False


class RunLogger:
    """单次运行的日志与指标收集器"""

    def __init__(self, settings: dict[str, Any], project_root: Path) -> None:
        self.settings = settings
        self.project_root = project_root
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.started_at = datetime.now().isoformat(timespec="seconds")
        self.t0 = time.perf_counter()

        log_dir_name = settings.get("run_log_dir", "output/logs")
        self.log_dir = (project_root / log_dir_name).resolve()
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.log_path = self.log_dir / f"run_{self.run_id}.log"
        self.report_path = self.log_dir / f"run_{self.run_id}_report.json"
        self.latest_report_path = self.log_dir / "latest_report.json"
        self.latest_log_path = self.log_dir / "latest.log"

        self._log_file: TextIO | None = None
        self._stdout_backup: TextIO | None = None
        self.steps: list[dict[str, Any]] = []
        self.extra: dict[str, Any] = {}

    def start(self) -> None:
        """开始记录：stdout 双写至日志文件"""
        self._log_file = open(self.log_path, "w", encoding="utf-8")
        self._stdout_backup = sys.stdout
        sys.stdout = _TeeWriter(self._stdout_backup, self._log_file)
        print(f"[运行日志] run_id={self.run_id}")
        print(f"[运行日志] 日志文件: {self.log_path}")

    def stop(self) -> None:
        """恢复 stdout 并关闭日志文件"""
        if self._stdout_backup is not None:
            sys.stdout = self._stdout_backup
            self._stdout_backup = None
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None
        # 维护 latest 软链接式副本（Windows 用复制）
        try:
            import shutil

            shutil.copy2(self.log_path, self.latest_log_path)
        except OSError:
            pass

    def record_step(
        self,
        name: str,
        duration_sec: float,
        **metrics: Any,
    ) -> None:
        """记录一个阶段的耗时与指标"""
        self.steps.append(
            {
                "name": name,
                "duration_sec": round(duration_sec, 2),
                **metrics,
            }
        )

    def set_extra(self, **kwargs: Any) -> None:
        """补充最终报告中的自定义字段"""
        self.extra.update(kwargs)

    def finalize(self, status: str = "success", error: str | None = None) -> Path:
        """写入 JSON 运行报告"""
        total_sec = round(time.perf_counter() - self.t0, 2)
        report: dict[str, Any] = {
            "run_id": self.run_id,
            "status": status,
            "error": error,
            "started_at": self.started_at,
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "total_duration_sec": total_sec,
            "test_mode": self.settings.get("test_mode", False),
            "trade_date": self.settings.get("trade_date"),
            "target_pool_size": self.settings.get("target_pool_size"),
            "enrich_workers": self.settings.get("enrich_workers"),
            "log_file": str(self.log_path),
            "steps": self.steps,
            **self.extra,
        }

        payload = json.dumps(report, ensure_ascii=False, indent=2)
        self.report_path.write_text(payload, encoding="utf-8")
        self.latest_report_path.write_text(payload, encoding="utf-8")
        return self.report_path
