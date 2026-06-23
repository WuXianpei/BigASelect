"""历史批跑：按交易日逐日执行筛选+打分，归档至 output/archive/，支持断点续跑"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.archive_manager import (  # noqa: E402
    archive_pipeline_outputs,
    ensure_archive_dirs,
    get_archive_root,
    is_date_archived,
    list_archived_dates,
    load_backfill_config,
    load_backfill_index,
    mark_date_failed,
    save_backfill_index,
)
from src.config_loader import load_settings  # noqa: E402
from src.data_fetcher import get_backfill_end_date, get_trading_days_window  # noqa: E402
from src.network_setup import setup_network  # noqa: E402


def _resolve_day_count(args: argparse.Namespace, cfg: dict) -> int:
    bf = cfg.get("backfill", {})
    days = args.days if args.days is not None else bf.get("default_days", 60)
    min_days = bf.get("min_days", 60)
    max_days = bf.get("max_days", 120)
    days = max(min_days, min(max_days, days))
    return days


def _resolve_end_date(args: argparse.Namespace, cfg: dict, settings: dict) -> str:
    if args.end_date:
        return str(args.end_date).replace("-", "")
    bf = cfg.get("backfill", {})
    configured = bf.get("end_date")
    if configured:
        return str(configured).replace("-", "")
    return get_backfill_end_date(settings)


def _plan_dates(
    end_date: str,
    day_count: int,
    *,
    force: bool,
    skip_existing: bool,
    root: Path,
) -> tuple[list[str], list[str]]:
    """返回 (待跑日期升序, 已跳过日期)"""
    all_dates = get_trading_days_window(end_date, day_count)
    pending: list[str] = []
    skipped: list[str] = []
    for d in all_dates:
        if not force and skip_existing and is_date_archived(root, d):
            skipped.append(d)
        else:
            pending.append(d)
    return pending, skipped


def run_backfill(args: argparse.Namespace) -> int:
    cfg = load_backfill_config()
    bf = cfg.get("backfill", {})
    settings = load_settings()
    setup_network(settings)

    day_count = _resolve_day_count(args, cfg)
    end_date = _resolve_end_date(args, cfg, settings)
    root = ensure_archive_dirs(get_archive_root(settings))

    skip_existing = bf.get("skip_existing", True) and not args.force
    continue_on_error = bf.get("continue_on_error", True) and not args.stop_on_error
    also_output = bf.get("also_write_output", True)

    pending, skipped = _plan_dates(
        end_date,
        day_count,
        force=args.force,
        skip_existing=skip_existing,
        root=root,
    )

    print("=" * 50)
    print("BigASelect - 历史批跑（筛选 + 打分 → archive）")
    print("=" * 50)
    print(f"截止交易日: {end_date}")
    print(f"回溯窗口: {day_count} 个交易日")
    print(f"归档目录: {root}")
    print(f"待跑: {len(pending)} 日，已跳过: {len(skipped)} 日")
    if skipped and not args.dry_run:
        print(f"跳过示例: {skipped[0]} … {skipped[-1]}" if len(skipped) > 1 else f"跳过: {skipped[0]}")
    print()

    if args.dry_run:
        print("[dry-run] 待跑日期:")
        for d in pending:
            print(f"  {d}")
        return 0

    if not pending:
        print("无需批跑，archive 已完整。")
        print(f"已归档 {len(list_archived_dates(root))} 个交易日。")
        return 0

    save_backfill_index(
        {
            **load_backfill_index(root),
            "run_plan": {
                "end_date": end_date,
                "day_count": day_count,
                "pending": pending,
                "skipped": skipped,
            },
        },
        root,
    )

    from main import run_for_trade_date

    ok_count = 0
    fail_count = 0
    t_all = time.perf_counter()

    for idx, trade_date in enumerate(pending, start=1):
        print()
        print("-" * 50)
        print(f"[{idx}/{len(pending)}] 交易日 {trade_date}")
        print("-" * 50)
        t_day = time.perf_counter()
        try:
            outputs = run_for_trade_date(
                trade_date,
                production=True,
                settings=settings,
                quiet=True,
            )
            result = archive_pipeline_outputs(
                outputs,
                root=root,
                settings=settings,
                duration_sec=time.perf_counter() - t_day,
            )
            pool_file = result["stock_pool"]
            stock_count = result["stock_count"]
            dur = time.perf_counter() - t_day
            ok_count += 1
            print(f"[完成] {trade_date} → {pool_file.name}，{stock_count} 只，耗时 {dur:.1f}s")
            if not also_output:
                for key, path in outputs.items():
                    if key != "trade_date" and path.is_file():
                        path.unlink(missing_ok=True)
        except Exception as exc:
            fail_count += 1
            err = f"{type(exc).__name__}: {exc}"
            mark_date_failed(trade_date, err, root=root)
            print(f"[失败] {trade_date}: {err}")
            traceback.print_exc()
            if not continue_on_error:
                print("已配置遇错即停，批跑中止。")
                break

    total_dur = time.perf_counter() - t_all
    archived_total = len(list_archived_dates(root))
    print()
    print("=" * 50)
    print(f"批跑结束: 成功 {ok_count}，失败 {fail_count}，archive 共 {archived_total} 日")
    print(f"总耗时 {total_dur:.1f}s")
    print("=" * 50)
    return 1 if fail_count else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="历史批跑：筛选+打分并归档")
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help="回溯交易日数量（默认 config/backfill_history.yaml，60~120）",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="截止交易日 YYYYMMDD（默认今天之前的最近交易日）",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重跑已归档日期",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅列出待跑日期",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="遇错立即停止（默认跳过失败日继续）",
    )
    args = parser.parse_args()
    sys.exit(run_backfill(args))


if __name__ == "__main__":
    main()
