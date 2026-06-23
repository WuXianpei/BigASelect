@echo off
chcp 65001 >nul
title BigASelect - 历史批跑

cd /d "%~dp0"

echo ==================================================
echo   BigASelect - 历史批跑（筛选 + 打分 -^> archive）
echo ==================================================
echo.
echo 默认: 截止「最新可用收盘日」（已收盘含当天），向前 60 个交易日
echo 归档: output\archive\stock_pool、market_context、sector_strength
echo 断点续跑: 已归档日期自动跳过（见 output\archive\meta\backfill_index.json）
echo.
echo 可选参数示例:
echo   python scripts\backfill_history.py --days 120
echo   python scripts\backfill_history.py --dry-run
echo   python scripts\backfill_history.py --force
echo.

python scripts\backfill_history.py %*
set EXIT_CODE=%ERRORLEVEL%

echo.
if %EXIT_CODE% equ 0 (
    echo [完成] 批跑结束。
) else (
    echo [失败] 存在失败日期，错误码 %EXIT_CODE%，可重新运行以续跑。
)
echo.
pause
exit /b %EXIT_CODE%
