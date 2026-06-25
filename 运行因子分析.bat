@echo off
chcp 65001 >nul
title BigASelect - 因子有效性分析

cd /d "%~dp0"

echo ==================================================
echo   BigASelect - 因子有效性分析
echo ==================================================
echo.
echo 数据源: output\archive\stock_pool（列 future_return_20）
echo 收益: future_return_20（20 个交易日 forward return）
echo 窗口: 最少 40，最多 100 个交易日（缺日跳过）
echo 报告: output\analysis\reports\
echo 失效时: output\analysis\proposed\factor_config.proposed.yaml
echo.
echo 可选: --days 120  --rebuild-returns  --dry-run
echo.

python scripts\analyze_factor_effectiveness.py %*
set EXIT_CODE=%ERRORLEVEL%

echo.
if %EXIT_CODE% equ 0 (
    echo [完成] 分析结束。
) else (
    echo [失败] 错误码 %EXIT_CODE%。
)
echo.
pause
exit /b %EXIT_CODE%
