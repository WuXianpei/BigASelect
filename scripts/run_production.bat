@echo off
chcp 65001 >nul
title BigASelect - A股正式筛选

REM BigASelect 正式全市场运行（日志写入 output/logs/）
cd /d "%~dp0.."

echo ==================================================
echo   BigASelect - A股股票筛选（正式全市场）
echo ==================================================
echo.
echo 运行日志: output\logs\latest.log
echo 运行报告: output\logs\latest_report.json
echo.
echo 正在启动，请稍候...
echo.

python main.py --production
set EXIT_CODE=%ERRORLEVEL%

echo.
if %EXIT_CODE% equ 0 (
    echo [完成] 筛选成功，请查看 output\ 目录下的 CSV 文件。
) else (
    echo [失败] 程序异常退出，错误码 %EXIT_CODE%，请查看 output\logs\ 日志。
)
echo.
pause
exit /b %EXIT_CODE%