@echo off

chcp 65001 >nul

title BigASelect - 市场环境 & 行业强度



REM 切换到批处理所在目录（项目根目录）

cd /d "%~dp0"



echo ==================================================

echo   BigASelect - 生成 market_context + sector_strength

echo ==================================================

echo.

echo 输出命名: market_context_YYYYMMDD.csv

echo           sector_strength_YYYYMMDD.csv（同日覆盖）

echo.

echo 数据规则: 最新可用数据；非交易时段取最近收盘（quote_mode=last_close）

echo.

echo 正在启动，请稍候...

echo.



python generate_market_sector.py

set EXIT_CODE=%ERRORLEVEL%



echo.

if %EXIT_CODE% equ 0 (

    echo [完成] 请查看 output\ 目录下的 CSV 文件。

) else (

    echo [失败] 程序异常退出，错误码 %EXIT_CODE%。

)

echo.

pause

exit /b %EXIT_CODE%

