@echo off
echo ========================================
echo    STS Dashboard Network Setup
echo ========================================
echo.

echo Finding your computer's IP address...
for /f "tokens=2 delims=:" %%i in ('ipconfig ^| findstr "IPv4"') do (
    set "ip=%%i"
    setlocal enabledelayedexpansion
    set "ip=!ip: =!"
    echo Your IP Address: !ip!
    echo.
    echo Dashboard will be available at:
    echo http://!ip!:8501
    echo.
    echo Share this URL with your team members on the same network.
    echo.
    endlocal
    goto :start_server
)

:start_server
echo Starting dashboard server...
echo Press Ctrl+C to stop the server
echo.
cd /d "c:\Users\311741\sts"
"C:/Users/311741/sts/.venv/Scripts/python.exe" -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501
