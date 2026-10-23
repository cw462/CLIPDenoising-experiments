@echo off
setlocal
cd /d "%~dp0"
set "PYTHONDONTWRITEBYTECODE=1"
"%~dp0.venv\Scripts\python.exe" -u tools\denoise_local.py %*
set "RUN_CODE=%ERRORLEVEL%"
if not "%RUN_CODE%"=="0" echo Demo failed. See the error above.
if "%~1"=="" pause
exit /b %RUN_CODE%
