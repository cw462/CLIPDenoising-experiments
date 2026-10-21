@echo off
cd /d "%~dp0"
set PYTHONDONTWRITEBYTECODE=1
call "%~dp0.venv\Scripts\activate.bat"
cmd /k
