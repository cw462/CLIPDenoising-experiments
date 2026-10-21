@echo off
setlocal
cd /d "%~dp0"
set "PYTHONDONTWRITEBYTECODE=1"
set "OMP_NUM_THREADS=4"
set "MKL_NUM_THREADS=4"
if not "%~1"=="" goto custom
"%~dp0.venv\Scripts\python.exe" -u tools\run_experiment.py --exp G1 --stage smoke --seed 0 --paths configs\local_paths.yaml --resume
set "RUN_CODE=%ERRORLEVEL%"
pause
exit /b %RUN_CODE%
:custom
"%~dp0.venv\Scripts\python.exe" -u tools\run_experiment.py %*
exit /b %ERRORLEVEL%
