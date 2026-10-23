@echo off
setlocal
cd /d "%~dp0"
set "PYTHONDONTWRITEBYTECODE=1"
set "OMP_NUM_THREADS=4"
set "MKL_NUM_THREADS=4"
if not "%~1"=="" goto custom
"%~dp0.venv\Scripts\python.exe" -u tools\run_pfa_experiment.py --group B0 --stage smoke --seed 100 --resume
if errorlevel 1 goto failed
"%~dp0.venv\Scripts\python.exe" -u tools\run_pfa_experiment.py --group B1 --stage smoke --seed 100 --resume
if errorlevel 1 goto failed
echo Paired proposal A smoke runs completed.
pause
exit /b 0
:custom
"%~dp0.venv\Scripts\python.exe" -u tools\run_pfa_experiment.py %*
exit /b %ERRORLEVEL%
:failed
echo Proposal A run failed. See the error above.
pause
exit /b 1
