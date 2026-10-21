@echo off
setlocal
cd /d "%~dp0"
set "PYTHONDONTWRITEBYTECODE=1"
if not exist "%~dp0tmp" mkdir "%~dp0tmp"
set "TEMP=%~dp0tmp"
set "TMP=%~dp0tmp"
set "OMP_NUM_THREADS=4"
set "MKL_NUM_THREADS=4"
"%~dp0.venv\Scripts\python.exe" -m pip check
if errorlevel 1 goto failed
"%~dp0.venv\Scripts\python.exe" -m pytest tests\test_data_contract.py tests\test_pipeline_capacity.py tests\test_spatial_pfa.py tests\test_structure_loss.py -q
if errorlevel 1 goto failed
"%~dp0.venv\Scripts\python.exe" -u tests\test_calibration_contract.py --weights pretrained\RN50.pt --device cpu --report reports\calibration_contract_cpu.json
if errorlevel 1 goto failed
"%~dp0.venv\Scripts\python.exe" -u tests\test_resume_contract.py --paths configs\local_paths.yaml --device cpu --report reports\resume_contract_cpu.json
if errorlevel 1 goto failed
"%~dp0.venv\Scripts\python.exe" -u tests\test_cache_contract.py --paths configs\local_paths.yaml --report reports\cache_contract.json
if errorlevel 1 goto failed
echo All local checks passed.
if "%~1"=="" pause
exit /b 0
:failed
echo Local checks failed. See the error above.
if "%~1"=="" pause
exit /b 1
