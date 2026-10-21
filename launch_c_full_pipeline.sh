#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
mkdir -p reports/c_only
exec .venv/bin/python tools/run_c_full_pipeline.py --paths configs/plan_c_only_server.yaml >> reports/c_only/pipeline_console.log 2>&1
