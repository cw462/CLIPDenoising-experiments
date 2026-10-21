#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"
mkdir -p reports
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
unset CUDA_VISIBLE_DEVICES
printf '%s\n' "Started/resumed at $(date -Iseconds)" >> reports/pipeline_console.log
.venv/bin/python -u tools/run_pipeline.py --mode main --paths configs/local_paths.yaml >> reports/pipeline_console.log 2>&1
code=$?
printf '{"exit_code":%s,"ended_at":"%s"}\n' "$code" "$(date -Iseconds)" > reports/pipeline_exit.json.tmp
mv reports/pipeline_exit.json.tmp reports/pipeline_exit.json
exit "$code"
