#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"
mkdir -p reports/ac
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
unset CUDA_VISIBLE_DEVICES
.venv/bin/python -u tools/run_ac_pipeline.py --plan-a configs/plan_a_server.yaml --plan-c configs/plan_c_server.yaml >> reports/ac/pipeline_console.log 2>&1
code=$?
printf '{"exit_code":%s,"ended_at":"%s"}\n' "$code" "$(date -Iseconds)" > reports/ac/pipeline_exit.json.tmp
mv reports/ac/pipeline_exit.json.tmp reports/ac/pipeline_exit.json
exit "$code"
