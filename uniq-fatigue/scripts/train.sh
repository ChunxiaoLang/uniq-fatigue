#!/usr/bin/env bash
set -euo pipefail
export PYTHONHASHSEED="${SEED:-42}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -m uniq_fatigue train "$@"
