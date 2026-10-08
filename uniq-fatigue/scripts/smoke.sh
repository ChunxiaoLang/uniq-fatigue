#!/usr/bin/env bash
set -euo pipefail
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -m uniq_fatigue synthetic --out data/smoke.npz
python -m uniq_fatigue split --data data/smoke.npz --protocol subject --out splits/smoke.json
python -m uniq_fatigue train --data data/smoke.npz --split splits/smoke.json --config configs/smoke.json --out runs/smoke
for mode in none sensor text; do
  python -m uniq_fatigue infer --data data/smoke.npz --split splits/smoke.json --checkpoint runs/smoke/best.pt --dropout "$mode" --out "runs/smoke/$mode.jsonl"
  python -m uniq_fatigue evaluate --data data/smoke.npz --predictions "runs/smoke/$mode.jsonl" --out "runs/smoke/$mode.metrics.json"
done
