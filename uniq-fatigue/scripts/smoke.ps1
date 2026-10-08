$ErrorActionPreference = 'Stop'
$env:PYTHONHASHSEED = '42'
$env:PYTHONUTF8 = '1'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
function Run-Python { & python @args; if ($LASTEXITCODE -ne 0) { throw 'Python command failed' } }
Run-Python -m uniq_fatigue synthetic --out data/smoke.npz
Run-Python -m uniq_fatigue split --data data/smoke.npz --protocol subject --out splits/smoke.json
Run-Python -m uniq_fatigue train --data data/smoke.npz --split splits/smoke.json --config configs/smoke.json --out runs/smoke
foreach ($mode in @('none','sensor','text')) {
    Run-Python -m uniq_fatigue infer --data data/smoke.npz --split splits/smoke.json --checkpoint runs/smoke/best.pt --dropout $mode --out "runs/smoke/$mode.jsonl"
    Run-Python -m uniq_fatigue evaluate --data data/smoke.npz --predictions "runs/smoke/$mode.jsonl" --out "runs/smoke/$mode.metrics.json"
}
