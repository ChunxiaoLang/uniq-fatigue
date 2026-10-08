import hashlib
import json
import os
import platform
import random
import subprocess
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def seed_all(seed, threads=4):
    # Set PYTHONHASHSEED before starting Python for hash determinism.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    import numpy as np
    import torch
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def manifest(data, split, config, root):
    import torch
    root = Path(root)
    tracked = sorted(p for directory in ("uniq_fatigue", "scripts", "prompts", "configs")
                     for p in (root / directory).rglob("*") if p.suffix in (".py", ".json", ".txt"))
    tracked += [root / "requirements.txt", root / "pyproject.toml"]
    model_lock = root / config.get("model_lock", "models.lock.json")
    if model_lock.exists():
        tracked.append(model_lock)
    return {
        "config": config, "data_sha256": sha256(data), "split_sha256": sha256(split),
        "files": {str(p.relative_to(root)): sha256(p) for p in tracked},
        "python": platform.python_version(), "platform": platform.platform(),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "packages": subprocess.check_output([os.sys.executable, "-m", "pip", "freeze"], text=True).splitlines(),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "deterministic_algorithms": True,
        "scope": "independent reconstruction; not a verification of the paper's reported numbers"
    }
