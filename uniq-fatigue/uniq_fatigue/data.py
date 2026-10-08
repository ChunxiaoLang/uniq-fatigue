import json
from pathlib import Path
import numpy as np


LABEL_MAP_WESAD = {1: 0, 4: 0, 3: 1, 2: 2}


def pm_label(value, scale):
    if scale != "paper10":
        raise ValueError("PMData native 1-5 mapping is unspecified. Supply an author-confirmed label map.")
    value = float(value)
    if not 1 <= value <= 10:
        raise ValueError("paper10 fatigue must be in [1,10]")
    return 0 if value <= 3 else 1 if value <= 7 else 2


def save_dataset(path, x, mask, records, channels):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, x=np.asarray(x, dtype=np.float32), mask=np.asarray(mask, dtype=bool),
                        records=np.asarray([json.dumps(r, ensure_ascii=False) for r in records]),
                        channels=np.asarray(channels))


def load_dataset(path):
    with np.load(path, allow_pickle=False) as d:
        x, mask = d["x"].copy(), d["mask"].copy()
        records = [json.loads(s) for s in d["records"].tolist()]
        channels = d["channels"].tolist()
    if x.ndim != 3 or mask.shape != x.shape or len(records) != len(x):
        raise ValueError("Expected x/mask [N,L,C] and N records")
    if len(channels) != x.shape[2] or not np.isfinite(x).all():
        raise ValueError("Invalid channels or nonfinite values; encode missingness using mask")
    if len({r["id"] for r in records}) != len(records):
        raise ValueError("Sample IDs must be unique")
    for r, m in zip(records, mask):
        if r["label"] not in (0, 1, 2) or r["end"] <= r["start"]:
            raise ValueError("Invalid label or half-open [start,end) interval")
        for field in ("subject", "session", "text", "provenance"):
            if field not in r:
                raise ValueError(f"Missing field {field}")
        if not m.any():
            raise ValueError("Dataset sample has no measured sensor values")
    return x, mask, records, channels


def context_text(x, mask, record, channels, dropout="none"):
    # Sensor summaries are computed only from the available sensor modality.
    # Free text must not already include copied sensor statistics or target fatigue ratings.
    parts = []
    if dropout != "sensor":
        for c, name in enumerate(channels):
            valid = mask[:, c]
            v = x[valid, c]
            if len(v):
                t = np.flatnonzero(valid).astype(float)
                slope = float(np.polyfit(t, v, 1)[0]) if len(v) > 1 else 0.0
                parts.append(f"{name}: min={v.min():.5g}, max={v.max():.5g}, slope/sample={slope:.5g}, variance={v.var():.5g}.")
    log = record["text"] if dropout != "text" else "Subjective log unavailable."
    return "[Observed sensor statistics] " + (" ".join(parts) or "Unavailable.") + "\n[Subjective context] " + log


def corrupt_statistics(text):
    import re
    # Only pollute the generated sensor-statistics section, not questionnaire fields.
    stat, log = text.split("\n[Subjective context]", 1)
    stat = re.sub(r"(?<==)[+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?", "-9999", stat)
    return stat + "\n[Subjective context]" + log
