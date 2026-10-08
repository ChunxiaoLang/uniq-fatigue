import random
from collections import defaultdict


def audit_split(records, split, allow_overlap=False):
    lookup = {r["id"]: r for r in records}
    seen = set()
    for subset in ("train", "val", "test"):
        ids = split[subset]
        if not ids or len(set(ids)) != len(ids) or seen.intersection(ids):
            raise ValueError("Empty, duplicate or intersecting split IDs")
        if not set(ids) <= lookup.keys():
            raise ValueError("Split references unknown IDs")
        seen.update(ids)
    groups = defaultdict(list)
    for subset in ("train", "val", "test"):
        for sid in split[subset]:
            r = lookup[sid]
            groups[(r["subject"], r["session"])].append((r["start"], r["end"], subset))
    overlap = 0
    for rows in groups.values():
        rows.sort()
        active = []
        for start, end, subset in rows:
            active = [(b, s) for b, s in active if b > start]
            overlap += sum(s != subset for b, s in active)
            active.append((end, subset))
    if overlap and not allow_overlap:
        raise ValueError(f"{overlap} cross-split overlapping raw intervals detected")
    if split["protocol"] in ("subject", "loso"):
        subjects = [{lookup[i]["subject"] for i in split[s]} for s in ("train", "val", "test")]
        if any(subjects[a] & subjects[b] for a, b in ((0,1),(0,2),(1,2))):
            raise ValueError("Subject leakage")
    return {"cross_split_overlaps": overlap, "excluded": len(records)-len(seen)}


def make_split(records, protocol="subject", seed=42, held_out=None):
    rng = random.Random(seed)
    out = {"protocol": protocol, "seed": seed, "train": [], "val": [], "test": []}
    if protocol in ("subject", "loso"):
        subjects = sorted({r["subject"] for r in records})
        if len(subjects) < 3:
            raise ValueError("Need at least 3 subjects")
        rng.shuffle(subjects)
        if protocol == "loso":
            if held_out not in subjects:
                raise ValueError("LOSO needs an existing --held-out subject")
            test = {held_out}
            rest = [s for s in subjects if s != held_out]
        else:
            nt = max(1, round(len(subjects)*0.3))
            test, rest = set(subjects[:nt]), subjects[nt:]
        nv = max(1, round(len(rest)*0.15))
        val = set(rest[:nv])
        for r in records:
            out["test" if r["subject"] in test else "val" if r["subject"] in val else "train"].append(r["id"])
    elif protocol == "random":
        ids = sorted(r["id"] for r in records)
        rng.shuffle(ids)
        nt = max(1, round(len(ids)*0.3))
        nv = max(1, round((len(ids)-nt)*0.15))
        out["test"], out["val"], out["train"] = ids[:nt], ids[nt:nt+nv], ids[nt+nv:]
        out["warning"] = "Paper-style random windows; can leak overlapping observations and subjects."
    elif protocol == "chronological":
        groups = defaultdict(list)
        for r in records:
            groups[(r["subject"], r["session"])].append(r)
        for rows in groups.values():
            rows.sort(key=lambda r: (r["start"], r["id"]))
            low, high = min(r["start"] for r in rows), max(r["end"] for r in rows)
            # 70% development, 30% test; 15% of development for validation.
            bv, bt = low+(high-low)*0.595, low+(high-low)*0.7
            for r in rows:
                if r["end"] <= bv:
                    out["train"].append(r["id"])
                elif r["start"] >= bv and r["end"] <= bt:
                    out["val"].append(r["id"])
                elif r["start"] >= bt:
                    out["test"].append(r["id"])
    else:
        raise ValueError(protocol)
    for key in ("train", "val", "test"):
        out[key].sort()
    out["audit"] = audit_split(records, out, allow_overlap=protocol == "random")
    return out
