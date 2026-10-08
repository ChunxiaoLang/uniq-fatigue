import csv
import json
from pathlib import Path
import numpy as np
from .audit import sha256, read, dump
from .data import save_dataset, LABEL_MAP_WESAD, pm_label


def moving_average(x,k=9):
    if k%2!=1:
        raise ValueError("moving average width must be odd")
    x=np.asarray(x,dtype=np.float64).reshape(-1)
    return np.convolve(np.pad(x,(k//2,k//2),mode="edge"),np.ones(k)/k,mode="valid")


def resample(x,source_hz,target_hz,n,antialias=False):
    x=np.asarray(x,dtype=np.float64).reshape(-1)
    if not len(x) or not np.isfinite(x).all():
        raise ValueError("Missing/nonfinite raw signal")
    if antialias and source_hz>target_hz:
        from scipy.signal import resample_poly
        from fractions import Fraction
        ratio=Fraction(target_hz/source_hz).limit_denominator()
        x=resample_poly(x,ratio.numerator,ratio.denominator)
        source_hz=target_hz
    return np.interp(np.arange(n)/target_hz,np.arange(len(x))/source_hz,x).astype(np.float32)


def wesad(raw,out,logs=None,window=100,stride=50,antialias=False,trusted_pickle=False):
    if not trusted_pickle:
        raise ValueError("Official WESAD uses pickle. Re-run with --trust-official-pickle ONLY for trusted official files.")
    import pickle
    files=sorted(Path(raw).rglob("S*.pkl"))
    if not files: raise ValueError("No S*.pkl files found")
    logmap=read(logs) if logs else {}
    xs,ms,records=[],[],[]
    channels=["chest_ecg","chest_resp","wrist_eda","wrist_bvp","wrist_temp"]
    source={}
    for path in files:
        source[str(path.relative_to(raw))]=sha256(path)
        with path.open("rb") as f: d=pickle.load(f,encoding="latin1")
        subject=path.stem
        label=np.asarray(d["label"]).reshape(-1)
        n=int(len(label)*100/700)
        signal=d["signal"]
        chest,wrist=signal["chest"],signal["wrist"]
        x=np.stack([
            resample(moving_average(chest["ECG"],9),700,100,n,antialias),
            resample(chest["Resp"],700,100,n,antialias),
            resample(wrist["EDA"],4,100,n),resample(wrist["BVP"],64,100,n),
            resample(wrist["TEMP"],4,100,n)],axis=-1)
        # Labels are never interpolated or majority-voted across state changes.
        for start in range(0,n-window+1,stride):
            a,b=start*7,(start+window)*7
            native=label[a:b]
            if len(native)!=window*7 or not np.all(native==native[0]): continue
            state=int(native[0])
            if state not in LABEL_MAP_WESAD: continue
            phase_key=f"{subject}:{state}"
            entry=logmap.get(phase_key,{})
            text=entry.get("text", "Subjective questionnaire log unavailable.")
            # State is stored only as provenance/label, never injected into text automatically.
            records.append({"id":f"{subject}:{start}","subject":subject,"session":subject,
                "start":start,"end":start+window,"label":LABEL_MAP_WESAD[state],"text":text,
                "reports":entry.get("reports"),"reference_report":entry.get("reference_report"),
                "text_source":entry.get("source","missing"),
                "provenance":{"dataset":"WESAD","original_state":state,"raw_sha256":source[str(path.relative_to(raw))],
                    "sampling_hz":100,"antialias":antialias,"reference_source":entry.get("reference_source","missing")}})
            xs.append(x[start:start+window]);ms.append(np.ones((window,5),dtype=bool))
    if not xs: raise ValueError("No valid WESAD windows")
    save_dataset(out,xs,ms,records,channels)
    dump(str(out)+".provenance.json",{"raw_files":source,"logs_sha256":sha256(logs) if logs else None,
          "window":window,"stride":stride,"antialias":antialias,"valid_pairs":len(xs),
          "warning":"Stress/affect labels are a paper-defined proxy, not validated sports-fatigue ground truth."})


def pmdata(raw,metadata,out,scale="native5",label_map=None,length=100,max_gap=300):
    """Deterministic explicit-interval import. Raw native HR JSON or timestamp/channel CSV.

    Interval/annotation metadata is required because the paper omits alignment rules.
    Native5 requires an externally specified map; paper10 accepts externally supplied 1-10 labels.
    """
    mapping=read(label_map) if label_map else None
    if scale=="native5" and mapping is None:
        raise ValueError("Native PMData fatigue uses 1-5. Provide --label-map; paper's 1-10 mapping cannot be silently applied.")
    with open(metadata,encoding="utf-8-sig",newline="") as f: rows=list(csv.DictReader(f))
    def timestamp(s): return np.datetime64(s,"ms").astype(np.int64)/1000
    streams={};raw_hashes={}
    for subject in sorted({r["subject"] for r in rows}):
        directory=Path(raw)/subject
        values={}
        for path in sorted(directory.rglob("*.csv")):
            with path.open(encoding="utf-8-sig",newline="") as f:
                reader=csv.DictReader(f)
                if not reader.fieldnames or "timestamp" not in reader.fieldnames: continue
                raw_hashes[str(path.relative_to(raw))]=sha256(path)
                for r in reader:
                    t=timestamp(r["timestamp"])
                    for name in reader.fieldnames:
                        if name!="timestamp" and r[name]!="":
                            values.setdefault(name,[]).append((t,float(r[name])))
        for path in sorted(directory.rglob("heart_rate*.json")):
            raw_hashes[str(path.relative_to(raw))]=sha256(path)
            obj=read(path)
            if not isinstance(obj,list): raise ValueError(f"Unsupported HR JSON schema: {path}; export timestamp CSV")
            for r in obj:
                if "dateTime" not in r or "value" not in r: raise ValueError(f"Unsupported HR record: {path}")
                bpm=r["value"]["bpm"] if isinstance(r["value"],dict) else r["value"]
                values.setdefault("heart_rate",[]).append((timestamp(r["dateTime"]),float(bpm)))
        # Duplicate timestamps are deterministically averaged, not arbitrarily selected.
        streams[subject]={}
        for name,points in values.items():
            t,v=np.asarray(sorted(points)).T
            unique,index,count=np.unique(t,return_index=True,return_counts=True)
            means=np.add.reduceat(v,index)/count
            streams[subject][name]=(unique,means)
    channels=sorted({name for values in streams.values() for name in values})
    if not channels: raise ValueError("No sensor channels found")
    xs,ms,records=[],[],[]
    for idx,r in enumerate(rows):
        a,b=timestamp(r["start_iso"]),timestamp(r["end_iso"])
        if b<=a: raise ValueError("Invalid interval")
        grid=np.linspace(a,b,length,endpoint=False)
        x=np.zeros((length,len(channels)),np.float32);mask=np.zeros_like(x,dtype=bool)
        for c,name in enumerate(channels):
            if name not in streams[r["subject"]]: continue
            t,v=streams[r["subject"]][name]
            # Restrict interpolation to this interval; no future or past boundary samples.
            valid=(t>=a)&(t<b)
            t,v=t[valid],v[valid]
            if not len(t): continue
            nearest=np.searchsorted(t,grid).clip(0,len(t)-1)
            previous=(nearest-1).clip(0,len(t)-1)
            gap=np.minimum(abs(t[nearest]-grid),abs(t[previous]-grid))
            usable=(grid>=t[0])&(grid<=t[-1])&(gap<=max_gap)
            x[:,c]=np.where(usable,np.interp(grid,t,v),0)
            mask[:,c]=usable
        if not mask.any(): continue
        label=pm_label(r["fatigue"],scale) if scale=="paper10" else int(mapping[str(int(float(r["fatigue"])) )])
        text=r.get("text","")
        if not text:
            fields=[f"{key}={r[key]}" for key in ("sleep_quality","soreness","mood","rpe") if r.get(key)]
            text="; ".join(fields) or "Subjective log unavailable."
        # Target fatigue is deliberately absent from subjective features.
        reportpath=r.get("reports_json","")
        ref=read(Path(metadata).parent/reportpath) if reportpath else {}
        records.append({"id":r.get("id") or f"{r['subject']}:{idx}","subject":r["subject"],
          "session":r.get("session") or "timeline","start":a,"end":b,"label":label,"text":text,
          "reports":ref.get("reports"),"reference_report":ref.get("reference_report"),"text_source":"observed",
          "provenance":{"dataset":"PMData","scale":scale,"reference_source":ref.get("reference_source","missing")}})
        xs.append(x);ms.append(mask)
    if not xs: raise ValueError("No aligned intervals")
    save_dataset(out,xs,ms,records,channels)
    dump(str(out)+".provenance.json",{"raw_files":raw_hashes,"metadata_sha256":sha256(metadata),
        "label_map_sha256":sha256(label_map) if label_map else None,"channels":channels,
        "valid_pairs":len(xs),"resampling":"100 evenly spaced points per supplied activity interval", "max_gap_seconds":max_gap})


def synthetic(out,seed=42):
    rng=np.random.default_rng(seed)
    xs,records=[],[]
    for subject in range(6):
        for j in range(12):
            label=j%3
            t=np.arange(64)/64
            signal=np.stack([np.sin(2*np.pi*(label+1)*t),np.cos(2*np.pi*(label+1)*t),t*(label+1)],-1)
            signal+=rng.normal(0,0.05,signal.shape)
            xs.append(signal)
            reports={k:f"Synthetic evidence pattern {label}; no clinical interpretation." for k in ("physio","kine","psych")}
            records.append({"id":f"demo{subject}:{j}","subject":f"demo{subject}","session":"demo",
                "start":j*64,"end":j*64+64,"label":label,"text":f"Synthetic observation pattern {label}.",
                "reports":reports,"reference_report":"Synthetic reference; not a paper result.","text_source":"synthetic",
                "provenance":{"dataset":"synthetic","reference_source":"synthetic","seed":seed}})
    x=np.asarray(xs,np.float32)
    save_dataset(out,x,np.ones_like(x,dtype=bool),records,["synthetic_a","synthetic_b","synthetic_c"])
