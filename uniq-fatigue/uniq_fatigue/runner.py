import json
import math
import os
import time
import hashlib
from pathlib import Path
import numpy as np
import torch
from .audit import dump,read,sha256,seed_all,manifest
from .data import load_dataset,context_text,corrupt_statistics
from .splits import audit_split
from .model import UniQFatigue
from .losses import objective
from .metrics import classification


ROOT=Path(__file__).resolve().parents[1]


def tensor_batch(x,mask,records,channels,indices,device,dropout="none",missing_rate=0.25,dropout_seed=42):
    if not 0<=missing_rate<=1:raise ValueError("missing_rate must be [0,1]")
    xx=x[indices].copy();mm=mask[indices].copy();texts=[]
    for j,i in enumerate(indices):
        seed=int.from_bytes(hashlib.sha256(f"{dropout_seed}:{records[i]['id']}".encode()).digest()[:8],"little")
        rng=np.random.default_rng(seed)
        text_missing=dropout=="text"
        if dropout=="sensor":mm[j]=False
        if dropout in ("sensor_partial","dual25"):
            rate=0.25 if dropout=="dual25" else missing_rate
            mm[j]&=rng.random(mm[j].shape)>=rate
        if dropout in ("text_partial","dual25"):
            rate=0.25 if dropout=="dual25" else missing_rate
            text_missing=bool(rng.random()<rate)
        xx[j]=np.where(mm[j],xx[j],0)
        texts.append(context_text(xx[j],mm[j],records[i],channels,"text" if text_missing else "sensor" if dropout=="sensor" else "none"))
    return torch.from_numpy(xx).to(device),torch.from_numpy(mm).to(device),texts


def indices_for(records,split,subset):
    lookup={r["id"]:i for i,r in enumerate(records)}
    return [lookup[s] for s in split[subset]]


def checkpoint(model,cfg,path):
    params={name:p.detach().cpu().clone() for name,p in model.named_parameters() if p.requires_grad}
    torch.save({"config":cfg,"parameters":params,"format_version":1},path)


def restore(path):
    d=torch.load(path,map_location="cpu",weights_only=True)
    cfg=d["config"]
    seed_all(cfg["seed"],cfg.get("num_threads",4))
    model=UniQFatigue(cfg,ROOT)
    actual={n:p for n,p in model.named_parameters() if p.requires_grad}
    if actual.keys()!=d["parameters"].keys(): raise ValueError("Checkpoint parameter names do not match architecture")
    with torch.no_grad():
        for name,p in actual.items(): p.copy_(d["parameters"][name])
    model.to(cfg["device"]).eval()
    return model,cfg


def predict(model,x,mask,records,channels,indices,dropout="none",batch_size=1,missing_rate=0.25,dropout_seed=42):
    rows=[]
    start=time.perf_counter()
    for a in range(0,len(indices),batch_size):
        ids=indices[a:a+batch_size]
        xx,mm,texts=tensor_batch(x,mask,records,channels,ids,next(model.parameters()).device,dropout,missing_rate,dropout_seed)
        outputs=model.infer(xx,mm,texts)
        for i,p in zip(ids,outputs):
            p.update({"id":records[i]["id"],"subject":records[i]["subject"],"label":records[i]["label"],
                      "dropout":dropout,"missing_rate":missing_rate,"dropout_seed":dropout_seed,"backend":model.cfg["backend"]})
            rows.append(p)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    return rows,time.perf_counter()-start


def joint_batch(model,x,mask,records,channels,ids,cfg):
    pieces=[]
    for a in range(0,len(ids),cfg["micro_batch_size"]):
        chunk=ids[a:a+cfg["micro_batch_size"]]
        xx,mm,texts=tensor_batch(x,mask,records,channels,chunk,cfg["device"])
        refs=[records[i].get("reports") for i in chunk]
        if any(not r or any(not r.get(k) for k in ("physio","kine","psych")) for r in refs):
            raise ValueError("Training requires three expert reference summaries per sample; no fabricated default targets.")
        labels=torch.tensor([records[i]["label"] for i in chunk],device=cfg["device"])
        pieces.append(model(xx,mm,texts,labels,refs,[corrupt_statistics(t) for t in texts]))
    # Contrastive negatives cover the full logical batch even with sequential LLM forwards.
    # Graphs remain live until backward: microbatching does not guarantee a memory reduction.
    out={}
    for key in pieces[0]:
        if key in ("fm","generation"):
            out[key]=sum(p[key]*len(p["ex"]) for p in pieces)/len(ids)
        else:
            out[key]=torch.cat([p[key] for p in pieces],0) if pieces[0][key] is not None else None
    labels=torch.tensor([records[i]["label"] for i in ids],device=cfg["device"])
    return objective(out,labels,cfg)


def train(data,split_path,config_path,out,seed=None,ablation=None):
    cfg=read(config_path)
    if seed is not None: cfg["seed"]=seed
    if ablation is not None: cfg["ablation"]=ablation
    if cfg["ablation"] not in ("full","A","B","C","D","E","F","G"): raise ValueError("Unknown ablation")
    if cfg["micro_batch_size"]<1 or cfg["batch_size"]<2: raise ValueError("Alignment training needs batch_size>=2")
    seed_all(cfg["seed"],cfg.get("num_threads",4))
    x,mask,records,channels=load_dataset(data)
    if len(channels)!=cfg["channels"]: raise ValueError(f"Data has {len(channels)} channels; update config.channels")
    split=read(split_path)
    if split.get("data_sha256") and split["data_sha256"]!=sha256(data):
        raise ValueError("Split was built from a different dataset")
    audit_split(records,split,allow_overlap=split["protocol"]=="random")
    if cfg["backend"]=="hf" and any(r["provenance"]["dataset"]=="synthetic" for r in records):
        raise ValueError("Use the smoke config for synthetic data")
    out=Path(out)
    if out.exists() and any(out.iterdir()): raise ValueError("Run directory must be empty; each experiment needs an immutable directory")
    out.mkdir(parents=True,exist_ok=True)
    provenance=manifest(data,split_path,cfg,ROOT)
    if Path(str(data)+".provenance.json").exists():
        provenance["data_provenance"]=read(str(data)+".provenance.json")
    if cfg.get("ts_checkpoint"):
        provenance["ts_checkpoint_sha256"]=sha256(cfg["ts_checkpoint"])
    dump(out/"manifest.json",provenance)
    dump(out/"split.json",split)
    dump(out/"config.json",cfg)
    model=UniQFatigue(cfg,ROOT).to(cfg["device"])
    parameters=[p for p in model.parameters() if p.requires_grad]
    dump(out/"parameters.json",{"trainable":sum(p.numel() for p in parameters),
        "total":sum(p.numel() for p in model.parameters()),"ts_is_pretrained":bool(cfg.get("ts_checkpoint")),
        "llm_hidden_dimension":model.language.hidden})
    optimizer=torch.optim.AdamW(parameters,lr=cfg["learning_rate"],betas=(0.9,0.999),weight_decay=cfg["weight_decay"])
    trainids=indices_for(records,split,"train")
    validids=indices_for(records,split,"val")
    if len(trainids)<2: raise ValueError("Too few train samples")
    steps=math.ceil(len(trainids)/cfg["batch_size"])*cfg["epochs"]
    warm=max(1,round(steps*cfg["warmup_ratio"]))
    rng=np.random.default_rng(cfg["seed"])
    best=-1;step=0;history=[]
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    for epoch in range(cfg["epochs"]):
        model.train()
        order=rng.permutation(trainids).tolist()
        epochlogs=[]
        started=time.perf_counter()
        for a in range(0,len(order),cfg["batch_size"]):
            ids=order[a:a+cfg["batch_size"]]
            progress=max(0,(step-warm)/max(1,steps-warm-1))
            lr=cfg["learning_rate"]*(step+1)/warm if step<warm else cfg["min_learning_rate"]+0.5*(cfg["learning_rate"]-cfg["min_learning_rate"])*(1+math.cos(math.pi*min(progress,1)))
            for group in optimizer.param_groups: group["lr"]=lr
            optimizer.zero_grad(set_to_none=True)
            loss,stats=joint_batch(model,x,mask,records,channels,ids,cfg)
            if not torch.isfinite(loss): raise FloatingPointError("Nonfinite loss")
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(parameters,1.0)
            if not torch.isfinite(norm): raise FloatingPointError("Nonfinite gradient norm")
            optimizer.step();step+=1;epochlogs.append(stats)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        training_seconds=time.perf_counter()-started
        model.eval()
        # Model selection uses free-running inference, never teacher-forced test summaries.
        valrows,valseconds=predict(model,x,mask,records,channels,validids,batch_size=cfg["micro_batch_size"])
        metric=classification([r["label"] for r in valrows],[r["class"] for r in valrows])
        line={"epoch":epoch+1,"lr":lr,"train_seconds":training_seconds,"val_seconds":valseconds,
              "losses":{k:float(np.mean([r[k] for r in epochlogs])) for k in epochlogs[0]},"validation":metric}
        history.append(line);dump(out/"history.json",history)
        print(json.dumps({"epoch":epoch+1,"loss":line["losses"]["total"],"val_macro_f1":metric["macro_f1"]}),flush=True)
        if metric["macro_f1"]>best:
            best=metric["macro_f1"];checkpoint(model,cfg,out/"best.pt")
            dump(out/"selection.json",{"best_epoch":epoch+1,"val_macro_f1":best,"criterion":"maximum validation macro F1; first tie retained"})
    checkpoint(model,cfg,out/"last.pt")
    dump(out/"checkpoint_hashes.json",{p.name:sha256(p) for p in (out/"best.pt",out/"last.pt")})
    dump(out/"efficiency.json",{"peak_training_memory_bytes":torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
         "device":cfg["device"],"epoch_train_seconds":[r["train_seconds"] for r in history],
         "includes_validation":False,"quantization":cfg.get("quantization","none"),
         "note":"Measured here; not copied from paper. Includes differentiable ODE unrolling."})


def infer(data,split_path,checkpoint_path,out,subset="test",dropout="none",missing_rate=0.25,dropout_seed=42):
    x,mask,records,channels=load_dataset(data)
    model,cfg=restore(checkpoint_path)
    if len(channels)!=cfg["channels"]: raise ValueError("Channel count mismatch")
    split=read(split_path)
    audit_split(records,split,allow_overlap=split["protocol"]=="random")
    run=Path(checkpoint_path).parent
    original=read(run/"manifest.json")
    if original["data_sha256"]!=sha256(data) or original["split_sha256"]!=sha256(split_path):
        raise ValueError("Evaluation data or split differs from training manifest")
    for name,digest in original["files"].items():
        if not (ROOT/name).exists() or sha256(ROOT/name)!=digest:
            raise ValueError(f"Source/prompt/model-lock changed since training: {name}")
    if cfg.get("ts_checkpoint") and sha256(cfg["ts_checkpoint"])!=original["ts_checkpoint_sha256"]:
        raise ValueError("Frozen TS checkpoint changed")
    indices=indices_for(records,split,subset)
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    rows,seconds=predict(model,x,mask,records,channels,indices,dropout,cfg["micro_batch_size"],missing_rate,dropout_seed)
    Path(out).parent.mkdir(parents=True,exist_ok=True)
    with open(out,"w",encoding="utf-8") as f:
        for row in rows: f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+"\n")
    dump(str(out)+".provenance.json",{"data_sha256":sha256(data),"split_sha256":sha256(split_path),
        "checkpoint_sha256":sha256(checkpoint_path),"predictions_sha256":sha256(out),"dropout":dropout,
        "subset":subset,"missing_rate":missing_rate,"dropout_seed":dropout_seed,"seconds":seconds,"samples_per_second":len(rows)/seconds,
        "peak_inference_memory_bytes":torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
        "backend":cfg["backend"],"status":cfg["status"]})
    return rows
