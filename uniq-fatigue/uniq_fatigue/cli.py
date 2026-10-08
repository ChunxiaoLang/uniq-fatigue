import argparse
import csv
import json
import random
import subprocess
import sys
from pathlib import Path
from .audit import read,dump,sha256
from .data import load_dataset,context_text
from .splits import make_split,audit_split


def jsonlines(path):
    with open(path,encoding="utf-8") as f: return [json.loads(s) for s in f if s.strip()]


def main(argv=None):
    parser=argparse.ArgumentParser(description="UniQ-Fatigue independent reconstruction")
    sub=parser.add_subparsers(dest="cmd",required=True)
    p=sub.add_parser("synthetic");p.add_argument("--out",required=True);p.add_argument("--seed",type=int,default=42)
    p=sub.add_parser("prepare-wesad")
    for name in ("raw","out"): p.add_argument("--"+name,required=True)
    p.add_argument("--logs");p.add_argument("--window",type=int,default=100);p.add_argument("--stride",type=int,default=50)
    p.add_argument("--antialias",action="store_true");p.add_argument("--trust-official-pickle",action="store_true")
    p=sub.add_parser("prepare-pmdata")
    for name in ("raw","metadata","out"):p.add_argument("--"+name,required=True)
    p.add_argument("--scale",choices=["native5","paper10"],default="native5");p.add_argument("--label-map")
    p.add_argument("--length",type=int,default=100);p.add_argument("--max-gap",type=float,default=300)
    p=sub.add_parser("split");p.add_argument("--data",required=True);p.add_argument("--out",required=True)
    p.add_argument("--protocol",choices=["subject","chronological","random","loso"],default="subject")
    p.add_argument("--seed",type=int,default=42);p.add_argument("--held-out")
    p=sub.add_parser("audit-data");p.add_argument("--data",required=True);p.add_argument("--split")
    p=sub.add_parser("train")
    for name in ("data","split","config","out"):p.add_argument("--"+name,required=True)
    p.add_argument("--seed",type=int);p.add_argument("--ablation",choices=["full","A","B","C","D","E","F","G"])
    p=sub.add_parser("infer")
    for name in ("data","split","checkpoint","out"):p.add_argument("--"+name,required=True)
    p.add_argument("--subset",choices=["train","val","test"],default="test");p.add_argument("--dropout",choices=["none","sensor","text","dual25","sensor_partial","text_partial"],default="none")
    p.add_argument("--missing-rate",type=float,default=0.25);p.add_argument("--dropout-seed",type=int,default=42)
    p=sub.add_parser("evaluate")
    for name in ("data","predictions","out"):p.add_argument("--"+name,required=True)
    p.add_argument("--text",action="store_true");p.add_argument("--language",choices=["en","zh"],default="en")
    p.add_argument("--bert-model",default="FacebookAI/roberta-large");p.add_argument("--model-lock",default="models.lock.json")
    p=sub.add_parser("lock-models");p.add_argument("--out",default="models.lock.json")
    p.add_argument("--models",nargs="+",default=["Qwen/Qwen2.5-14B-Instruct","FacebookAI/roberta-large","hfl/chinese-roberta-wwm-ext"])
    p.add_argument("--revision",default="main");p.add_argument("--download",action="store_true")
    p=sub.add_parser("suite")
    for name in ("data","config","out"):p.add_argument("--"+name,required=True)
    p.add_argument("--seeds",type=int,nargs="+",default=[42,2024,3407]);p.add_argument("--loso",action="store_true")
    p.add_argument("--ablations",nargs="+",choices=["full","A","B","C","D","E","F","G"],default=["full"])
    p.add_argument("--protocol",choices=["subject","random","chronological"],default="subject")
    p=sub.add_parser("aggregate");p.add_argument("--runs",required=True);p.add_argument("--out",required=True)
    p=sub.add_parser("blind-export");p.add_argument("--data",required=True);p.add_argument("--predictions",nargs="+",required=True);p.add_argument("--out",required=True);p.add_argument("--seed",type=int,default=42)
    p=sub.add_parser("score-panel");p.add_argument("--scores",required=True);p.add_argument("--key",required=True);p.add_argument("--out",required=True)
    args=parser.parse_args(argv)
    if args.cmd=="synthetic":
        from .prepare import synthetic
        synthetic(args.out,args.seed)
    elif args.cmd=="prepare-wesad":
        from .prepare import wesad
        wesad(args.raw,args.out,args.logs,args.window,args.stride,args.antialias,args.trust_official_pickle)
    elif args.cmd=="prepare-pmdata":
        from .prepare import pmdata
        pmdata(args.raw,args.metadata,args.out,args.scale,args.label_map,args.length,args.max_gap)
    elif args.cmd=="split":
        _,_,records,_=load_dataset(args.data)
        split=make_split(records,args.protocol,args.seed,args.held_out)
        split["data_sha256"]=sha256(args.data)
        dump(args.out,split)
    elif args.cmd=="audit-data":
        x,m,records,channels=load_dataset(args.data)
        result={"samples":len(x),"shape":list(x.shape),"channels":channels,
                "subjects":sorted({r["subject"] for r in records}),"missing_fraction":float(1-m.mean()),
                "reports_missing":sum(not r.get("reports") for r in records),"sha256":sha256(args.data)}
        if args.split:
            s=read(args.split);result["split_audit"]=audit_split(records,s,s["protocol"]=="random")
        print(json.dumps(result,ensure_ascii=False,indent=2))
    elif args.cmd=="train":
        from .runner import train
        train(args.data,args.split,args.config,args.out,args.seed,args.ablation)
    elif args.cmd=="infer":
        from .runner import infer
        infer(args.data,args.split,args.checkpoint,args.out,args.subset,args.dropout,args.missing_rate,args.dropout_seed)
    elif args.cmd=="evaluate":
        from .metrics import classification,bootstrap_subjects,text_metrics
        rows=jsonlines(args.predictions)
        _,_,records,_=load_dataset(args.data)
        lookup={r["id"]:r for r in records}
        if len({r["id"] for r in rows})!=len(rows):raise ValueError("Duplicate prediction IDs")
        for r in rows:
            if r["id"] not in lookup or r["label"]!=lookup[r["id"]]["label"] or r["subject"]!=lookup[r["id"]]["subject"]:
                raise ValueError("Prediction metadata does not match ground truth")
        result=classification([r["label"] for r in rows],[r["class"] for r in rows])
        result.update(bootstrap_subjects(rows))
        result.update({"data_sha256":sha256(args.data),"predictions_sha256":sha256(args.predictions),
                       "scope":"new measured result, not a paper-verified number"})
        if args.text:
            lock=read(args.model_lock)
            result["text"]=text_metrics(rows,records,args.language,args.bert_model,lock["models"][args.bert_model]["revision"])
        dump(args.out,result)
    elif args.cmd=="lock-models":
        from huggingface_hub import HfApi,snapshot_download
        if Path(args.out).exists():raise ValueError("Model lock already exists; choose a new path to prevent silently changing versions")
        models={}
        for name in args.models:
            info=HfApi().model_info(name,revision=args.revision)
            entry={"revision":info.sha,"requested_revision":args.revision}
            if args.download:
                local=snapshot_download(name,revision=info.sha)
                entry["snapshot_sha256"]={str(p.relative_to(local)):sha256(p) for p in Path(local).rglob("*") if p.is_file()}
            models[name]=entry
        dump(args.out,{"models":models,"status":"resolved for this reconstruction; authors' model revisions unavailable"})
    elif args.cmd=="suite":
        root=Path(args.out);root.mkdir(parents=True,exist_ok=True)
        _,_,records,_=load_dataset(args.data)
        heldouts=sorted({r["subject"] for r in records}) if args.loso else [None]
        for seed in args.seeds:
            for subject in heldouts:
                tag=f"seed{seed}"+(f"_held{subject}" if subject else "")
                splitpath=root/"splits"/(tag+".json")
                split=make_split(records,"loso" if args.loso else args.protocol,seed,subject)
                split["data_sha256"]=sha256(args.data);dump(splitpath,split)
                for ablation in args.ablations:
                    run=root/(tag+"_"+ablation)
                    commands=[["train","--data",args.data,"--split",str(splitpath),"--config",args.config,"--out",str(run),"--seed",str(seed),"--ablation",ablation]]
                    for dropout in ("none","sensor","text","dual25"):
                        prediction=str(run/(dropout+".jsonl"))
                        commands.extend([["infer","--data",args.data,"--split",str(splitpath),"--checkpoint",str(run/"best.pt"),"--out",prediction,"--dropout",dropout],
                                         ["evaluate","--data",args.data,"--predictions",prediction,"--out",str(run/(dropout+".metrics.json"))]])
                    for cmd in commands:subprocess.run([sys.executable,"-m","uniq_fatigue"]+cmd,check=True)
    elif args.cmd=="aggregate":
        import numpy as np
        grouped={}
        for p in sorted(Path(args.runs).rglob("*.metrics.json")):
            run=read(p.parent/"config.json");split=read(p.parent/"split.json");m=read(p)
            key=f"{split['protocol']}:{run['ablation']}:{p.name}"
            grouped.setdefault(key,[]).append({"path":str(p),"seed":run["seed"],"metric":m})
        # LOSO folds averaged inside each seed first; seed std is never mixed with fold std.
        results={}
        for key,entries in grouped.items():
            byseed={}
            for e in entries:byseed.setdefault(e["seed"],[]).append(e["metric"])
            res={"runs":len(entries),"seeds":len(byseed),"per_seed":{}}
            for metric in ("accuracy","macro_f1","kappa"):
                means=[]
                for seed,items in byseed.items():
                    values=[m[metric] for m in items if m[metric] is not None]
                    if not values:continue
                    mean=float(np.mean(values));means.append(mean)
                    res["per_seed"].setdefault(str(seed),{})[metric]={"mean_over_runs":mean,"runs":len(values),"fold_std":float(np.std(values,ddof=1)) if len(values)>1 else None}
                res[metric]={"mean_over_seeds":float(np.mean(means)) if means else None,"seed_std":float(np.std(means,ddof=1)) if len(means)>1 else None}
            results[key]=res
        dump(args.out,results)
    elif args.cmd=="blind-export":
        x,mask,records,channels=load_dataset(args.data)
        lookup={r["id"]:(i,r) for i,r in enumerate(records)};items=[]
        for path in args.predictions:
            for row in jsonlines(path):items.append((str(Path(path)),row))
        random.Random(args.seed).shuffle(items)
        out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
        anonymous=[];key={}
        for i,(source,row) in enumerate(items):
            token=f"R{i:06d}"
            key[token]={"source":source,"sample_id":row["id"]}
            index,r=lookup[row["id"]]
            anonymous.append({"anonymous_id":token,"observed_evidence":context_text(x[index],mask[index],r,channels),
                              "report":row["report"],"note":"Latent model claims are not ground truth; no target labels provided to judges."})
        dump(out/"blind_reports.json",anonymous);dump(out/"private_key.json",key)
        with (out/"scores.csv").open("w",newline="",encoding="utf-8") as f:
            writer=csv.writer(f);writer.writerow(["anonymous_id","judge_id","judge_kind","model_version","metabolic","kinetics","action"])
            for r in anonymous:writer.writerow([r["anonymous_id"],"","human_or_llm","","","",""])
    elif args.cmd=="score-panel":
        import numpy as np
        key=read(args.key);groups={}
        with open(args.scores,encoding="utf-8-sig",newline="") as f:rows=list(csv.DictReader(f))
        seen=set()
        for r in rows:
            if not r["judge_id"] or r["judge_kind"] not in ("human","llm"):raise ValueError("Missing judge metadata")
            if r["judge_kind"]=="llm" and not r["model_version"]:raise ValueError("LLM judge needs a pinned model version")
            pair=(r["anonymous_id"],r["judge_id"])
            if pair in seen:raise ValueError("Duplicate judge/report score")
            seen.add(pair)
            values=[float(r[k]) for k in ("metabolic","kinetics","action")]
            if any(v<1 or v>5 for v in values):raise ValueError("Scores must be 1-5")
            source=key[r["anonymous_id"]]["source"]
            groups.setdefault(source+":"+r["judge_kind"],[]).append(values)
        result={k:{"ratings":len(v),"dimension_means":np.mean(v,axis=0).tolist(),"mean":float(np.mean(v))} for k,v in groups.items()}
        dump(args.out,{"scores":result,"scores_sha256":sha256(args.scores),"warning":"Descriptive means; not proof of complete blinding or inter-rater agreement"})


if __name__=="__main__":main()
