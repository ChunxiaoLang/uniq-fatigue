import argparse
from pathlib import Path
import torch
import numpy as np
from uniq_fatigue.audit import read,dump,seed_all,sha256
from uniq_fatigue.data import load_dataset
from uniq_fatigue.model import ConditionalFlow
from uniq_fatigue.splits import audit_split

def main():
    p=argparse.ArgumentParser()
    for k in ("data","split","config","out"):p.add_argument("--"+k,required=True)
    p.add_argument("--epochs",type=int,default=10)
    args=p.parse_args()
    cfg=read(args.config);cfg["ablation"]="full"
    seed_all(cfg["seed"],cfg["num_threads"])
    x,m,records,channels=load_dataset(args.data)
    split=read(args.split);audit_split(records,split,split["protocol"]=="random")
    if cfg["channels"]!=len(channels):raise ValueError("Channel count mismatch")
    lookup={r["id"]:i for i,r in enumerate(records)}
    ids=[lookup[i] for i in split["train"]]
    model=ConditionalFlow(cfg).to(cfg["device"]).train()
    # Keep the projected target coordinate system fixed during standalone pretraining
    # to prevent learning a trivially collapsed projection. This is an extra assumption.
    for name,param in model.named_parameters():
        if name.startswith("patch.") or name in ("channel","gamma","beta","null"):
            param.requires_grad=False
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=cfg["learning_rate"],weight_decay=cfg["weight_decay"])
    rng=np.random.default_rng(cfg["seed"]);history=[]
    for epoch in range(args.epochs):
        shuffled=rng.permutation(ids);losses=[]
        for a in range(0,len(ids),cfg["batch_size"]):
            ix=shuffled[a:a+cfg["batch_size"]]
            xx=torch.from_numpy(x[ix]).to(cfg["device"]);mm=torch.from_numpy(m[ix]).to(cfg["device"])
            cond,valid=model.encode(xx,mm)
            target,_=model.encode(model.smooth(xx,mm),mm)
            prior=torch.randn_like(target)*valid[...,None]
            t=torch.rand(len(xx),device=cfg["device"])
            z=prior*(1-t[:,None,None,None])+target.detach()*t[:,None,None,None]
            error=(model.velocity(z,t,cond,valid)-(target.detach()-prior)).square()
            loss=(error*valid[...,None]).sum()/(valid.sum()*error.shape[-1]).clamp_min(1)
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
            losses.append(float(loss.detach()))
        history.append({"epoch":epoch+1,"flow_loss":float(np.mean(losses))})
        print(history[-1],flush=True)
    path=Path(args.out)
    if path.exists():raise ValueError("Output already exists")
    path.parent.mkdir(parents=True,exist_ok=True)
    torch.save(model.cpu().state_dict(),path)
    dump(str(path)+".json",{"config":cfg,"data_sha256":sha256(args.data),"split_sha256":sha256(args.split),
        "checkpoint_sha256":sha256(path),"epochs":args.epochs,"history":history,
        "status":"independent scratch CFM pretraining, not official UniTS weights"})

if __name__=="__main__":main()
