import argparse
from pathlib import Path
import numpy as np
import torch
from uniq_fatigue.audit import read
from uniq_fatigue.data import load_dataset
from uniq_fatigue.runner import restore,tensor_batch,indices_for

def main():
    p=argparse.ArgumentParser()
    for k in ("data","split","checkpoint","out"):p.add_argument("--"+k,required=True)
    args=p.parse_args();model,cfg=restore(args.checkpoint)
    x,m,records,channels=load_dataset(args.data);ids=indices_for(records,read(args.split),"val")
    exs,ets,hfs=[],[],[]
    with torch.no_grad():
        for i in ids:
            xx,mm,texts=tensor_batch(x,m,records,channels,[i],cfg["device"])
            ex,et,hf,_,_=model.features(xx,mm,texts)
            exs.append(ex.cpu().numpy());ets.append(et.cpu().numpy());hfs.append(hf.cpu().numpy())
    Path(args.out).parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.out,ex=np.concatenate(exs),et=np.concatenate(ets),hf=np.concatenate(hfs),
                        labels=np.array([records[i]["label"] for i in ids]),ids=np.array([records[i]["id"] for i in ids]))

if __name__=="__main__":main()
