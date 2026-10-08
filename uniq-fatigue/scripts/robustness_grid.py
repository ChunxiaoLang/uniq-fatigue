import argparse
import subprocess
import sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    for k in ("data","split","checkpoint","out"):p.add_argument("--"+k,required=True)
    p.add_argument("--seed",type=int,default=42)
    args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    for modality in ("sensor_partial","text_partial"):
        for rate in range(0,101,10):
            name=f"{modality}_{rate:03d}"
            pred=str(out/(name+".jsonl"))
            subprocess.run([sys.executable,"-m","uniq_fatigue","infer","--data",args.data,"--split",args.split,
                "--checkpoint",args.checkpoint,"--subset","val","--dropout",modality,"--missing-rate",str(rate/100),
                "--dropout-seed",str(args.seed),"--out",pred],check=True)
            subprocess.run([sys.executable,"-m","uniq_fatigue","evaluate","--data",args.data,"--predictions",pred,
                "--out",str(out/(name+".metrics.json"))],check=True)

if __name__=="__main__":main()
