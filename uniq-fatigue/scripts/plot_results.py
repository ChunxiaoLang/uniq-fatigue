import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from uniq_fatigue.audit import read

def main():
    p=argparse.ArgumentParser();p.add_argument("--run",required=True);p.add_argument("--grid");p.add_argument("--out",required=True)
    args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    history=read(Path(args.run)/"history.json")
    fig,ax=plt.subplots(figsize=(6,4))
    ax.plot([r["epoch"] for r in history],[100*r["validation"]["macro_f1"] for r in history],marker="o")
    ax.set(xlabel="Epoch",ylabel="Validation macro F1 (%)",title="Measured validation convergence")
    ax.grid(alpha=.25);fig.tight_layout();fig.savefig(out/"convergence.png",dpi=200);plt.close(fig)
    if args.grid:
        fig,ax=plt.subplots(figsize=(6,4))
        for modality in ("sensor_partial","text_partial"):
            paths=sorted(Path(args.grid).glob(modality+"_*.metrics.json"))
            x=[int(p.name.split("_")[-1].split(".")[0]) for p in paths]
            y=[100*read(p)["macro_f1"] for p in paths]
            ax.plot(x,y,marker="o",label=modality)
        ax.set(xlabel="Missing rate (%)",ylabel="Validation macro F1 (%)",title="Measured robustness; independent missingness")
        ax.legend();ax.grid(alpha=.25);fig.tight_layout();fig.savefig(out/"robustness.png",dpi=200);plt.close(fig)

if __name__=="__main__":main()
