import argparse
from pathlib import Path
import numpy as np
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser();p.add_argument("--embeddings",required=True);p.add_argument("--out",required=True)
    p.add_argument("--seed",type=int,default=42);args=p.parse_args()
    with np.load(args.embeddings,allow_pickle=False) as d:
        ex,et,labels=d["ex"],d["et"],d["labels"]
    joined=np.concatenate([ex,et]);n=len(ex)
    if len(joined)<4:raise ValueError("Need at least four embeddings")
    z=TSNE(n_components=2,perplexity=min(30,len(joined)-1),random_state=args.seed,init="pca",learning_rate="auto").fit_transform(joined)
    fig,ax=plt.subplots(figsize=(6,5))
    for c in range(3):
        valid=labels==c
        ax.scatter(z[:n][valid,0],z[:n][valid,1],marker="o",label=f"Sensor class {c}",alpha=.7)
        ax.scatter(z[n:][valid,0],z[n:][valid,1],marker="x",label=f"Text class {c}",alpha=.7)
    ax.set(title="Measured modality embeddings; t-SNE is not proof of PID",xlabel="t-SNE 1",ylabel="t-SNE 2")
    ax.legend(fontsize=8);fig.tight_layout();Path(args.out).parent.mkdir(parents=True,exist_ok=True);fig.savefig(args.out,dpi=200)

if __name__=="__main__":main()
