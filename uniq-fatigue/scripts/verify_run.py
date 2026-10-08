import argparse
from pathlib import Path
from uniq_fatigue.audit import read,sha256

def main():
    p=argparse.ArgumentParser()
    for k in ("run","data","split"):p.add_argument("--"+k,required=True)
    args=p.parse_args();root=Path(__file__).resolve().parents[1];run=Path(args.run)
    m=read(run/"manifest.json")
    assert sha256(args.data)==m["data_sha256"],"Data changed"
    assert sha256(args.split)==m["split_sha256"],"Split changed"
    for name,digest in m["files"].items():
        assert sha256(root/name)==digest,f"Source/prompt/config/model version changed: {name}"
    for name,digest in read(run/"checkpoint_hashes.json").items():
        assert sha256(run/name)==digest,f"Checkpoint changed: {name}"
    for path in run.glob("*.jsonl.provenance.json"):
        pred=Path(str(path)[:-len(".provenance.json")]);meta=read(path)
        assert sha256(pred)==meta["predictions_sha256"],f"Predictions changed: {pred}"
        assert meta["checkpoint_sha256"] in read(run/"checkpoint_hashes.json").values(),"Unknown prediction checkpoint"
    print("All recorded source, data, split, checkpoint and prediction hashes match.")

if __name__=="__main__":main()
