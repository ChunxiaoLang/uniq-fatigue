import argparse
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from uniq_fatigue.audit import read,dump,sha256,seed_all

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True);p.add_argument("--out",required=True)
    p.add_argument("--model",default="Qwen/Qwen2.5-14B-Instruct")
    p.add_argument("--lock",default="models.lock.json");p.add_argument("--device",default="cuda")
    p.add_argument("--seed",type=int,default=42)
    args=p.parse_args()
    if Path(args.out).exists():raise ValueError("Output exists; preserve cached logs")
    seed_all(args.seed)
    fields=read(args.input)
    allowed={"panas_positive","panas_negative","panas_items","age","sex","sleep_quality","soreness","mood","rpe","ambient_temperature"}
    for entry in fields.values():
        if not entry or not set(entry)<=allowed:raise ValueError("Unsupported fields: no experimental state, target label or fatigue rating allowed")
    sha=read(args.lock)["models"][args.model]["revision"]
    if len(sha)!=40:raise ValueError("Expected resolved model SHA")
    tokenizer=AutoTokenizer.from_pretrained(args.model,revision=sha)
    model=AutoModelForCausalLM.from_pretrained(args.model,revision=sha,torch_dtype=torch.bfloat16 if args.device=="cuda" else torch.float32,attn_implementation="eager").to(args.device).eval()
    prompt=Path("prompts/reconstruction.txt").read_text(encoding="utf-8")
    outputs={}
    with torch.no_grad():
        for key,entry in sorted(fields.items()):
            # The phase key is never included in the LLM prompt.
            ids=tokenizer.apply_chat_template([{"role":"user","content":prompt+"\n"+json.dumps(entry,ensure_ascii=False)}],tokenize=True,add_generation_prompt=True,return_tensors="pt").to(args.device)
            out=model.generate(ids,attention_mask=torch.ones_like(ids),do_sample=False,max_new_tokens=192,pad_token_id=tokenizer.eos_token_id)
            outputs[key]={"text":tokenizer.decode(out[0,len(ids[0]):],skip_special_tokens=True),"source":"model_generated",
                          "generation":{"model":args.model,"revision":sha,"seed":args.seed,"do_sample":False,
                                        "max_new_tokens":192,"prompt_sha256":sha256("prompts/reconstruction.txt")}}
    dump(args.out,outputs)
    dump(args.out+".provenance.json",{"input_sha256":sha256(args.input),"output_sha256":sha256(args.out),"model":args.model,"revision":sha})

if __name__=="__main__":main()
