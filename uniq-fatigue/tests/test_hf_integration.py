import unittest
from unittest.mock import patch
from pathlib import Path
import torch
from transformers import Qwen2Config,Qwen2ForCausalLM
from uniq_fatigue.audit import read,seed_all
from uniq_fatigue.model import UniQFatigue
from uniq_fatigue.losses import objective

ROOT=Path(__file__).resolve().parents[1]


class TestTokenizer:
    eos_token="EOS";eos_token_id=2;pad_token_id=2
    def encode(self,text,add_special_tokens=False):return [3+b%253 for b in text.encode()]
    def apply_chat_template(self,messages,tokenize=True,add_generation_prompt=True,return_tensors=None):
        ids=[1]+self.encode(messages[0]["content"])[:20]+[1]
        return torch.tensor([ids]) if return_tensors=="pt" else ids
    def __call__(self,texts,padding=True,truncation=True,max_length=24,return_tensors="pt"):
        rows=[([1]+self.encode(t)+[2])[:max_length] for t in texts]
        n=max(map(len,rows));ids=[];mask=[]
        for row in rows:
            ids.append(row+[2]*(n-len(row)));mask.append([1]*len(row)+[0]*(n-len(row)))
        return {"input_ids":torch.tensor(ids),"attention_mask":torch.tensor(mask)}
    def decode(self,ids,skip_special_tokens=True):return " ".join(str(int(i)) for i in ids if int(i)>2)


class HFIntegrationTests(unittest.TestCase):
    def test_real_qwen_lora_prefix_teacher_forcing_and_generation(self):
        seed_all(42,2)
        cfg=read(ROOT/"configs/smoke.json")
        cfg.update(backend="hf",model_id="Qwen/Qwen2.5-14B-Instruct",model_lock="models.lock.json",
                   text_max_length=24,report_max_length=8,max_new_tokens=3)
        config=Qwen2Config(vocab_size=256,hidden_size=32,intermediate_size=64,num_hidden_layers=1,
               num_attention_heads=4,num_key_value_heads=2,bos_token_id=1,eos_token_id=2,pad_token_id=2)
        backbone=Qwen2ForCausalLM(config)
        with patch("transformers.AutoTokenizer.from_pretrained",return_value=TestTokenizer()),patch("transformers.AutoModelForCausalLM.from_pretrained",return_value=backbone):
            model=UniQFatigue(cfg,ROOT).train()
        x=torch.randn(2,64,3);m=torch.ones_like(x,dtype=torch.bool);y=torch.tensor([0,1])
        targets=[{k:"ref" for k in ("physio","kine","psych")}] * 2
        out=model(x,m,["short","a longer log"],y,targets,["bad","bad"])
        loss,_=objective(out,y,cfg);loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(model.bridge.weight.grad.abs().sum()),0)
        lora=[p for n,p in model.language.named_parameters() if "lora_B" in n]
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in lora))
        frozen=[p for n,p in model.language.named_parameters() if "lora_" not in n]
        self.assertTrue(all(not p.requires_grad for p in frozen))
        model.eval()
        # Avoid synthesis tokenizer tensor branch in this focused integration test;
        # exercise the actual three expert generate + final generated-token re-forward.
        with torch.no_grad():
            _,_,hf,_,_=model.features(x,m,["short","a longer log"])
            prob,local,weights,_,reports=model.expert_forward(hf,["short","a longer log"])
        self.assertEqual(prob.shape,(2,3));self.assertEqual(local.shape,(2,3,3))
        self.assertEqual(len(reports),3)
        torch.testing.assert_close(weights.sum(1),torch.ones(2))
        rows=model.infer(x,m,["short","a longer log"])
        self.assertEqual(len(rows),2)
        self.assertTrue(all(isinstance(r["report"],str) for r in rows))


if __name__=="__main__":unittest.main()
