import unittest
from pathlib import Path
import torch
from uniq_fatigue.audit import read,seed_all
from uniq_fatigue.model import UniQFatigue,ConditionalFlow,LoRALinear
from uniq_fatigue.losses import objective,alignment_losses

ROOT=Path(__file__).resolve().parents[1]


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.cfg=read(ROOT/"configs/smoke.json")
        self.cfg["text_max_length"]=24;self.cfg["report_max_length"]=16;self.cfg["max_new_tokens"]=4
        seed_all(42,2)

    def test_lora_starts_zero(self):
        m=LoRALinear(8,2,4,0)
        x=torch.randn(2,3,8)
        torch.testing.assert_close(m(x),m.base(x))

    def test_flow_deterministic_and_all_missing(self):
        m=ConditionalFlow(self.cfg).eval()
        x=torch.randn(2,64,3);mask=torch.ones_like(x,dtype=torch.bool)
        torch.testing.assert_close(m(x,mask)[0],m(x,mask)[0],rtol=0,atol=0)
        h,_=m(x,torch.zeros_like(mask))
        self.assertTrue(torch.isfinite(h).all())
        torch.testing.assert_close(h,m.null[None].expand(2,-1))

    def test_joint_backward_reaches_all_modules(self):
        model=UniQFatigue(self.cfg,ROOT).train()
        x=torch.randn(3,64,3);mask=torch.ones_like(x,dtype=torch.bool)
        refs=[{k:"Observed pattern." for k in ("physio","kine","psych")} for _ in range(3)]
        y=torch.tensor([0,1,2])
        out=model(x,mask,["context"]*3,y,refs,["corrupt"]*3)
        loss,_=objective(out,y,self.cfg);loss.backward()
        self.assertTrue(torch.isfinite(loss))
        for module in (model.flow,model.bridge,model.language,model.nan,model.text_project):
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in module.parameters()))
        torch.testing.assert_close(out["weights"].sum(1),torch.ones(3))
        torch.testing.assert_close(out["probabilities"].sum(1),torch.ones(3))

    def test_free_running_inference(self):
        model=UniQFatigue(self.cfg,ROOT).eval()
        x=torch.randn(2,64,3);mask=torch.ones_like(x,dtype=torch.bool)
        rows=model.infer(x,mask,["observations"]*2)
        self.assertEqual(len(rows),2)
        for r in rows:
            self.assertEqual(len(r["experts"]),3)
            self.assertIn(r["class"],(0,1,2))
            self.assertAlmostEqual(sum(r["weights"]),1,places=5)

    def test_ablation_paths(self):
        for ablation in ("A","B","C","D","E","F","G"):
            cfg=dict(self.cfg,ablation=ablation)
            model=UniQFatigue(cfg,ROOT).train()
            x=torch.randn(2,64,3);mask=torch.ones_like(x,dtype=torch.bool)
            refs=[{k:"Reference." for k in ("physio","kine","psych")}]*2
            out=model(x,mask,["log"]*2,torch.tensor([0,1]),refs,["bad"]*2)
            loss,_=objective(out,torch.tensor([0,1]),cfg)
            self.assertTrue(torch.isfinite(loss))
            if ablation=="A":self.assertEqual(float(out["fm"]),0)
            if ablation=="E":torch.testing.assert_close(out["weights"],torch.full((2,3),1/3))
            if ablation=="G":self.assertEqual(out["weights"].shape[1],1)

    def test_refuse_frozen_random_backbone(self):
        with self.assertRaises(ValueError):UniQFatigue(dict(self.cfg,freeze_ts=True),ROOT)

    def test_missing_mask_is_batch_partition_independent(self):
        import numpy as np
        from uniq_fatigue.runner import tensor_batch
        x=np.ones((3,64,3),dtype=np.float32);m=np.ones_like(x,dtype=bool)
        records=[{"id":str(i),"text":"self report"} for i in range(3)]
        a=tensor_batch(x,m,records,["a","b","c"],[0,1,2],"cpu","dual25",dropout_seed=123)
        b=tensor_batch(x,m,records,["a","b","c"],[1],"cpu","dual25",dropout_seed=123)
        torch.testing.assert_close(a[1][1],b[1][0]);self.assertEqual(a[2][1],b[2][0])


if __name__=="__main__":unittest.main()
