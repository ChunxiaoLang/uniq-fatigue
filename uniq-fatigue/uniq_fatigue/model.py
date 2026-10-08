import math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from .audit import read


class LoRALinear(nn.Module):
    def __init__(self, width, rank, alpha, dropout):
        super().__init__()
        self.base = nn.Linear(width, width)
        self.a = nn.Parameter(torch.randn(rank, width) * 0.02)
        self.b = nn.Parameter(torch.zeros(width, rank))
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(self.dropout(x), self.a), self.b) * self.scale


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg["d_model"]
        self.heads = cfg["heads"]
        if d % self.heads:
            raise ValueError("d_model must be divisible by heads")
        self.q = LoRALinear(d, cfg["ts_rank"], cfg["ts_alpha"], cfg["lora_dropout"])
        self.k = nn.Linear(d, d)
        self.v = LoRALinear(d, cfg["ts_rank"], cfg["ts_alpha"], cfg["lora_dropout"])
        self.o = nn.Linear(d, d)

    def forward(self, x, valid):
        b, n, d = x.shape
        def reshape(z):
            return z.reshape(b, n, self.heads, d//self.heads).transpose(1, 2)
        q, k, v = map(reshape, (self.q(x), self.k(x), self.v(x)))
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(d//self.heads)
        scores = scores.masked_fill(~valid[:, None, None, :], -1e4)
        # Fully missing streams yield zero values, not NaNs.
        attn = scores.softmax(-1)
        v = v * valid[:, None, :, None]
        y = (attn @ v).transpose(1, 2).reshape(b, n, d)
        return self.o(y) * valid[:, :, None]


class FlowBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg["d_model"]
        self.spatial = Attention(cfg)
        self.temporal = Attention(cfg)
        self.ln_s, self.ln_t, self.ln_f = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.cond_norm = nn.GroupNorm(1, d)
        self.acor = nn.Linear(d, 2*d)
        self.time_mod = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, 2*d))
        self.ff = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))

    def forward(self, x, condition, time_embedding, valid):
        b, n, c, d = x.shape
        scale, shift = self.time_mod(time_embedding).chunk(2, -1)
        cond = self.cond_norm(condition.reshape(-1, d, 1)).reshape(b,n,c,d)
        cs, cb = self.acor(cond).chunk(2, -1)
        y = self.ln_s(x) * (1 + scale[:,None,None]) + shift[:,None,None]
        y = y * (1 + cs.tanh()) + cb
        x = x + self.spatial(y.reshape(b*n,c,d), valid.reshape(b*n,c)).reshape(b,n,c,d)
        y = self.ln_t(x).permute(0,2,1,3).reshape(b*c,n,d)
        vt = valid.permute(0,2,1).reshape(b*c,n)
        x = x + self.temporal(y,vt).reshape(b,c,n,d).permute(0,2,1,3)
        return (x+self.ff(self.ln_f(x))) * valid[...,None]


class ConditionalFlow(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d, c, p = cfg["d_model"], cfg["channels"], cfg["patch_length"]
        self.patch = nn.Linear(p, d)
        self.channel = nn.Parameter(torch.randn(c, d)*0.02)
        self.blocks = nn.ModuleList([FlowBlock(cfg) for _ in range(cfg["layers"])])
        self.output = nn.Linear(d,d)
        self.gamma = nn.Parameter(torch.ones(c))
        self.beta = nn.Parameter(torch.zeros(c))
        self.null = nn.Parameter(torch.zeros(d))

    def encode(self, x, mask):
        p, step = self.cfg["patch_length"], self.cfg["patch_stride"]
        if x.shape[1] < p:
            pad = p-x.shape[1]
            x, mask = F.pad(x,(0,0,0,pad)), F.pad(mask,(0,0,0,pad),value=False)
        # Normalize each channel inside each segment, as in the paper's Eq.8 prose.
        z = x.unfold(1,p,step)  # B,N,C,P
        m = mask.unfold(1,p,step)
        count = m.sum(-1,keepdim=True).clamp_min(1)
        mean = (z*m).sum(-1,keepdim=True)/count
        var = ((z-mean).square()*m).sum(-1,keepdim=True)/count
        norm = ((z-mean)/(var+1e-5).sqrt()) * self.gamma[None,None,:,None] + self.beta[None,None,:,None]
        h = self.patch(norm*m)+self.channel[None,None]
        valid = m.any(-1)
        # Fixed sinusoidal patch positions avoid discarding temporal order.
        n,d = h.shape[1],h.shape[-1]
        position = torch.arange(n,device=x.device)[:,None]
        frequencies = torch.exp(torch.arange(0,d,2,device=x.device)*(-math.log(10000)/d))
        pos = torch.zeros(n,d,device=x.device,dtype=h.dtype)
        pos[:,0::2],pos[:,1::2] = (position*frequencies).sin(), (position*frequencies[:d//2]).cos()
        return (h+pos[None,:,None])*valid[...,None],valid

    def smooth(self,x,mask):
        k = self.cfg["filter_size"]
        if k%2 != 1:
            raise ValueError("filter_size must be odd")
        m = mask.transpose(1,2).float()
        numerator = F.avg_pool1d((x*mask).transpose(1,2),k,1,k//2,count_include_pad=False)
        denominator = F.avg_pool1d(m,k,1,k//2,count_include_pad=False)
        return (numerator/denominator.clamp_min(1e-8)).transpose(1,2)

    def velocity(self,z,t,cond,valid):
        d = z.shape[-1]
        freqs = torch.exp(torch.arange(0,d,2,device=z.device)*(-math.log(10000)/d))
        times = t[:,None]*freqs[None]*1000
        emb = torch.cat([times.sin(),times.cos()],-1)[:,:d]
        h = z
        for block in self.blocks:
            h = block(h,cond,emb,valid)
        return self.output(h)*valid[...,None]

    def forward(self,x,mask):
        cond,valid = self.encode(x,mask)
        if self.cfg["ablation"] == "A":
            z = cond
            fm = x.new_zeros(())
        else:
            z = torch.zeros_like(cond)
            steps = self.cfg["ode_steps"]
            for s in range(steps):
                t = x.new_full((len(x),),s/steps)
                z = z+self.velocity(z,t,cond,valid)/steps
            fm = x.new_zeros(())
            if self.training:
                target,_ = self.encode(self.smooth(x,mask),mask)
                target = target.detach()  # explicit reconstruction choice, prevents target gradients
                prior = torch.randn_like(target)*valid[...,None]
                t = torch.rand(len(x),device=x.device)
                intermediate = prior*(1-t[:,None,None,None])+target*t[:,None,None,None]
                error = (self.velocity(intermediate,t,cond,valid)-(target-prior)).square()
                fm = (error*valid[...,None]).sum()/(valid.sum()*z.shape[-1]).clamp_min(1)
        pooled = (z*valid[...,None]).sum((1,2))/valid.sum((1,2)).clamp_min(1)[:,None]
        missing = ~valid.any((1,2))
        return torch.where(missing[:,None],self.null[None],pooled),fm


class TinyLanguage(nn.Module):
    """Byte GRU smoke backend; intentionally not claimed to be Qwen."""
    hidden = 48
    def __init__(self,cfg):
        super().__init__()
        self.cfg=cfg
        self.embedding=nn.Embedding(259,self.hidden,padding_idx=0)
        self.gru=nn.GRU(self.hidden,self.hidden,batch_first=True)
        self.output=nn.Linear(self.hidden,259)

    def tokens(self,text,max_length):
        return [1]+[b+3 for b in text.encode("utf-8")[:max_length-2]]+[2]

    def encode(self,texts):
        return torch.stack([self._last(self.tokens(t,self.cfg["text_max_length"])) for t in texts])

    def _last(self,ids,prefix=None):
        device=self.embedding.weight.device
        emb=self.embedding(torch.tensor(ids,device=device))[None]
        if prefix is not None:
            emb=torch.cat([prefix[None,None],emb],1)
        h,_=self.gru(emb)
        return h[0,-1]

    def expert(self,prefix,texts,targets=None):
        hs,losses,reports=[],[],[]
        for i,text in enumerate(texts):
            prompt=self.tokens(text,self.cfg["text_max_length"])
            if self.training and targets is not None:
                target=self.tokens(targets[i],self.cfg["report_max_length"])[1:]
                ids=prompt+target
                emb=self.embedding(torch.tensor(ids,device=prefix.device))[None]
                h,_=self.gru(torch.cat([prefix[i,None,None],emb],1))
                logits=self.output(h[0,len(prompt):len(prompt)+len(target)])
                losses.append(F.cross_entropy(logits,torch.tensor(target,device=prefix.device)))
                hs.append(h[0,-1]); reports.append(targets[i])
            else:
                ids=prompt[:]
                for _ in range(self.cfg["max_new_tokens"]):
                    h=self._last(ids,prefix[i])
                    token=int(self.output(h).argmax())
                    ids.append(token)
                    if token == 2: break
                hs.append(self._last(ids,prefix[i]))
                reports.append(bytes(t-3 for t in ids[len(prompt):] if 3<=t<=258).decode("utf-8",errors="replace"))
        loss=torch.stack(losses).mean() if losses else prefix.new_zeros(())
        return torch.stack(hs),loss,reports

    def synthesize(self,text):
        return "Synthetic smoke backend: calibrated probabilities and expert weights are recorded; byte-model text is not a validated fatigue assessment."


class HFLanguage(nn.Module):
    def __init__(self,cfg,root):
        super().__init__()
        from transformers import AutoTokenizer,AutoModelForCausalLM
        from peft import LoraConfig,get_peft_model
        lock=read(Path(root)/cfg["model_lock"])
        entry=lock["models"][cfg["model_id"]]
        revision=entry["revision"]
        if len(revision)!=40 or any(c not in "0123456789abcdef" for c in revision):
            raise ValueError("Model revision must be a resolved 40-character commit SHA")
        if cfg.get("quantization","none")!="none":
            raise ValueError("Quantization requires a separately audited implementation/config")
        self.cfg=cfg
        self.tokenizer=AutoTokenizer.from_pretrained(cfg["model_id"],revision=revision,trust_remote_code=False)
        self.tokenizer.pad_token=self.tokenizer.eos_token
        self.tokenizer.padding_side="right"
        dtype=getattr(torch,cfg["dtype"])
        base=AutoModelForCausalLM.from_pretrained(cfg["model_id"],revision=revision,
                  torch_dtype=dtype,attn_implementation="eager",trust_remote_code=False)
        self.hidden=base.config.hidden_size
        self.llm=get_peft_model(base,LoraConfig(r=cfg["llm_rank"],lora_alpha=cfg["llm_alpha"],
                    lora_dropout=cfg["lora_dropout"],target_modules=["q_proj","v_proj"],task_type="CAUSAL_LM"))
        if cfg["gradient_checkpointing"]:
            self.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
            self.llm.enable_input_require_grads()
        self.llm.config.use_cache=False

    def encode(self,texts):
        batch=self.tokenizer(texts,padding=True,truncation=True,max_length=self.cfg["text_max_length"],return_tensors="pt")
        batch={k:v.to(next(self.llm.parameters()).device) for k,v in batch.items()}
        out=self.llm(**batch,output_hidden_states=True,use_cache=False)
        h=out.hidden_states[-1].float()
        m=batch["attention_mask"].float()
        return (h*m[...,None]).sum(1)/m.sum(1).clamp_min(1)[:,None]

    def _prompt(self,text):
        return self.tokenizer.apply_chat_template([{"role":"user","content":text}],tokenize=True,add_generation_prompt=True)[:self.cfg["text_max_length"]]

    def expert(self,prefix,texts,targets=None):
        hs,losses,reports=[],[],[]
        embed=self.llm.get_input_embeddings()
        for i,text in enumerate(texts):
            prompt=self._prompt(text)
            p=torch.tensor(prompt,device=prefix.device)
            if self.training and targets is not None:
                target=self.tokenizer.encode(targets[i],add_special_tokens=False)[:self.cfg["report_max_length"]-1]+[self.tokenizer.eos_token_id]
                ids=torch.tensor(prompt+target,device=prefix.device)
                emb=torch.cat([prefix[i,None].to(embed.weight.dtype),embed(ids)],0)[None]
                labels=torch.full((1,emb.shape[1]),-100,device=prefix.device,dtype=torch.long)
                labels[0,1+len(prompt):]=torch.tensor(target,device=prefix.device)
                out=self.llm(inputs_embeds=emb,attention_mask=torch.ones_like(labels),labels=labels,output_hidden_states=True,use_cache=False)
                hs.append(out.hidden_states[-1][0,-1].float());losses.append(out.loss.float());reports.append(targets[i])
            else:
                emb=torch.cat([prefix[i,None].to(embed.weight.dtype),embed(p)],0)[None]
                attention=torch.ones(emb.shape[:2],device=prefix.device,dtype=torch.long)
                generated=self.llm.generate(inputs_embeds=emb,attention_mask=attention,do_sample=False,
                    max_new_tokens=self.cfg["max_new_tokens"],pad_token_id=self.tokenizer.pad_token_id,use_cache=True)
                # inputs_embeds generation yields newly generated IDs; re-forward the full sequence
                # to classify from the actual final generated token instead of the prompt token.
                new_emb=embed(generated[0])
                full=torch.cat([emb,new_emb[None]],1)
                out=self.llm(inputs_embeds=full,attention_mask=torch.ones(full.shape[:2],device=prefix.device,dtype=torch.long),output_hidden_states=True,use_cache=False)
                hs.append(out.hidden_states[-1][0,-1].float())
                reports.append(self.tokenizer.decode(generated[0],skip_special_tokens=True))
        return torch.stack(hs),torch.stack(losses).mean() if losses else prefix.new_zeros(()),reports

    def synthesize(self,text):
        ids=self.tokenizer.apply_chat_template([{"role":"user","content":text}],tokenize=True,add_generation_prompt=True,return_tensors="pt").to(next(self.llm.parameters()).device)
        # Synthesis is inference-only and has no soft prefix; inputs carry reports and calibration.
        out=self.llm.generate(ids,attention_mask=torch.ones_like(ids),do_sample=False,max_new_tokens=self.cfg["max_new_tokens"],pad_token_id=self.tokenizer.pad_token_id,use_cache=True)
        return self.tokenizer.decode(out[0,ids.shape[1]:],skip_special_tokens=True)


class UniQFatigue(nn.Module):
    def __init__(self,cfg,root):
        super().__init__()
        self.cfg=cfg
        self.flow=ConditionalFlow(cfg)
        if cfg.get("ts_checkpoint"):
            self.flow.load_state_dict(torch.load(cfg["ts_checkpoint"],map_location="cpu",weights_only=True),strict=True)
        if cfg.get("freeze_ts"):
            if not cfg.get("ts_checkpoint"):
                raise ValueError("Refusing to freeze a random flow backbone. Provide compatible pretraining weights.")
            for n,p in self.flow.named_parameters():
                p.requires_grad=(n.endswith(".a") or n.endswith(".b") or n.startswith("patch") or n in ("gamma","beta","null"))
        self.language=TinyLanguage(cfg) if cfg["backend"]=="tiny" else HFLanguage(cfg,root)
        d,h=cfg["d_model"],self.language.hidden
        self.text_project=nn.Linear(h,d)
        self.fusion=nn.Sequential(nn.Linear(2*d,cfg["fusion_hidden"]),nn.GELU(),nn.Linear(cfg["fusion_hidden"],d))
        self.fusion_classifier=nn.Linear(d,3)
        self.bridge=nn.Linear(d,h)
        self.local_classifier=nn.Linear(h,3)
        self.nan=nn.Sequential(nn.Linear(h,cfg["nan_hidden"]),nn.GELU(),nn.Linear(cfg["nan_hidden"],1))
        self.expert_prompts=read(Path(root)/"prompts/experts.json")
        self.synthesis_prompt=(Path(root)/"prompts/synthesis.txt").read_text(encoding="utf-8")

    def features(self,x,mask,texts,corrupted=None):
        ex,fm=self.flow(x,mask)
        et=self.text_project(self.language.encode(texts))
        hf=self.fusion(torch.cat([ex,et],-1))
        ec=self.text_project(self.language.encode(corrupted)) if corrupted is not None else None
        return ex,et,hf,ec,fm

    def expert_forward(self,hf,texts,targets=None):
        prefix=self.bridge(hf)
        states,losses,reports=[],[],[]
        keys=["physio"] if self.cfg["ablation"]=="G" else ["physio","kine","psych"]
        for key in keys:
            prompts=[self.expert_prompts[key]+"\n"+t for t in texts]
            refs=[t[key] for t in targets] if targets is not None else None
            h,loss,report=self.language.expert(prefix,prompts,refs)
            states.append(h);losses.append(loss);reports.append(report)
        states=torch.stack(states,1)
        local=self.local_classifier(states).softmax(-1)
        weights=self.nan(states).squeeze(-1).softmax(-1)
        if self.cfg["ablation"]=="E":
            weights=torch.full_like(weights,1/len(keys))
        global_p=(local*weights[...,None]).sum(1)
        return global_p,local,weights,torch.stack(losses).mean(),reports

    def forward(self,x,mask,texts,labels,targets,corrupted=None):
        ex,et,hf,ec,fm=self.features(x,mask,texts,corrupted)
        p,local,w,lm,reports=self.expert_forward(hf,texts,targets)
        return {"ex":ex,"et":et,"hf":hf,"ec":ec,"fm":fm,"probabilities":p,
                "local":local,"weights":w,"generation":lm,"fusion_logits":self.fusion_classifier(hf)}

    @torch.no_grad()
    def infer(self,x,mask,texts):
        _,_,hf,_,_=self.features(x,mask,texts)
        p,local,w,_,reports=self.expert_forward(hf,texts)
        outputs=[]
        for i in range(len(x)):
            bundle={"class":int(p[i].argmax()),"probabilities":p[i].tolist(),"weights":w[i].tolist(),
                    "local_probabilities":local[i].tolist(),"experts":[r[i] for r in reports]}
            bundle["report"]=self.language.synthesize(self.synthesis_prompt+"\nObserved context:\n"+texts[i]+"\n"+str(bundle))
            outputs.append(bundle)
        return outputs
