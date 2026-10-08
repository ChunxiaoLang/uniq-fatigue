import torch
from torch.nn import functional as F


def alignment_losses(ex,et,hf,ec,tau,eta=1.0):
    # A mathematically explicit surrogate reconstructed from prose; not a PID estimator.
    ex,et,hf=map(lambda x:F.normalize(x,dim=-1),(ex,et,hf))
    b=len(ex)
    labels=torch.arange(b,device=ex.device)
    cross=(ex@et.T)/tau
    if ec is not None:
        ec=F.normalize(ec,dim=-1)
        hard=(ex*ec).sum(-1,keepdim=True)/tau
        forward=F.cross_entropy(torch.cat([cross,hard],1),labels)
    else:
        forward=F.cross_entropy(cross,labels)
    reverse=F.cross_entropy(cross.T,labels)
    redundant=(forward+eta*reverse)/(1+eta)
    # Each single modality of sample i is a positive for fused i; both modalities
    # from other samples are negatives. Average the two positive log-probabilities.
    logits=hf@torch.cat([ex,et],0).T/tau
    logp=logits.log_softmax(-1)
    synergy=-0.5*(logp[labels,labels]+logp[labels,labels+b]).mean()
    return redundant,synergy


def objective(out,labels,cfg):
    rdn,syn=alignment_losses(out["ex"],out["et"],out["hf"],out["ec"],cfg["tau"],cfg["redundancy_eta"])
    pred=F.cross_entropy(out["fusion_logits"],labels)
    p=out["probabilities"].clamp_min(1e-8)
    nan=F.nll_loss(p.log(),labels)
    # KL(local || global); global is not detached. Direction is an explicit assumption.
    local=out["local"].clamp_min(1e-8)
    kl=(local*(local.log()-p[:,None].log())).sum(-1).mean()
    rweight=0 if cfg["ablation"]=="B" else cfg["lambda_rdn"]
    sweight=0 if cfg["ablation"] in ("C","D") else cfg["lambda_syn"]
    if cfg["ablation"]=="D":
        rweight=cfg["lambda_rdn"]
    kweight=0 if cfg["ablation"]=="F" else cfg["lambda_consistency"]
    total=pred+nan+kweight*kl+cfg["lambda_generation"]*out["generation"]+rweight*rdn+sweight*syn+cfg["lambda_fm"]*out["fm"]
    return total,{k:float(v.detach()) for k,v in {"total":total,"prediction":pred,"nan":nan,"consistency":kl,
                          "generation":out["generation"],"redundancy":rdn,"synergy":syn,"flow":out["fm"]}.items()}
