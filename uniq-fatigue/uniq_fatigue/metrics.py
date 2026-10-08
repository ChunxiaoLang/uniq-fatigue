import numpy as np


def classification(y,p):
    cm=np.zeros((3,3),dtype=np.int64)
    for a,b in zip(y,p): cm[int(a),int(b)]+=1
    n=int(cm.sum())
    if not n: raise ValueError("No predictions")
    tp=np.diag(cm).astype(float)
    true,pred=cm.sum(1),cm.sum(0)
    precision=np.divide(tp,pred,out=np.zeros(3),where=pred!=0)
    recall=np.divide(tp,true,out=np.zeros(3),where=true!=0)
    f1=np.divide(2*precision*recall,precision+recall,out=np.zeros(3),where=(precision+recall)!=0)
    accuracy=float(tp.sum()/n)
    pe=float((true*pred).sum()/n**2)
    kappa=(accuracy-pe)/(1-pe) if pe<1 else None
    # Paper specifies ordinary kappa; ordinal quadratic kappa is additional, not substituted.
    weights=(np.arange(3)[:,None]-np.arange(3)[None,:])**2/4
    expected=true[:,None]*pred[None,:]/n
    denom=float((weights*expected).sum())
    quadratic=1-float((weights*cm).sum())/denom if denom else None
    return {"n":n,"accuracy":accuracy,"macro_f1":float(f1.mean()),"kappa":kappa,
      "quadratic_kappa_extra":quadratic,"per_class_precision":precision.tolist(),"per_class_recall":recall.tolist(),
      "per_class_f1":f1.tolist(),"support":true.tolist(),"confusion_matrix":cm.tolist(),
      "macro_f1_policy":"fixed labels [0,1,2], absent classes score zero"}


def bootstrap_subjects(rows,repeats=1000,seed=42):
    subjects=sorted({r["subject"] for r in rows})
    rng=np.random.default_rng(seed)
    values=[]
    grouped={s:[r for r in rows if r["subject"]==s] for s in subjects}
    for _ in range(repeats):
        sample=[r for s in rng.choice(subjects,len(subjects),replace=True) for r in grouped[s]]
        values.append(classification([r["label"] for r in sample],[r["class"] for r in sample])["macro_f1"])
    return {"macro_f1_subject_bootstrap_ci95":np.quantile(values,[0.025,0.975]).tolist(),
            "bootstrap_seed":seed,"repeats":repeats,"subjects":len(subjects)}


def text_metrics(predictions,records,language,bert_model,bert_revision):
    from sacrebleu.metrics import BLEU
    from rouge_score import rouge_scorer
    from bert_score import BERTScorer
    from transformers import AutoTokenizer,AutoModel
    from pathlib import Path
    import tempfile
    references={r["id"]:r for r in records}
    refs,hyp=[],[]
    for row in predictions:
        ref=references[row["id"]].get("reference_report")
        source=references[row["id"]]["provenance"].get("reference_source","missing")
        if not ref or source in ("synthetic","missing","model_generated"):
            raise ValueError("Text evaluation requires independent, provenance-tagged reference reports for EVERY sample")
        refs.append(ref);hyp.append(row["report"])
    tokenizer="zh" if language=="zh" else "13a"
    bleu=BLEU(tokenize=tokenizer,effective_order=False)
    b=bleu.corpus_score(hyp,[refs])
    # Chinese ROUGE uses explicit character spacing; English uses package word tokenization.
    class CharacterTokenizer:
        def tokenize(self,text):return [c for c in text if not c.isspace()]
    scorer=rouge_scorer.RougeScorer(["rougeL"],use_stemmer=language=="en",
                                  tokenizer=CharacterTokenizer() if language=="zh" else None)
    def rouge_text(t): return t
    rouge=np.mean([scorer.score(rouge_text(r),rouge_text(h))["rougeL"].fmeasure for r,h in zip(refs,hyp)])
    if len(bert_revision)!=40: raise ValueError("BERTScore model needs an immutable SHA")
    # Save resolved revision to local directory: BERTScorer otherwise silently loads main.
    with tempfile.TemporaryDirectory(prefix="uniq_bertscore_") as tmp:
        AutoTokenizer.from_pretrained(bert_model,revision=bert_revision).save_pretrained(tmp)
        model=AutoModel.from_pretrained(bert_model,revision=bert_revision)
        layers=model.config.num_hidden_layers
        model.save_pretrained(tmp)
        bert=BERTScorer(model_type=tmp,num_layers=layers,lang=language,rescale_with_baseline=False,device="cpu")
        _,_,f=bert.score(hyp,refs)
    return {"bleu4_percent":float(b.score),"bleu_signature":str(bleu.get_signature()),
        "rougeL":float(rouge),"bertscore_f1":float(f.mean()),"bert_model":bert_model,
        "bert_revision":bert_revision,"bert_num_layers":layers,"bert_rescale_with_baseline":False,
        "note":"Tokenizer, BERT layer selection and baseline rescaling were not disclosed by the paper."}
