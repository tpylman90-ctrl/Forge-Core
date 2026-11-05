"""Matched Forge Core mistake-curriculum study; original scratch-trained model only.

Three arms: ordinary replay, model-surprisal replay, and surprisal replay with
source-verified answer-first correction traces. The "teacher" here is a
deterministic source oracle, not an external pretrained model.
"""
from __future__ import annotations
import hashlib, json, random, time
from pathlib import Path
import torch
from torch.nn import functional as F
from tokenizers import Tokenizer
from forge_core.neural.model import ByteTransformer, ModelConfig
from forge_core.neural.seed_v07 import heldout
from forge_core.neural.train import make_batch

SEQ=128
BATCH=4
ARMS=("uniform","surprise","source_teacher")

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

@torch.no_grad()
def score_windows(model, tokens, starts, batch=16):
    model.eval()
    scores=[]
    for lo in range(0,len(starts),batch):
        ss=starts[lo:lo+batch]
        x=torch.stack([tokens[s:s+SEQ] for s in ss])
        y=torch.stack([tokens[s+1:s+SEQ+1] for s in ss])
        logits,_=model(x,y)
        loss=F.cross_entropy(logits.reshape(-1,logits.size(-1)),y.reshape(-1),
                             reduction="none").view(len(ss),SEQ).mean(1)
        scores.extend(loss.cpu().tolist())
    return scores

def make_pools(model,tokens,codec,out):
    g=torch.Generator().manual_seed(770031)
    max_start=tokens.numel()-SEQ-1
    candidate_count=min(2048,max_start)
    starts=torch.randperm(max_start,generator=g)[:candidate_count].tolist()
    losses=score_windows(model,tokens,starts)
    ranked=sorted(zip(losses,starts),reverse=True)
    hard=[s for _,s in ranked[:max(1,len(ranked)//5)]]
    # Source oracle: preserve a wrong top-1 guess, the verified next token,
    # and a short evidence-based reason. No validation/test text is used.
    traces=[]
    model.eval()
    for _,start in ranked[:min(512,len(ranked))]:
        x=tokens[start:start+SEQ].unsqueeze(0)
        y=tokens[start+1:start+SEQ+1].unsqueeze(0)
        with torch.no_grad():
            logits,_=model(x,y)
        token_losses=F.cross_entropy(logits[0],y[0],reduction="none")
        pos=int(token_losses.argmax())
        guess=int(logits[0,pos].argmax())
        gold=int(y[0,pos])
        if guess==gold:
            continue
        context=codec.decode(tokens[start:start+pos+1].tolist())[-240:]
        guess_text=codec.decode([guess])
        gold_text=codec.decode([gold])
        record=(f"Context: {context}\n"
                f"Forge's guess: {guess_text}\n"
                f"Correct answer: {gold_text}\n"
                f"Why: the verified source continues with {gold_text}; "
                f"the guess {guess_text} conflicts with that observed continuation.\n")
        traces.extend(codec.encode(record).ids+[codec.token_to_id("[UNK]")])
    if len(traces)<SEQ+1:
        raise RuntimeError("Not enough source-verified correction traces")
    teacher_tokens=torch.tensor(traces,dtype=torch.long)
    teacher_starts=list(range(0,teacher_tokens.numel()-SEQ-1,SEQ//2))
    out.mkdir(parents=True,exist_ok=True)
    (out/"curriculum.json").write_text(json.dumps({
        "source_oracle":"training-text continuation only",
        "external_ai_teacher_used":False,
        "scored_windows":len(starts),"hard_pool_windows":len(hard),
        "teacher_trace_count":len(traces),"hard_fraction":0.20,
        "teacher_update_fraction":0.10,
        "scoring_pool_sha256":hashlib.sha256(
          json.dumps(starts,separators=(",",":")).encode()).hexdigest()
    },indent=2)+"\n")
    return hard,teacher_tokens,teacher_starts

def batch_at(tokens,starts,g):
    picked=torch.randint(len(starts),(BATCH,),generator=g).tolist()
    ix=[starts[i] for i in picked]
    x=torch.stack([tokens[s:s+SEQ] for s in ix])
    y=torch.stack([tokens[s+1:s+SEQ+1] for s in ix])
    return x,y

def run(source:Path,out:Path,updates:int=3000):
    if not 1<=updates<=4000:
        raise ValueError("Updates must be 1..4000")
    torch.set_num_threads(min(2,torch.get_num_threads()))
    manifest=json.loads((source/"source-manifest.json").read_text())
    parent_path=source/"checkpoint.pt"
    parent_hash=sha(parent_path)
    ckpt=torch.load(parent_path,map_location="cpu",weights_only=True)
    if ckpt["step"]!=76000 or manifest["revision"]!="f54c09fd23315a6f9c86f9dc80f725de7d8f9c64":
        raise ValueError("Unexpected parent step or data revision")
    if manifest.get("license")!="cdla-sharing-1.0":
        raise ValueError("Unapproved source license")
    for split in ("train","valid"):
        digest=sha(source/f"{split}.txt")
        if digest!=manifest.get(f"{split}_sha256") or digest!=ckpt.get(f"{split}_sha256"):
            raise ValueError(f"{split} source hash mismatch")
    if sha(source/"tokenizer.json")!=ckpt["tokenizer_sha256"]:
        raise ValueError("Tokenizer/checkpoint hash mismatch")
    codec=Tokenizer.from_file(str(source/"tokenizer.json"))
    train_ids=torch.tensor(codec.encode((source/"train.txt").read_text()).ids,dtype=torch.long)
    valid_ids=torch.tensor(codec.encode((source/"valid.txt").read_text()).ids,dtype=torch.long)
    if len(train_ids)<=SEQ*BATCH or len(valid_ids)<=SEQ*BATCH:
        raise ValueError("Insufficient tokenized data")
    config=ModelConfig(**ckpt["configuration"])
    parent=ByteTransformer(config)
    parent.load_state_dict(ckpt["model"],strict=True)
    hard,teacher_tokens,teacher_starts=make_pools(parent,train_ids,codec,out)
    parent_256=heldout(parent,valid_ids,windows=256,seq_len=SEQ,batch=BATCH)
    summary={"schema":1,"status":"research_only_not_promoted",
      "method":"model-surprisal replay plus source-verified correction traces",
      "parent_step":76000,"parent_checkpoint_sha256":parent_hash,
      "train_sha256":manifest["train_sha256"],"valid_sha256":manifest["valid_sha256"],
      "tokenizer_sha256":ckpt["tokenizer_sha256"],"parent_validation_256":round(parent_256,6),
      "updates_per_candidate":updates,"seeds":[11,43,73],"arms":{}}
    t0=time.monotonic()
    for arm in ARMS:
        rows=[]
        for seed in (11,43,73):
            model=ByteTransformer(config)
            model.load_state_dict(ckpt["model"],strict=True)
            opt=torch.optim.AdamW(model.parameters(),lr=ckpt["lr"])
            opt.load_state_dict(ckpt["optimizer"])
            for group in opt.param_groups: group["lr"]=ckpt["lr"]
            rng=torch.Generator().manual_seed(seed*1009+7000)
            random.seed(seed)
            for step in range(1,updates+1):
                if arm=="source_teacher" and step%10==0:
                    x,y=batch_at(teacher_tokens,teacher_starts,rng)
                elif arm=="uniform":
                    x,y=make_batch(train_ids,BATCH,SEQ,rng)
                else:
                    starts=[]
                    for _ in range(BATCH):
                        use_hard=(torch.rand((),generator=rng).item()<0.5)
                        starts.append(random.choice(hard) if use_hard else
                                      int(torch.randint(train_ids.numel()-SEQ-1,(1,),generator=rng)))
                    x=torch.stack([train_ids[s:s+SEQ] for s in starts])
                    y=torch.stack([train_ids[s+1:s+SEQ+1] for s in starts])
                model.train()
                _,loss=model(x,y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                opt.step()
            end256=heldout(model,valid_ids,windows=256,seq_len=SEQ,batch=BATCH)
            end64=heldout(model,valid_ids,windows=64,seq_len=SEQ,batch=BATCH)
            folder=out/arm/f"seed-{seed}"
            folder.mkdir(parents=True,exist_ok=True)
            cp={**ckpt,"step":76000+updates,"model":model.state_dict(),
                "optimizer":opt.state_dict(),"torch_rng_state":torch.get_rng_state(),
                "study_arm":arm,"study_seed":seed}
            torch.save(cp,folder/"checkpoint.pt")
            (folder/"tokenizer.json").write_bytes((source/"tokenizer.json").read_bytes())
            (folder/"source-manifest.json").write_bytes((source/"source-manifest.json").read_bytes())
            capability={"model_step":76000+updates,"validation_loss_256":round(end256,6),
                        "validation_loss_64":round(end64,6),
                        "capability_note":"Run the common preschool evaluator on each candidate."}
            (folder/"metrics.json").write_text(json.dumps(capability,indent=2)+"\n")
            rows.append({"seed":seed,"validation_256":round(end256,6),
                         "validation_64":round(end64,6),
                         "validation_delta_256":round(end256-parent_256,6),
                         "checkpoint_sha256":sha(folder/"checkpoint.pt")})
            print(json.dumps({"arm":arm,**rows[-1]}),flush=True)
        summary["arms"][arm]=rows
    summary["runtime_seconds"]=round(time.monotonic()-t0,2)
    summary["external_ai_teacher_used"]=False
    summary["teacher_limit"]="A source-verified token correction, not a semantic explanation."
    (out/"results.json").write_text(json.dumps(summary,indent=2)+"\n")
    return summary

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--updates",type=int,default=3000)
    a=p.parse_args()
    run(a.source,a.out,a.updates)
