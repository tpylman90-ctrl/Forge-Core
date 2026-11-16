"""Random-init image/text -> procedural object concept branch; isolated from Forge Seed."""
from __future__ import annotations
import argparse,copy,hashlib,json,random,re,time
from pathlib import Path
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader,Dataset

CATEGORIES=("crate","stool","chair","mug","lamp","hovercraft")
VIEWS=("front","side","top","three_quarter")
TOKEN_RE=re.compile(r"[a-z0-9]+")
SIZE=64
PARAM_DIM=8


def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()


def words(text): return TOKEN_RE.findall(text.lower())[:48]


def read_records(root,split):
    return [json.loads(x) for x in (root/f"{split}.jsonl").read_text().splitlines() if x]


def make_vocab(records):
    ws=sorted({w for r in records for w in words(r["prompt"])})
    return {"<pad>":0,"<unk>":1,**{w:i+2 for i,w in enumerate(ws)}}


def encode(text,vocab):
    ids=[vocab.get(w,1) for w in words(text)]
    return torch.tensor(ids+[0]*(48-len(ids)),dtype=torch.long)


class ObjectData(Dataset):
    def __init__(self,root,rows,vocab,mode,seed):
        self.root,self.rows,self.vocab,self.mode,self.seed=root,rows,vocab,mode,seed; self.epoch=0
    def __len__(self): return len(self.rows)
    def set_mode(self, mode): self.mode=mode
    def __getitem__(self,i):
        r=self.rows[i]; rng=random.Random(self.seed+i+self.epoch*100003)
        imgs=torch.zeros((4,3,SIZE,SIZE)); mask=torch.zeros(4)
        if self.mode=="train":
            modality=rng.choice(("image","text","both")); count=rng.choice((1,2,4))
            chosen=sorted(rng.sample(range(4),count)); use_img=modality!="text"; use_text=modality!="image"
        elif self.mode=="text": chosen=[]; use_img=False; use_text=True
        elif self.mode=="single": chosen=[0]; use_img=True; use_text=False
        elif self.mode=="all_views": chosen=list(range(4)); use_img=True; use_text=False
        elif self.mode=="both": chosen=list(range(4)); use_img=True; use_text=True
        else: raise ValueError(self.mode)
        if use_img:
            for vi in chosen:
                path=self.root/r["views"][VIEWS[vi]]
                with Image.open(path) as im:
                    im=im.convert("RGB").resize((SIZE,SIZE),Image.Resampling.BILINEAR)
                    b=torch.frombuffer(bytearray(im.tobytes()),dtype=torch.uint8)
                    imgs[vi]=b.reshape(SIZE,SIZE,3).permute(2,0,1).float()/255
                mask[vi]=1
        ids=encode(r["prompt"],self.vocab) if use_text else torch.zeros(48,dtype=torch.long)
        return imgs,mask,ids,CATEGORIES.index(r["category"]),torch.tensor(r["params"],dtype=torch.float32),r["id"]


class ObjectConceptNet(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()
        self.vision=nn.Sequential(nn.Conv2d(3,24,5,2,2),nn.GroupNorm(6,24),nn.GELU(),
            nn.Conv2d(24,48,3,2,1),nn.GroupNorm(8,48),nn.GELU(),
            nn.Conv2d(48,96,3,2,1),nn.GroupNorm(8,96),nn.GELU(),nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),nn.Linear(96,128),nn.GELU())
        self.embedding=nn.Embedding(vocab_size,64,padding_idx=0)
        self.text=nn.Sequential(nn.Linear(64,96),nn.GELU(),nn.Linear(96,64),nn.GELU())
        self.fuse=nn.Sequential(nn.Linear(194,128),nn.GELU(),nn.Linear(128,128),nn.GELU())
        self.category=nn.Linear(128,len(CATEGORIES)); self.shape=nn.Linear(128,PARAM_DIM)
    def forward(self,images,view_mask,ids):
        b=images.size(0); vf=self.vision(images.reshape(b*4,3,SIZE,SIZE)).reshape(b,4,128)
        vm=view_mask.unsqueeze(-1); vision=(vf*vm).sum(1)/vm.sum(1).clamp(min=1)
        tm=(ids!=0).float().unsqueeze(-1); emb=self.embedding(ids)
        pooled=(emb*tm).sum(1)/tm.sum(1).clamp(min=1); text=self.text(pooled)
        flags=torch.stack((view_mask.sum(1).gt(0).float(),tm.sum((1,2)).gt(0).float()),1)
        fused=self.fuse(torch.cat((vision,text,flags),1))
        return self.category(fused),torch.sigmoid(self.shape(fused))


def evaluate(model,root,rows,vocab,mode,seed,batch):
    ds=ObjectData(root,rows,vocab,mode,seed); loader=DataLoader(ds,batch_size=batch,shuffle=False)
    model.eval(); correct=0; n=0; err=torch.zeros(PARAM_DIM); samples=[]
    with torch.no_grad():
        for im,mask,ids,labels,params,record_ids in loader:
            logits,shape=model(im,mask,ids); pred=logits.argmax(-1)
            correct+=int((pred==labels).sum()); n+=len(labels); err+=(shape-params).abs().sum(0)
            if mode=="both":
                for j,rid in enumerate(record_ids):
                    samples.append({"id":rid,"predicted_category":CATEGORIES[int(pred[j])],
                        "target_category":CATEGORIES[int(labels[j])],
                        "predicted_params":[round(float(v),6) for v in shape[j]],
                        "target_params":[round(float(v),6) for v in params[j]]})
    return {"category_accuracy":correct/max(1,n),"parameter_mae":float(err.sum()/(max(1,n)*PARAM_DIM)),
            "parameter_mae_by_dimension":[round(float(v/max(1,n)),6) for v in err]},samples


def train(root,out,seed,epochs=30,batch=32):
    random.seed(seed); torch.manual_seed(seed); torch.set_num_threads(min(2,torch.get_num_threads()))
    manifest=json.loads((root/"dataset_manifest.json").read_text()); msha=sha(root/"dataset_manifest.json")
    train_rows=read_records(root,"train"); valid=read_records(root,"valid"); test=read_records(root,"test")
    train_params=torch.tensor([r["params"] for r in train_rows],dtype=torch.float32)
    test_params=torch.tensor([r["params"] for r in test],dtype=torch.float32)
    baseline_params=train_params.median(dim=0).values
    baseline_mae=float((test_params-baseline_params).abs().mean())
    train_labels=[r["category"] for r in train_rows]
    baseline_class_accuracy=max(train_labels.count(c) for c in CATEGORIES)/max(1,len(test))
    vocab=make_vocab(train_rows); ds=ObjectData(root,train_rows,vocab,"train",seed)
    loader=DataLoader(ds,batch_size=batch,shuffle=True,num_workers=0,generator=torch.Generator().manual_seed(seed))
    model=ObjectConceptNet(len(vocab)); optimizer=torch.optim.AdamW(model.parameters(),lr=0.001,weight_decay=1e-4)
    out.mkdir(parents=True,exist_ok=True); best=float("inf"); best_state=None; best_opt=None; curve=[]; start=time.monotonic()
    for epoch in range(1,epochs+1):
        ds.epoch=epoch; model.train(); losses=[]; ds.set_mode("train")
        for im,mask,ids,label,params,_ in loader:
            logits,shape=model(im,mask,ids)
            loss=nn.functional.cross_entropy(logits,label)+0.75*nn.functional.smooth_l1_loss(shape,params)
            optimizer.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step(); losses.append(float(loss.detach()))
        val=[]
        for mode in ("text","all_views","both"):
            score,_=evaluate(model,root,valid,vocab,mode,seed,batch)
            val.append(1-score["category_accuracy"]+score["parameter_mae"])
        v=sum(val)/len(val); point={"epoch":epoch,"train_loss":sum(losses)/max(1,len(losses)),
             "validation_score":v,"elapsed_seconds":round(time.monotonic()-start,1)}
        curve.append(point); print(json.dumps({"event":"epoch",**point}),flush=True)
        if v<best:
            best=v; best_state={k:t.detach().cpu().clone() for k,t in model.state_dict().items()}
            best_opt=copy.deepcopy(optimizer.state_dict())
    model.load_state_dict(best_state)
    modes={}
    for mode in ("text","single","all_views","both"):
        modes[mode],preds=evaluate(model,root,test,vocab,mode,seed+101,batch)
        if mode=="both": (out/"predictions.json").write_text(json.dumps(preds,indent=2)+"\n")
    ck={"schema":1,"model":best_state,"optimizer":best_opt,"seed":seed,"epoch_count":epochs,
        "architecture":"ObjectConceptNet-v1","categories":list(CATEGORIES),"vocabulary":vocab,
        "weights_origin":"fresh_random_initialization","forge_seed_parent":None,
        "dataset_manifest_sha256":msha,"training_curve":curve}
    torch.save(ck,out/"checkpoint.pt")
    metrics={"schema":1,"status":"research_only_not_promoted","seed":seed,"updates":epochs*len(loader),
        "training_examples":len(train_rows),"validation_examples":len(valid),"test_examples":len(test),
        "dataset_manifest_sha256":msha,"dataset_license":manifest["license"],
        "weights_origin":"fresh_random_initialization","used_forge_language_checkpoint":False,
        "validation_best_score":best,"test_by_modality":modes,
        "baseline":{"category_accuracy":baseline_class_accuracy,"parameter_mae":baseline_mae,
                    "definition":"Most-common train class and per-dimension training median; no image or prompt input."},
        "training_curve":curve,
        "parameters":sum(p.numel() for p in model.parameters()),"android_integrated":False,
        "promotion":"none; task review and repeatable held-out gains required",
        "checkpoint_sha256":sha(out/"checkpoint.pt")}
    (out/"metrics.json").write_text(json.dumps(metrics,indent=2)+"\n")
    (out/"lineage.json").write_text(json.dumps({"seed":seed,"parent_checkpoint":None,
        "initialization":"fresh_random","dataset_manifest_sha256":msha,
        "memory_in_weights":False,"language_lane_weights_loaded":False,
        "checkpoint_sha256":metrics["checkpoint_sha256"]},indent=2)+"\n")
    print(json.dumps({"event":"final",**metrics},indent=2),flush=True)
    return metrics


def main():
    p=argparse.ArgumentParser(); p.add_argument("--data",type=Path,required=True); p.add_argument("--out",type=Path,required=True)
    p.add_argument("--seed",type=int,required=True); p.add_argument("--epochs",type=int,default=30); p.add_argument("--batch-size",type=int,default=32)
    a=p.parse_args(); train(a.data,a.out,a.seed,a.epochs,a.batch_size)

if __name__=="__main__": main()
