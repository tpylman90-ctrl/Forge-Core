"""Random-init image/text -> procedural object concept branch; isolated from Forge Seed."""
from __future__ import annotations
import argparse,copy,hashlib,json,random,re,time
from pathlib import Path
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader,Dataset

CATEGORIES=("crate","stool","chair","mug","lamp","hovercraft")
SEEN_CATEGORIES=("crate","stool","lamp","hovercraft")
HELDOUT_CATEGORIES=("chair","mug")
VIEWS=("front","side","top","three_quarter")
TOKEN_RE=re.compile(r"[a-z0-9]+")
SIZE=64
PARAM_DIM=8

# Only score controls that the procedural geometry actually uses for each class.
# p7 is intentionally inactive for every class; several other controls are category-specific.
ACTIVE_PARAMETERS=(
    (1,1,1,1,1,0,0,0), # crate: dimensions, bevel, rib detail
    (1,1,1,1,1,1,0,0), # stool: dimensions, bevel, seat height, leg width
    (1,1,1,1,1,1,1,0), # chair: adds back-panel height
    (1,1,1,0,1,0,0,0), # mug: dimensions and handle thickness
    (1,1,1,0,1,1,1,0), # lamp: dimensions, base, stem, shade details
    (1,1,1,1,1,1,0,0), # hovercraft: dimensions, bevel, hull, pod width
)


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
        imgs=torch.zeros((4,4,SIZE,SIZE)); mask=torch.zeros(4); part_masks=torch.zeros((4,SIZE,SIZE),dtype=torch.long)
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
                    rgb=b.reshape(SIZE,SIZE,3).permute(2,0,1).float()/255
                    border=torch.cat((rgb[:,0,:],rgb[:,-1,:],rgb[:,:,0],rgb[:,:,-1]),dim=1)
                    background=border.median(dim=1).values.view(3,1,1)
                    foreground=((rgb-background).abs().mean(0,keepdim=True)>0.045).float()
                    imgs[vi]=torch.cat((rgb,foreground),dim=0)
                    part_path=self.root/r["part_views"][VIEWS[vi]]
                    with Image.open(part_path) as pm:
                        pm=pm.convert("RGB").resize((SIZE,SIZE),Image.Resampling.NEAREST)
                        pb=torch.frombuffer(bytearray(pm.tobytes()),dtype=torch.uint8)
                        colors=pb.reshape(SIZE,SIZE,3).to(torch.int16)
                        palette=torch.tensor([[5,5,5],[230,20,20],[20,230,20],[20,20,230],[230,209,20]],dtype=torch.int16)
                        distances=((colors.unsqueeze(2)-palette.view(1,1,5,3))**2).sum(-1)
                        part_masks[vi]=distances.argmin(2)
                mask[vi]=1
        ids=encode(r["prompt"],self.vocab) if use_text else torch.zeros(48,dtype=torch.long)
        return imgs,mask,ids,CATEGORIES.index(r["category"]),torch.tensor(r["params"],dtype=torch.float32),part_masks,r["id"]


class ObjectConceptNetV3(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()
        self.backbone=nn.Sequential(nn.Conv2d(4,24,5,2,2),nn.GroupNorm(6,24),nn.GELU(),
            nn.Conv2d(24,48,3,2,1),nn.GroupNorm(8,48),nn.GELU(),
            nn.Conv2d(48,96,3,2,1),nn.GroupNorm(8,96),nn.GELU())
        self.global_pool=nn.AdaptiveAvgPool2d((2,2))
        self.vision_project=nn.Sequential(nn.Linear(96*4,128),nn.GELU())
        self.part_head=nn.Sequential(nn.Conv2d(96,64,3,padding=1),nn.GELU(),nn.Conv2d(64,5,1))
        self.embedding=nn.Embedding(vocab_size,64,padding_idx=0)
        self.text=nn.Sequential(nn.Linear(64,96),nn.GELU(),nn.Linear(96,64),nn.GELU())
        self.fuse=nn.Sequential(nn.Linear(194,128),nn.GELU(),nn.Linear(128,128),nn.GELU())
        self.category=nn.Linear(128,len(CATEGORIES)); self.shape=nn.Linear(128,PARAM_DIM)
    def forward(self,images,view_mask,ids):
        b=images.size(0)
        maps=self.backbone(images.reshape(b*4,4,SIZE,SIZE))
        vf=self.vision_project(self.global_pool(maps).flatten(1)).reshape(b,4,128)
        vm=view_mask.unsqueeze(-1); vision=(vf*vm).sum(1)/vm.sum(1).clamp(min=1)
        tm=(ids!=0).float().unsqueeze(-1); emb=self.embedding(ids)
        pooled=(emb*tm).sum(1)/tm.sum(1).clamp(min=1); text=self.text(pooled)
        flags=torch.stack((view_mask.sum(1).gt(0).float(),tm.sum((1,2)).gt(0).float()),1)
        fused=self.fuse(torch.cat((vision,text,flags),1))
        part=nn.functional.interpolate(self.part_head(maps),size=(SIZE,SIZE),mode="bilinear",align_corners=False)
        part=part.reshape(b,4,5,SIZE,SIZE)
        return self.category(fused),torch.sigmoid(self.shape(fused)),part


def active_parameter_mask(labels):
    table=torch.tensor(ACTIVE_PARAMETERS,dtype=torch.float32,device=labels.device)
    return table[labels]


def _part_iou(conf):
    ious=[]
    for i in range(1,5):
        union=int(conf[i,:].sum()+conf[:,i].sum()-conf[i,i])
        if union>0: ious.append(float(conf[i,i])/union)
    return sum(ious)/max(1,len(ious))

def evaluate(model,root,rows,vocab,mode,seed,batch):
    ds=ObjectData(root,rows,vocab,mode,seed); loader=DataLoader(ds,batch_size=batch,shuffle=False)
    model.eval(); correct={"seen":0,"heldout":0}; counts={"seen":0,"heldout":0}
    errors={k:torch.zeros(PARAM_DIM) for k in ("all","seen","heldout")}
    active_counts={k:torch.zeros(PARAM_DIM) for k in ("all","seen","heldout")}
    conf={k:torch.zeros((5,5),dtype=torch.long) for k in ("all","seen","heldout")}
    samples=[]
    with torch.no_grad():
        for im,mask,ids,labels,params,part_targets,record_ids in loader:
            logits,shape,part_logits=model(im,mask,ids); pred=logits.argmax(-1)
            active=active_parameter_mask(labels).cpu()
            family=torch.tensor([0 if CATEGORIES[int(y)] in SEEN_CATEGORIES else 1 for y in labels],dtype=torch.long)
            for group,select in (("seen",family==0),("heldout",family==1)):
                if bool(select.any()):
                    correct[group]+=int((pred[select]==labels[select]).sum()); counts[group]+=int(select.sum())
                    errors[group]+=((shape[select]-params[select]).abs()*active[select]).sum(0)
                    active_counts[group]+=active[select].sum(0)
            errors["all"]+=((shape-params).abs()*active).sum(0); active_counts["all"]+=active.sum(0)
            if bool(mask.any()):
                part_pred=part_logits.argmax(2)
                for bi in range(len(labels)):
                    group="seen" if family[bi]==0 else "heldout"
                    for vi in range(4):
                        if not bool(mask[bi,vi]): continue
                        target=part_targets[bi,vi].reshape(-1)
                        guess=part_pred[bi,vi].reshape(-1)
                        bins=torch.bincount(target*5+guess,minlength=25).reshape(5,5)
                        conf["all"]+=bins; conf[group]+=bins
            if mode=="both":
                for j,rid in enumerate(record_ids):
                    samples.append({"id":rid,"family_split":"seen" if family[j]==0 else "heldout",
                        "predicted_category":CATEGORIES[int(pred[j])],"target_category":CATEGORIES[int(labels[j])],
                        "predicted_params":[round(float(v),6) for v in shape[j]],
                        "target_params":[round(float(v),6) for v in params[j]]})
    out={}
    for group in ("all","seen","heldout"):
        total=float(active_counts[group].sum())
        out[group+"_parameter_mae"]=float(errors[group].sum()/max(1.0,total))
        out[group+"_active_parameter_count"]=int(total)
        if group!="all":
            out[group+"_category_accuracy"]=correct[group]/max(1,counts[group])
        if int(conf[group].sum())>0: out[group+"_part_miou_foreground"]=_part_iou(conf[group])
        else: out[group+"_part_miou_foreground"]=None
    out["category_accuracy"]=(correct["seen"]+correct["heldout"])/max(1,counts["seen"]+counts["heldout"])
    out["parameter_mae"]=out["all_parameter_mae"]
    out["part_miou_foreground"]=out["all_part_miou_foreground"]
    return out,samples


def train(root,out,seed,epochs=60,batch=32):
    random.seed(seed); torch.manual_seed(seed); torch.set_num_threads(min(2,torch.get_num_threads()))
    manifest=json.loads((root/"dataset_manifest.json").read_text()); msha=sha(root/"dataset_manifest.json")
    train_rows=read_records(root,"train"); valid=read_records(root,"valid"); test=read_records(root,"test")
    train_params=torch.tensor([r["params"] for r in train_rows],dtype=torch.float32)
    test_params=torch.tensor([r["params"] for r in test],dtype=torch.float32)
    baseline_params=train_params.median(dim=0).values
    test_labels=torch.tensor([CATEGORIES.index(r["category"]) for r in test],dtype=torch.long)
    test_active=active_parameter_mask(test_labels)
    baseline_mae=float(((test_params-baseline_params).abs()*test_active).sum()/test_active.sum().clamp(min=1))
    train_labels=[r["category"] for r in train_rows]
    majority=max(CATEGORIES,key=lambda c:train_labels.count(c))
    baseline_class_accuracy=sum(r["category"]==majority for r in test)/max(1,len(test))
    baseline_seen=sum(r["category"]==majority for r in test if r["category"] in SEEN_CATEGORIES)/max(1,sum(r["category"] in SEEN_CATEGORIES for r in test))
    vocab=make_vocab(train_rows); ds=ObjectData(root,train_rows,vocab,"train",seed)
    loader=DataLoader(ds,batch_size=batch,shuffle=True,num_workers=0,generator=torch.Generator().manual_seed(seed))
    model=ObjectConceptNetV3(len(vocab)); optimizer=torch.optim.AdamW(model.parameters(),lr=0.001,weight_decay=1e-4)
    part_weights=torch.tensor([0.2,1.0,1.4,1.4,1.2],dtype=torch.float32)
    out.mkdir(parents=True,exist_ok=True); best=float("inf"); best_state=None; best_opt=None; curve=[]; start=time.monotonic()
    for epoch in range(1,epochs+1):
        ds.epoch=epoch; model.train(); losses=[]; ds.set_mode("train")
        for im,mask,ids,label,params,part_targets,_ in loader:
            logits,shape,part_logits=model(im,mask,ids)
            active=active_parameter_mask(label)
            per_value=nn.functional.smooth_l1_loss(shape,params,reduction="none")
            shape_loss=(per_value*active).sum()/active.sum().clamp(min=1)
            raw_part=nn.functional.cross_entropy(part_logits.reshape(-1,5,SIZE,SIZE),
                part_targets.reshape(-1,SIZE,SIZE),weight=part_weights,reduction="none")
            view_weight=mask.reshape(-1,1,1)
            part_loss=(raw_part*view_weight).sum()/(view_weight.sum()*SIZE*SIZE).clamp(min=1)
            loss=nn.functional.cross_entropy(logits,label)+2.0*shape_loss+0.5*part_loss
            optimizer.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step(); losses.append(float(loss.detach()))
        val=[]
        for mode in ("text","all_views","both"):
            score,_=evaluate(model,root,valid,vocab,mode,seed,batch)
            miou=score["seen_part_miou_foreground"] or 0.0
            val.append(1-score["seen_category_accuracy"]+score["seen_parameter_mae"]+0.5*(1-miou))
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
    ck={"schema":3,"model":best_state,"optimizer":best_opt,"seed":seed,"epoch_count":epochs,
        "architecture":"ObjectConceptNetV3-part-segmentation-family-holdout","categories":list(CATEGORIES),
        "seen_categories":list(SEEN_CATEGORIES),"heldout_categories":list(HELDOUT_CATEGORIES),
        "vocabulary":vocab,"weights_origin":"fresh_random_initialization","forge_seed_parent":None,
        "dataset_manifest_sha256":msha,"training_curve":curve}
    torch.save(ck,out/"checkpoint.pt")
    metrics={"schema":3,"status":"research_only_not_promoted","seed":seed,"updates":epochs*len(loader),
        "training_examples":len(train_rows),"validation_examples":len(valid),"test_examples":len(test),
        "seen_categories":list(SEEN_CATEGORIES),"heldout_categories":list(HELDOUT_CATEGORIES),
        "dataset_manifest_sha256":msha,"dataset_license":manifest["license"],
        "weights_origin":"fresh_random_initialization","used_forge_language_checkpoint":False,
        "validation_best_score":best,"test_by_modality":modes,
        "baseline":{"category_accuracy":baseline_class_accuracy,"seen_category_accuracy":baseline_seen,
                    "parameter_mae":baseline_mae,"part_miou_foreground":0.0,
                    "definition":"Most-common train class, per-dimension training median, and all-background part mask. Parameter MAE scores only active geometry controls. Chair and mug are held out from both training and validation."},
        "training_curve":curve,"parameters":sum(p.numel() for p in model.parameters()),
        "android_integrated":False,
        "promotion":"none; repeatable held-out-family part and geometry gains required",
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
