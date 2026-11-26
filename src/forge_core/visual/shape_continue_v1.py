"""Image-only continuation of archived V5 visual weights; no language training."""
import argparse
import hashlib
import json
import math
import random
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from PIL import Image
from .object_branch_v5 import ObjectConceptNetV5, CATEGORIES, VIEWS, SIZE, active_parameter_mask, read_records, sha

def verified_dataset(root):
    manifest=json.loads((root/"dataset_manifest.json").read_text())
    if manifest.get("license")!="CC0-1.0" or not manifest.get("synthetic_only"):
        raise ValueError("Expected original synthetic dataset")
    for relative,digest in manifest["file_sha256"].items():
        p=(root/relative).resolve()
        if not p.is_relative_to(root.resolve()) or sha(p)!=digest:
            raise ValueError("Dataset hash mismatch: "+relative)
    rows={s:read_records(root,s) for s in ("train","valid","test")}
    seen=set()
    for split,items in rows.items():
        ids={x["id"] for x in items}
        if len(ids)!=len(items) or ids&seen:
            raise ValueError("Overlapping IDs")
        seen|=ids
        if len(items)!=manifest["counts_by_split"][split]:
            raise ValueError("Split count mismatch")
    return manifest,rows

def tensors(root,rows):
    images=[]; labels=[]; params=[]; family=[]
    for row in rows:
        views=[]
        for view in VIEWS:
            with Image.open(root/row["views"][view]) as im:
                im=im.convert("RGB").resize((SIZE,SIZE),Image.Resampling.BILINEAR)
                rgb=torch.frombuffer(bytearray(im.tobytes()),dtype=torch.uint8).reshape(SIZE,SIZE,3).permute(2,0,1).float()/255
            border=torch.cat((rgb[:,0,:],rgb[:,-1,:],rgb[:,:,0],rgb[:,:,-1]),dim=1)
            bg=border.median(dim=1).values.view(3,1,1)
            fg=((rgb-bg).abs().mean(0,keepdim=True)>0.045).float()
            views.append(torch.cat((rgb,fg),0))
        images.append(torch.stack(views))
        labels.append(CATEGORIES.index(row["category"]))
        params.append(row["params"])
        family.append(int(row.get("family_split")=="heldout"))
    return TensorDataset(torch.stack(images),torch.tensor(labels),torch.tensor(params,dtype=torch.float32),torch.tensor(family))

def cache_features(model, dataset, batch_size):
    # GroupNorm operates independently on each view. Masking cached views
    # therefore preserves the image-only prediction for either view mode.
    model.eval()
    features = []
    with torch.no_grad():
        for images, in DataLoader(TensorDataset(dataset.tensors[0]), batch_size=batch_size):
            _, pooled = model.image_features(images)
            features.append(pooled)
    return TensorDataset(torch.cat(features), *dataset.tensors[1:])

def infer(model, features, mode):
    batch, views = features.shape[:2]
    mask = features.new_ones((batch, views))
    if mode == "single":
        mask[:, 1:] = 0
    ids = torch.zeros(batch, 48, dtype=torch.long, device=features.device)
    return model.predict_from_features(features, mask, ids)

def evaluate(model,ds,batch,median):
    model.eval(); result={}
    with torch.no_grad():
        for mode in ("single","all_views"):
            sums={k:[0.,0.,0,0,0.] for k in ("all","seen","heldout")}
            for im,lab,params,fam in DataLoader(ds,batch_size=batch):
                logits,pred=infer(model,im,mode)
                active=active_parameter_mask(lab)
                for k,sel in (("all",torch.ones(len(lab),dtype=torch.bool)),("seen",fam==0),("heldout",fam==1)):
                    if not sel.any(): continue
                    a=active[sel]
                    sums[k][0]+=float(((pred[sel]-params[sel]).abs()*a).sum())
                    sums[k][1]+=float(a.sum())
                    sums[k][2]+=int((logits[sel].argmax(1)==lab[sel]).sum())
                    sums[k][3]+=int(sel.sum())
                    sums[k][4]+=float(((median-params[sel]).abs()*a).sum())
            result[mode]={k:{"mae":v[0]/v[1],"category_accuracy":v[2]/v[3],"count":v[3],"median_mae":v[4]/v[1]} for k,v in sums.items() if v[3]}
    return result

def score(metrics):
    return sum(metrics[m]["all"]["mae"] for m in ("single","all_views"))/2

def save(path,model,opt,epoch,parent_sha,manifest_sha,seed,metrics,vocab):
    temp=path.with_suffix(".tmp")
    torch.save({"schema":1,"architecture":"ObjectConceptNetV5","model":model.state_dict(),"optimizer":opt.state_dict(),
                "epoch":epoch,"seed":seed,"parent_sha256":parent_sha,"dataset_manifest_sha256":manifest_sha,
                "validation":metrics,"vocabulary":vocab,"torch_rng":torch.get_rng_state(),"scope":"image_only_shape_continuation"},temp)
    temp.replace(path)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",type=Path,required=True);p.add_argument("--parent",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True);p.add_argument("--seed",type=int,required=True)
    p.add_argument("--epochs",type=int,default=12);p.add_argument("--batch",type=int,default=32)
    a=p.parse_args()
    if not 1<=a.epochs<=20: raise ValueError("Bounded epoch budget")
    torch.set_num_threads(2);torch.manual_seed(a.seed);random.seed(a.seed)
    a.out.mkdir(parents=True,exist_ok=False)
    manifest,rows=verified_dataset(a.data); msha=sha(a.data/"dataset_manifest.json")
    parent_file=a.parent/"checkpoint.pt"; parent_sha=sha(parent_file)
    pm=json.loads((a.parent/"metrics.json").read_text())
    if pm["checkpoint_sha256"]!=parent_sha or pm["dataset_manifest_sha256"]!=msha or pm["seed"]!=a.seed:
        raise ValueError("Parent checkpoint provenance mismatch")
    ck=torch.load(parent_file,map_location="cpu",weights_only=True)
    if ck["seed"]!=a.seed or ck["dataset_manifest_sha256"]!=msha or ck["forge_seed_parent"] is not None:
        raise ValueError("Wrong checkpoint lineage")
    model=ObjectConceptNetV5(len(ck["vocabulary"]));model.load_state_dict(ck["model"])
    # Preserve the learned image backbone, segmentation and text branch.
    # Only downstream geometry/classification projections are trainable.
    for name,param in model.named_parameters():
        param.requires_grad_(name.split(".")[0] in ("vision_project","fuse","category","shape"))
    trainable = {name: param for name, param in model.named_parameters() if param.requires_grad}
    opt = torch.optim.AdamW(list(trainable.values()), lr=0.0002, weight_decay=1e-4)
    train_images = tensors(a.data, rows["train"])
    train = cache_features(model, train_images, a.batch)
    # Verify both cached paths against the checkpoint's regular forward pass.
    sample = train_images.tensors[0][:2]
    with torch.no_grad():
        _, pooled = model.image_features(sample)
        for mode in ("single", "all_views"):
            mask = sample.new_ones(sample.shape[:2])
            if mode == "single":
                mask[:, 1:] = 0
            inputs = sample * mask[:, :, None, None, None]
            expected = model(inputs, mask, torch.zeros(len(sample), 48, dtype=torch.long))[:2]
            for actual, reference in zip(infer(model, pooled, mode), expected):
                torch.testing.assert_close(actual, reference)
    del train_images
    valid = cache_features(model, tensors(a.data, rows["valid"]), a.batch)
    updates = 0
    median=train.tensors[2].median(dim=0).values
    baseline=evaluate(model,valid,a.batch,median);best=score(baseline);selected=0;stale=0;curve=[]
    save(a.out/"selected.pt",model,opt,0,parent_sha,msha,a.seed,baseline,ck["vocabulary"])
    print(json.dumps({"event":"parent_validation","metrics":baseline}),flush=True)
    generator=torch.Generator().manual_seed(a.seed+917)
    loader=DataLoader(train,batch_size=a.batch,shuffle=True,generator=generator)
    for epoch in range(1,a.epochs+1):
        model.train(); losses=[]
        for i,(im,lab,params,_) in enumerate(loader):
            mode="single" if (epoch+i)%2 else "all_views"
            logits,pred=infer(model,im,mode)
            active=active_parameter_mask(lab)
            loss=((pred-params).abs()*active).sum()/active.sum()+0.35*nn.functional.cross_entropy(logits,lab)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite CPU visual training loss")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(list(trainable.values()), 1., error_if_nonfinite=True)
            opt.step()
            updates += 1
            losses.append(float(loss.detach()))
        val=evaluate(model,valid,a.batch,median);v=score(val)
        retain=all(val[m]["all"]["category_accuracy"]>=baseline[m]["all"]["category_accuracy"]-0.01 and val[m]["all"]["mae"]<=baseline[m]["all"]["mae"]*1.01 for m in ("single","all_views"))
        improved=retain and v<best-0.0001
        if improved:
            best=v;selected=epoch;stale=0
            save(a.out/"selected.pt",model,opt,epoch,parent_sha,msha,a.seed,val,ck["vocabulary"])
        else: stale+=1
        save(a.out/"latest.pt",model,opt,epoch,parent_sha,msha,a.seed,val,ck["vocabulary"])
        curve.append({"epoch":epoch,"optimizer_updates":updates,"loss":sum(losses)/len(losses),"validation":val,"retention_pass":retain,"selected":improved})
        (a.out/"curve.json").write_text(json.dumps(curve,indent=2))
        print(json.dumps({"event":"epoch",**curve[-1]}),flush=True)
        if stale>=3: break
    # Test is loaded only after the selection is locked.
    chosen = torch.load(a.out/"selected.pt", map_location="cpu", weights_only=True)
    changed = sorted(name for name, tensor in chosen["model"].items()
                     if not torch.equal(tensor, ck["model"][name]))
    if set(changed) - trainable.keys():
        raise RuntimeError("Protected visual checkpoint tensors changed")
    model.load_state_dict(ck["model"])
    test = cache_features(model, tensors(a.data, rows["test"]), a.batch)
    parent_test = evaluate(model, test, a.batch, median)
    model.load_state_dict(chosen["model"]);candidate_test=evaluate(model,test,a.batch,median)
    gate=selected>0 and bool(changed) and all(candidate_test[m]["heldout"]["mae"]<min(parent_test[m]["heldout"]["mae"],candidate_test[m]["heldout"]["median_mae"]) and candidate_test[m]["heldout"]["category_accuracy"]>=parent_test[m]["heldout"]["category_accuracy"]-0.01 for m in ("single","all_views"))
    report={"status":"research_only","seed":a.seed,"source_run":44011991242,"parent_checkpoint_sha256":parent_sha,"dataset_manifest_sha256":msha,
        "selected_epoch":selected,"epochs_attempted":len(curve),"optimizer_reset":True,"parent_validation":baseline,"parent_test":parent_test,"candidate_test":candidate_test,
        "heldout_gate_pass":gate,"weights_changed_from_parent":bool(changed),"changed_tensors":changed,
        "protected_tensors_unchanged":True,"optimizer_updates":updates,"selected_optimizer_updates":curve[selected-1]["optimizer_updates"] if selected else 0,
        "latest_sha256":sha(a.out/"latest.pt"),"selected_sha256":sha(a.out/"selected.pt"),"promotion":"none; require multi-seed independent evidence",
        "limitations":["Same synthetic V5 family and historical holdout; not a new independent real-world test","No mesh reconstruction or texture learning","No language checkpoint used","Selection uses validation only"]}
    (a.out/"metrics.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2),flush=True)
if __name__=="__main__": main()
