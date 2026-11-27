"""CUDA-only CAD surface reconstruction pilot using original CC0 Forge Basics geometry."""
import argparse
import hashlib
import importlib.util
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

GRID=32
# Fixed object-disjoint split. Similar primitives can occur across splits;
# this is not an independent real-world generalization benchmark.
SPLITS={
 "train":["cube","sphere","cylinder","pipe","wedge","fairing","armour_panel","nozzle","bolt","radiator_grille","intake_frame","hub"],
 "valid":["plate","cone","hex_nut"],
 "test":["torus","aero_fin","thruster_housing"]
}
SOURCE_COMMIT="c2f050a54418d69a8ddc082d10110fb4d55e985d"
SOURCE_BLOB="29365713aa30d6034c38b825551441ae60e23219"

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def generator(path):
    raw=Path(path).read_bytes()
    blob=hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()
    if blob!=SOURCE_BLOB: raise ValueError("CAD generator differs from pinned reviewed blob")
    spec=importlib.util.spec_from_file_location("forge_basics",path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    if set(sum(SPLITS.values(),[]))!={a[0] for a in mod.BASICS}: raise ValueError("Asset inventory mismatch")
    return mod

def make_data(mod, variants, seed):
    result={}
    for si,(split,ids) in enumerate(SPLITS.items()):
        x=[];y=[];records=[]
        for ai,asset in enumerate(ids):
            mesh=np.asarray(mod.make_asset(asset).data,dtype=np.float32).reshape(-1,6)[:,:3].reshape(-1,3,3)
            low=mesh.min(axis=(0,1));high=mesh.max(axis=(0,1))
            mesh=(mesh-(low+high)/2)/max(high-low)
            area=np.linalg.norm(np.cross(mesh[:,1]-mesh[:,0],mesh[:,2]-mesh[:,0]),axis=1)
            prob=area/area.sum()
            for j in range(variants):
                rng=np.random.default_rng(seed+si*1000000+ai*10000+j)
                # Deterministic axis scaling and quarter-turn rotations; no new object family.
                scale=rng.uniform(.55,1.,3)
                tris=mesh*scale
                for _ in range(int(rng.integers(0,4))):
                    tris=tris[..., [2,1,0]]*np.array([-1,1,1])
                selected=tris[rng.choice(len(tris),size=16000,p=prob)]
                a=rng.random((len(selected),1));b=rng.random((len(selected),1))
                flip=a+b>1;a[flip]=1-a[flip];b[flip]=1-b[flip]
                points=selected[:,0]+a*(selected[:,1]-selected[:,0])+b*(selected[:,2]-selected[:,0])
                # Conservative sampled surface voxels; not watertight solid occupancy.
                ijk=np.clip(np.floor((points+0.5)*(GRID-2)+1).astype(int),0,GRID-1)
                surface=np.zeros((GRID,GRID,GRID),np.uint8);surface[tuple(ijk.T)]=1
                views=np.stack([surface.max(axis=i) for i in range(3)])
                x.append(views);y.append(surface);records.append({"asset":asset,"variant":j,"scale":scale.tolist()})
        result[split]=(torch.from_numpy(np.stack(x)),torch.from_numpy(np.stack(y)),records)
    return result

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc=nn.Sequential(nn.Conv2d(6,32,5,2,2),nn.GroupNorm(4,32),nn.GELU(),
          nn.Conv2d(32,64,3,2,1),nn.GroupNorm(8,64),nn.GELU(),
          nn.Conv2d(64,96,3,2,1),nn.GELU(),nn.Flatten(),nn.Linear(96*4*4,256),nn.GELU())
        self.dec=nn.Sequential(nn.Linear(256,64*4*4*4),nn.Unflatten(1,(64,4,4,4)),
          nn.ConvTranspose3d(64,32,4,2,1),nn.GroupNorm(4,32),nn.GELU(),
          nn.ConvTranspose3d(32,16,4,2,1),nn.GroupNorm(4,16),nn.GELU(),
          nn.ConvTranspose3d(16,1,4,2,1))
    def forward(self,x,mask):
        flags=mask.expand(-1,-1,GRID,GRID)
        return self.dec(self.enc(torch.cat((x*flags,flags),1)))[:,0]

def mode_input(x,mode):
    mask=torch.ones((len(x),3,1,1),device=x.device)
    if mode=="single": mask[:,1:]=0
    return mask

def iou(p,y):
    return ((p&y).sum((1,2,3)).float()/((p|y).sum((1,2,3)).float().clamp_min(1)))

def evaluate(net,loader,device):
    net.eval();total={m:[] for m in ("single","multi")};base={m:[] for m in total}
    with torch.no_grad():
        for x,y in loader:
            x=x.to(device).float();y=y.to(device).bool()
            for m in total:
                pred=net(x,mode_input(x,m)).sigmoid()>=.5
                total[m].extend(iou(pred,y).cpu().tolist())
                # Extruded silhouette (one view) or intersection visual hull (three).
                hull=x[:,0,None,:,:].bool().expand(-1,GRID,-1,-1)
                if m=="multi": hull=hull & x[:,1,:,None,:].bool() & x[:,2,:,:,None].bool()
                base[m].extend(iou(hull,y).cpu().tolist())
    return {m:{"surface_iou":float(np.mean(total[m])),"visual_hull_iou":float(np.mean(base[m])),"count":len(total[m])} for m in total}

def model_sha256(net):
    digest = hashlib.sha256()
    for name, tensor in net.state_dict().items():
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()

def checkpoint(path,net,opt,seed,epoch,metrics,manifest):
    tmp=path.with_suffix(".tmp")
    torch.save({"architecture":"CADSurface32-v1","model":net.state_dict(),"optimizer":opt.state_dict(),
                "seed":seed,"epoch":epoch,"metrics":metrics,"dataset_sha256":manifest,"torch_rng":torch.get_rng_state(),
                "cuda_rng":torch.cuda.get_rng_state_all() if next(net.parameters()).is_cuda else [],"no_language_weights":True},tmp)
    tmp.replace(path)

def run(data,root,seed,epochs,batch,manifest,*,device="cuda",limitations=None,objective=None):
    device = torch.device(device)
    on_cuda = device.type == "cuda"
    torch.manual_seed(seed)
    if on_cuda:
        torch.cuda.manual_seed_all(seed)
    out=root/f"seed-{seed}";out.mkdir(exist_ok=False)
    net=Net().to(device);opt=torch.optim.AdamW(net.parameters(),lr=.001)
    def loader(split,shuffle=False):
        x,y,_=data[split]
        return DataLoader(TensorDataset(x,y),batch_size=batch,shuffle=shuffle,pin_memory=on_cuda,
                          generator=torch.Generator().manual_seed(seed))
    train,valid,test=loader("train",True),loader("valid"),loader("test")
    if on_cuda:
        torch.cuda.reset_peak_memory_stats(device)
    initial=evaluate(net,valid,device)
    initial_model_sha256 = model_sha256(net)
    updates = 0
    def score(m): return (m["single"]["surface_iou"]+m["multi"]["surface_iou"])/2
    best=score(initial);selected=0;stale=0;curve=[];start=time.monotonic()
    checkpoint(out/"initial.pt",net,opt,seed,0,initial,manifest)
    checkpoint(out/"selected.pt",net,opt,seed,0,initial,manifest)
    for epoch in range(1,epochs+1):
        net.train();n=0;loss_sum=0
        for i,(x,y) in enumerate(train):
            x=x.to(device,non_blocking=True).float();y=y.to(device,non_blocking=True).float()
            opt.zero_grad(set_to_none=True)
            pred=net(x,mode_input(x,"single" if (epoch+i)%2 else "multi"))
            loss = (objective(pred, y) if objective is not None else
                    nn.functional.binary_cross_entropy_with_logits(pred,y,pos_weight=torch.tensor(5.,device=device)))
            if not torch.isfinite(loss): raise RuntimeError("Nonfinite training loss")
            loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
            updates += 1
            loss_sum+=float(loss.detach())*len(x);n+=len(x)
        val=evaluate(net,valid,device);s=score(val)
        if s>best+0.001:
            best=s;selected=epoch;stale=0
            checkpoint(out/"selected.pt",net,opt,seed,epoch,val,manifest)
        else: stale+=1
        checkpoint(out/"latest.pt",net,opt,seed,epoch,val,manifest)
        curve.append({"epoch":epoch,"updates":updates,"loss":loss_sum/n,"validation":val,"seconds":time.monotonic()-start})
        (out/"curve.json").write_text(json.dumps(curve,indent=2))
        print(json.dumps({"seed":seed,**curve[-1]}),flush=True)
        if stale>=4 or time.monotonic()-start>1800: break
    chosen=torch.load(out/"selected.pt",map_location=device,weights_only=True)
    net.load_state_dict(chosen["model"])
    selected_model_sha256 = model_sha256(net)
    result=evaluate(net,test,device)
    report={"seed":seed,"selected_epoch":selected,"epochs_attempted":len(curve),"initial_validation":initial,
            "selected_validation":chosen["metrics"],"test":result,"checkpoint_sha256":sha(out/"selected.pt"),
            "initial_checkpoint_sha256":sha(out/"initial.pt"),
            "initial_model_sha256":initial_model_sha256,"selected_model_sha256":selected_model_sha256,
            "weights_changed_from_initial":selected_model_sha256 != initial_model_sha256,
            "updates":updates,"batch_size":batch,"epoch_limit":epochs,
            "peak_cuda_allocated_bytes":torch.cuda.max_memory_allocated(device) if on_cuda else None,
            "dataset_sha256":manifest,"device":device.type,"gpu":torch.cuda.get_device_name(0) if on_cuda else None,"seconds":time.monotonic()-start,
            "status":"research_only","gate":selected>0 and selected_model_sha256 != initial_model_sha256 and all(result[m]["surface_iou"]>result[m]["visual_hull_iou"] for m in result),
            "limitations":limitations or "Synthetic surface voxels, fixed 18 CAD templates, correlated families; no mesh, texture, or arbitrary-photo reconstruction."}
    (out/"metrics.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)
    return report

def main():
    p=argparse.ArgumentParser();p.add_argument("--generator",required=True,type=Path);p.add_argument("--out",required=True,type=Path)
    p.add_argument("--variants",type=int,default=64);p.add_argument("--epochs",type=int,default=20);p.add_argument("--batch",type=int,default=16)
    a=p.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("Enable Kaggle GPU accelerator; no CPU fallback")
    if not 8<=a.variants<=128 or not 1<=a.epochs<=30 or not 4<=a.batch<=32: raise ValueError("Bounded budget exceeded")
    a.out.mkdir(parents=True,exist_ok=False)
    mod=generator(a.generator);data=make_data(mod,a.variants,20261125)
    metadata={"license":"CC0-1.0","source":"Native-model-forge/tools/generate_forge_basics.py","source_commit":SOURCE_COMMIT,"source_blob":SOURCE_BLOB,
      "grid":GRID,"variants":a.variants,"splits":SPLITS,"generator_seed":20261125,
      "data_hashes":{s:hashlib.sha256(x.numpy().tobytes()+y.numpy().tobytes()).hexdigest() for s,(x,y,_) in data.items()},
      "records":{s:r for s,(_,_,r) in data.items()},"private_data":False}
    mf=a.out/"dataset_manifest.json";mf.write_text(json.dumps(metadata,indent=2));h=sha(mf)
    results=[]
    for seed in (43,53,73):
        results.append(run(data,a.out,seed,a.epochs,a.batch,h))
        (a.out/"summary.json").write_text(json.dumps(results,indent=2))
    print("Completed three isolated CUDA candidates; no promotion.")
if __name__=="__main__": main()
