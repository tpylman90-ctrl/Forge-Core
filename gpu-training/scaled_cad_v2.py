"""Scaled CAD + licensed scan surface dataset for CUDA pilot. Geometry only."""
import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path
import numpy as np
import torch
from cad_shape_v1 import generator, run, SPLITS, SOURCE_COMMIT, SOURCE_BLOB, GRID

GSO_REPO="hugging-3d/google-scanned-objects"
GSO_REV="b944bf0e56d93c82652b8b7243187be237434ed4"
SEED=20261125
PRESETS={"small":(256,16,4),"medium":(512,32,43),"large":(1024,128,43)}

def digest(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def split_for(identifier):
    n=int(hashlib.sha256(("forge-gso-v1:"+identifier).encode()).hexdigest()[:8],16)%100
    return "train" if n<80 else "valid" if n<90 else "test"

def points(tri, key):
    tri=np.asarray(tri,dtype=np.float32)
    if tri.ndim!=3 or tri.shape[1:]!=(3,3) or not np.isfinite(tri).all():
        raise ValueError("invalid triangles")
    lo=tri.min(axis=(0,1)); hi=tri.max(axis=(0,1)); extent=float(max(hi-lo))
    if extent<=1e-9: raise ValueError("degenerate model")
    tri=(tri-(lo+hi)/2)/extent
    area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)
    if not np.isfinite(area.sum()) or area.sum()<=0: raise ValueError("zero area")
    rng=np.random.default_rng(int(hashlib.sha256(key.encode()).hexdigest()[:16],16))
    chosen=tri[rng.choice(len(tri),size=24000,p=area/area.sum())]
    a=rng.random((len(chosen),1));b=rng.random((len(chosen),1))
    flip=a+b>1;a[flip]=1-a[flip];b[flip]=1-b[flip]
    return (chosen[:,0]+a*(chosen[:,1]-chosen[:,0])+b*(chosen[:,2]-chosen[:,0])).astype(np.float32)

def accepted_license(value):
    normalized="".join(c for c in str(value).lower() if c.isalnum())
    return normalized in {"ccby40","creativecommonsattribution40international",
      "creativecommonsattribution40","httpscreativecommonsorglicensesby40",
      "httpcreativecommonsorglicensesby40"}

def scan_sources(shard_count, cache):
    from huggingface_hub import hf_hub_download, HfApi
    import trimesh
    files=sorted(p for p in HfApi().list_repo_files(GSO_REPO,repo_type="dataset",revision=GSO_REV) if p.startswith("data/") and p.endswith(".tar"))
    if not files: raise RuntimeError("No GSO shards available")
    seen_ids=set();seen_mesh=set()
    for shard in files[:shard_count]:
        local=hf_hub_download(GSO_REPO,shard,repo_type="dataset",revision=GSO_REV,cache_dir=cache)
        with tarfile.open(local,"r:*") as tf:
            members={m.name:m for m in tf.getmembers() if m.isfile()}
            for name,m in sorted(members.items()):
                if not name.endswith(".json") or m.size>1<<20: continue
                meta=json.loads(tf.extractfile(m).read())
                ident=str(meta.get("object_id",""))
                if not ident or ident in seen_ids or not accepted_license(meta.get("license_id")):
                    continue
                meshname=name[:-5]+".glb"
                glb=members.get(meshname)
                if glb is None or glb.size>100*1024*1024: continue
                raw=tf.extractfile(glb).read(); h=hashlib.sha256(raw).hexdigest()
                if h in seen_mesh: continue
                try:
                    scene=trimesh.load(io.BytesIO(raw),file_type="glb",force="scene",process=False)
                    mesh=scene.dump(concatenate=True)
                    if len(mesh.faces)>1_000_000: continue
                    cloud=points(mesh.triangles,"gso:"+ident)
                except (ValueError,TypeError,IndexError,RuntimeError):
                    continue
                seen_ids.add(ident);seen_mesh.add(h)
                yield {"id":"gso:"+ident,"split":split_for(ident),"cloud":cloud,
                       "license":"CC-BY-4.0","source":GSO_REPO,"revision":GSO_REV,
                       "copyright":str(meta.get("copyright") or "Google LLC"),
                       "shard":shard,"mesh_sha256":h}
        print("Processed GSO shard",shard,"unique approved objects",len(seen_ids),flush=True)
    if not seen_ids: raise RuntimeError("No individually licensed GSO mesh passed intake; inspect metadata rather than bypassing gate")

def materialize(root, sources, cad_variants, scan_variants):
    data={};records={}; hashes={}
    for split in ("train","valid","test"):
        subset=[s for s in sources if s["split"]==split]
        count=sum(cad_variants if s["source"]=="Forge Basics" else scan_variants for s in subset)
        if not count: raise RuntimeError("Empty split")
        xp=root/f"{split}-views.npy";yp=root/f"{split}-surface.npy"
        x=np.lib.format.open_memmap(xp,mode="w+",dtype=np.uint8,shape=(count,3,GRID,GRID))
        y=np.lib.format.open_memmap(yp,mode="w+",dtype=np.uint8,shape=(count,GRID,GRID,GRID))
        offset=0; rec=[]
        for s in subset:
            cloud=s["cloud"]
            variants=cad_variants if s["source"]=="Forge Basics" else scan_variants
            begin=offset
            for j in range(variants):
                rng=np.random.default_rng(int(hashlib.sha256((s["id"]+":"+str(j)).encode()).hexdigest()[:16],16))
                # Preserve one unmodified normalized source, vary proportions and orientation afterwards.
                scale=np.ones(3) if j==0 else rng.uniform(.55,1.,3)
                theta=0. if j==0 else rng.uniform(-np.pi,np.pi)
                c,ss=np.cos(theta),np.sin(theta)
                rot=np.array([[c,0,ss],[0,1,0],[-ss,0,c]])
                p=(cloud*scale)@rot.T
                p=p/max(1.,float(np.abs(p).max())*2)
                idx=np.clip(np.floor((p+.5)*(GRID-2)+1).astype(np.int32),0,GRID-1)
                v=np.zeros((GRID,GRID,GRID),np.uint8);v[tuple(idx.T)]=1
                y[offset]=v;x[offset]=np.stack([v.max(axis=i) for i in range(3)])
                offset+=1
            rec.append({k:v for k,v in s.items() if k!="cloud"}|{"first_row":begin,"count":variants})
        x.flush();y.flush()
        records[split]=rec;hashes[split]={"views":digest(xp),"surface":digest(yp),"samples":count}
        data[split]=(torch.from_numpy(x),torch.from_numpy(y),rec)
        print(split,count,"samples from",len(subset),"sources",flush=True)
    return data,records,hashes

def main():
    p=argparse.ArgumentParser();p.add_argument("--generator",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    p.add_argument("--preset",choices=PRESETS,default="medium");p.add_argument("--cad-only",action="store_true")
    p.add_argument("--epochs",type=int,default=8);p.add_argument("--batch",type=int,default=32);p.add_argument("--seed",type=int,choices=[43,53,73])
    a=p.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required; no CPU fallback")
    if not 1<=a.epochs<=20 or not 4<=a.batch<=32: raise ValueError("Invalid compute bounds")
    a.out.mkdir(parents=True,exist_ok=False)
    cad_n,gso_n,shards=PRESETS[a.preset]
    mod=generator(a.generator);sources=[]
    for split,ids in SPLITS.items():
        for ident in ids:
            tri=np.asarray(mod.make_asset(ident).data,dtype=np.float32).reshape(-1,6)[:,:3].reshape(-1,3,3)
            sources.append({"id":"cad:"+ident,"source":"Forge Basics","revision":SOURCE_COMMIT,
                            "license":"CC0-1.0","split":split,"cloud":points(tri,ident)})
    if not a.cad_only: sources.extend(scan_sources(shards,str(a.out/"hf-cache")))
    data,records,hashes=materialize(a.out,sources,cad_n,gso_n)
    meta={"schema":2,"preset":a.preset,"cad_source_blob":SOURCE_BLOB,"gso_revision":None if a.cad_only else GSO_REV,
          "records":records,"split_hashes":hashes,"grid":GRID,"target":"sampled surface voxels; not solid occupancy",
          "source_object_count":len(sources),"sample_count":sum(v["samples"] for v in hashes.values()),
          "limitations":["Augmentations are not independent assets","Near-duplicate geometry across distinct scan IDs not excluded","No RGB/photo or texture training"]}
    mf=a.out/"dataset_manifest.json";mf.write_text(json.dumps(meta,indent=2));h=digest(mf)
    reports=[]
    for seed in ([a.seed] if a.seed else [43,53,73]):
        reports.append(run(data,a.out,seed,a.epochs,a.batch,h))
        (a.out/"summary.json").write_text(json.dumps(reports,indent=2))
    print("Completed CUDA candidates; no automatic promotion.")
if __name__=="__main__": main()
