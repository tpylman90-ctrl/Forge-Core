"""Generate a deterministic CC0 multi-view object dataset with Blender."""
import argparse, hashlib, json, random, sys
from pathlib import Path
import bpy
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from object_geometry_v1 import CATEGORIES,aim_camera,build_object,clear_scene,setup_render,view_directions

COLORS={"red":(0.72,0.20,0.16,1),"blue":(0.16,0.36,0.72,1),"green":(0.22,0.58,0.35,1),
        "yellow":(0.78,0.58,0.14,1),"gray":(0.42,0.48,0.55,1),"orange":(0.82,0.34,0.10,1)}
STYLES=("angular","rounded","ribbed","compact","wide","tall")
MATERIALS=("painted metal","matte plastic","brushed alloy","ceramic")


def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(1<<20),b""): h.update(block)
    return h.hexdigest()


def main():
    raw=sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else []
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",type=Path,required=True); ap.add_argument("--seed",type=int,default=202611)
    ap.add_argument("--train-per-category",type=int,default=48); ap.add_argument("--valid-per-category",type=int,default=12)
    ap.add_argument("--test-per-category",type=int,default=12); ap.add_argument("--resolution",type=int,default=96)
    a=ap.parse_args(raw); out=a.out.resolve(); out.mkdir(parents=True,exist_ok=True)
    image_root=out/"images"; image_root.mkdir(exist_ok=True)
    cam=setup_render(a.resolution); scene=bpy.context.scene
    counts={"train":a.train_per_category,"valid":a.valid_per_category,"test":a.test_per_category}
    splits={k:[] for k in counts}; total=0
    for ci,cat in enumerate(CATEGORIES):
        for si,split in enumerate(("train","valid","test")):
            for j in range(counts[split]):
                rng=random.Random(a.seed+ci*100000+si*10000+j)
                p=[rng.uniform(0.05,0.95) for _ in range(8)]
                color_name=rng.choice(tuple(COLORS)); color=COLORS[color_name]
                dims=[0.45+1.25*p[k] for k in range(3)]
                style=STYLES[min(5,int(p[3]*6))]; material_name=MATERIALS[min(3,int(p[4]*4))]
                prompt=(f"make a {style} {color_name} {cat} from {material_name}; width {dims[0]:.1f} "
                        f"meters, depth {dims[1]:.1f} meters, height {dims[2]:.1f} meters")
                sid=f"{cat}_{split}_{j:04d}"
                clear_scene(); build_object(cat,p,color); center=(0,0,dims[2]*0.5); scale=max(dims)*1.8
                views={}
                for name,direction in view_directions().items():
                    aim_camera(cam,center,direction,scale)
                    path=image_root/f"{sid}_{name}.png"; scene.render.filepath=str(path)
                    bpy.ops.render.render(write_still=True); views[name]=str(path.relative_to(out))
                splits[split].append({"id":sid,"category":cat,"prompt":prompt,
                    "params":[round(float(x),6) for x in p],"dimensions_m":[round(x,5) for x in dims],
                    "views":views,"color_name":color_name,"style":style})
                total+=1
                if total%25==0: print(json.dumps({"event":"generated","objects":total}),flush=True)
    for split,records in splits.items():
        records.sort(key=lambda r:r["id"])
        (out/f"{split}.jsonl").write_text("".join(json.dumps(r,sort_keys=True)+"\n" for r in records))
    hashes={}
    for path in sorted(out.rglob("*")):
        if path.is_file() and path.name!="dataset_manifest.json": hashes[str(path.relative_to(out))]=sha(path)
    manifest={"schema":1,"name":"Forge procedural object images v1","license":"CC0-1.0",
        "synthetic_only":True,"source":"Repository Blender geometry; no third-party images or meshes.",
        "generator_seed":a.seed,"blender_version":bpy.app.version_string,"resolution":a.resolution,
        "categories":list(CATEGORIES),"counts_by_split":{k:len(v) for k,v in splits.items()},
        "views_per_object":list(view_directions()),"file_sha256":hashes}
    mt=json.dumps(manifest,sort_keys=True,indent=2)+"\n"
    (out/"dataset_manifest.json").write_text(mt)
    print(json.dumps({"event":"dataset_complete","objects":total,
      "manifest_sha256":hashlib.sha256(mt.encode()).hexdigest()},indent=2),flush=True)

if __name__=="__main__": main()
