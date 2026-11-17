"""Generate CC0 multi-view object RGB images and semantic part maps.

Training uses four object families. Chair and mug are withheld entirely from
training and validation and appear only in the family-held-out test split.
"""
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
SEEN_CATEGORIES=("crate","stool","lamp","hovercraft")
HELDOUT_CATEGORIES=("chair","mug")
PART_PALETTE={
    "background":(0.02,0.02,0.02,1),
    "body":(0.90,0.08,0.08,1),
    "support":(0.08,0.90,0.08,1),
    "feature":(0.08,0.08,0.90,1),
    "trim":(0.90,0.82,0.08,1),
}
def part_for(name):
    n=name.lower()
    if "rim" in n or "ring" in n: return "trim"
    if any(x in n for x in ("leg","stem","lift pod")): return "support"
    if any(x in n for x in ("rib","handle","thruster","back panel","cockpit","nose armor")): return "feature"
    return "body"
def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(1<<20),b""): h.update(block)
    return h.hexdigest()
def main():
    raw=sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else []
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",type=Path,required=True); ap.add_argument("--seed",type=int,default=202611)
    ap.add_argument("--train-per-category",type=int,default=128); ap.add_argument("--valid-per-category",type=int,default=16)
    ap.add_argument("--test-per-category",type=int,default=16); ap.add_argument("--resolution",type=int,default=96)
    a=ap.parse_args(raw); out=a.out.resolve(); out.mkdir(parents=True,exist_ok=True)
    image_root=out/"images"; image_root.mkdir(exist_ok=True)
    part_root=out/"parts"; part_root.mkdir(exist_ok=True)
    cam=setup_render(a.resolution); scene=bpy.context.scene
    counts={"train":a.train_per_category,"valid":a.valid_per_category,"test":a.test_per_category}
    splits={k:[] for k in counts}; total=0
    for ci,cat in enumerate(CATEGORIES):
        for si,split in enumerate(("train","valid","test")):
            if split in ("train","valid") and cat in HELDOUT_CATEGORIES: continue
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
                mesh_objects=[obj for obj in bpy.context.scene.objects if obj.type=="MESH"]
                for obj in mesh_objects: obj.color=PART_PALETTE[part_for(obj.name)]
                views={}; part_views={}
                for name,direction in view_directions().items():
                    aim_camera(cam,center,direction,scale)
                    image_path=image_root/f"{sid}_{name}.png"; scene.render.filepath=str(image_path)
                    scene.display.shading.color_type="MATERIAL"
                    scene.display.shading.show_shadows=True; scene.display.shading.show_cavity=True
                    bpy.ops.render.render(write_still=True); views[name]=str(image_path.relative_to(out))
                    scene.display.shading.color_type="OBJECT"
                    scene.display.shading.light="FLAT"; scene.display.shading.show_shadows=False
                    scene.display.shading.show_cavity=False; scene.world.color=(0,0,0)
                    part_path=part_root/f"{sid}_{name}_parts.png"; scene.render.filepath=str(part_path)
                    bpy.ops.render.render(write_still=True); part_views[name]=str(part_path.relative_to(out))
                scene.display.shading.color_type="MATERIAL"
                splits[split].append({"id":sid,"category":cat,"object_family":cat,
                    "family_split":"heldout" if cat in HELDOUT_CATEGORIES else "seen",
                    "prompt":prompt,"params":[round(float(x),6) for x in p],
                    "dimensions_m":[round(x,5) for x in dims],"views":views,"part_views":part_views,
                    "color_name":color_name,"style":style})
                total+=1
                if total%25==0: print(json.dumps({"event":"generated","objects":total}),flush=True)
    for split,records in splits.items():
        records.sort(key=lambda r:r["id"])
        (out/f"{split}.jsonl").write_text("".join(json.dumps(r,sort_keys=True)+"\n" for r in records))
    hashes={}
    for path in sorted(out.rglob("*")):
        if path.is_file() and path.name!="dataset_manifest.json": hashes[str(path.relative_to(out))]=sha(path)
    mt=json.dumps({"schema":2,"name":"Forge semantic object-part images v3","license":"CC0-1.0",
        "synthetic_only":True,"source":"Repository Blender geometry; no third-party images or meshes.",
        "generator_seed":a.seed,"blender_version":bpy.app.version_string,"resolution":a.resolution,
        "categories":list(CATEGORIES),"seen_categories":list(SEEN_CATEGORIES),
        "heldout_categories":list(HELDOUT_CATEGORIES),"counts_by_split":{k:len(v) for k,v in splits.items()},
        "views_per_object":list(view_directions()),"part_palette":{k:list(v) for k,v in PART_PALETTE.items()},
        "part_names":["background","body","support","feature","trim"],"file_sha256":hashes},sort_keys=True,indent=2)+"\n"
    (out/"dataset_manifest.json").write_text(mt)
    print(json.dumps({"event":"dataset_complete","objects":total,
      "manifest_sha256":hashlib.sha256(mt.encode()).hexdigest(),
      "counts_by_split":{k:len(v) for k,v in splits.items()}},indent=2),flush=True)
if __name__=="__main__": main()
