"""Lane-isolated answer-masked scene-spec SFT v2 with composition/template holdouts."""
from __future__ import annotations
import argparse, copy, hashlib, json, math, random
from pathlib import Path
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from forge_core.neural.model import ByteTransformer, ModelConfig
from forge_core.neural.scene_spec_sft_v1 import (
    ROOT_SHA, TOKENIZER_SHA, sha, sha_file, jsonl, encode_record,
    masked_loss, greedy_answer,
)

ROOT="a7bcdb996e440f44d7496b887b54d19324927b4296fe9ed2f0f0293c88c63a0a"
ORIGINAL={
 "blender":"672db59bfc44d026f26e38d781c84dd24f92882c65b888639ec66a790002ed68",
 "godot":"de7de5052f0a56d821ea3995f1f2e5fbb80d10750e4fb3a564f3c6c4a2a0afdf",
}
V1_BEST={
 "blender":"b5f987b41ebbbe12be2aa2d3312b9f61d12dedd47d665f0982e8c05f38fa26a7",
 "godot":"43f5d8df4e3c6fea3986766cc51186bd4e75874b0f23fbd06668e07706fc9c26",
}
NAMES=("Beacon","Crate","Marker","Pillar","Module","Reactor","Console","Canister")
ROOTS=("World","Room","TestScene","PropSet","Workshop","Arena","Level","RootNode")
CHILDREN=("Beacon","Marker","Orb","Item","Prop","Sphere","Pickup","Actor")
WHY={
 "blender":{
  "cube":"Size sets dimensions; bevel width and segments shape the edges.",
  "uv_sphere":"Radius sets size; segments and rings set mesh detail.",
  "cylinder":"Radius and depth set size; vertices set radial detail.",
 },
 "godot":{
  "sphere":"Radius sets size; the mesh instance displays the sphere.",
  "box":"Uniform size sets dimensions; the mesh instance displays the box.",
  "capsule":"Radius and height define the capsule dimensions.",
 },
}
TRAIN_TEMPLATES={
 "blender":{
  "cube":("Blender cube {name}, size {size}, bevel width {bevel_width}, bevel segments {bevel_segments}. Return JSON and a short why.",
          "Create a Blender cube named {name}. Set size {size}; set bevel width {bevel_width} and bevel segments {bevel_segments}. JSON plus why."),
  "uv_sphere":("Blender UV sphere {name}, radius {radius}, segments {segments}, rings {rings}. Return JSON and a short why.",
               "Create a Blender UV sphere named {name}; radius {radius}; segments {segments}; rings {rings}. JSON plus why."),
  "cylinder":("Blender cylinder {name}, radius {radius}, depth {depth}, vertices {vertices}, bevel width {bevel_width}, bevel segments {bevel_segments}. Return JSON and a short why.",
              "Create a Blender cylinder named {name}; radius {radius}; depth {depth}; vertices {vertices}; bevel width {bevel_width}; bevel segments {bevel_segments}. JSON plus why."),
 },
 "godot":{
  "sphere":("Godot scene root {root_name} with SphereMesh child {child_name}, radius {radius}. Return JSON and a short why.",
            "Create a Godot SphereMesh node named {child_name} under root {root_name}; set radius {radius}. JSON plus why."),
  "box":("Godot scene root {root_name} with BoxMesh child {child_name}, uniform size {extent}. Return JSON and a short why.",
         "Create a Godot BoxMesh node named {child_name} under root {root_name}; set every size axis to {extent}. JSON plus why."),
  "capsule":("Godot scene root {root_name} with CapsuleMesh child {child_name}, radius {radius}, height {height}. Return JSON and a short why.",
             "Create a Godot CapsuleMesh node named {child_name} under root {root_name}; set radius {radius} and height {height}. JSON plus why."),
 },
}
TEST_TEMPLATES={
 "blender":{
  "cube":("Blender task: create a cube called {name}, size {size}, edge bevel width {bevel_width}, bevel segment count {bevel_segments}. JSON and short why.",
          "For Blender, produce a cube spec named {name}: side length {size}, bevel width {bevel_width}, with {bevel_segments} bevel segments. Respond in JSON with a short why."),
  "uv_sphere":("Blender task: create a UV sphere called {name}, radius {radius}, mesh segments {segments}, ring count {rings}. JSON and short why.",
               "For Blender, produce a UV sphere spec named {name}: radius {radius}, {segments} segments, {rings} rings. Respond in JSON with a short why."),
  "cylinder":("Blender task: create a cylinder called {name}, radius {radius}, depth {depth}, vertices {vertices}, bevel width {bevel_width}, bevel segment count {bevel_segments}. JSON and short why.",
              "For Blender, produce a cylinder spec named {name}: radius {radius}, depth {depth}, {vertices} vertices, bevel width {bevel_width}, {bevel_segments} bevel segments. Respond in JSON with a short why."),
 },
 "godot":{
  "sphere":("Godot task: create root {root_name} and child mesh node {child_name} using SphereMesh, radius {radius}. JSON and short why.",
            "For Godot, specify a scene rooted at {root_name} with {child_name} as a SphereMesh node, radius {radius}. Respond in JSON with a short why."),
  "box":("Godot task: create root {root_name} and child mesh node {child_name} using BoxMesh, uniform axis size {extent}. JSON and short why.",
         "For Godot, specify a scene rooted at {root_name} with {child_name} as a BoxMesh node, all size axes {extent}. Respond in JSON with a short why."),
  "capsule":("Godot task: create root {root_name} and child mesh node {child_name} using CapsuleMesh, radius {radius}, height {height}. JSON and short why.",
             "For Godot, specify a scene rooted at {root_name} with {child_name} as a CapsuleMesh node, radius {radius} and height {height}. Respond in JSON with a short why."),
 }
}

def _fmt(v): return str(v)
def _row(lane:str,kind:str,values:dict,ident:str,bucket:int)->dict:
    vals={**values,"kind":kind}
    if lane=="blender":
        names={k:vals.get(k) for k in ("name","size","bevel_width","bevel_segments","radius","segments","rings","depth","vertices") if k in vals}
        prompt_tpl=(TRAIN_TEMPLATES[lane][kind][bucket%2] if bucket<8 else TEST_TEMPLATES[lane][kind][bucket%2])
    else:
        names={k:vals.get(k) for k in ("root_name","child_name","radius","height","extent","mesh_type") if k in vals}
        prompt_tpl=(TRAIN_TEMPLATES[lane][kind][bucket%2] if bucket<8 else TEST_TEMPLATES[lane][kind][bucket%2])
    prompt=prompt_tpl.format(**names)
    target={**vals,"why":WHY[lane][kind]}
    return {"id":ident,"lane":lane,"kind":kind,"prompt":prompt,"target":json.dumps(target,sort_keys=True,separators=(",",":")),
            "expected":vals,"why":WHY[lane][kind],"license":"CC0-1.0"}

def make_records(lane:str)->dict[str,list[dict]]:
    if lane not in ("blender","godot"): raise ValueError("lane must be blender or godot")
    rows=[]
    if lane=="blender":
        for name in NAMES:
            for size in (0.8,1.0,1.5,2.0,2.5,3.0):
                for width in (0.03,0.08,0.1,0.12,0.2):
                    for seg in (1,2,3):
                        v={"name":name,"size":size,"bevel_width":width,"bevel_segments":seg}
                        rows.append(_row(lane,"cube",v,f"b-cube-{name}-{size}-{width}-{seg}",0))
            for radius in (0.4,0.75,1.0,1.25,1.5,2.0):
                for segments in (16,24,32):
                    for rings in (8,12,16):
                        v={"name":name,"radius":radius,"segments":segments,"rings":rings}
                        rows.append(_row(lane,"uv_sphere",v,f"b-sphere-{name}-{radius}-{segments}-{rings}",0))
            for radius in (0.4,0.75,1.0,1.25,1.5,2.0):
                for depth in (0.5,1.0,1.5,2.0):
                    for vertices in (16,24,32):
                        for width in (0.03,0.08,0.12):
                            v={"name":name,"radius":radius,"depth":depth,"vertices":vertices,
                               "bevel_width":width,"bevel_segments":2}
                            rows.append(_row(lane,"cylinder",v,f"b-cylinder-{name}-{radius}-{depth}-{vertices}-{width}",0))
        probes=[
          ("b-cube-Beacon-2.0-0.1-3",{"kind":"cube","name":"Beacon","size":2.0,"bevel_width":0.1,"bevel_segments":3}),
          ("b-sphere-Beacon-1.0-24-16",{"kind":"uv_sphere","name":"Beacon","radius":1.0,"segments":24,"rings":16}),
          ("b-cylinder-Beacon-1.0-2.0-24-0.08",{"kind":"cylinder","name":"Beacon","radius":1.0,"depth":2.0,"vertices":24,"bevel_width":0.08,"bevel_segments":2}),
        ]
    else:
        for root in ROOTS:
            for child in CHILDREN:
                for radius in (0.25,0.5,0.75,1.0,1.5,2.0):
                    vals={"root_name":root,"child_name":child,"mesh_type":"SphereMesh","radius":radius}
                    rows.append(_row(lane,"sphere",vals,f"g-sphere-{root}-{child}-{radius}",0))
                for extent in (0.5,0.75,1.0,1.5,2.0,3.0):
                    vals={"root_name":root,"child_name":child,"mesh_type":"BoxMesh","extent":extent}
                    rows.append(_row(lane,"box",vals,f"g-box-{root}-{child}-{extent}",0))
                for radius in (0.25,0.5,0.75,1.0):
                    for height in (1.0,1.5,2.0,3.0):
                        vals={"root_name":root,"child_name":child,"mesh_type":"CapsuleMesh","radius":radius,"height":height}
                        rows.append(_row(lane,"capsule",vals,f"g-capsule-{root}-{child}-{radius}-{height}",0))
        probes=[
          ("g-sphere-World-Beacon-1.0",{"kind":"sphere","root_name":"World","child_name":"Beacon","mesh_type":"SphereMesh","radius":1.0}),
          ("g-box-World-Beacon-1.0",{"kind":"box","root_name":"World","child_name":"Beacon","mesh_type":"BoxMesh","extent":1.0}),
          ("g-capsule-World-Beacon-0.5-2.0",{"kind":"capsule","root_name":"World","child_name":"Beacon","mesh_type":"CapsuleMesh","radius":0.5,"height":2.0}),
        ]
    splits={"train":[],"valid":[],"test":[]}
    for row in rows:
        bucket=int(hashlib.sha256(row["id"].encode()).hexdigest()[:8],16)%10
        row["prompt"]= (TRAIN_TEMPLATES[lane][row["kind"]][bucket%2].format(**row["expected"])
                        if bucket<8 else TEST_TEMPLATES[lane][row["kind"]][bucket%2].format(**row["expected"]))
        split="train" if bucket<7 else ("valid" if bucket==7 else "test")
        splits[split].append(row)
    # Keep three fixed probes and their task compositions in the test split.
    probe_ids={p[0] for p in probes}
    for row in rows:
        if row["id"] in probe_ids: row["fixed_probe"]=True
    for probe_id in probe_ids:
        found=next((r for group in splits.values() for r in group if r["id"]==probe_id),None)
        if found is None: raise AssertionError(f"Missing probe {probe_id}")
        for group_name in ("train","valid"):
            if found in splits[group_name]:
                moved=splits[group_name].pop(splits[group_name].index(found))
                replacement=splits["test"].pop(0)
                splits[group_name].append(replacement)
                splits["test"].append(moved)
    # Test uses only never-seen prompt phrasings; train and validation use familiar templates.
    for split in ("train","valid"):
        for r in splits[split]:
            r["prompt"]=TRAIN_TEMPLATES[lane][r["kind"]][int(hashlib.sha256(r["id"].encode()).hexdigest()[-2:],16)%2].format(**r["expected"])
    for r in splits["test"]:
        templates=TEST_TEMPLATES[lane][r["kind"]]
        idx=int(hashlib.sha256((r["id"]+":test-template").encode()).hexdigest()[:2],16)%len(templates)
        r["prompt"]=templates[idx].format(**r["expected"])
    for group in splits.values(): group.sort(key=lambda r:r["id"])
    return splits

def prepare_data(lane:str,out:Path)->dict:
    splits=make_records(lane); out.mkdir(parents=True,exist_ok=True)
    manifest={"schema":2,"lane":lane,"dataset":"Forge original multi-family scene-spec tasks v2",
      "license":"CC0-1.0","generator":"scene_spec_sft_v2.py","generator_version":2,
      "source_references":[],"examples_original":True,"reference_prose_copied":False,
      "split_policy":"hash split by full task composition; train/valid prompt templates are disjoint from two held-out test templates; three fixed probes forced into test",
      "families":sorted({r["kind"] for r in splits["train"]}),"probe_compositions_held_out":True,
      "probe_ids":[r["id"] for r in splits["test"] if r.get("fixed_probe")],
      "splits":{}}
    for split,rows in splits.items():
        data=jsonl(rows); (out/f"{split}.jsonl").write_bytes(data)
        manifest["splits"][split]={"count":len(rows),"sha256":sha(data),
          "family_counts":{k:sum(r["kind"]==k for r in rows) for k in sorted({r["kind"] for r in rows})}}
    manifest["manifest_sha256"]=sha(json.dumps(manifest,sort_keys=True,separators=(",",":")).encode())
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    (out/"ATTRIBUTION.txt").write_text("All training examples are original Forge-Core-authored combinatorial tasks, CC0-1.0. No model-generated teacher outputs or copied documentation prose.\n",encoding="utf-8")
    return manifest

def read_rows(path:Path)->list[dict]:
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]

def sample_test_rows(rows:list[dict],per_family:int=34)->list[dict]:
    chosen=[]
    for family in sorted({r["kind"] for r in rows}):
        pool=[r for r in rows if r["kind"]==family]
        pool.sort(key=lambda r:hashlib.sha256((r["id"]+":v2eval").encode()).hexdigest())
        selected=pool[:per_family]
        for probe in (r for r in pool if r.get("fixed_probe")):
            if probe not in selected:
                selected[-1]=probe
        chosen.extend(selected)
    return chosen

def parse_answer(text:str,lane:str,kind:str)->dict|None:
    import re
    m=re.search(r"\{[^{}]*\}",text,re.DOTALL)
    if not m: return None
    try: obj=json.loads(m.group(0))
    except json.JSONDecodeError: return None
    if not isinstance(obj,dict) or not isinstance(obj.get("why"),str) or not obj["why"].strip(): return None
    obj={k:v for k,v in obj.items() if k!="why"}
    kind_keys={
      "blender":{"cube":{"kind","name","size","bevel_width","bevel_segments"},
        "uv_sphere":{"kind","name","radius","segments","rings"},
        "cylinder":{"kind","name","radius","depth","vertices","bevel_width","bevel_segments"}},
      "godot":{"sphere":{"kind","root_name","child_name","mesh_type","radius"},
        "box":{"kind","root_name","child_name","mesh_type","extent"},
        "capsule":{"kind","root_name","child_name","mesh_type","radius","height"}},
    }
    if set(obj)!=kind_keys[lane][kind] or obj.get("kind")!=kind: return None
    return obj

def exact_match(got:dict|None,want:dict)->bool:
    if got is None or set(got)!=set(want): return False
    for k,v in want.items():
        x=got[k]
        if isinstance(v,float):
            if isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isclose(float(x),v,rel_tol=0,abs_tol=1e-5): return False
        elif type(x) is not type(v) or x!=v: return False
    return True

@torch.no_grad()
def generation_metrics(model,tokenizer,rows:list[dict],lane:str)->dict:
    samples=[]; model.eval()
    for row in rows:
        raw=greedy_answer(model,tokenizer,row["prompt"],max_new=110)
        parsed=parse_answer(raw,lane,row["kind"])
        samples.append({"id":row["id"],"kind":row["kind"],"prompt":row["prompt"],"raw":raw,
                        "schema_valid":parsed is not None,"exact_fields":exact_match(parsed,row["expected"])})
    return {"count":len(samples),"schema_valid":sum(x["schema_valid"] for x in samples),
            "exact_field_match":sum(x["exact_fields"] for x in samples),"samples":samples}

def load_model(checkpoint_dir:Path,lane:str,cfg=None):
    ckpath=checkpoint_dir/"checkpoint.pt"; tokpath=checkpoint_dir/"tokenizer.json"
    if not ckpath.exists() or not tokpath.exists(): raise FileNotFoundError(checkpoint_dir)
    ck=torch.load(ckpath,map_location="cpu",weights_only=True)
    if ck.get("lane")!=lane: raise ValueError("Checkpoint lane mismatch")
    if sha_file(tokpath)!=TOKENIZER_SHA: raise ValueError("Tokenizer SHA mismatch")
    cfg=ModelConfig(**ck["configuration"])
    model=ByteTransformer(cfg); model.load_state_dict(ck["model"],strict=True); model.eval()
    tok=Tokenizer.from_file(str(tokpath))
    return ck,model,tok,cfg

def train(a):
    data=prepare_data(a.lane,a.data_dir)
    dirs={"start":a.parent_dir,"baseline":a.baseline_dir,"comparison":a.comparison_dir}
    expected_start=V1_BEST[a.lane] if a.lane=="blender" else ORIGINAL[a.lane]
    if sha_file(dirs["start"]/"checkpoint.pt")!=expected_start: raise ValueError("Unexpected v2 start checkpoint SHA")
    if sha_file(dirs["baseline"]/"checkpoint.pt")!=ORIGINAL[a.lane]: raise ValueError("Original baseline checkpoint SHA mismatch")
    expected_comparison=V1_BEST[a.lane]
    if sha_file(dirs["comparison"]/"checkpoint.pt")!=expected_comparison: raise ValueError("v1 comparison SHA mismatch")
    for split in ("train","valid","test"):
        raw=(a.data_dir/f"{split}.jsonl").read_bytes()
        if sha(raw)!=data["splits"][split]["sha256"]: raise ValueError(f"{split} data hash mismatch")
    parent,model,tokenizer,cfg=load_model(a.parent_dir,a.lane)
    initial_model=ByteTransformer(cfg); initial_model.load_state_dict(parent["model"],strict=True); initial_model.eval()
    base,baseline,_,_=load_model(a.baseline_dir,a.lane)
    compare,comparison,_,_=load_model(a.comparison_dir,a.lane)
    if parent.get("parent_sha256")!=ROOT: raise ValueError("Wrong Forge root ancestry")
    if tokenizer.get_vocab_size(with_added_tokens=True)!=cfg.vocab_size: raise ValueError("Tokenizer/model mismatch")
    encoded={s:[encode_record(r,tokenizer,cfg.max_seq_len) for r in read_rows(a.data_dir/f"{s}.jsonl")] for s in ("train","valid","test")}
    if min(len(encoded[s]) for s in encoded)<64: raise ValueError("Dataset split too small")
    torch.set_num_threads(min(2,torch.get_num_threads()))
    torch.manual_seed(a.seed); rng=random.Random(a.seed)
    start_loss=masked_loss(model,encoded["valid"]); best=start_loss; best_step=0; stale=0; curve=[]
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=0.01)
    best_state=copy.deepcopy(model.state_dict()); best_opt=copy.deepcopy(opt.state_dict())
    last_state=copy.deepcopy(best_state); last_opt=copy.deepcopy(best_opt); update=0
    while update<a.steps:
        batch=[rng.choice(encoded["train"]) for _ in range(a.batch_size)]
        width=max(len(x["ids"])-1 for x in batch)
        x=torch.zeros((len(batch),width),dtype=torch.long); y=torch.zeros_like(x); mask=torch.zeros_like(x,dtype=torch.bool)
        for i,item in enumerate(batch):
            ids=item["ids"]; n=len(ids)-1; p=item["prompt_len"]-1
            x[i,:n]=torch.tensor(ids[:-1]); y[i,:n]=torch.tensor(ids[1:]); mask[i,p:n]=True
        logits,_=model(x); losses=F.cross_entropy(logits.reshape(-1,cfg.vocab_size),y.reshape(-1),reduction="none").reshape_as(y)
        loss=losses[mask].mean(); opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); update+=1
        if update%a.eval_every==0 or update==a.steps:
            score=masked_loss(model,encoded["valid"])
            curve.append({"updates":update,"valid_answer_loss":round(score,8),"train_answer_loss":round(float(loss.detach()),8)})
            last_state=copy.deepcopy(model.state_dict()); last_opt=copy.deepcopy(opt.state_dict())
            if score<best-a.min_delta:
                best=score; best_step=update; best_state=copy.deepcopy(last_state); best_opt=copy.deepcopy(last_opt); stale=0
            else: stale+=1
            print(json.dumps({"lane":a.lane,"seed":a.seed,"updates":update,"valid_answer_loss":round(score,8),
              "best_valid_answer_loss":round(best,8),"stale_evaluations":stale,"patience":a.patience}),flush=True)
            if stale>=a.patience: break
    a.out.mkdir(parents=True,exist_ok=True)
    for f in ("tokenizer.json",): (a.out/f).write_bytes((a.parent_dir/f).read_bytes())
    for f in ("manifest.json","train.jsonl","valid.jsonl","test.jsonl","ATTRIBUTION.txt"):
        (a.out/f).write_bytes((a.data_dir/f).read_bytes())
    def save(state,opt_state,path,steps):
        ck=dict(parent); ck.update({"model":state,"optimizer":opt_state,"lane":a.lane,
          "v2_parent_checkpoint_sha256":sha_file(a.parent_dir/"checkpoint.pt"),
          "sft_data_manifest_sha256":data["manifest_sha256"],"sft_seed":a.seed,
          "sft_updates_this_run":steps,"lane_updates":int(parent.get("lane_updates",0))+steps,
          "sft_optimizer_policy":"fresh AdamW; answer-only loss; prompt targets masked",
          "sft_torch_rng_state":torch.get_rng_state(),"sft_python_rng_state":rng.getstate()})
        torch.save(ck,path)
    save(last_state,last_opt,a.out/"last-checkpoint.pt",update)
    selected_parent=(best_step==0)
    if selected_parent:
        (a.out/"checkpoint.pt").write_bytes((a.parent_dir/"checkpoint.pt").read_bytes())
        selected=ByteTransformer(cfg); selected.load_state_dict(parent["model"],strict=True)
    else:
        save(best_state,best_opt,a.out/"checkpoint.pt",best_step)
        selected=ByteTransformer(cfg); selected.load_state_dict(best_state,strict=True)
    test_rows=read_rows(a.data_dir/"test.jsonl"); eval_rows=sample_test_rows(test_rows,34)
    roles={"original_parent":(baseline,base),"v1_candidate":(comparison,compare),
           "training_start":(initial_model,parent),"selected":(selected,parent)}
    gen={name:generation_metrics(m,tokenizer,eval_rows,a.lane) for name,(m,_) in roles.items()}
    losses={name:{"valid":round(masked_loss(m,encoded["valid"]),8),
                  "test":round(masked_loss(m,encoded["test"]),8)} for name,(m,_) in roles.items()}
    report={"schema":2,"status":"research_only_no_promotion","lane":a.lane,"seed":a.seed,
      "root_parent_sha256":ROOT,"training_start_sha256":sha_file(a.parent_dir/"checkpoint.pt"),
      "original_parent_sha256":sha_file(a.baseline_dir/"checkpoint.pt"),
      "v1_comparison_sha256":sha_file(a.comparison_dir/"checkpoint.pt"),
      "selected_checkpoint_sha256":sha_file(a.out/"checkpoint.pt"),
      "training_start_lane_updates":parent.get("lane_updates"),"selected_updates":best_step,
      "updates_attempted":update,"selected_from":"unchanged_training_start" if selected_parent else "SFT_candidate",
      "losses":losses,"generated":{"sample_count":len(eval_rows),"by_role":gen},
      "families":data["families"],"data_manifest_sha256":data["manifest_sha256"],
      "split_hashes":{k:v["sha256"] for k,v in data["splits"].items()},"tokenizer_sha256":TOKENIZER_SHA,
      "curve":curve,"weights_promoted":False,"android_integrated":False,
      "training":{"objective":"answer-only next-token cross entropy; prompt tokens masked",
       "optimizer":"fresh AdamW","learning_rate":a.lr,"batch_size":a.batch_size,
       "eval_every":a.eval_every,"patience":a.patience,"seed":a.seed},
      "interpretation":"Three-family procedural scene-spec diagnostic. New task combinations and prompt templates were held out. Scores do not establish general Blender/Godot competence."}
    (a.out/"metrics.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    lineage={"lane":a.lane,"seed":a.seed,"training_start_sha256":report["training_start_sha256"],
      "selected_sha256":report["selected_checkpoint_sha256"],"parent_lane_updates":parent.get("lane_updates"),
      "selected_lane_updates":int(parent.get("lane_updates",0))+best_step,"updates_attempted":update,
      "manifest_sha256":data["manifest_sha256"],"split_hashes":report["split_hashes"],
      "optimizer_state_resumed":False,"forge_weights_resumed":True,"promoted":False}
    (a.out/"lineage.json").write_text(json.dumps(lineage,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"lane":a.lane,"seed":a.seed,"selected_from":report["selected_from"],
      "updates_attempted":update,"updates_selected":best_step,
      "selected_exact":gen["selected"]["exact_field_match"],"selected_total":gen["selected"]["count"],
      "parent_exact":gen["original_parent"]["exact_field_match"],"v1_exact":gen["v1_candidate"]["exact_field_match"],
      "selected_sha256":report["selected_checkpoint_sha256"]},indent=2),flush=True)
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lane",choices=("blender","godot"),required=True)
    p.add_argument("--parent-dir",type=Path,required=True); p.add_argument("--baseline-dir",type=Path,required=True)
    p.add_argument("--comparison-dir",type=Path,required=True); p.add_argument("--data-dir",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True); p.add_argument("--seed",type=int,required=True)
    p.add_argument("--steps",type=int,default=5000); p.add_argument("--eval-every",type=int,default=250)
    p.add_argument("--patience",type=int,default=8); p.add_argument("--lr",type=float,default=1e-5)
    p.add_argument("--batch-size",type=int,default=4); p.add_argument("--min-delta",type=float,default=1e-5)
    a=p.parse_args()
    if not 250<=a.steps<=12000 or not 50<=a.eval_every<=1000 or not 2<=a.patience<=16: p.error("Pilot bounds exceeded")
    if not 1e-6<=a.lr<=5e-5 or not 1<=a.batch_size<=16: p.error("Invalid optimizer bounds")
    train(a)

if __name__=="__main__": main()
