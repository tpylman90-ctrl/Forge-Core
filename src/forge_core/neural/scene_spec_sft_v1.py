"""Answer-masked, reproducible CC0 scene-spec SFT for isolated Forge tool lanes.

The Forge model is resumed from its verified lane checkpoint. This module uses
only original combinatorial task/spec/why examples and a fresh optimizer because
the objective differs from manual prose continuation. Each seed is a separate
branch; a candidate is selected only if answer-only validation loss improves.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, math, random, re
from pathlib import Path
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from forge_core.neural.model import ByteTransformer, ModelConfig

ROOT_SHA="a7bcdb996e440f44d7496b887b54d19324927b4296fe9ed2f0f0293c88c63a0a"
TOKENIZER_SHA="4ecd7803d4c369e80bda2e74c30cfb769b7e3a942f931df83383eec9f794f04c"
PARENTS={
 "blender":{"run":42541109934,"sha":"672db59bfc44d026f26e38d781c84dd24f92882c65b888639ec66a790002ed68"},
 "godot":{"run":42541109934,"sha":"de7de5052f0a56d821ea3995f1f2e5fbb80d10750e4fb3a564f3c6c4a2a0afdf"},
}
ATTRIBUTION={
 "blender":{"version":"Blender 5.4 Manual","license":"CC BY-SA 4.0 (references only; no prose copied)","url":"https://docs.blender.org/manual/en/5.4/modeling/modifiers/introduction.html"},
 "godot":{"version":"Godot 4.7 documentation","license":"CC BY 3.0 (references only; no prose copied)","url":"https://docs.godotengine.org/en/4.7/tutorials/3d/procedural_geometry/arraymesh.html"},
}
BLENDER_NAMES=("Beacon","Crate","Marker","Pillar","Module","Reactor","Console","Canister")
ROOT_NAMES=("World","Room","TestScene","PropSet","Workshop","Arena","Level","RootNode")
CHILD_NAMES=("Beacon","Marker","Orb","Item","Prop","Sphere","Pickup","Actor")
BLENDER_SIZES=(0.8,1.0,1.25,1.5,2.0,2.5,3.0,4.0)
BEVEL_WIDTHS=(0.03,0.05,0.08,0.1,0.12)
BEVEL_SEGMENTS=(1,2,3,4)
GODOT_RADII=(0.25,0.5,0.75,1.0,1.25,1.5,2.0,3.0)
WHY={
 "blender":"Bevel width controls how far cube edges round; segments control the smoothness of that rounding.",
 "godot":"SphereMesh radius sets the sphere size, and MeshInstance3D displays that mesh as a scene child.",
}

def sha(data:bytes)->str: return hashlib.sha256(data).hexdigest()
def sha_file(path:Path)->str: return sha(path.read_bytes())
def jsonl(rows:list[dict])->bytes:
 return ("".join(json.dumps(row,sort_keys=True,separators=(",",":"),ensure_ascii=False)+"\n" for row in rows)).encode()

def _target(lane:str,values:dict)->str:
 payload={**values,"why":WHY[lane]}
 return json.dumps(payload,separators=(",",":"),ensure_ascii=False)

def _record(lane:str,values:dict,ident:str)->dict:
 if lane=="blender":
  prompt=(f"Blender cube {values['object_name']}, size {values['size']}, bevel width "
          f"{values['bevel_width']}, {values['bevel_segments']} segments. Return JSON and a short why.")
 else:
  prompt=(f"Godot scene root {values['root_name']} with SphereMesh child {values['child_name']}, "
          f"radius {values['radius']}. Return JSON and a short why.")
 return {"id":ident,"lane":lane,"prompt":prompt,"target":_target(lane,values),
         "expected":values,"why":WHY[lane],"license":"CC0-1.0"}

def make_records(lane:str)->dict[str,list[dict]]:
 if lane not in PARENTS: raise ValueError("lane must be blender or godot")
 rows=[]
 if lane=="blender":
  for n in BLENDER_NAMES:
   for size in BLENDER_SIZES:
    for width in BEVEL_WIDTHS:
     for seg in BEVEL_SEGMENTS:
      vals={"object_name":n,"size":size,"bevel_width":width,"bevel_segments":seg}
      ident=f"b-{n}-{size}-{width}-{seg}"
      rows.append(_record(lane,vals,ident))
 else:
  for root in ROOT_NAMES:
   for child in CHILD_NAMES:
    for radius in GODOT_RADII:
     vals={"root_name":root,"child_name":child,"mesh_type":"SphereMesh","radius":radius}
     ident=f"g-{root}-{child}-{radius}"
     rows.append(_record(lane,vals,ident))
 # Hash split is deterministic and independent of run seed.
 splits={"train":[],"valid":[],"test":[]}
 for row in rows:
  bucket=int(hashlib.sha256(row["id"].encode()).hexdigest()[:8],16)%10
  splits["train" if bucket<8 else ("valid" if bucket==8 else "test")].append(row)
 # Keep the exact current engine probe as a held-out composition.
 probe=({"object_name":"Beacon","size":2.0,"bevel_width":0.1,"bevel_segments":3}
        if lane=="blender" else
        {"root_name":"World","child_name":"Beacon","mesh_type":"SphereMesh","radius":1.0})
 probe_id=next(r["id"] for r in rows if r["expected"]==probe)
 for src in ("train","valid"):
  found=next((i for i,r in enumerate(splits[src]) if r["id"]==probe_id),None)
  if found is not None:
   moving=splits[src].pop(found)
   replacement=splits["test"].pop(0)
   splits[src].append(replacement)
   splits["test"].append(moving)
 if not any(r["id"]==probe_id for r in splits["test"]):
  raise AssertionError("fixed engine probe must be in held-out test")
 for items in splits.values(): items.sort(key=lambda r:r["id"])
 return splits

def prepare_data(lane:str,out:Path)->dict:
 splits=make_records(lane); out.mkdir(parents=True,exist_ok=True)
 manifest={
  "schema":1,"lane":lane,"dataset":"Forge original scene-spec tasks",
  "license":"CC0-1.0","generator":"scene_spec_sft_v1.py","generator_version":1,
  "source_references":[ATTRIBUTION[lane]],"reference_prose_copied":False,
  "field_semantics":{"blender":"Blender cube size and bevel modifier settings",
                     "godot":"Godot Node3D root with SphereMesh child"},
  "probe_held_out":True,
  "splits":{},
 }
 for split,rows in splits.items():
  data=jsonl(rows); (out/f"{split}.jsonl").write_bytes(data)
  manifest["splits"][split]={"count":len(rows),"sha256":sha(data)}
 manifest["total_examples"]=sum(len(v) for v in splits.values())
 manifest["train_sha256"]=manifest["splits"]["train"]["sha256"]
 manifest["valid_sha256"]=manifest["splits"]["valid"]["sha256"]
 manifest["test_sha256"]=manifest["splits"]["test"]["sha256"]
 manifest["manifest_sha256"]=sha(json.dumps(manifest,sort_keys=True,separators=(",",":")).encode())
 (out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
 (out/"ATTRIBUTION.txt").write_text(
  f"Training examples are original Forge-Core-authored combinatorial examples, licensed CC0-1.0.\n"
  f"Official source used to verify terminology, without copying prose: {ATTRIBUTION[lane]['version']}\n"
  f"{ATTRIBUTION[lane]['url']}\nReference documentation license: {ATTRIBUTION[lane]['license']}.\n",encoding="utf-8")
 return manifest

def read_rows(path:Path)->list[dict]:
 return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

def encode_record(row:dict,tokenizer:Tokenizer,max_seq:int)->dict:
 p=tokenizer.encode(row["prompt"]).ids
 a=tokenizer.encode(row["target"]).ids
 if not p or not a or len(p)+len(a)>max_seq:
  raise ValueError(f"Prompt/answer exceeds model context: {row['id']} ({len(p)}+{len(a)}>{max_seq})")
 return {"ids":p+a,"prompt_len":len(p),"row":row}

def masked_loss(model:ByteTransformer,encoded:list[dict],batch_size:int=8)->float:
 model.eval(); device=torch.device("cpu")
 total=0.0; count=0
 with torch.no_grad():
  for start in range(0,len(encoded),batch_size):
   batch=encoded[start:start+batch_size]
   width=max(len(x["ids"])-1 for x in batch)
   x=torch.zeros((len(batch),width),dtype=torch.long,device=device)
   y=torch.zeros_like(x); mask=torch.zeros_like(x,dtype=torch.bool)
   for i,item in enumerate(batch):
    ids=item["ids"]; n=len(ids)-1
    x[i,:n]=torch.tensor(ids[:-1]); y[i,:n]=torch.tensor(ids[1:])
    first=item["prompt_len"]-1
    mask[i,first:n]=True
   logits,_=model(x)
   per=F.cross_entropy(logits.reshape(-1,model.cfg.vocab_size),y.reshape(-1),reduction="none").reshape_as(y)
   total+=float(per[mask].sum()); count+=int(mask.sum())
 return total/max(1,count)

def parse_json_answer(text:str,lane:str)->dict|None:
 match=re.search(r"\{[^{}]*\}",text,flags=re.DOTALL)
 if not match: return None
 try: obj=json.loads(match.group(0))
 except json.JSONDecodeError: return None
 if not isinstance(obj,dict) or not isinstance(obj.get("why"),str) or not obj["why"].strip():
  return None
 keys=set(obj)-{"why"}
 expected_keys=({"object_name","size","bevel_width","bevel_segments"} if lane=="blender"
                else {"root_name","child_name","mesh_type","radius"})
 if keys!=expected_keys: return None
 return {k:obj[k] for k in expected_keys}

@torch.no_grad()
def greedy_answer(model:ByteTransformer,tokenizer:Tokenizer,prompt:str,max_new:int=100)->str:
 ids=tokenizer.encode(prompt).ids
 if len(ids)>=model.cfg.max_seq_len: return ""
 model.eval(); out=list(ids)
 for _ in range(min(max_new,model.cfg.max_seq_len-len(ids))):
  logits,_=model(torch.tensor([out[-model.cfg.max_seq_len:]],dtype=torch.long))
  token=int(torch.argmax(logits[0,-1]).item())
  out.append(token)
 return tokenizer.decode(out[len(ids):],skip_special_tokens=True).strip()

def generate_metrics(model:ByteTransformer,tokenizer:Tokenizer,rows:list[dict],lane:str)->dict:
 samples=[]
 for row in rows:
  raw=greedy_answer(model,tokenizer,row["prompt"])
  parsed=parse_json_answer(raw,lane)
  want=row["expected"]
  exact=parsed==want
  samples.append({"id":row["id"],"prompt":row["prompt"],"raw":raw,
                  "valid_schema":parsed is not None,"exact_fields":exact,
                  "why_present":bool(parsed is not None),"expected":want})
 return {"count":len(samples),"schema_valid":sum(x["valid_schema"] for x in samples),
         "exact_field_match":sum(x["exact_fields"] for x in samples),
         "samples":samples}

def train(args)->dict:
 parent_path=args.parent_dir/"checkpoint.pt"; tokenizer_path=args.parent_dir/"tokenizer.json"
 parent_sha=sha_file(parent_path)
 expected=PARENTS[args.lane]["sha"]
 if parent_sha!=expected: raise ValueError(f"Verified {args.lane} parent mismatch: {parent_sha}")
 if sha_file(tokenizer_path)!=TOKENIZER_SHA: raise ValueError("Parent tokenizer SHA mismatch")
 parent=torch.load(parent_path,map_location="cpu",weights_only=True)
 if parent.get("lane")!=args.lane or parent.get("lane_updates")!=1500:
  raise ValueError("Specialist parent lane/update mismatch")
 if parent.get("parent_sha256")!=ROOT_SHA: raise ValueError("Wrong Forge root ancestry")
 data_dir=args.data_dir; manifest=json.loads((data_dir/"manifest.json").read_text(encoding="utf-8"))
 if manifest["lane"]!=args.lane or manifest["license"]!="CC0-1.0": raise ValueError("Dataset lane/license mismatch")
 for split in ("train","valid","test"):
  raw=(data_dir/f"{split}.jsonl").read_bytes()
  if sha(raw)!=manifest["splits"][split]["sha256"]: raise ValueError(f"{split} hash mismatch")
 tokenizer=Tokenizer.from_file(str(tokenizer_path))
 cfg=ModelConfig(**parent["configuration"])
 if tokenizer.get_vocab_size(with_added_tokens=True)!=cfg.vocab_size: raise ValueError("Tokenizer/model vocabulary mismatch")
 data={s:[encode_record(row,tokenizer,cfg.max_seq_len) for row in read_rows(data_dir/f"{s}.jsonl")] for s in ("train","valid","test")}
 if min(len(data[s]) for s in data)<16: raise ValueError("Each split must have at least 16 examples")
 torch.set_num_threads(min(2,torch.get_num_threads()))
 model=ByteTransformer(cfg); model.load_state_dict(parent["model"],strict=True)
 torch.manual_seed(args.seed); sample_rng=random.Random(args.seed)
 # Fresh optimizer: answer-only SFT is a new objective; weights still descend
 # from this exact Forge specialist checkpoint.
 optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=0.01)
 pvalid=masked_loss(model,data["valid"]); ptest=masked_loss(model,data["test"])
 best_valid=pvalid; best_step=0; stale=0; curve=[]
 best_state=copy.deepcopy(model.state_dict()); best_opt=copy.deepcopy(optimizer.state_dict())
 last_state=copy.deepcopy(best_state); last_opt=copy.deepcopy(best_opt)
 update=0
 model.train()
 while update<args.steps:
  chosen=[sample_rng.choice(data["train"]) for _ in range(args.batch_size)]
  width=max(len(item["ids"])-1 for item in chosen)
  x=torch.zeros((len(chosen),width),dtype=torch.long)
  y=torch.zeros_like(x); mask=torch.zeros_like(x,dtype=torch.bool)
  for i,item in enumerate(chosen):
   ids=item["ids"]; n=len(ids)-1
   x[i,:n]=torch.tensor(ids[:-1]); y[i,:n]=torch.tensor(ids[1:])
   mask[i,item["prompt_len"]-1:n]=True
  logits,_=model(x)
  per=F.cross_entropy(logits.reshape(-1,cfg.vocab_size),y.reshape(-1),reduction="none").reshape_as(y)
  loss=per[mask].mean()
  optimizer.zero_grad(set_to_none=True); loss.backward()
  torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step(); update+=1
  if update%args.eval_every==0 or update==args.steps:
   score=masked_loss(model,data["valid"])
   curve.append({"updates":update,"valid_answer_loss":round(score,8),"train_answer_loss":round(float(loss.detach()),8)})
   last_state=copy.deepcopy(model.state_dict()); last_opt=copy.deepcopy(optimizer.state_dict())
   if score < best_valid-args.min_delta:
    best_valid=score; best_step=update; best_state=copy.deepcopy(last_state); best_opt=copy.deepcopy(last_opt); stale=0
   else: stale+=1
   print(json.dumps({"lane":args.lane,"seed":args.seed,"updates":update,
    "valid_answer_loss":round(score,8),"best_valid_answer_loss":round(best_valid,8),
    "stale_evaluations":stale,"patience":args.patience}),flush=True)
   if stale>=args.patience: break
 out=args.out; out.mkdir(parents=True,exist_ok=True)
 for f in ("tokenizer.json",): (out/f).write_bytes(tokenizer_path.read_bytes())
 (out/"dataset-manifest.json").write_bytes((data_dir/"manifest.json").read_bytes())
 for f in ("train.jsonl","valid.jsonl","test.jsonl","ATTRIBUTION.txt"):
  (out/f).write_bytes((data_dir/f).read_bytes())
 def save_state(state,opt_state,path,selected_updates):
  ck=dict(parent)
  ck.update({"model":state,"optimizer":opt_state,"lane":args.lane,
    "sft_parent_checkpoint_sha256":parent_sha,"sft_parent_run":PARENTS[args.lane]["run"],
    "sft_data_manifest_sha256":manifest["manifest_sha256"],
    "sft_train_sha256":manifest["train_sha256"],"sft_valid_sha256":manifest["valid_sha256"],
    "sft_test_sha256":manifest["test_sha256"],"sft_seed":args.seed,
    "sft_updates_this_run":selected_updates,"lane_updates":1500+selected_updates,
    "sft_optimizer_policy":"fresh AdamW for answer-masked objective",
    "sft_torch_rng_state":torch.get_rng_state(),
    "sft_python_rng_state":sample_rng.getstate()})
  torch.save(ck,path)
 save_state(last_state,last_opt,out/"last-checkpoint.pt",update)
 selected_parent=(best_step==0)
 if selected_parent:
  (out/"checkpoint.pt").write_bytes(parent_path.read_bytes())
  selected_model=ByteTransformer(cfg); selected_model.load_state_dict(parent["model"],strict=True)
 else:
  save_state(best_state,best_opt,out/"checkpoint.pt",best_step)
  selected_model=ByteTransformer(cfg); selected_model.load_state_dict(best_state,strict=True)
 valid_gen=generate_metrics(selected_model,tokenizer,[x["row"] for x in data["valid"][:24]],args.lane)
 test_gen=generate_metrics(selected_model,tokenizer,[x["row"] for x in data["test"][:24]],args.lane)
 parent_model=ByteTransformer(cfg); parent_model.load_state_dict(parent["model"],strict=True)
 parent_test_gen=generate_metrics(parent_model,tokenizer,[x["row"] for x in data["test"][:24]],args.lane)
 report={
  "schema":1,"status":"research_only_no_promotion","lane":args.lane,"seed":args.seed,
  "root_parent_sha256":ROOT_SHA,"immediate_parent_run":PARENTS[args.lane]["run"],
  "immediate_parent_checkpoint_sha256":parent_sha,
  "selected_checkpoint_sha256":sha_file(out/"checkpoint.pt"),
  "parent_lane_updates":1500,"updates_attempted":update,"updates_selected":best_step,
  "selected_from":"unchanged_parent" if selected_parent else "SFT_candidate",
  "parent_valid_answer_loss":round(pvalid,8),"selected_valid_answer_loss":round(best_valid,8),
  "parent_test_answer_loss":round(ptest,8),
  "selected_test_answer_loss":round(masked_loss(selected_model,data["test"]),8),
  "sft_data_manifest_sha256":manifest["manifest_sha256"],
  "train_sha256":manifest["train_sha256"],"valid_sha256":manifest["valid_sha256"],
  "test_sha256":manifest["test_sha256"],"tokenizer_sha256":TOKENIZER_SHA,
  "valid_generation":valid_gen,"test_generation_selected":test_gen,"test_generation_parent":parent_test_gen,
  "curve":curve,"training":{"objective":"answer-only token cross-entropy; prompt targets masked",
   "optimizer":"fresh AdamW, lane-local","learning_rate":args.learning_rate,
   "batch_size":args.batch_size,"eval_every":args.eval_every,"patience":args.patience,
   "stop_reason":"early_stopping" if stale>=args.patience else "max_updates",
   "seed":args.seed},"weights_promoted":False,"android_integrated":False,
  "interpretation":"Controlled scene-spec SFT diagnostics; exact-schema and engine checks are separate from held-out loss. No general tool competence claim."
 }
 (out/"metrics.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
 (out/"lineage.json").write_text(json.dumps({
  "lane":args.lane,"seed":args.seed,"root_parent_sha256":ROOT_SHA,
  "immediate_parent_run":PARENTS[args.lane]["run"],"immediate_parent_sha256":parent_sha,
  "selected_checkpoint_sha256":report["selected_checkpoint_sha256"],
  "optimizer_state_resumed":False,"weights_resumed_from_verified_forge_checkpoint":True,
  "lane_updates_parent":1500,"lane_updates_selected":1500+best_step,
  "updates_attempted":update,"sft_data_manifest_sha256":manifest["manifest_sha256"],
  "train_sha256":manifest["train_sha256"],"valid_sha256":manifest["valid_sha256"],
  "test_sha256":manifest["test_sha256"],"tokenizer_sha256":TOKENIZER_SHA,
  "promoted":False,"android_integrated":False},indent=2)+"\n",encoding="utf-8")
 print(json.dumps({"lane":args.lane,"seed":args.seed,"selected_from":report["selected_from"],
  "updates_attempted":update,"updates_selected":best_step,
  "valid_schema":test_gen["schema_valid"],"test_exact_match":test_gen["exact_field_match"],
  "parent_exact_match":parent_test_gen["exact_field_match"],
  "selected_sha256":report["selected_checkpoint_sha256"]},indent=2),flush=True)
 return report

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument("--lane",choices=("blender","godot"),required=True)
 p.add_argument("--parent-dir",type=Path,required=True)
 p.add_argument("--data-dir",type=Path,required=True)
 p.add_argument("--out",type=Path,required=True)
 p.add_argument("--seed",type=int,required=True)
 p.add_argument("--steps",type=int,default=7500)
 p.add_argument("--eval-every",type=int,default=250)
 p.add_argument("--patience",type=int,default=6)
 p.add_argument("--learning-rate",type=float,default=1e-5)
 p.add_argument("--batch-size",type=int,default=4)
 p.add_argument("--min-delta",type=float,default=1e-4)
 a=p.parse_args()
 if not 250<=a.steps<=20000 or not 50<=a.eval_every<=1000 or not 2<=a.patience<=12:
  p.error("steps/eval-every/patience outside safe pilot bounds")
 if not 1e-6<=a.learning_rate<=5e-5 or not 1<=a.batch_size<=16: p.error("learning rate or batch size outside bounds")
 prepare_data(a.lane,a.data_dir)
 train(a)

if __name__=="__main__": main()
