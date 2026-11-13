"""Godot-only compact scene-spec SFT v3; exact JSON fields, no prose target."""
from __future__ import annotations
import argparse, copy, json, math, random, re, shutil, tempfile
from pathlib import Path
import torch
import torch.nn.functional as F
from forge_core.neural.model import ByteTransformer
from forge_core.neural.scene_spec_sft_v1 import encode_record, masked_loss, greedy_answer
from forge_core.neural.scene_spec_sft_v2 import (
    ORIGINAL, ROOT, TOKENIZER_SHA, make_records, sha, sha_file, jsonl,
    load_model, read_rows, sample_test_rows,
)
from forge_core.neural.scene_spec_v2_engine_eval import build_godot

FIELDS=("kind","root_name","child_name","mesh_type","radius","extent","height")
EXPECTED_START="f3dbfb896c1e525a2aef97a9fc2aaf06a1fc0b12b822acbd31bbd2f52e5d40dc"
PROMPT_PREFIX=("Return exactly one compact JSON object with keys in this order: "
               "kind, root_name, child_name, mesh_type, radius, extent, height. "
               "Use null for numeric fields that do not apply. No explanation. ")

def canonical_expected(values:dict)->dict:
    return {key:values.get(key,None) for key in FIELDS}

def prepare_data(out:Path)->dict:
    splits=make_records("godot")
    out.mkdir(parents=True,exist_ok=True)
    manifest={"schema":3,"lane":"godot","dataset":"Godot compact exact-field scene specs v3",
      "license":"CC0-1.0","source":"Forge-Core-authored procedural task grid",
      "policy":"reuse v2 composition split; remove prose targets; fixed seven-key output schema; two held-out prompt forms",
      "fields":list(FIELDS),"families":sorted({r["kind"] for r in splits["train"]}),
      "splits":{}}
    for split,source_rows in splits.items():
        rows=[]
        for source in source_rows:
            prompt=source["prompt"]
            prompt=re.sub(r"(?i)(?:Return JSON and a short why|JSON plus why|Respond in JSON with a short why|Output JSON and a short why)\.","Return JSON only.",prompt)
            prompt=PROMPT_PREFIX+prompt
            expected=canonical_expected(source["expected"])
            target=json.dumps(expected,ensure_ascii=False,separators=(",",":"))
            rows.append({"id":source["id"],"lane":"godot","kind":source["kind"],
              "prompt":prompt,"target":target,"expected":expected,"license":"CC0-1.0"})
        payload=jsonl(rows)
        (out/f"{split}.jsonl").write_bytes(payload)
        manifest["splits"][split]={"count":len(rows),"sha256":sha(payload)}
    manifest["manifest_sha256"]=sha(json.dumps(manifest,sort_keys=True,separators=(",",":")).encode())
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    (out/"ATTRIBUTION.txt").write_text(
      "Original Godot task examples are generated from Forge-Core-authored CC0-1.0 grids. "
      "Targets contain exact scene fields only; no teacher outputs or explanatory prose.\n",encoding="utf-8")
    return manifest

def parse_compact(raw:str, expected_kind:str)->dict|None:
    match=re.search(r"\{[^{}]*\}",raw,re.DOTALL)
    if not match: return None
    try: obj=json.loads(match.group(0))
    except json.JSONDecodeError: return None
    if not isinstance(obj,dict): return None
    obj.pop("why",None)
    for key in FIELDS:
        obj.setdefault(key,None)
    if set(obj)!=set(FIELDS): return None
    if obj.get("kind") not in ("sphere","box","capsule"): return None
    for key in ("root_name","child_name","mesh_type"):
        if not isinstance(obj.get(key),str): return None
    for key in ("radius","extent","height"):
        value=obj.get(key)
        if value is not None and (isinstance(value,bool) or not isinstance(value,(int,float))): return None
    return obj

def field_match(got:dict|None,want:dict)->dict:
    if got is None: return {key:False for key in FIELDS}
    results={}
    for key in FIELDS:
        x,v=got.get(key),want.get(key)
        if isinstance(v,float):
            results[key]=(not isinstance(x,bool) and isinstance(x,(int,float))
                          and math.isclose(float(x),v,rel_tol=0,abs_tol=1e-5))
        else:
            results[key]=(type(x) is type(v) and x==v)
    return results

@torch.no_grad()
def evaluate_generation(model,tokenizer,rows:list[dict])->tuple[dict,list[dict]]:
    samples=[]; exact_specs=[]
    model.eval()
    for row in rows:
        raw=greedy_answer(model,tokenizer,row["prompt"],max_new=96)
        parsed=parse_compact(raw,row["kind"])
        per_field=field_match(parsed,row["expected"])
        exact=all(per_field.values())
        matched_text=re.search(r"\{[^{}]*\}",raw,re.DOTALL)
        clean=bool(matched_text and raw.strip()==matched_text.group(0))
        samples.append({"id":row["id"],"kind":row["kind"],"prompt":row["prompt"],
          "raw":raw,"schema_valid":parsed is not None,"exact_fields":exact,
          "field_correct":per_field,"clean_single_json":clean})
        if exact: exact_specs.append(parsed)
    counts={key:sum(s["field_correct"][key] for s in samples) for key in FIELDS}
    return {"count":len(samples),"schema_valid":sum(s["schema_valid"] for s in samples),
      "exact_field_match":sum(s["exact_fields"] for s in samples),
      "clean_single_json":sum(s["clean_single_json"] for s in samples),
      "field_correct_counts":counts,"samples":samples},exact_specs

def engine_check(binary:str,rows:list[dict],specs:list[dict])->dict:
    controls=[]
    for kind in sorted({row["kind"] for row in rows}):
        controls.append(next(row["expected"] for row in rows if row["kind"]==kind))
    inputs=controls+specs
    with tempfile.TemporaryDirectory(prefix="forge-godot-fields-v3-") as tmp:
        result=build_godot(binary,inputs,Path(tmp))
    items=result.get("items",[])
    result["positive_controls"]={"count":len(controls),
      "valid":len(items)>=len(controls) and all(x.get("valid",False) for x in items[:len(controls)])}
    generated=items[len(controls):]
    result["generated_exact_scenes"]={"count":len(specs),
      "valid":all(x.get("valid",False) for x in generated) if generated else None}
    result["engine_version"]=torch.__version__ if False else result.get("engine_version")
    return result

def train(a):
    manifest=prepare_data(a.data_dir)
    start_sha=sha_file(a.start_dir/"checkpoint.pt")
    if start_sha!=EXPECTED_START: raise ValueError(f"Unexpected v2 start checkpoint: {start_sha}")
    if sha_file(a.original_dir/"checkpoint.pt")!=ORIGINAL["godot"]: raise ValueError("Original checkpoint hash mismatch")
    if sha_file(a.v1_dir/"checkpoint.pt")!="43f5d8df4e3c6fea3986766cc51186bd4e75874b0f23fbd06668e07706fc9c26":
        raise ValueError("v1 comparison checkpoint hash mismatch")
    for split in ("train","valid","test"):
        if sha((a.data_dir/f"{split}.jsonl").read_bytes())!=manifest["splits"][split]["sha256"]:
            raise ValueError(f"{split} hash mismatch")
    parent,model,tokenizer,cfg=load_model(a.start_dir,"godot")
    roles={}
    for name,path in (("original_parent",a.original_dir),("v1_candidate",a.v1_dir),("v2_start",a.start_dir)):
        ck,m,t,c=load_model(path,"godot")
        roles[name]=(ck,m)
    if parent.get("parent_sha256")!=ROOT: raise ValueError("Wrong Forge root ancestry")
    encoded={split:[encode_record(row,tokenizer,cfg.max_seq_len) for row in read_rows(a.data_dir/f"{split}.jsonl")]
             for split in ("train","valid","test")}
    torch.set_num_threads(min(2,torch.get_num_threads()))
    torch.manual_seed(a.seed); rng=random.Random(a.seed)
    optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=0.01)
    start_loss=masked_loss(model,encoded["valid"]); best=start_loss; best_step=0; stale=0
    best_state=copy.deepcopy(model.state_dict()); best_opt=copy.deepcopy(optimizer.state_dict())
    last_state=copy.deepcopy(best_state); last_opt=copy.deepcopy(optimizer.state_dict())
    curve=[]; update=0
    while update<a.steps:
        batch=[rng.choice(encoded["train"]) for _ in range(a.batch_size)]
        width=max(len(item["ids"])-1 for item in batch)
        x=torch.zeros((len(batch),width),dtype=torch.long)
        y=torch.zeros_like(x); mask=torch.zeros_like(x,dtype=torch.bool)
        for i,item in enumerate(batch):
            ids=item["ids"]; n=len(ids)-1; p=item["prompt_len"]-1
            x[i,:n]=torch.tensor(ids[:-1]); y[i,:n]=torch.tensor(ids[1:]); mask[i,p:n]=True
        logits,_=model(x)
        losses=F.cross_entropy(logits.reshape(-1,cfg.vocab_size),y.reshape(-1),reduction="none").reshape_as(y)
        loss=losses[mask].mean(); optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step(); update+=1
        if update%a.eval_every==0 or update==a.steps:
            score=masked_loss(model,encoded["valid"])
            curve.append({"updates":update,"valid_compact_json_loss":round(score,8)})
            last_state=copy.deepcopy(model.state_dict()); last_opt=copy.deepcopy(optimizer.state_dict())
            if score<best-a.min_delta:
                best=score; best_step=update; best_state=copy.deepcopy(last_state)
                best_opt=copy.deepcopy(last_opt); stale=0
            else: stale+=1
            print(json.dumps({"lane":"godot","seed":a.seed,"updates":update,
              "valid_loss":round(score,8),"best":round(best,8),"stale":stale}),flush=True)
            if stale>=a.patience: break
    a.out.mkdir(parents=True,exist_ok=True)
    shutil.copy2(a.start_dir/"tokenizer.json",a.out/"tokenizer.json")
    for name in ("manifest.json","train.jsonl","valid.jsonl","test.jsonl","ATTRIBUTION.txt"):
        shutil.copy2(a.data_dir/name,a.out/name)
    def save(state,opt_state,path,steps):
        checkpoint=dict(parent)
        checkpoint.update({"model":state,"optimizer":opt_state,"lane":"godot",
          "godot_field_v3_parent_sha256":start_sha,"sft_data_manifest_sha256":manifest["manifest_sha256"],
          "sft_seed":a.seed,"sft_updates_this_run":steps,
          "lane_updates":int(parent.get("lane_updates",0))+steps,
          "sft_optimizer_policy":"fresh AdamW; prompt-masked compact exact-field JSON; no explanation target",
          "sft_torch_rng_state":torch.get_rng_state(),"sft_python_rng_state":rng.getstate()})
        torch.save(checkpoint,path)
    save(last_state,last_opt,a.out/"last-checkpoint.pt",update)
    if best_step==0:
        shutil.copy2(a.start_dir/"checkpoint.pt",a.out/"checkpoint.pt")
        selected=ByteTransformer(cfg); selected.load_state_dict(parent["model"],strict=True)
    else:
        save(best_state,best_opt,a.out/"checkpoint.pt",best_step)
        selected=ByteTransformer(cfg); selected.load_state_dict(best_state,strict=True)
    test_rows=sample_test_rows(read_rows(a.data_dir/"test.jsonl"),per_family=34)
    generated={}
    matched={}
    for name,(_,m) in roles.items():
        generated[name],matched[name]=evaluate_generation(m,tokenizer,test_rows)
    generated["selected"],matched["selected"]=evaluate_generation(selected,tokenizer,test_rows)
    engine={}
    for name,(_,m) in roles.items():
        _,specs=evaluate_generation(m,tokenizer,test_rows)
        engine[name]=engine_check(a.engine,test_rows,specs)
    engine["selected"]=engine_check(a.engine,test_rows,matched["selected"])
    losses={}
    for name,(_,m) in roles.items():
        losses[name]={"valid":round(masked_loss(m,encoded["valid"]),8),
                      "test":round(masked_loss(m,encoded["test"]),8)}
    losses["selected"]={"valid":round(masked_loss(selected,encoded["valid"]),8),
                        "test":round(masked_loss(selected,encoded["test"]),8)}
    report={"schema":3,"status":"research_only_no_promotion","lane":"godot","seed":a.seed,
      "original_parent_sha256":sha_file(a.original_dir/"checkpoint.pt"),
      "v1_comparison_sha256":sha_file(a.v1_dir/"checkpoint.pt"),
      "v2_training_start_sha256":start_sha,"selected_checkpoint_sha256":sha_file(a.out/"checkpoint.pt"),
      "starting_lane_updates":parent.get("lane_updates"),"selected_updates":best_step,
      "updates_attempted":update,"losses":losses,"generated":generated,"engine":engine,
      "data_manifest_sha256":manifest["manifest_sha256"],
      "split_hashes":manifest["splits"],"curve":curve,
      "optimizer":{"name":"AdamW","learning_rate":a.lr,"batch_size":a.batch_size,
                   "eval_every":a.eval_every,"patience":a.patience,"seed":a.seed},
      "weights_promoted":False,"android_integrated":False,
      "interpretation":"Compact seven-field Godot output with no prose target; exact-field and headless-engine checks are narrow diagnostics, not broad tool-competence certification."}
    (a.out/"metrics.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    lineage={"lane":"godot","seed":a.seed,"parent_sha256":start_sha,
      "selected_sha256":report["selected_checkpoint_sha256"],"parent_lane_updates":parent.get("lane_updates"),
      "selected_lane_updates":int(parent.get("lane_updates",0))+best_step,
      "updates_attempted":update,"manifest_sha256":manifest["manifest_sha256"],
      "optimizer_state_resumed":False,"forge_weights_resumed":True,"promoted":False}
    (a.out/"lineage.json").write_text(json.dumps(lineage,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"seed":a.seed,"selected_updates":best_step,
      "heldout_exact":generated["selected"]["exact_field_match"],
      "heldout_total":generated["selected"]["count"],
      "per_field":generated["selected"]["field_correct_counts"],
      "engine_exact":engine["selected"]["generated_exact_scenes"]},indent=2),flush=True)
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start-dir",type=Path,required=True)
    p.add_argument("--original-dir",type=Path,required=True)
    p.add_argument("--v1-dir",type=Path,required=True)
    p.add_argument("--data-dir",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--engine",required=True)
    p.add_argument("--seed",type=int,required=True)
    p.add_argument("--steps",type=int,default=5000)
    p.add_argument("--eval-every",type=int,default=250)
    p.add_argument("--patience",type=int,default=8)
    p.add_argument("--lr",type=float,default=1e-5)
    p.add_argument("--batch-size",type=int,default=4)
    p.add_argument("--min-delta",type=float,default=1e-5)
    a=p.parse_args()
    if not 250<=a.steps<=12000 or not 50<=a.eval_every<=1000 or not 2<=a.patience<=16:
        p.error("Pilot bounds exceeded")
    if not 1e-6<=a.lr<=5e-5 or not 1<=a.batch_size<=16: p.error("Invalid optimizer bounds")
    train(a)
if __name__=="__main__": main()
