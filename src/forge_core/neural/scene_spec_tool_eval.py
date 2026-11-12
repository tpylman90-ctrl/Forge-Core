"""Evaluation-only, headless Blender/Godot scene-spec benchmark.

Forge emits a tiny JSON description. Only exact whitelisted fields and values
are accepted; fixed harness code builds and inspects scenes. Model text is
never executed. This diagnostic does not update weights or establish practical
tool competence.
"""
from __future__ import annotations
import argparse, hashlib, json, math, re, subprocess, tempfile
from pathlib import Path
import torch
from tokenizers import Tokenizer
from forge_core.neural.model import ByteTransformer, ModelConfig

PROMPTS = {
    "blender": 'Blender cube Beacon, size 2.0, bevel width 0.1, 3 segments. Return JSON and a short why.',
    "godot": 'Godot scene root World with SphereMesh child Beacon, radius 1.0. Return JSON and a short why.',
}
EXPECTED = {
    "blender": {"object_name": "Beacon", "size": 2.0, "bevel_width": 0.1, "bevel_segments": 3},
    "godot": {"root_name": "World", "child_name": "Beacon", "mesh_type": "SphereMesh", "radius": 1.0},
}

def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
    return h.hexdigest()

def generate(checkpoint: Path, lane: str, seed: int, max_tokens: int):
    ckpt=torch.load(checkpoint,map_location="cpu",weights_only=True)
    cfg=ModelConfig(**ckpt["configuration"])
    if ckpt.get("lane")!=lane: raise ValueError(f"Checkpoint lane mismatch: {lane} vs {ckpt.get('lane')}")
    tokenizer_path=checkpoint.parent/"tokenizer.json"
    tokenizer=Tokenizer.from_file(str(tokenizer_path))
    tokenizer_sha=sha256(tokenizer_path)
    recorded_sha=ckpt.get("tokenizer_sha256")
    if recorded_sha and recorded_sha!=tokenizer_sha: raise ValueError("Archived tokenizer hash does not match checkpoint metadata")
    if tokenizer.get_vocab_size(with_added_tokens=True)!=cfg.vocab_size:
        raise ValueError("Tokenizer vocabulary size does not match model configuration")
    prefix_ids=tokenizer.encode(PROMPTS[lane],add_special_tokens=False).ids
    if not prefix_ids or len(prefix_ids)>=cfg.max_seq_len: raise ValueError("Prompt token length is outside the checkpoint context")
    with torch.random.fork_rng():
        model=ByteTransformer(cfg); model.load_state_dict(ckpt["model"],strict=True); model.eval()
        torch.manual_seed(seed)
        prefix=torch.tensor([prefix_ids],dtype=torch.long)
        with torch.no_grad(): out=model.generate(prefix,new_tokens=max_tokens,temperature=0.7)[0].tolist()
    raw=tokenizer.decode(out[len(prefix_ids):],skip_special_tokens=True)
    return raw,{"step":ckpt.get("step"),"lane_updates":ckpt.get("lane_updates"),
                "checkpoint_sha256":sha256(checkpoint),"tokenizer_sha256":tokenizer_sha,
                "prompt":PROMPTS[lane],"seed":seed,"temperature":0.7,
                "generated_tokens":max_tokens,"prompt_tokens":len(prefix_ids)}
def parse_candidate(raw: str,lane: str):
    match=re.search(r"\{[^{}]*\}",raw,flags=re.DOTALL)
    if not match: return None,"no_single_level_json_object"
    try: obj=json.loads(match.group(0))
    except json.JSONDecodeError: return None,"invalid_json"
    if (not isinstance(obj,dict) or set(obj)!=(set(EXPECTED[lane])|{"why"})
        or not isinstance(obj.get("why"),str) or not obj["why"].strip()):
        return None,"wrong_schema_or_missing_why"
    obj={k:v for k,v in obj.items() if k!="why"}
    if lane=="blender":
        ok=(obj["object_name"]=="Beacon" and type(obj["size"]) in (int,float)
            and math.isclose(float(obj["size"]),2.0,abs_tol=1e-5)
            and type(obj["bevel_width"]) in (int,float)
            and math.isclose(float(obj["bevel_width"]),0.1,abs_tol=1e-5)
            and type(obj["bevel_segments"]) is int and obj["bevel_segments"]==3)
    else:
        ok=(obj["root_name"]=="World" and obj["child_name"]=="Beacon"
            and obj["mesh_type"]=="SphereMesh" and type(obj["radius"]) in (int,float)
            and math.isclose(float(obj["radius"]),1.0,abs_tol=1e-5))
    return (obj,"schema_valid") if ok else (None,"schema_value_mismatch")

def blender_build_and_verify(binary: str,spec: dict,work: Path):
    spec_path,result_path=work/"spec.json",work/"blender-result.json"
    spec_path.write_text(json.dumps(spec),encoding="utf-8")
    script=work/"build_blender.py"
    script.write_text(r'''
import bpy,json,sys
from pathlib import Path
args=sys.argv[sys.argv.index("--")+1:]
spec=json.loads(Path(args[0]).read_text()); out=Path(args[1])
bpy.ops.object.select_all(action="SELECT"); bpy.ops.object.delete(use_global=False)
bpy.ops.mesh.primitive_cube_add(size=float(spec["size"]))
obj=bpy.context.object; obj.name=spec["object_name"]
mod=obj.modifiers.new("Edge_Bevel","BEVEL")
mod.width=float(spec["bevel_width"]); mod.segments=int(spec["bevel_segments"])
bpy.ops.wm.save_as_mainfile(filepath=str(out.with_suffix(".blend")))
dims=[float(x) for x in obj.dimensions]
ok=(obj.type=="MESH" and obj.name=="Beacon" and all(abs(x-2.0)<1e-4 for x in dims)
    and len(obj.modifiers)==1 and obj.modifiers[0].type=="BEVEL"
    and abs(obj.modifiers[0].width-0.1)<1e-5 and obj.modifiers[0].segments==3)
result={"engine":"Blender","object_name":obj.name,"object_type":obj.type,
        "dimensions":dims,"modifier":obj.modifiers[0].type,
        "bevel_width":float(obj.modifiers[0].width),
        "bevel_segments":obj.modifiers[0].segments,"valid":ok}
out.write_text(json.dumps(result,indent=2))
if not ok: raise SystemExit("Blender structural validation failed")
''',encoding="utf-8")
    run=subprocess.run([binary,"--background","--factory-startup","--python",str(script),"--",str(spec_path),str(result_path)],
        cwd=work,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=180)
    (work/"blender.log").write_text(run.stdout,encoding="utf-8")
    if run.returncode!=0 or not result_path.exists():
        return {"valid":False,"returncode":run.returncode,"log_tail":run.stdout[-3000:]}
    return json.loads(result_path.read_text(encoding="utf-8"))

def godot_build_and_verify(binary: str,spec: dict,work: Path):
    project=work/"godot_project"; project.mkdir(parents=True,exist_ok=True)
    (project/"project.godot").write_text('config_version=5\n[application]\nconfig/name="ForgeSceneEval"\n',encoding="utf-8")
    scene=f'''[gd_scene load_steps=2 format=3]

[sub_resource type="SphereMesh" id="SphereMesh_beacon"]
radius = {float(spec["radius"])}
height = {float(spec["radius"])*2.0}

[node name="{spec["root_name"]}" type="Node3D"]

[node name="{spec["child_name"]}" type="MeshInstance3D" parent="."]
mesh = SubResource("SphereMesh_beacon")
'''
    (project/"scene.tscn").write_text(scene,encoding="utf-8")
    (project/"validate.gd").write_text(r'''
extends SceneTree
func _initialize() -> void:
    var packed := load("res://scene.tscn") as PackedScene
    if packed == null:
        printerr("scene_load_failed")
        quit(2)
        return
    var root := packed.instantiate()
    var child := root.get_node_or_null("Beacon")
    var ok := root is Node3D and root.name == "World" and child is MeshInstance3D
    if ok:
        ok = child.mesh is SphereMesh and absf((child.mesh as SphereMesh).radius - 1.0) < 0.00001
    print("FORGE_SCENE_VALID=" + str(ok))
    root.free()
    quit(0 if ok else 3)
''',encoding="utf-8")
    run=subprocess.run([binary,"--headless","--path",str(project),"--script","res://validate.gd"],
        cwd=work,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=180)
    (work/"godot.log").write_text(run.stdout,encoding="utf-8")
    valid=run.returncode==0 and "FORGE_SCENE_VALID=true" in run.stdout
    return {"engine":"Godot","root_name":spec["root_name"],"child_name":spec["child_name"],
        "mesh_type":spec["mesh_type"],"radius":spec["radius"],"valid":valid,
        "returncode":run.returncode,"log_tail":run.stdout[-3000:] if not valid else ""}

def version(binary: str,lane: str) -> str:
    p=subprocess.run([binary,"--version"],text=True,stdout=subprocess.PIPE,
                     stderr=subprocess.STDOUT,timeout=30)
    return p.stdout.strip()[:500]

def evaluate_one(binary,lane,spec,work):
    return blender_build_and_verify(binary,spec,work) if lane=="blender" else godot_build_and_verify(binary,spec,work)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lane",choices=("blender","godot"),required=True)
    p.add_argument("--checkpoint-dir",type=Path,required=True)
    p.add_argument("--engine",required=True); p.add_argument("--out",type=Path,required=True)
    p.add_argument("--seed",type=int,default=20261109)
    p.add_argument("--generated-tokens",type=int,default=160)
    p.add_argument("--role",choices=("parent","candidate"),required=True)
    a=p.parse_args()
    ckpt_path=a.checkpoint_dir/"checkpoint.pt"
    if not ckpt_path.is_file() or not (a.checkpoint_dir/"tokenizer.json").is_file():
        raise FileNotFoundError("Expected checkpoint.pt and tokenizer.json")
    raw,lineage=generate(ckpt_path,a.lane,a.seed,a.generated_tokens)
    candidate,parse_status=parse_candidate(raw,a.lane)
    result={"schema":1,"status":"evaluation_only_no_training","lane":a.lane,"role":a.role,
        "lineage":lineage,"engine_version":version(a.engine,a.lane),"spec_generation":{"raw_text":raw,"parse_status":parse_status,
        "schema_valid":candidate is not None},"positive_control":None,"forge_scene":None,
        "interpretation":"A tiny constrained scene-spec diagnostic; it does not establish general Blender/Godot competence."}
    with tempfile.TemporaryDirectory(prefix=f"forge-{a.lane}-scene-eval-") as tmp:
        root=Path(tmp); control=root/"positive_control"; control.mkdir()
        result["positive_control"]=evaluate_one(a.engine,a.lane,EXPECTED[a.lane],control)
        if not result["positive_control"].get("valid"): raise RuntimeError(f"{a.lane} positive-control harness failed")
        if candidate is None:
            result["forge_scene"]={"valid":False,"engine_invoked":False,"failure":parse_status}
        else:
            run_dir=root/"forge_candidate"; run_dir.mkdir()
            result["forge_scene"]=evaluate_one(a.engine,a.lane,candidate,run_dir)
            result["forge_scene"]["engine_invoked"]=True
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2,ensure_ascii=False),flush=True)

if __name__=="__main__": main()
