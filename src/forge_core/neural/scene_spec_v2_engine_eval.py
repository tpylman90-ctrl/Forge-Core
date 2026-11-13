"""Headless evaluation for generated v2 Blender/Godot scene specs; generated text is never executed."""
from __future__ import annotations
import argparse,json,subprocess,tempfile
from pathlib import Path
import torch
from tokenizers import Tokenizer
from forge_core.neural.model import ByteTransformer,ModelConfig
from forge_core.neural.scene_spec_sft_v2 import (
    sha_file,TOKENIZER_SHA,read_rows,sample_test_rows,greedy_answer,parse_answer,exact_match,
)

def model_outputs(checkpoint_dir:Path,data_dir:Path,lane:str):
    ckpt=checkpoint_dir/"checkpoint.pt"; tokfile=checkpoint_dir/"tokenizer.json"
    ck=torch.load(ckpt,map_location="cpu",weights_only=True)
    if ck.get("lane")!=lane: raise ValueError("checkpoint lane mismatch")
    if sha_file(tokfile)!=TOKENIZER_SHA: raise ValueError("tokenizer hash mismatch")
    cfg=ModelConfig(**ck["configuration"]); tok=Tokenizer.from_file(str(tokfile))
    if tok.get_vocab_size(with_added_tokens=True)!=cfg.vocab_size: raise ValueError("tokenizer/model mismatch")
    model=ByteTransformer(cfg); model.load_state_dict(ck["model"],strict=True); model.eval()
    rows=sample_test_rows(read_rows(data_dir/"test.jsonl"),per_family=6)
    results=[]; matched=[]
    for row in rows:
        raw=greedy_answer(model,tok,row["prompt"],max_new=110)
        parsed=parse_answer(raw,lane,row["kind"])
        exact=exact_match(parsed,row["expected"])
        results.append({"id":row["id"],"kind":row["kind"],"raw":raw,"schema_valid":parsed is not None,"exact_fields":exact})
        if exact: matched.append(parsed)
    return ck,rows,results,matched

def build_blender(binary:str,specs:list[dict],work:Path)->dict:
    payload=work/"specs.json"; result=work/"engine-result.json"; script=work/"build.py"
    payload.write_text(json.dumps(specs),encoding="utf-8")
    script.write_text(r'''
import bpy,json,sys
from pathlib import Path
specs=json.loads(Path(sys.argv[sys.argv.index("--")+1]).read_text())
items=[]
for spec in specs:
    bpy.ops.object.select_all(action="SELECT"); bpy.ops.object.delete(use_global=False)
    kind=spec["kind"]
    if kind=="cube":
        bpy.ops.mesh.primitive_cube_add(size=float(spec["size"]))
    elif kind=="uv_sphere":
        bpy.ops.mesh.primitive_uv_sphere_add(segments=int(spec["segments"]),ring_count=int(spec["rings"]),radius=float(spec["radius"]))
    elif kind=="cylinder":
        bpy.ops.mesh.primitive_cylinder_add(vertices=int(spec["vertices"]),radius=float(spec["radius"]),depth=float(spec["depth"]))
    else:
        raise ValueError("disallowed kind")
    obj=bpy.context.object; obj.name=spec["name"]
    if "bevel_width" in spec:
        mod=obj.modifiers.new("Edge_Bevel","BEVEL")
        mod.width=float(spec["bevel_width"]); mod.segments=int(spec["bevel_segments"])
    if kind=="cube": dims=[float(spec["size"])]*3
    elif kind=="uv_sphere": dims=[float(spec["radius"])*2]*3
    else: dims=[float(spec["radius"])*2,float(spec["radius"])*2,float(spec["depth"])]
    ok=obj.name==spec["name"] and obj.type=="MESH" and all(abs(a-b)<1e-4 for a,b in zip(obj.dimensions,dims))
    if "bevel_width" in spec:
        ok=ok and len(obj.modifiers)==1 and obj.modifiers[0].type=="BEVEL"
        ok=ok and abs(obj.modifiers[0].width-float(spec["bevel_width"]))<1e-5
        ok=ok and obj.modifiers[0].segments==int(spec["bevel_segments"])
    else:
        ok=ok and len(obj.modifiers)==0
    items.append({"kind":kind,"name":obj.name,"valid":bool(ok)})
out=Path(sys.argv[sys.argv.index("--")+2]); out.write_text(json.dumps(items,indent=2))
if not all(x["valid"] for x in items): raise SystemExit("Blender structural check failed")
''',encoding="utf-8")
    p=subprocess.run([binary,"--background","--factory-startup","--python",str(script),"--",str(payload),str(result)],
        cwd=work,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=180)
    (work/"blender.log").write_text(p.stdout,encoding="utf-8")
    if p.returncode or not result.exists(): return {"valid":False,"count":0,"log_tail":p.stdout[-3000:]}
    items=json.loads(result.read_text())
    return {"valid":all(x["valid"] for x in items),"count":len(items),"items":items,"log_tail":""}

def build_godot(binary:str,specs:list[dict],work:Path)->dict:
    project=work/"project"; project.mkdir()
    (project/"project.godot").write_text("""config_version=5
[application]
config/name="ForgeSceneSpecV2"
""",encoding="utf-8")
    (project/"specs.json").write_text(json.dumps(specs),encoding="utf-8")
    (project/"validate.gd").write_text(r'''
extends SceneTree
func _initialize() -> void:
    var specs = JSON.parse_string(FileAccess.get_file_as_string("res://specs.json"))
    if not specs is Array:
        printerr("bad_specs")
        quit(2)
        return
    var checks: Array[bool] = []
    for spec in specs:
        var root := Node3D.new()
        root.name = str(spec["root_name"])
        var child := MeshInstance3D.new()
        child.name = str(spec["child_name"])
        var mesh: PrimitiveMesh
        match str(spec["kind"]):
            "sphere":
                var sphere := SphereMesh.new()
                sphere.radius = float(spec["radius"])
                sphere.height = float(spec["radius"]) * 2.0
                mesh = sphere
            "box":
                var box := BoxMesh.new()
                var extent := float(spec["extent"])
                box.size = Vector3(extent, extent, extent)
                mesh = box
            "capsule":
                var capsule := CapsuleMesh.new()
                capsule.radius = float(spec["radius"])
                capsule.height = float(spec["height"])
                mesh = capsule
            _:
                printerr("disallowed_kind")
                quit(3)
                return
        child.mesh = mesh
        root.add_child(child)
        var ok := root.name == str(spec["root_name"]) and child.name == str(spec["child_name"])
        if spec["kind"] == "sphere":
            ok = ok and mesh is SphereMesh and absf((mesh as SphereMesh).radius - float(spec["radius"])) < 0.00001
        elif spec["kind"] == "box":
            ok = ok and mesh is BoxMesh and (mesh as BoxMesh).size.is_equal_approx(Vector3.ONE * float(spec["extent"]))
        else:
            ok = ok and mesh is CapsuleMesh and absf((mesh as CapsuleMesh).radius - float(spec["radius"])) < 0.00001
            ok = ok and absf((mesh as CapsuleMesh).height - float(spec["height"])) < 0.00001
        checks.append(ok)
        root.free()
    print("FORGE_V2_CHECKS=" + JSON.stringify(checks))
    quit(0 if checks.all(func(v): return v) else 4)
''',encoding="utf-8")
    p=subprocess.run([binary,"--headless","--path",str(project),"--script","res://validate.gd"],
        cwd=work,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=180)
    (work/"godot.log").write_text(p.stdout,encoding="utf-8")
    marker=next((x for x in p.stdout.splitlines() if x.startswith("FORGE_V2_CHECKS=")),None)
     if marker is None: return {"valid":False,"count":0,"log_tail":p.stdout[-3000:]}
     checks=json.loads(marker.split("=",1)[1])
     return {"valid":p.returncode==0 and all(checks),"count":len(checks),
             "items":[{"valid":x} for x in checks],
             "log_tail":p.stdout[-3000:] if p.returncode else ""}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lane",choices=("blender","godot"),required=True)
    p.add_argument("--role",choices=("original_parent","v1_candidate","selected"),required=True)
    p.add_argument("--checkpoint-dir",type=Path,required=True);p.add_argument("--data-dir",type=Path,required=True)
    p.add_argument("--engine",required=True);p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    ck,rows,samples,matched=model_outputs(a.checkpoint_dir,a.data_dir,a.lane)
    controls=[]
    for kind in sorted({r["kind"] for r in rows}):
        controls.append(next(r["expected"] for r in rows if r["kind"]==kind))
    specs=controls+matched
    with tempfile.TemporaryDirectory(prefix=f"forge-v2-{a.lane}-") as tmp:
        result=build_blender(a.engine,specs,Path(tmp)) if a.lane=="blender" else build_godot(a.engine,specs,Path(tmp))
        # The first three inputs are positive controls; remaining scenes came from exact model outputs.
        items=result.get("items",[])
        control_items=items[:len(controls)]
        generated_items=items[len(controls):]
        result["positive_controls"]={"count":len(controls),"valid":len(control_items)==len(controls) and all(x.get("valid",False) for x in control_items)}
        result["generated_exact_scenes"]={"count":len(matched),"valid":all(x.get("valid",False) for x in generated_items) if generated_items else None}
        result["engine_version"]=subprocess.run([a.engine,"--version"],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=30).stdout.strip()[:300]
    report={"schema":1,"status":"evaluation_only","lane":a.lane,"role":a.role,
      "checkpoint_sha256":sha_file(a.checkpoint_dir/"checkpoint.pt"),
      "fixed_probe_tasks":len(rows),"schema_valid":sum(x["schema_valid"] for x in samples),
      "exact_field_match":sum(x["exact_fields"] for x in samples),"samples":samples,
      "headless_scene_build":result,
      "interpretation":"The engine builder accepts only exact task-matching JSON specs and constructs scenes from a whitelist. Text is never executed; this small benchmark does not establish broad tool competence."}
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2),flush=True)
if __name__=="__main__": main()
