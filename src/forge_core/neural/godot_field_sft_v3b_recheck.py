"""Re-run pinned headless Godot checks against saved v3b generations."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from forge_core.neural.godot_field_sft_v3 import engine_check, parse_compact
from forge_core.neural.scene_spec_sft_v2 import read_rows, sample_test_rows

ROLES=("original_parent","v1_candidate","v2_start","selected")

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifacts-root",type=Path,required=True)
    p.add_argument("--engine",required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    report={"schema":1,"source_run":43430665839,"status":"research_only_recheck","seeds":{}}
    for seed in (871,872,873):
        root=a.artifacts_root/f"seed{seed}"
        metrics=json.loads((root/"checkpoints/candidate/metrics.json").read_text(encoding="utf-8"))
        rows=sample_test_rows(read_rows(root/"data/test.jsonl"),per_family=34)
        roles={}
        for role in ROLES:
            specs=[]
            for sample in metrics["generated"][role]["samples"]:
                if not sample.get("exact_fields"):
                    continue
                parsed=parse_compact(sample["raw"],sample["kind"])
                if parsed is not None:
                    specs.append(parsed)
            check=engine_check(a.engine,rows,specs)
            roles[role]={
                "exact_outputs":len(specs),
                "positive_controls":check.get("positive_controls"),
                "generated_exact_scenes":check.get("generated_exact_scenes"),
                "godot_valid":check.get("valid"),
                "godot_count":check.get("count"),
                "log_tail":check.get("log_tail",""),
            }
        report["seeds"][str(seed)]=roles
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({seed:roles["selected"] for seed,roles in report["seeds"].items()},indent=2),flush=True)

if __name__=="__main__":
    main()
