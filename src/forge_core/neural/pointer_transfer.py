"""Untouched cross-template examination of Forge's trained neural pointer.

No training and NO checkpoint promotion. Novel names, objects, and
sentence constructions are fixed in this module *before* evaluating the
new pointer head. Three predetermined seeds yield 360 original cases.
Original Forge language backbone remains frozen.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

import torch

from forge_core.neural.model import ByteTransformer, ModelConfig
from forge_core.neural.preschool_copy import (
    TRAIN_PEOPLE, EVAL_PEOPLE, TRAIN_WORDS, EVAL_WORDS,
)
from forge_core.neural.preschool_direct import DirectExample
from forge_core.neural.preschool_qa import verify_source
from forge_core.neural.span_pointer_lab import (
    WordPointer, features_for, evaluate,
)

EXAM_PEOPLE = ("Yara","Dax","Mabel","Rafi","Zuri","Pavel",
               "Toni","Ines","Rohan","Yuki","Sofia","Miles")
EXAM_WORDS = ("compass","lantern","chalk","satchel","whistle","marble",
              "drum","berry","flag","feather","bicycle","badge")
COLORS = ("red","blue","green","yellow","purple",
          "orange","pink","black","white","brown")
NUMBERS = ("two","three","four","five","six","seven",
           "eight","nine","ten","eleven")
SKILLS = ("copy","select","recipient","giver","color","quantity")


def make_exam(seed:int, per_skill:int=20) -> list[DirectExample]:
    if not 20 <= per_skill <= 60:
        raise ValueError("Exam case budget outside allowed range")
    if (set(EXAM_PEOPLE) & (set(TRAIN_PEOPLE)|set(EVAL_PEOPLE))
        or set(EXAM_WORDS) & (set(TRAIN_WORDS)|set(EVAL_WORDS))):
        raise ValueError("External-template exam vocabulary leaked into previous tasks")
    rng = random.Random(seed)
    examples=[]
    for skill in SKILLS:
        for i in range(per_skill):
            a,b = rng.sample(EXAM_PEOPLE,2)
            x,y,z = rng.sample(EXAM_WORDS,3)
            c1,c2 = rng.sample(COLORS,2)
            n1,n2 = rng.sample(NUMBERS,2)
            # A different set of narrative/question templates from prior
            # Forge Core copying and role training.
            if skill=="copy":
                target = a if i%2 else x
                prompt=(
                    f"School exercise: {b} mentioned {y}. Copy the exact "
                    f"word between these parentheses ({target})."
                    if i%2 else
                    f"Teacher's note: Ignore {z} and {b}; the written "
                    f"target is '{target}'. Echo only that target."
                )
                gold=target
            elif skill=="select":
                selected=i%3
                words=(x,y,z)
                prompt=(
                    f"Ordered labels: one = {x}; two = {y}; three = {z}. "
                    f"What word is label {('one','two','three')[selected]}?"
                    if i%2 else
                    f"Index of words: slot 1 contains {x}, slot 2 contains "
                    f"{y}, slot 3 contains {z}. Return the contents of slot "
                    f"{selected+1}."
                )
                gold=words[selected]
            elif skill=="recipient":
                prompt=(
                    f"After {a} passed the {x} along, {b} held it. "
                    f"Identify the person who ended up with the {x}."
                    if i%2 else
                    f"The {x} moved away from {a} and into {b}'s hands. "
                    f"Name the final holder."
                )
                gold=b
            elif skill=="giver":
                prompt=(
                    f"At the beginning, {a} held the {x}. Next {b} "
                    f"received it from {a}. Who supplied the {x}?"
                    if i%2 else
                    f"{a} sent the {x} over to {b}. Identify the sender."
                )
                gold=a
            elif skill=="color":
                target=b if i%2 else a
                prompt=(
                    f"Before the show, {a} chose a {c1} {x}; {b} selected "
                    f"a {c2} {y}. Report the shade chosen by {target}."
                    if i%2 else
                    f"Items at school: {b}'s {y} is {c2}, whereas "
                    f"{a}'s {x} is {c1}. Which color belongs to {target}?"
                )
                gold=c2 if i%2 else c1
            else:
                target=b if i%2 else a
                prompt=(
                    f"Report for {a}: {n1} stones. Report for {b}: {n2} "
                    f"shells. State the quantity recorded for {target}."
                    if i%2 else
                    f"{b} counted {n2} shells; {a} counted {n1} stones. "
                    f"How many did {target} count?"
                )
                gold=n2 if i%2 else n1
            # Non-semantic session marker prevents accidental duplicates
            # across predetermined exam seeds; it encodes NO answer or skill.
            session = {5101:"morning",5102:"afternoon",5103:"evening"}.get(
                seed, f"session-{seed}")
            prompt = (f"Story: During the {session} class, {prompt}"
                      "\nQuestion: Answer using one source word.\nAnswer:")
            if gold.casefold() not in prompt.casefold():
                raise AssertionError("Exam truth absent from source")
            examples.append(DirectExample(
                f"novel-exam:{seed}:{skill}:{i}",skill,prompt,gold))
    if len({x.prompt for x in examples}) != len(examples):
        raise ValueError("Duplicate unseen exam questions")
    return examples


def examine(*, source:Path, parent:Path, pointer:Path,
            output:Path, seeds:tuple[int,...]=(5101,5102,5103)) -> dict:
    torch.set_num_threads(min(2,torch.get_num_threads()))
    original,manifest,codec=verify_source(source)
    root_sha=hashlib.sha256((source/"checkpoint.pt").read_bytes()).hexdigest()
    parent_path=parent/"verified-research.pt"
    research=torch.load(parent_path,map_location="cpu",weights_only=True)
    if (research.get("schema")!=4 or not research.get("experimental_only")
        or research.get("arm")!="verified"
        or research.get("source_checkpoint_sha256")!=root_sha
        or research.get("configuration")!=original["configuration"]):
        raise ValueError("Invalid parent research checkpoint")
    parent_sha=hashlib.sha256(parent_path.read_bytes()).hexdigest()
    backbone=ByteTransformer(ModelConfig(**original["configuration"]))
    backbone.load_state_dict(research["model"],strict=True)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)
    heads={}
    for name in ("verified","wrong-source"):
        data=torch.load(pointer/f"{name}-pointer.pt",
                        map_location="cpu",weights_only=True)
        if (data.get("schema")!=1 or not data.get("experimental_only")
            or data.get("arm")!=name
            or data.get("source_checkpoint_sha256")!=root_sha
            or data.get("parent_sha256")!=parent_sha
            or data.get("width")!=backbone.cfg.n_embd):
            raise ValueError("Pointer provenance mismatch")
        model=WordPointer(backbone.cfg.n_embd)
        model.load_state_dict(data["model"],strict=True)
        heads[name]=model
    reports=[]
    all_rows=[]
    for seed in seeds:
        rows=make_exam(seed)
        all_rows.extend(rows)
        features=[features_for(backbone,codec,row) for row in rows]
        reports.append({
            "seed":seed,
            "verified":evaluate(heads["verified"],features),
            "wrong_source":evaluate(heads["wrong-source"],features),
        })
    if len({x.prompt for x in all_rows})!=len(all_rows):
        raise ValueError("Exam has duplicate questions across seeds")
    totals={name:sum(x[name]["correct"] for x in reports)
            for name in ("verified","wrong_source")}
    report={
        "schema":1,"status":"external_template_family_internal_synthetic_exam",
        "exam_seeds":list(seeds),"cases":len(all_rows),
        "original_model_sha256":root_sha,
        "parent_model_sha256":parent_sha,
        "source_train_sha256":manifest["train_sha256"],
        "no_training_or_checkpoint_changes":True,
        "new_names_and_objects_disjoint":True,
        "new_sentence_templates":True,
        "verified_correct":totals["verified"],
        "wrong_control_correct":totals["wrong_source"],
        "verified_accuracy":round(totals["verified"]/len(all_rows),4),
        "wrong_control_accuracy":round(totals["wrong_source"]/len(all_rows),4),
        "by_seed":reports,
        "claim":"Neural source-word copying on synthetic novel templates only",
        "limitations":[
            "Internally constructed exam; not independently annotated natural-language material",
            "New templates and novel names but only six constrained source-word skills",
            "No third-party pretrained teacher weights used by Forge",
            "No general preschool comprehension or causal reasoning certification",
            "Do not promote experimental pointer or student weights automatically",
        ],
    }
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({
        "cases":len(all_rows),
        "verified_correct":report["verified_correct"],
        "wrong_control_correct":report["wrong_control_correct"],
        "verified_accuracy":report["verified_accuracy"],
        "by_seed":[{"seed":r["seed"],"verified":r["verified"]["correct"],
                    "control":r["wrong_source"]["correct"]}
                   for r in reports],
    },indent=2),flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--parent",type=Path,required=True)
    p.add_argument("--pointer",type=Path,required=True)
    p.add_argument("--out",type=Path,default=Path("artifacts/pointer-transfer.json"))
    a=p.parse_args()
    examine(source=a.source,parent=a.parent,pointer=a.pointer,output=a.out)


if __name__=="__main__":
    main()
