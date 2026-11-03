"""Bounded Forge Teacher v1 neural student study; research-only.

A verified lesson bundle is the ONLY accepted source for supervised
question/answer targets. Training has two arms from identical original
Forge research weights: gold rule labels versus within-skill shuffled
labels. Both receive the same public-language rehearsal and schedules.

This does not run a pretrained teacher: optional local GGUF provides
advisory *planning* only. This trains Forge's own scratch-origin weights.
"""
from __future__ import annotations
import argparse
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import time

import torch

from forge_core.learning.teacher_v1 import audit_bundle, SKILL_SET
from forge_core.neural.model import ByteTransformer, ModelConfig
from forge_core.neural.preschool_copy import shuffle_within_skill
from forge_core.neural.preschool_direct import DirectExample, answer_loss, encode, evaluate_direct
from forge_core.neural.preschool_qa import verify_source
from forge_core.neural.seed_v07 import heldout
from forge_core.neural.train import make_batch


def student_study(*, original_dir: Path, candidate_dir: Path,
                  lesson_dir: Path, output: Path, seed: int = 1729,
                  updates: int = 280, batch_size: int = 6,
                  learning_rate: float = 1e-4,
                  language_weight: float = 0.25) -> dict:
    if not (1 <= seed <= 1_000_000 and 1 <= updates <= 2000
            and 2 <= batch_size <= 6 and 0 < learning_rate <= 0.001
            and 0 <= language_weight < 0.8):
        raise ValueError("Out-of-bounds teacher student experiment")
    torch.set_num_threads(min(2,torch.get_num_threads()))
    train_l, dev_l=audit_bundle(lesson_dir)
    train=[DirectExample(row.id,row.skill,row.prompt,row.answer) for row in train_l]
    dev=[DirectExample(row.id,row.skill,row.prompt,row.answer) for row in dev_l]
    if {row.prompt for row in train} & {row.prompt for row in dev}:
        raise ValueError("Training/development data overlap")
    source, manifest, codec=verify_source(original_dir)
    original_hash=hashlib.sha256((original_dir/"checkpoint.pt").read_bytes()).hexdigest()
    parent_path=candidate_dir/"verified-research.pt"
    research=torch.load(parent_path,map_location="cpu",weights_only=True)
    if (research.get("schema")!=4 or not research.get("experimental_only")
        or research.get("arm")!="verified"
        or research.get("configuration")!=source["configuration"]
        or research.get("source_checkpoint_sha256")!=original_hash):
        raise ValueError("Unapproved parent neural research checkpoint")
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        base=ByteTransformer(ModelConfig(**source["configuration"]))
        base.load_state_dict(research["model"],strict=True)
        verified=copy.deepcopy(base)
        control=copy.deepcopy(base)

    shuffled=shuffle_within_skill(train,seed+41)
    if [x.prompt for x in train] != [x.prompt for x in shuffled]:
        raise AssertionError("The shuffled control received different prompts")
    pool={skill:[i for i,row in enumerate(train) if row.skill==skill]
          for skill in SKILL_SET}
    if any(not pool[skill] for skill in SKILL_SET):
        raise ValueError("Missing curriculum skill")
    rng=random.Random(seed+17)
    # A fixed index schedule, with one question from each category per step.
    skills=tuple(sorted(SKILL_SET))
    schedule=[
        [rng.choice(pool[skill]) for skill in rng.sample(skills,batch_size)]
        for _ in range(updates)
    ]
    gold_encoded=encode(train,codec,base.cfg.max_seq_len)
    shuffled_encoded=encode(shuffled,codec,base.cfg.max_seq_len)
    public_train=torch.tensor(
        codec.encode((original_dir/"train.txt").read_text()).ids,dtype=torch.long)
    public_valid=torch.tensor(
        codec.encode((original_dir/"valid.txt").read_text()).ids,dtype=torch.long)

    def assess(model):
        return {
            "development":evaluate_direct(model,codec,dev),
            "heldout_language_loss":round(
                heldout(model,public_valid,windows=64,seq_len=128),6),
        }

    # Treat development as a monitoring set, NOT an independent final exam.
    baseline=assess(base)
    def adapt(model, records, arm):
        model.train()
        optimizer=torch.optim.AdamW(model.parameters(),lr=learning_rate)
        lang_rng=torch.Generator().manual_seed(seed+900)
        history=[]
        for step,indices in enumerate(schedule,1):
            qa_loss=answer_loss(model,[records[i] for i in indices])
            if language_weight:
                x,y=make_batch(public_train,2,64,lang_rng)
                _,lm_loss=model(x,y)
                total=(1-language_weight)*qa_loss+language_weight*lm_loss
            else:
                total=qa_loss
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step()
            if step==1 or step%100==0 or step==updates:
                history.append({
                    "step":step,"task_loss":round(float(qa_loss.detach()),5),
                })
        print(json.dumps({"arm":arm,"curve":history}),flush=True)
        return history

    start=time.monotonic()
    gold_curve=adapt(verified,gold_encoded,"verified")
    shuffle_curve=adapt(control,shuffled_encoded,"shuffled")
    final_verified=assess(verified)
    final_shuffle=assess(control)
    # Aggregate feedback is the ONLY item permitted to influence future plans.
    dev_progress={
        skill:{
            "correct":final_verified["development"]["by_skill"][skill]["correct"],
            "total":final_verified["development"]["by_skill"][skill]["total"],
        }
        for skill in skills
    }
    retention=final_verified["heldout_language_loss"] <= (
        baseline["heldout_language_loss"]*1.05)
    report={
        "schema":1,"status":"research_only_not_promoted",
        "teacher_mode":"verified-rule-curriculum_selection",
        "external_pretrained_ai_teacher_was_run":False,
        "original_checkpoint_sha256":original_hash,
        "parent_candidate_sha256":hashlib.sha256(parent_path.read_bytes()).hexdigest(),
        "source_train_sha256":manifest["train_sha256"],
        "lesson_manifest_sha256":hashlib.sha256(
            (lesson_dir/"manifest.json").read_bytes()).hexdigest(),
        "train_examples":len(train),"development_examples":len(dev),
        "seed":seed,"updates_per_arm":updates,
        "qa_batch_size":batch_size,
        "language_rehearsal_ratio":language_weight,
        "same_starting_weights_and_minibatches":True,
        "baseline":baseline,"verified":final_verified,
        "shuffled":final_shuffle,
        "language_retention_within_five_percent":retention,
        "verified_curve":gold_curve,"shuffled_curve":shuffle_curve,
        "duration_seconds":round(time.monotonic()-start,2),
        "next_development_scores":dev_progress,
        "limitations":[
            "Model selection uses synthetic development questions, not final untouched exams",
            "AI planning is OPTIONAL; this automatic run uses deterministic rule-selection",
            "A single small study cannot establish understanding or preschool ability",
            "Do not promote either arm, overwrite archived weights or enable Android actions",
        ]
    }
    output.mkdir(parents=True,exist_ok=True)
    (output/"teacher-student-results.json").write_text(
        json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    (output/"next-development-scores.json").write_text(
        json.dumps(dev_progress,indent=2,sort_keys=True)+"\n")
    for arm,network in (("verified",verified),("shuffled",control)):
        torch.save({
            "schema":5,"experimental_only":True,"arm":arm,
            "method":"forge_teacher_v1_bounded_adapter",
            "source_checkpoint_sha256":original_hash,
            "parent_sha256":report["parent_candidate_sha256"],
            "lesson_manifest_sha256":report["lesson_manifest_sha256"],
            "configuration":asdict(network.cfg),
            "model":network.state_dict(),
        },output/f"{arm}-research.pt")
    print(json.dumps({
        "development_before":baseline["development"]["correct"],
        "development_verified":final_verified["development"]["correct"],
        "development_shuffled":final_shuffle["development"]["correct"],
        "development_total":len(dev),
        "language_loss_before":baseline["heldout_language_loss"],
        "language_loss_verified":final_verified["heldout_language_loss"],
        "language_retained":retention,
    },indent=2),flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--original",type=Path,required=True)
    p.add_argument("--parent",type=Path,required=True)
    p.add_argument("--lessons",type=Path,required=True)
    p.add_argument("--out",type=Path,default=Path("artifacts/teacher-student"))
    p.add_argument("--seed",type=int,default=1729)
    p.add_argument("--updates",type=int,default=280)
    a=p.parse_args()
    student_study(original_dir=a.original,candidate_dir=a.parent,
                  lesson_dir=a.lessons,output=a.out,
                  seed=a.seed,updates=a.updates)


if __name__=="__main__":
    main()
