"""Forge English prerequisite v3: copy unfamiliar LETTERS before grammar.

An isolated, controlled adaptation of the original scratch-trained Forge Seed
v0.7 Transformer. Four original verified tasks progress through character
copying, spaced sequence copying, joining, then appending an unseen letter.

The same exact training examples are seen in staged vs mixed order; another
arm receives shuffled output labels, and one rehearses only original stories.
All models start from identical audited source weights. No pretrained teachers,
user data, Android or experimental-weight promotion.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import re
import time

import torch
from torch import nn
from torch.nn import functional as F

from forge_core.neural.model import ByteTransformer, ModelConfig
from forge_core.neural.preschool_qa import verify_source
from forge_core.neural.train import make_batch
from forge_core.neural.english_production_v2 import (
    Lesson, encode_samples, train_loss, comparable_story_loss,
)

SKILLS=("copy_letter","copy_spaced","join_letters","append_letter")
ARMS=("verified_staged","verified_mixed","shuffled_staged","story_only")
SEEDS=(9231,9232)
CONSONANTS="bcdfghjklmnpqrstvwxyz"
VOWELS="aeiou"
LETTERS=CONSONANTS+VOWELS

QUESTIONS={
    "train":{
        "copy_letter":"Copy the single letter written after letter=.",
        "copy_spaced":"Write the same characters with spaces between them.",
        "join_letters":"Join these characters into one unspaced word.",
        "append_letter":"Make one unspaced word and add the ending letter.",
    },
    "development":{
        "copy_letter":"What exactly is the specified lowercase character?",
        "copy_spaced":"Repeat the written sequence including the separating spaces.",
        "join_letters":"Give the contiguous letters without spaces.",
        "append_letter":"Produce the joined letters followed by the new final character.",
    },
    "exam":{
        "copy_letter":"Provide only the character identified in the source.",
        "copy_spaced":"Reproduce the character series in the same order, spaced.",
        "join_letters":"Combine the ordered letter symbols into one string.",
        "append_letter":"Combine the letters and attach the single new suffix character.",
    },
}


def split_bucket(word: str) -> str:
    """Character strings across train/dev/exam have disjoint hash buckets."""
    if not re.fullmatch("[a-z]+",word):
        raise ValueError("Invalid letter sequence")
    value=hashlib.sha256(("forge-char-v3:"+word).encode()).digest()[0]%10
    return ("train" if value<7 else "development" if value==7 else "exam")


def random_letters(rng: random.Random, length: int) -> str:
    if not 3<=length<=7:
        raise ValueError("Unsupported copy string length")
    return "".join(rng.choice(CONSONANTS if i%2==0 else VOWELS)
                   for i in range(length))


def expected(skill: str, source: str) -> str:
    if skill=="copy_letter":
        m=re.fullmatch(r"letter=([a-z]);context=([a-z]+)",source)
        if m:
            return m.group(1)
    elif skill in ("copy_spaced","join_letters"):
        m=re.fullmatch(r"letters=([a-z](?: [a-z]){2,6})",source)
        if m:
            return m.group(1) if skill=="copy_spaced" else m.group(1).replace(" ","")
    elif skill=="append_letter":
        m=re.fullmatch(r"letters=([a-z](?: [a-z]){2,6});append=([a-z])",source)
        if m:
            return m.group(1).replace(" ","")+m.group(2)
    raise ValueError("Unrecognized character lesson source")


def prompt(split: str, skill: str, source: str) -> str:
    return (f"Letter practice\nSource: {source}\n"
            f"{QUESTIONS[split][skill]}\nAnswer:")


def validate(rows: list[Lesson], split: str | None=None) -> None:
    if not rows or len(set(x.id for x in rows))!=len(rows):
        raise ValueError("Invalid exercise identifiers")
    if len(set(x.prompt for x in rows))!=len(rows):
        raise ValueError("Repeated exercise prompt")
    for row in rows:
        if row.split not in QUESTIONS or row.skill not in SKILLS:
            raise ValueError("Invalid character exercise category")
        if split is not None and row.split!=split:
            raise ValueError("Cross-split exercise")
        if row.prompt!=prompt(row.split,row.skill,row.source):
            raise ValueError("Unreviewed character question content")
        if row.answer!=expected(row.skill,row.source):
            raise ValueError("Independent character-rule answer mismatch")
        # We know how to write individual letters from English's
        # universal alphabet. Longer novel sequences are bucketed.
        if row.skill=="copy_letter":
            m=re.fullmatch(r"letter=[a-z];context=([a-z]+)",row.source)
            key=m.group(1)
        else:
            key=row.answer.replace(" ","")
        if split_bucket(key)!=row.split:
            raise ValueError("Character-string test leakage across splits")


def make_lessons(split: str, per_skill: int,
                 seed: int) -> list[Lesson]:
    if split not in QUESTIONS or not 4<=per_skill<=150 or per_skill%2:
        raise ValueError("Outside approved character curriculum size")
    rng=random.Random(seed+{"train":17,"development":1901,"exam":9917}[split])
    records=[]
    for skill in SKILLS:
        seen=set()
        for idx in range(per_skill):
            for _ in range(10000):
                base=random_letters(rng,rng.randint(3,6))
                if skill=="copy_letter":
                    letter=rng.choice(LETTERS)
                    source=f"letter={letter};context={base}"
                    key=base
                elif skill=="copy_spaced":
                    source="letters="+" ".join(base)
                    key=base
                elif skill=="join_letters":
                    source="letters="+" ".join(base)
                    key=base
                else:
                    suffix=rng.choice(LETTERS)
                    source="letters="+" ".join(base)+";append="+suffix
                    key=base+suffix
                if source not in seen and split_bucket(key)==split:
                    seen.add(source)
                    records.append(Lesson(
                        f"char-v3:{split}:{skill}:{idx}",skill,split,source,
                        prompt(split,skill,source),expected(skill,source)))
                    break
            else:
                raise ValueError("Unable to construct disjoint character examples")
    validate(records,split)
    return records


def shuffle_answers(rows: list[Lesson],seed: int) -> list[Lesson]:
    rng=random.Random(seed)
    replacements={}
    for skill in SKILLS:
        original=[r.answer for r in rows if r.skill==skill]
        for attempt in range(40):
            perm=original.copy()
            rng.shuffle(perm)
            if sum(x!=y for x,y in zip(original,perm))>=len(original)//2:
                break
        else:
            raise ValueError("No meaningful character null label shuffle")
        replacements[skill]=iter(perm)
    return [Lesson(r.id,r.skill,r.split,r.source,r.prompt,
                   next(replacements[r.skill])) for r in rows]


def schedules(rows: list[Lesson],seed: int,
              updates_per_level: int = 120):
    if not 4<=updates_per_level<=250:
        raise ValueError("Unbounded copying curriculum")
    buckets={skill:[i for i,r in enumerate(rows) if r.skill==skill]
             for skill in SKILLS}
    if any(not v for v in buckets.values()):
        raise ValueError("Missing character lesson category")
    rng=random.Random(seed)
    ordered=[]
    for phase,focus in enumerate(SKILLS):
        for i in range(updates_per_level):
            old=SKILLS[(seed+i)%phase] if phase else focus
            ordered.append([rng.choice(buckets[focus]) for _ in range(3)]+
                           [rng.choice(buckets[old])])
    mixed=ordered.copy()
    rng.shuffle(mixed)
    if sorted(map(tuple,ordered))!=sorted(map(tuple,mixed)):
        raise AssertionError("Mixed control mismatched target exposure")
    return ordered,mixed


def levenshtein(a: str,b: str) -> int:
    prev=list(range(len(b)+1))
    for i,char in enumerate(a,1):
        row=[i]
        for j,other in enumerate(b,1):
            row.append(min(row[-1]+1,prev[j]+1,
                           prev[j-1]+int(char!=other)))
        prev=row
    return prev[-1]


def normalized(raw: str) -> str:
    return " ".join(raw.split("\n",1)[0].strip().casefold().split())


@torch.no_grad()
def evaluate(model: ByteTransformer,rows: list[Lesson],
             codec,max_tokens: int=20) -> dict:
    if not rows or not 2<=max_tokens<=32:
        raise ValueError("Unsupported character exam")
    model.eval()
    totals={skill:{"total":0,"exact":0,"first_character":0,
                   "edit_distance":0,"target_characters":0,
                   "terminated":0} for skill in SKILLS}
    previews=[]
    for case in rows:
        prefix=codec.encode(case.prompt).ids
        if not prefix or len(prefix)>model.cfg.max_seq_len:
            raise ValueError("Character question exceeds original context")
        ids=list(prefix)
        tokens=[]
        finished=False
        for _ in range(max_tokens):
            x=torch.tensor([ids[-model.cfg.max_seq_len:]],dtype=torch.long)
            logits,_=model(x)
            following=int(logits[0,-1].argmax())
            ids.append(following)
            tokens.append(following)
            if "\n" in codec.decode(tokens):
                finished=True
                break
        raw=codec.decode(tokens)
        guess=normalized(raw)
        gold=normalized(case.answer)
        # Expose character-level progress despite zero exact outputs.
        compact_guess="".join(c for c in guess if c in LETTERS)
        compact_gold="".join(c for c in gold if c in LETTERS)
        bucket=totals[case.skill]
        bucket["total"]+=1
        bucket["exact"]+=int(guess==gold)
        bucket["first_character"]+=int(bool(compact_guess) and
                                       compact_guess[0]==compact_gold[0])
        bucket["edit_distance"]+=levenshtein(compact_gold,compact_guess)
        bucket["target_characters"]+=len(compact_gold)
        bucket["terminated"]+=int(finished)
        if len(previews)<len(SKILLS) and not any(
                x["skill"]==case.skill for x in previews):
            previews.append({"skill":case.skill,"target":case.answer,
                             "generated":raw[:160],"exact":guess==gold})
    n=len(rows)
    return {
        "exact":sum(v["exact"] for v in totals.values()),
        "total":n,
        "exact_accuracy":round(sum(v["exact"] for v in totals.values())/n,4),
        "by_skill":{skill:{
            **v,"exact_accuracy":round(v["exact"]/max(1,v["total"]),4),
            "first_character_rate":round(v["first_character"]/max(1,v["total"]),4),
            "character_error_rate":round(v["edit_distance"]/
                                         max(1,v["target_characters"]),4),
            "newline_terminated_rate":round(v["terminated"]/
                                            max(1,v["total"]),4),
        } for skill,v in totals.items()},
        "examples":previews,
        "scoring":"Open-ended greedy exact text, first char, Levenshtein and newline; no answer choices",
    }


def study(*,folder: Path,out: Path,updates_per_level: int=120,
          seeds: tuple[int,...]=SEEDS,lr: float=.00012,
          rehearsal_weight: float=.2) -> dict:
    if (not 4<=updates_per_level<=250 or not seeds or len(seeds)>2
            or len(set(seeds))!=len(seeds) or not 0<lr<=.001
            or not 0<=rehearsal_weight<=.4):
        raise ValueError("Unapproved character bridge CPU workload")
    torch.set_num_threads(min(2,torch.get_num_threads()))
    checkpoint,manifest,codec=verify_source(folder)
    cfg=ModelConfig(**checkpoint["configuration"])
    source_sha=hashlib.sha256((folder/"checkpoint.pt").read_bytes()).hexdigest()
    source_tokens=torch.tensor(codec.encode((folder/"train.txt").read_text()).ids,
                               dtype=torch.long)
    valid_tokens=torch.tensor(codec.encode((folder/"valid.txt").read_text()).ids,
                              dtype=torch.long)
    dev=make_lessons("development",per_skill=10,seed=77333)
    exam=make_lessons("exam",per_skill=24,seed=99173)
    original=ByteTransformer(cfg)
    original.load_state_dict(checkpoint["model"],strict=True)
    original_lm=comparable_story_loss(original,valid_tokens)
    base=evaluate(original,exam,codec)
    torch.set_num_threads(min(2,torch.get_num_threads()))
    out.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    report=[]
    for seed in seeds:
        train=make_lessons("train",per_skill=100,seed=seed)
        if {r.prompt for r in train} & {r.prompt for r in dev+exam}:
            raise ValueError("Train/test character prompt collision")
        teacher=encode_samples(train,codec,cfg.max_seq_len)
        null=shuffle_answers(train,seed+218)
        wrong=encode_samples(null,codec,cfg.max_seq_len)
        for rows in (dev,exam):
            encode_samples(rows,codec,cfg.max_seq_len)
        staged,mixed=schedules(train,seed=seed,
                               updates_per_level=updates_per_level)
        plans={"verified_staged":staged,"verified_mixed":mixed,
               "shuffled_staged":staged,"story_only":mixed}
        data={"seed":seed,"arms":{}}
        for arm in ARMS:
            model=ByteTransformer(cfg)
            model.load_state_dict(checkpoint["model"],strict=True)
            optim=torch.optim.AdamW(model.parameters(),lr=lr)
            g=torch.Generator().manual_seed(seed+3101)
            samples=wrong if arm=="shuffled_staged" else teacher
            best=None
            winner=(-1.,-1.,float("-inf"),-1)
            curve=[]
            for update,indices in enumerate(plans[arm],1):
                model.train()
                bx,by=make_batch(source_tokens,2,64,g)
                _,story_loss=model(bx,by)
                lesson=None
                if arm=="story_only":
                    loss=story_loss
                else:
                    lesson=train_loss(model,[samples[i] for i in indices])
                    loss=(1-rehearsal_weight)*lesson+rehearsal_weight*story_loss
                optim.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),1)
                optim.step()
                if update==1 or update%updates_per_level==0:
                    diagnostic=evaluate(model,dev,codec)
                    first=sum(v["first_character"] for v in
                              diagnostic["by_skill"].values())
                    edit=sum(v["edit_distance"] for v in
                             diagnostic["by_skill"].values())
                    rank=(diagnostic["exact"],first,-edit,update)
                    if rank>winner:
                        winner=rank
                        best={"step":update,
                              "weights":copy.deepcopy(model.state_dict())}
                    curve.append({
                        "update":update,"development_exact":diagnostic["exact"],
                        "development_first_character":first,
                        "development_edit_distance":edit,
                        "train_loss":round(float(loss.detach()),5),
                        "lesson_loss":round(float(lesson.detach()),5)
                        if lesson is not None else None,
                    })
                    print(json.dumps({"seed":seed,"arm":arm,**curve[-1]}),flush=True)
            if best is None:
                raise RuntimeError("No checkpoint selected using development")
            last=copy.deepcopy(model.state_dict())
            last_optim=optim.state_dict()
            model.load_state_dict(best["weights"])
            shifted=evaluate(model,exam,codec)
            development=evaluate(model,dev,codec)
            pub_loss=comparable_story_loss(model,valid_tokens)
            torch.save({
                "schema":1,"research_only":True,"seed":seed,"arm":arm,
                "source_checkpoint_sha256":source_sha,
                "tokenizer_sha256":checkpoint["tokenizer_sha256"],
                "configuration":asdict(cfg),
                "selected_dev_step":best["step"],"best_weights":best["weights"],
                "last_weights":last,"optimizer_last":last_optim,
                "rehearsal_rng":g.get_state(),"torch_rng":torch.get_rng_state(),
            },out/f"seed-{seed}-{arm}-research.pt")
            data["arms"][arm]={
                "selected_dev_step":best["step"],
                "development":development,
                "heldout_char_exam":shifted,
                "public_story_loss":pub_loss,
                "public_story_loss_ratio":round(pub_loss/original_lm,5),
                "curve":curve,
            }
        report.append(data)
        (out/f"seed-{seed}-char-metrics.json").write_text(
            json.dumps(data,indent=2)+"\n",encoding="utf-8")
    result={
        "schema":1,"study":"forge-char-copy-bridge-v3",
        "status":"research_only_no_promotion",
        "source_run":41572257363,
        "source_sha256":source_sha,
        "tokenizer_sha256":checkpoint["tokenizer_sha256"],
        "train_corpus_sha256":manifest["train_sha256"],
        "valid_corpus_sha256":manifest["valid_sha256"],
        "same_original_pretraining":True,
        "updates_per_arm_seed":len(staged),
        "seeds":list(seeds),
        "training_cases_per_seed":len(train),
        "development_cases":len(dev),"heldout_cases":len(exam),
        "original":{"exam":base,"public_story_loss":original_lm},
        "results":report,
        "seconds":round(time.monotonic()-started,2),
        "promoted":False,
        "limitations":[
            "This is a micro-literacy copying diagnostic, not conversational English",
            "Letters recur across splits; whole strings are disjoint by SHA256 bucket",
            "Results measure synthetic transformation formats, not new real nouns",
            "The original BPE tokenizer need not naturally isolate single letters",
            "Learning tasks and controls differ in teacher information by design",
            "The same formal structure occurs in train and exam; no external literacy test",
            "No downloaded pretrained weights, private user data or Android changes",
            "Any new trained checkpoint stays research-only",
        ],
    }
    (out/"english-char-copy-v3.json").write_text(
        json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({
        "seconds":result["seconds"],
        "frozen_exact":base["exact"],
        "runs":[{
            "seed":row["seed"],
            "arms":{a:{
                "exact":d["heldout_char_exam"]["exact"],
                "first_char":sum(v["first_character"] for v in
                                 d["heldout_char_exam"]["by_skill"].values()),
                "errors":sum(v["edit_distance"] for v in
                              d["heldout_char_exam"]["by_skill"].values()),
                "story_ratio":d["public_story_loss_ratio"],
            } for a,d in row["arms"].items()}
        } for row in report],
    },indent=2),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir",type=Path,required=True)
    parser.add_argument("--out",type=Path,default=Path("results/english-char-copy-v3"))
    parser.add_argument("--updates-per-level",type=int,default=120)
    parser.add_argument("--seeds",type=int,choices=(1,2),default=2)
    args=parser.parse_args()
    study(folder=args.checkpoint_dir,out=args.out,
          updates_per_level=args.updates_per_level,seeds=SEEDS[:args.seeds])


if __name__=="__main__":
    main()
