"""Forge English v4: source-position letter selection with explicit assembly.

The original scratch-trained v0.7 language model is asked to name the
letter at a requested position in a previously unseen 3–7 letter string.
A deterministic (external) evaluator assembles the predicted letters.
It must NEVER be called unaided, natural, whole-word generation.

Research-only supervision: source-derived correct per-position letter;
controls: same-exposure staged/mixed, shuffled-letter, first-only, and
public-story-only. No pretrained AI, no Android changes, no promotion.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
import re
import time

import torch
from torch import nn

from forge_core.neural.model import ByteTransformer, ModelConfig
from forge_core.neural.preschool_qa import verify_source
from forge_core.neural.english_production_v2 import (
    Lesson, encode_samples, train_loss, comparable_story_loss,
)
from forge_core.neural.train import make_batch

ARMS=("position_staged","position_mixed","position_shuffled",
      "first_only","story_only")
SEEDS=(10331,10332)
ALPHABET="abcdefghijklmnopqrstuvwxyz"
VOWELS="aeiou"
CONSONANTS="bcdfghjklmnpqrstvwxyz"
SPLITS=("train","development","exam")
QUESTIONS={
    "train":"Which lowercase character is in the indicated position? Write just that letter.",
    "development":"Return only the letter that occupies the requested numbered place.",
    "exam":"Identify the single character at this one-based index in the written series.",
}


def bucket(value: str) -> str:
    if not re.fullmatch("[a-z]{3,7}",value):
        raise ValueError("Only bounded lowercase sequences are accepted")
    digest=hashlib.sha256(("forge-pos-v4:"+value).encode()).digest()
    b=digest[0]%10
    return "train" if b<7 else "development" if b==7 else "exam"


def source_string(rng: random.Random) -> str:
    n=rng.randint(3,7)
    return "".join(rng.choice(CONSONANTS if i%2==0 else VOWELS)
                   for i in range(n))


@dataclass(frozen=True)
class PositionLesson:
    id: str
    split: str
    string: str
    index: int
    prompt: str
    answer: str


def make_prompt(split: str, string: str, position: int) -> str:
    if split not in SPLITS or not 1<=position<=len(string) or bucket(string)!=split:
        raise ValueError("Cannot prompt outside registered split/positions")
    spaced=" ".join(string)
    return (f"Letter position exercise\nLetters: {spaced}\n"
            f"Index: {position}\n{QUESTIONS[split]}\nAnswer:")


def generate(split: str, *, count: int, seed: int) -> list[PositionLesson]:
    if split not in SPLITS or not 6<=count<=180:
        raise ValueError("Unregistered split size")
    rng=random.Random(seed+{"train":11,"development":1837,"exam":8113}[split])
    seen=set()
    rows=[]
    for sample in range(count):
        for attempt in range(20000):
            word=source_string(rng)
            if word not in seen and bucket(word)==split:
                seen.add(word)
                break
        else:
            raise ValueError("Could not sample independent source strings")
        for position in range(1,len(word)+1):
            rows.append(PositionLesson(
                f"forge-position-v4:{split}:{sample:03}:{position:02}",
                split,word,position,make_prompt(split,word,position),
                word[position-1]))
    verify(rows,split)
    return rows


def verify(rows: list[PositionLesson], split: str | None=None):
    if not rows or len(set(r.id for r in rows))!=len(rows):
        raise ValueError("Duplicate or missing source positions")
    if len(set(r.prompt for r in rows))!=len(rows):
        raise ValueError("Repeated question prompts")
    by_word={}
    for row in rows:
        if row.split not in SPLITS or (split and row.split!=split):
            raise ValueError("Cross-split teaching leakage")
        if bucket(row.string)!=row.split or not 1<=row.index<=len(row.string):
            raise ValueError("Incorrect split or source position")
        if row.prompt!=make_prompt(row.split,row.string,row.index):
            raise ValueError("Question/source mismatch")
        if row.answer!=row.string[row.index-1]:
            raise ValueError("Corrupted answer is not the source character")
        by_word.setdefault(row.string,set()).add(row.index)
    for word,positions in by_word.items():
        if positions!=set(range(1,len(word)+1)):
            raise ValueError("Incomplete source character demonstration")


def corrupt_labels(rows: list[PositionLesson],seed: int):
    rng=random.Random(seed)
    bucketed={i:[r.answer for r in rows if r.index==i]
              for i in range(1,8)}
    for index,values in bucketed.items():
        if not values:
            continue
        for attempt in range(100):
            wrong=values.copy()
            rng.shuffle(wrong)
            if sum(x!=y for x,y in zip(values,wrong))>=len(values)//2:
                bucketed[index]=iter(wrong)
                break
        else:
            raise ValueError("Failed source-blind shuffled-label control")
    # Corrupted rows deliberately fail the independent gold validator;
    # never pass them to training pipeline as verified teaching data.
    return [PositionLesson(r.id,r.split,r.string,r.index,r.prompt,
                           next(bucketed[r.index])) for r in rows]


def sequence_schedules(rows: list[PositionLesson], *, seed: int,
                       steps_per_phase: int):
    if not 4<=steps_per_phase<=250:
        raise ValueError("Outside approved study budget")
    buckets={pos:[i for i,r in enumerate(rows) if r.index==pos]
             for pos in range(1,8)}
    if any(not values for values in buckets.values()):
        raise ValueError("No source cases for a letter position")
    rng=random.Random(seed)
    batches=[]
    # Stage 1 focuses positions 1/2; stage 2 1–3; stage 3 1–5;
    # stage 4 mixes ALL positions 1–7.
    stages=((1,2),(1,2,3),(1,2,3,4,5),(1,2,3,4,5,6,7))
    for allowed in stages:
        for step in range(steps_per_phase):
            # Every batch has four one-letter training cases.
            # At least one high index in later stages.
            indexes=[rng.choice(buckets[rng.choice(allowed)])
                     for _ in range(3)]
            indexes.append(rng.choice(buckets[max(allowed)]))
            batches.append(indexes)
    mixed=batches.copy()
    rng.shuffle(mixed)
    if sorted(map(tuple,batches))!=sorted(map(tuple,mixed)):
        raise AssertionError("Position curriculum control changed exposures")
    return batches,mixed


def make_encoded(rows,codec,max_context):
    # Preserve existing source-only-prefix / target-only-loss behavior.
    wrapped=[Lesson(x.id,"position",x.split,
                    f"letters={x.string};index={x.index}",
                    x.prompt,x.answer) for x in rows]
    return encode_samples(wrapped,codec,max_context)


def normalize_generated(raw: str) -> str:
    return raw.split("\n",1)[0].strip().lower()


@torch.no_grad()
def evaluate(model: ByteTransformer,rows: list[PositionLesson],
             codec,max_tokens: int=5) -> dict:
    if not rows or not 2<=max_tokens<=12:
        raise ValueError("Invalid generated-character exam")
    model.eval()
    by_word={}
    by_position={str(i):{"correct":0,"total":0} for i in range(1,8)}
    examples=[]
    correct=0
    finished=0
    for row in rows:
        prefix=codec.encode(row.prompt).ids
        if not prefix or len(prefix)>model.cfg.max_seq_len:
            raise ValueError("Prompt beyond causal language context")
        state=list(prefix)
        continuation=[]
        did_finish=False
        for _ in range(max_tokens):
            logits,_=model(torch.tensor(
                [state[-model.cfg.max_seq_len:]],dtype=torch.long))
            next_id=int(logits[0,-1].argmax())
            state.append(next_id)
            continuation.append(next_id)
            if "\n" in codec.decode(continuation):
                did_finish=True
                break
        guess=normalize_generated(codec.decode(continuation))
        good=int(guess==row.answer)
        correct+=good
        finished+=int(did_finish)
        detail=by_position[str(row.index)]
        detail["correct"]+=good
        detail["total"]+=1
        key=row.string
        by_word.setdefault(key,{})[row.index]=guess
        if len(examples)<7 and not any(x["position"]==row.index
                                      for x in examples):
            examples.append({
                "word":key,"position":row.index,
                "gold_letter":row.answer,"generated_letter":guess,
                "exact":bool(good),
            })
    assembled=0
    samples=[]
    for original,predictions in by_word.items():
        if set(predictions)!=set(range(1,len(original)+1)):
            raise ValueError("Missing model predictions for a string index")
        # Explicit NON-NEURAL assembly is a scaffold, not evidence
        # that the language model autonomously wrote this string.
        output="".join(predictions[i] for i in range(1,len(original)+1))
        success=output==original
        assembled+=int(success)
        if len(samples)<6:
            samples.append({"source":original,"assembled":output,"exact":success})
    total=len(rows)
    return {
        "letter_correct":correct,"letter_total":total,
        "letter_accuracy":round(correct/total,4),
        "nonfirst_correct":sum(v["correct"] for k,v in
                               by_position.items() if k!="1"),
        "nonfirst_total":sum(v["total"] for k,v in
                             by_position.items() if k!="1"),
        "assembled_exact":assembled,"whole_strings":len(by_word),
        "assembled_rate":round(assembled/len(by_word),4),
        "terminated":finished,"termination_rate":round(finished/total,4),
        "by_position":{k:{**v,"accuracy":round(
            v["correct"]/v["total"],4) if v["total"] else None}
            for k,v in by_position.items()},
        "sample_letters":examples,"sample_assembled":samples,
        "scoring":"Greedy one-letter answers at explicit positions; externally joined strings",
    }


def study(*, source: Path,out: Path,
          steps_per_phase: int=110,
          seeds: tuple[int,...]=SEEDS,
          lr: float=.00012,
          rehearsal_weight: float=.20) -> dict:
    if (not 4<=steps_per_phase<=250 or not 1<=len(seeds)<=2 or
        len(set(seeds))!=len(seeds) or not 0<lr<=.001 or
        not 0<=rehearsal_weight<=.4):
        raise ValueError("Outside registered proof-of-copy CPU budget")
    torch.set_num_threads(min(2,torch.get_num_threads()))
    checkpoint,manifest,codec=verify_source(source)
    source_sha=hashlib.sha256((source/"checkpoint.pt").read_bytes()).hexdigest()
    cfg=ModelConfig(**checkpoint["configuration"])
    frozen=ByteTransformer(cfg)
    frozen.load_state_dict(checkpoint["model"],strict=True)
    public_train=torch.tensor(
        codec.encode((source/"train.txt").read_text()).ids,dtype=torch.long)
    public_valid=torch.tensor(
        codec.encode((source/"valid.txt").read_text()).ids,dtype=torch.long)
    development=generate("development",count=28,seed=30013)
    exam=generate("exam",count=48,seed=80517)
    frozen_loss=comparable_story_loss(frozen,public_valid)
    frozen_exam=evaluate(frozen,exam,codec)
    results=[]
    out.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    for seed in seeds:
        examples=generate("train",count=150,seed=seed)
        old_words={r.string for r in examples}
        if old_words.intersection(r.string for r in development+exam):
            raise ValueError("Training strings leaked into test")
        correct_data=make_encoded(examples,codec,cfg.max_seq_len)
        null_data=make_encoded(
            corrupt_labels(examples,seed+113),codec,cfg.max_seq_len)
        first_indexes={r.string:i for i,r in enumerate(examples)
                       if r.index==1}
        for part in (development,exam):
            make_encoded(part,codec,cfg.max_seq_len)
        staged,mixed=sequence_schedules(examples,seed=seed,
                                        steps_per_phase=steps_per_phase)
        schedules={
            "position_staged":staged,"position_mixed":mixed,
            "position_shuffled":staged,"first_only":staged,
            "story_only":mixed,
        }
        report={"seed":seed,"arms":{}}
        for arm in ARMS:
            model=ByteTransformer(cfg)
            model.load_state_dict(checkpoint["model"],strict=True)
            optimizer=torch.optim.AdamW(model.parameters(),lr=lr)
            rng=torch.Generator().manual_seed(seed+2222)
            best=None
            best_key=(-1,-1,-1,-1)
            curve=[]
            samples=null_data if arm=="position_shuffled" else correct_data
            for step,indices in enumerate(schedules[arm],1):
                if arm=="first_only":
                    indices=[first_indexes[examples[k].string]
                             for k in indices]
                model.train()
                xx,yy=make_batch(public_train,2,64,rng)
                _,story_loss=model(xx,yy)
                lesson_loss=None
                if arm=="story_only":
                    loss=story_loss
                else:
                    lesson_loss=train_loss(model,[samples[k] for k in indices])
                    loss=(1-rehearsal_weight)*lesson_loss+rehearsal_weight*story_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),1.)
                optimizer.step()
                if step==1 or step%steps_per_phase==0:
                    dev=evaluate(model,development,codec)
                    key=(dev["letter_correct"],dev["nonfirst_correct"],
                         dev["assembled_exact"],step)
                    if key>best_key:
                        best_key=key
                        best={"step":step,
                              "weights":copy.deepcopy(model.state_dict())}
                    row={
                        "update":step,"dev_char_correct":dev["letter_correct"],
                        "dev_later_char_correct":dev["nonfirst_correct"],
                        "dev_assembled":dev["assembled_exact"],
                        "train_loss":round(float(loss.detach()),5),
                        "teacher_loss":round(float(lesson_loss.detach()),5)
                        if lesson_loss is not None else None,
                    }
                    curve.append(row)
                    print(json.dumps({"seed":seed,"arm":arm,**row}),flush=True)
            if best is None:
                raise RuntimeError("No development checkpoint")
            latest=copy.deepcopy(model.state_dict())
            last_optimizer=optimizer.state_dict()
            model.load_state_dict(best["weights"])
            score=evaluate(model,exam,codec)
            story=comparable_story_loss(model,public_valid)
            torch.save({
                "schema":1,"research_only":True,
                "arm":arm,"seed":seed,
                "original_checkpoint_sha256":source_sha,
                "tokenizer_sha256":checkpoint["tokenizer_sha256"],
                "configuration":asdict(cfg),
                "best_development_step":best["step"],
                "best_weights":best["weights"],
                "last_weights":latest,
                "optimizer_last":last_optimizer,
                "public_rehearsal_rng":rng.get_state(),
            },out/f"seed-{seed}-{arm}-research.pt")
            report["arms"][arm]={
                "selected_dev_step":best["step"],
                "development_curve":curve,
                "final_position_exam":score,
                "public_story_loss":story,
                "story_loss_ratio":round(story/frozen_loss,5),
            }
        results.append(report)
        (out/f"seed-{seed}-position-results.json").write_text(
            json.dumps(report,indent=2)+"\n",encoding="utf-8")
    output={
        "schema":1,"study":"forge-english-position-copy-v4",
        "status":"research_only_unpromoted",
        "original_forge_seed_checkpoint_unchanged":True,
        "source_run_id":41572257363,
        "source_checkpoint_sha256":source_sha,
        "source_tokenizer_sha256":checkpoint["tokenizer_sha256"],
        "source_train_sha256":manifest["train_sha256"],
        "source_valid_sha256":manifest["valid_sha256"],
        "arms":list(ARMS),"seeds":list(seeds),
        "updates_per_arm_seed":len(staged),
        "exam_whole_strings":48,
        "dev_whole_strings":28,
        "train_whole_strings_per_seed":150,
        "frozen":{"position_exam":frozen_exam,
                  "public_story_loss":frozen_loss},
        "results":results,
        "elapsed_seconds":round(time.monotonic()-started,2),
        "exam_is_now_used":True,
        "promoted":False,
        "limitations":[
            "External scaffolding requests each letter position separately and assembles strings; NOT unaided word generation",
            "These are independently generated synthetic sequences, not actual English word literacy",
            "Training and exam share source formatting with disjoint whole strings and different question wordings",
            "The position and whole-string length are explicitly provided; no learned stopping or autonomous copy loop",
            "The original model uses a BPE tokenizer, not a specialized character tokenizer",
            "Shuffled and story-only arms receive less valid teacher information than correct-label arms",
            "No independent external reading test, human dialogue or general reasoning benchmark",
            "Original public text loss and checkpoint lineage audited; no Android or private data",
        ],
    }
    (out/"english-position-copy-v4.json").write_text(
        json.dumps(output,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({
        "seconds":output["elapsed_seconds"],
        "frozen":{"letters":frozen_exam["letter_correct"],
                  "assembled":frozen_exam["assembled_exact"]},
        "results":[{
            "seed":r["seed"],
            "arms":{a:{
                "letters":v["final_position_exam"]["letter_correct"],
                "later_letters":v["final_position_exam"]["nonfirst_correct"],
                "assembled":v["final_position_exam"]["assembled_exact"],
                "story_loss_ratio":v["story_loss_ratio"],
            } for a,v in r["arms"].items()}
        } for r in results],
    },indent=2),flush=True)
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir",type=Path,required=True)
    parser.add_argument("--out",type=Path,default=Path("results/english-position-v4"))
    parser.add_argument("--steps-per-phase",type=int,default=110)
    parser.add_argument("--seeds",type=int,choices=(1,2),default=2)
    args=parser.parse_args()
    study(source=args.checkpoint_dir,out=args.out,
          steps_per_phase=args.steps_per_phase,seeds=SEEDS[:args.seeds])


if __name__=="__main__":
    main()
