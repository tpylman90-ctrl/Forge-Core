"""Forge v7: controlled source-copy length curriculum (research only).

Hypothesis: v6's near-perfect 3-8 character copying fails at 9-12
because long source positions were never taught. Two scratch-initialized
neural copy decoders share weights, update counts and data budgets;
only their training length distributions differ. No pretrained weights.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import time

import torch
from torch import nn

from forge_core.neural.preschool_qa import verify_source
from forge_core.neural.english_learned_pointer_v6 import (
    ALPHABET, Pair, NeuralCopyDecoder, assert_firewall, evaluate,
    expanded, generate_pairs, objective, random_pair, sequence_bucket,
    validate_pairs, word_pairs,
)

SEEDS=(13741,13742)
ARMS=("short_only","length_curriculum")


def mixed_training(count: int, seed: int) -> list[Pair]:
    """Create verified train-partition twins at 3-12 chars."""
    if not 16 <= count <= 1000:
        raise ValueError("Invalid training pair count")
    rng=random.Random(seed+3901)
    out=[]
    seen=set()
    # Equal sample coverage across short and long length ranges.
    for i in range(count):
        lo,hi=(3,8) if i%2==0 else (9,12)
        for _ in range(10000):
            p=random_pair(rng,lo,hi,"train")
            if p.original not in seen and p.twin not in seen:
                out.append(p)
                seen.update((p.original,p.twin))
                break
        else:
            raise ValueError("Failed to construct disjoint training pairs")
    validate_pairs(out)
    return out


def experiment(*,source:Path,out:Path,updates:int=500,
               seeds:tuple[int,...]=SEEDS) -> dict:
    if (not 20 <= updates <= 1000 or not 1 <= len(seeds) <= 2
        or len(set(seeds)) != len(seeds) or
        any(s not in SEEDS for s in seeds)):
        raise ValueError("Outside preregistered v7 CPU budget")
    original,manifest,_=verify_source(source)
    origin_sha=hashlib.sha256((source/"checkpoint.pt").read_bytes()).hexdigest()
    torch.set_num_threads(min(2,torch.get_num_threads()))
    # Never inspect these exam scores during training or model selection.
    dev=generate_pairs("dev",80,20261105)
    exam_short=generate_pairs("exam",128,20261106)
    exam_long=generate_pairs("long",120,20261107)
    natural=word_pairs()
    pools={}
    for seed in seeds:
        short=generate_pairs("train",512,seed)
        mixed=mixed_training(512,seed)
        assert_firewall(short,dev,exam_short,exam_long,natural)
        assert_firewall(mixed,dev,exam_short,exam_long,natural)
        pools[seed]={"short_only":short,"length_curriculum":mixed}
    out.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    rows=[]
    for seed in seeds:
        torch.manual_seed(seed+55)
        initial=NeuralCopyDecoder()
        initial_state=copy.deepcopy(initial.state_dict())
        seed_result={"seed":seed,"arms":{}}
        for arm in ARMS:
            model=NeuralCopyDecoder()
            model.load_state_dict(initial_state,strict=True)
            optimizer=torch.optim.AdamW(model.parameters(),lr=.003)
            pool=pools[seed][arm]
            rng=random.Random(seed+731)
            curve=[]
            # Same number of optimizer updates and same number of
            # original/twin examples, different length distributions.
            for step in range(1,updates+1):
                if arm=="length_curriculum":
                    # 1/3 short in first stage; balanced thereafter.
                    # Long-source exposure is the only intended change.
                    if step<=updates//3:
                        eligible=[p for p in pool if len(p.original)<=8]
                    else:
                        eligible=pool
                else:
                    eligible=pool
                chosen=[rng.choice(eligible) for _ in range(4)]
                words=expanded(chosen)
                loss=objective(model,words,alignment_weight=.35)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),1.)
                optimizer.step()
                if step in (1,updates//2,updates):
                    curve.append({"step":step,
                                  "train_loss":round(float(loss.detach()),6)})
            scores={
                "dev_short":evaluate(model,dev),
                "unseen_short":evaluate(model,exam_short),
                "unseen_long_9_12":evaluate(model,exam_long),
                "curated_words":evaluate(model,natural),
            }
            torch.save({
                "schema":1,"experimental_only":True,
                "seed":seed,"arm":arm,
                "source_checkpoint_sha256":origin_sha,
                "configuration":{"width":model.width,"pointer":model.pointer},
                "state_dict":model.state_dict(),
            },out/f"{seed}-{arm}.pt")
            seed_result["arms"][arm]={
                "training_pairs":len(pool),
                "training_short_pairs":sum(len(p.original)<=8 for p in pool),
                "training_long_pairs":sum(len(p.original)>8 for p in pool),
                "updates":updates,"curve":curve,"evaluation":scores,
            }
            print(json.dumps({
                "seed":seed,"arm":arm,
                "short_exact":scores["unseen_short"]["exact"],
                "short_total":scores["unseen_short"]["total"],
                "long_exact":scores["unseen_long_9_12"]["exact"],
                "long_total":scores["unseen_long_9_12"]["total"],
                "long_pair_exact":scores["unseen_long_9_12"]["pair_exact"],
            }),flush=True)
        rows.append(seed_result)
    report={
        "schema":1,"research_only":True,"promoted":False,
        "experiment":"v7-length-generalization",
        "source_checkpoint_sha256":origin_sha,
        "source_train_sha256":manifest["train_sha256"],
        "original_model_weights_loaded_into_auxiliary":False,
        "original_model_weights_modified":False,
        "seeds":list(seeds),"updates_per_arm_seed":updates,
        "train_pairs_per_arm":512,
        "training_range_short_only":"3-8",
        "training_range_curriculum":"3-12 with initial short-only stage",
        "dev_seed":20261105,"short_exam_seed":20261106,
        "long_exam_seed":20261107,
        "results":rows,"duration_seconds":round(time.monotonic()-started,2),
        "limitations":[
            "Standalone copy decoder, not integrated with Forge's language model",
            "Length curriculum differs in sample distribution and source lengths",
            "Exam material is procedural and narrow, not human language comprehension",
            "No pretrained teacher weights, personal data or Android observer changes",
            "No checkpoint promotion regardless of result",
        ],
    }
    (out/"length-results.json").write_text(
        json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--out",type=Path,default=Path("results/length-v7"))
    p.add_argument("--updates",type=int,default=500)
    p.add_argument("--seeds",type=int,choices=(1,2),default=2)
    args=p.parse_args()
    experiment(source=args.source,out=args.out,
               updates=args.updates,seeds=SEEDS[:args.seeds])


if __name__=="__main__":
    main()
