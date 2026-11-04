"""Forge English v5: unbiased character copying with counterfactual twins.

Research-only continuation of v4 on the ORIGINAL source-verified,
random-initialized Forge Seed v0.7 language model, not a new pretrained AI.

Every short string uses independently uniform a-z characters: unlike v4,
vowels are NOT predictable from position. For each source string, construct
a near-identical twin differing in exactly one character. Evaluate whether
the model outputs the actual changed character for BOTH twins.

Four equal-update arms:
 - paired: correctly supervised twins in the same gradient minibatch
 - shuffled_batches: identical examples and targets, separate batch order
 - permuted_labels: same pairs but wrong position-matched targets
 - source_masked: same pairs and valid targets, but source letters hidden

The completed-string "assembly" metric concatenates individual answers
externally and must NOT be called unaided writing or learned iteration.
No checkpoint promotion, Android access, user data, or external code.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass, asdict
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
from forge_core.neural.train import make_batch
from forge_core.neural.english_production_v2 import (
    Lesson, encode_samples, train_loss, comparable_story_loss,
)

ALPHABET = "abcdefghijklmnopqrstuvwxyz"
SPLITS = ("train", "development", "exam")
ARMS = ("paired_verified", "unpaired_verified", "permuted_labels",
        "source_masked")
SEEDS = (11631, 11632)
PROMPT_RULE = {
    "train": "Count from the left starting at one. Write the letter at Position.",
    "development": "Which single character appears at this numbered place?",
    "exam": "Return only the character occupying the one-based source index.",
}


def bucket(value: str) -> str:
    if not re.fullmatch("[a-z]{4,8}", value):
        raise ValueError("Bounded lowercase source string expected")
    h = hashlib.sha256(("forge-english-v5-unbiased-family:" + value).encode()).digest()
    n = h[0] % 12
    return "train" if n < 8 else "development" if n < 10 else "exam"


@dataclass(frozen=True)
class Task:
    id: str
    split: str
    family: str
    variant: int
    text: str
    changed_position: int
    index: int
    prompt: str
    answer: str


def prompt_for(split: str, text: str, index: int,
               *, hide_source: bool = False) -> str:
    if split not in SPLITS or not re.fullmatch("[a-z]{4,8}", text):
        raise ValueError("Invalid source text or registered split")
    if not 1 <= index <= len(text):
        raise ValueError("Requested index is outside the source")
    chars = ("? " * len(text)).strip() if hide_source else " ".join(text)
    return (f"Letter reasoning\nLetters: {chars}\nPosition: {index}\n"
            f"{PROMPT_RULE[split]}\nAnswer:")


def make_tasks(split: str, *, families: int,
               seed: int) -> list[Task]:
    if split not in SPLITS or not 6 <= families <= 220:
        raise ValueError("Unsupported number of source families")
    rng = random.Random(seed + {
        "train": 107, "development": 2111, "exam": 11171}[split])
    output = []
    words_seen = set()
    for family_index in range(families):
        for attempt in range(100000):
            length = rng.randint(4, 8)
            original = "".join(rng.choice(ALPHABET) for _ in range(length))
            change = rng.randint(1, length)
            choices = [c for c in ALPHABET if c != original[change-1]]
            different = rng.choice(choices)
            modified = original[:change-1] + different + original[change:]
            if (bucket(original) == split and original not in words_seen
                    and modified not in words_seen and original != modified):
                words_seen.add(original)
                words_seen.add(modified)
                break
        else:
            raise ValueError("Unable to sample disjoint counterfactual family")
        for variant, word in enumerate((original, modified)):
            for index in range(1, length + 1):
                output.append(Task(
                    id=f"forge-english-v5:{split}:{family_index:04}:{variant}:{index}",
                    split=split, family=original, variant=variant, text=word,
                    changed_position=change, index=index,
                    prompt=prompt_for(split, word, index),
                    answer=word[index-1],
                ))
    validate(output, split=split)
    return output


def validate(rows: list[Task], *, split: str | None = None) -> None:
    if not rows or len(set(x.id for x in rows)) != len(rows):
        raise ValueError("Missing or duplicate experiment IDs")
    if len(set(x.prompt for x in rows)) != len(rows):
        raise ValueError("Duplicate textual questions")
    families = {}
    for row in rows:
        if row.split not in SPLITS or (split and row.split != split):
            raise ValueError("Leaked data split")
        if bucket(row.family) != row.split or row.variant not in (0, 1):
            raise ValueError("Unregistered source family")
        if not 1 <= row.changed_position <= len(row.text):
            raise ValueError("Bad counterfactual position")
        if not 1 <= row.index <= len(row.text):
            raise ValueError("Bad requested position")
        if row.answer != row.text[row.index-1]:
            raise ValueError("Wrong character answer")
        if row.prompt != prompt_for(row.split, row.text, row.index):
            raise ValueError("Prompt alteration or possible answer leakage")
        families.setdefault(row.family, {}).setdefault(row.variant, {})[row.index] = row
    for family, variants in families.items():
        if set(variants) != {0, 1}:
            raise ValueError("A twin counterfactual source is missing")
        base, twin = variants[0], variants[1]
        if set(base) != set(range(1, len(family)+1)):
            raise ValueError("Incomplete base sequence positions")
        if set(twin) != set(base):
            raise ValueError("Incomplete twin sequence positions")
        exemplar = base[1]
        alt = twin[1]
        if exemplar.text != family or exemplar.text == alt.text:
            raise ValueError("Malformed twin source")
        difference = [i for i,(a,b) in enumerate(
            zip(exemplar.text,alt.text),1) if a != b]
        if (len(exemplar.text) != len(alt.text) or
                difference != [exemplar.changed_position] or
                alt.changed_position != exemplar.changed_position):
            raise ValueError("Counterfactual twins must differ at one position")


def source_strings(rows: list[Task]) -> set[str]:
    return {row.text for row in rows}


def permute_targets(rows: list[Task], *, seed: int) -> list[Task]:
    """Preserve answer frequencies within source indices; corrupt labels."""
    rng = random.Random(seed)
    replacement = {}
    for index in range(1, 9):
        selected = [r.answer for r in rows if r.index == index]
        if not selected:
            continue
        for attempt in range(150):
            wrong = selected.copy()
            rng.shuffle(wrong)
            if sum(x != y for x,y in zip(selected,wrong)) >= len(selected)*.75:
                replacement[index] = iter(wrong)
                break
        else:
            raise ValueError("Null arm failed to separate true source labels")
    return [Task(**{**asdict(row), "answer": next(replacement[row.index])})
            for row in rows]


def train_batches(rows: list[Task], *, seed: int, updates: int):
    if not 20 <= updates <= 1600:
        raise ValueError("Outside approved training compute")
    families = {}
    for i,row in enumerate(rows):
        families.setdefault(row.family, {})[(row.variant,row.index)] = i
    rng = random.Random(seed)
    family_names = list(families)
    paired = []
    for step in range(updates):
        batch = []
        for _ in range(2):
            family = rng.choice(family_names)
            index_map = families[family]
            length = max(index for _,index in index_map)
            changed = rows[index_map[0,1]].changed_position
            index = changed if rng.random() < .6 else rng.randint(1, length)
            batch.extend([index_map[0,index], index_map[1,index]])
        paired.append(batch)
    unpaired = [i for batch in paired for i in batch]
    rng.shuffle(unpaired)
    alternative = [unpaired[i:i+4] for i in range(0,len(unpaired),4)]
    if sorted(i for row in paired for i in row) != sorted(unpaired):
        raise AssertionError("Comparison got unequal sample exposure")
    if any(len(batch) != 4 for batch in alternative):
        raise AssertionError("A training control changed minibatch size")
    return paired, alternative


def make_encoded(rows: list[Task], codec, max_context: int,
                 *, hide_source: bool = False):
    wrapped = [Lesson(
        row.id, "position", row.split,
        f"string={row.text};index={row.index}",
        prompt_for(row.split,row.text,row.index,hide_source=hide_source),
        row.answer) for row in rows]
    return encode_samples(wrapped, codec, max_context)


def normalized(text: str) -> str:
    return text.split("\n",1)[0].strip().lower()


@torch.no_grad()
def evaluate(model: ByteTransformer, rows: list[Task], codec,
             *, hide_source: bool = False, max_tokens: int = 5) -> dict:
    if not rows or not 2 <= max_tokens <= 8:
        raise ValueError("Invalid bounded evaluation request")
    model.eval()
    letters = 0
    terminated = 0
    samples = []
    by_position = {str(i): {"correct": 0, "total": 0} for i in range(1,9)}
    grouped = {}
    for row in rows:
        source = prompt_for(row.split,row.text,row.index,
                            hide_source=hide_source)
        tokens = codec.encode(source).ids
        if not tokens or len(tokens) > model.cfg.max_seq_len:
            raise ValueError("Question exceeds original language context")
        history = list(tokens)
        generated = []
        did_stop = False
        for _ in range(max_tokens):
            logits,_ = model(torch.tensor(
                [history[-model.cfg.max_seq_len:]],dtype=torch.long))
            token = int(logits[0,-1].argmax())
            history.append(token)
            generated.append(token)
            if "\n" in codec.decode(generated):
                did_stop = True
                break
        value = normalized(codec.decode(generated))
        good = int(value == row.answer)
        letters += good
        terminated += int(did_stop)
        by_position[str(row.index)]["correct"] += good
        by_position[str(row.index)]["total"] += 1
        grouped.setdefault(row.family, {}).setdefault(
            row.variant, {})[row.index] = value
        if len(samples) < 10 and not any(
                x["index"] == row.index for x in samples):
            samples.append({
                "source":row.text, "index":row.index,
                "gold":row.answer, "generated":value, "correct":bool(good),
            })
    assembled = 0
    whole_count = 0
    hard_pair_both = 0
    hard_pair_changed = 0
    for family, variants in grouped.items():
        if set(variants) != {0,1}:
            raise ValueError("Missing family twin predictions")
        change = next(row.changed_position for row in rows
                      if row.family == family)
        for variant in (0,1):
            answers = variants[variant]
            # All source positions must be evaluated, otherwise this
            # explicitly external assembly number would be misleading.
            if set(answers) != set(range(1,len(family)+1)):
                raise ValueError("Incomplete letter positions in exam")
            target = (family if variant == 0 else next(
                row.text for row in rows
                if row.family == family and row.variant == 1))
            assembled += int("".join(answers[j] for j in
                                     range(1,len(target)+1)) == target)
            whole_count += 1
        a,b = variants[0][change],variants[1][change]
        gold_a = family[change-1]
        gold_b = next(row.text[change-1] for row in rows
                      if row.family == family and row.variant == 1)
        hard_pair_both += int(a == gold_a and b == gold_b)
        hard_pair_changed += int(a != b)
    later = sum(v["correct"] for k,v in by_position.items() if k!="1")
    later_total = sum(v["total"] for k,v in by_position.items() if k!="1")
    deeper = sum(v["correct"] for k,v in by_position.items() if int(k)>=3)
    deeper_total = sum(v["total"] for k,v in by_position.items() if int(k)>=3)
    for item in by_position.values():
        item["rate"] = round(item["correct"]/item["total"],4) if item["total"] else None
    return {
        "letters_correct":letters, "letter_total":len(rows),
        "letter_accuracy":round(letters/len(rows),4),
        "nonfirst_correct":later,"nonfirst_total":later_total,
        "third_plus_correct":deeper,"third_plus_total":deeper_total,
        "paired_both_correct":hard_pair_both,
        "paired_predictions_differ":hard_pair_changed,
        "paired_families":len(grouped),
        "externally_assembled_exact":assembled,
        "externally_assembled_total":whole_count,
        "terminated":terminated,
        "termination_rate":round(terminated/len(rows),4),
        "by_position":by_position,
        "examples":samples,
        "scoring":"Greedy one-letter generated answer, exact; assembly is external",
        "masked_source_control":hide_source,
    }


def experiment(*, source: Path, out: Path, updates: int = 900,
               seeds: tuple[int,...] = SEEDS,
               lr: float = .00015, rehearsal_fraction: float = .2) -> dict:
    if (not 20<=updates<=1600 or not seeds or len(seeds)>2
            or len(set(seeds))!=len(seeds)
            or not 0<lr<=.001 or not 0<=rehearsal_fraction<=.4):
        raise ValueError("Outside bounded English v5 research")
    torch.set_num_threads(min(2,torch.get_num_threads()))
    original,manifest,codec = verify_source(source)
    original_sha=hashlib.sha256((source/"checkpoint.pt").read_bytes()).hexdigest()
    cfg=ModelConfig(**original["configuration"])
    public_training=torch.tensor(
        codec.encode((source/"train.txt").read_text()).ids,dtype=torch.long)
    public_valid=torch.tensor(
        codec.encode((source/"valid.txt").read_text()).ids,dtype=torch.long)
    dev=make_tasks("development",families=18,seed=7811)
    exam=make_tasks("exam",families=40,seed=10413)
    frozen=ByteTransformer(cfg)
    frozen.load_state_dict(original["model"],strict=True)
    story_reference=comparable_story_loss(frozen,public_valid)
    frozen_score=evaluate(frozen,exam,codec)
    out.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    data=[]
    for seed in seeds:
        rows=make_tasks("train",families=180,seed=seed)
        if (source_strings(rows)&source_strings(dev+exam)
                or source_strings(dev)&source_strings(exam)):
            raise ValueError("Train, dev and exam contain overlapping strings")
        for dataset in (rows,dev,exam):
            make_encoded(dataset,codec,cfg.max_seq_len)
        verified=make_encoded(rows,codec,cfg.max_seq_len)
        wrong=make_encoded(permute_targets(rows,seed=seed+821),
                           codec,cfg.max_seq_len)
        blind=make_encoded(rows,codec,cfg.max_seq_len,hide_source=True)
        paired,unpaired=train_batches(rows,seed=seed,updates=updates)
        schedules={
            "paired_verified":paired,"unpaired_verified":unpaired,
            "permuted_labels":paired,"source_masked":paired,
        }
        row={"seed":seed,"arms":{}}
        for arm in ARMS:
            # Exact same source network, no pretrained teacher weights.
            torch.manual_seed(seed+301)
            policy=ByteTransformer(cfg)
            policy.load_state_dict(original["model"],strict=True)
            optim=torch.optim.AdamW(policy.parameters(),lr=lr)
            g=torch.Generator().manual_seed(seed+901)
            encoded=(wrong if arm=="permuted_labels" else blind
                     if arm=="source_masked" else verified)
            best=None
            winner=(-1,-1,-1,-1)
            curve=[]
            for update,indices in enumerate(schedules[arm],1):
                policy.train()
                teaching=train_loss(policy,[encoded[i] for i in indices])
                bx,by=make_batch(public_training,2,64,g)
                _,story=policy(bx,by)
                loss=(1-rehearsal_fraction)*teaching+rehearsal_fraction*story
                optim.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(),1)
                optim.step()
                if update==1 or update%150==0 or update==updates:
                    diag=evaluate(policy,dev,codec,
                                  hide_source=(arm=="source_masked"))
                    key=(diag["paired_both_correct"],
                         diag["third_plus_correct"],diag["letters_correct"],
                         update)
                    if key>winner:
                        best={"step":update,
                              "weights":copy.deepcopy(policy.state_dict())}
                        winner=key
                    curve.append({
                        "update":update,
                        "development_letters":diag["letters_correct"],
                        "development_later":diag["nonfirst_correct"],
                        "development_hard_pairs":diag["paired_both_correct"],
                        "development_assembled":diag["externally_assembled_exact"],
                        "train_loss":round(float(loss.detach()),5),
                    })
                    print(json.dumps({"arm":arm,"seed":seed,**curve[-1]}),
                          flush=True)
            if best is None:
                raise RuntimeError("Missing development-selected checkpoint")
            last=copy.deepcopy(policy.state_dict())
            opt_state=optim.state_dict()
            policy.load_state_dict(best["weights"])
            final=evaluate(policy,exam,codec,
                           hide_source=(arm=="source_masked"))
            dev_final=evaluate(policy,dev,codec,
                               hide_source=(arm=="source_masked"))
            story_loss=comparable_story_loss(policy,public_valid)
            torch.save({
                "schema":1,"research_only":True,"arm":arm,"seed":seed,
                "original_checkpoint_sha256":original_sha,
                "source_tokenizer_sha256":original["tokenizer_sha256"],
                "configuration":asdict(cfg),"best_dev_step":best["step"],
                "best_weights":best["weights"],"last_weights":last,
                "last_optimizer":opt_state,
                "rehearsal_rng_state":g.get_state(),
            },out/f"seed-{seed}-{arm}-research.pt")
            row["arms"][arm]={
                "selected_dev_update":best["step"],
                "development_curve":curve,
                "development":dev_final,
                "heldout_unbiased_exam":final,
                "public_story_loss":story_loss,
                "story_loss_ratio":round(story_loss/story_reference,5),
            }
        data.append(row)
        (out/f"seed-{seed}-results.json").write_text(
            json.dumps(row,indent=2)+"\n",encoding="utf-8")
    report={
        "schema":1,"study":"forge-english-unbiased-counterfactual-v5",
        "source_training_run_id":41572257363,
        "source_checkpoint_sha256":original_sha,
        "source_tokenizer_sha256":original["tokenizer_sha256"],
        "original_corpus_train_sha256":manifest["train_sha256"],
        "original_corpus_valid_sha256":manifest["valid_sha256"],
        "scratch_trained_original_checkpoint_unchanged":True,
        "research_only":True,"promoted":False,
        "unbiased_uniform_letters":True,
        "one_change_counterfactual_twins":True,
        "fresh_split_hash_namespace":True,
        "train_families_per_seed":180,
        "development_families":18,
        "exam_families":40,
        "seeds":list(seeds),"arms":list(ARMS),
        "updates_per_arm_seed":updates,
        "training_examples_per_step":4,
        "public_story_rehearsal_fraction":rehearsal_fraction,
        "checkpoint_selection":"development-only hard-pair, later-position, total letters",
        "frozen":{"story_loss":story_reference,"exam":frozen_score},
        "results":data,
        "duration_seconds":round(time.monotonic()-started,2),
        "exam_now_used":True,
        "limitations":[
            "All 40 twin families in the final exam are the same across seeds",
            "New strings are synthetic, uniformly random letters and not real English words",
            "Index is externally supplied; neural model does not decide how to loop/stop",
            "External assembly joins separate single-letter predictions; not autonomous spelling",
            "Train and test reuse the same underlying source indexing operation",
            "Paired vs unpaired groups share identical examples but have distinct gradients",
            "Masked-source controls provide labels for unanswerable problems",
            "No externally authored literacy or natural-language understanding benchmark",
            "Only original source language weights, isolated temporary research adaptations",
            "No phone, encrypted memory, Android or production model changes",
        ],
    }
    (out/"english-unbiased-copy-v5.json").write_text(
        json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({
        "seconds":report["duration_seconds"],
        "frozen_correct":frozen_score["letters_correct"],
        "runs":[{
            "seed":entry["seed"],
            "arms":{name:{
                "letters":x["heldout_unbiased_exam"]["letters_correct"],
                "questions":x["heldout_unbiased_exam"]["letter_total"],
                "third_plus":x["heldout_unbiased_exam"]["third_plus_correct"],
                "hard_pair":x["heldout_unbiased_exam"]["paired_both_correct"],
                "assembled":x["heldout_unbiased_exam"]["externally_assembled_exact"],
                "story_loss_ratio":x["story_loss_ratio"],
            } for name,x in entry["arms"].items()}
        } for entry in data],
    },indent=2),flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint-dir",type=Path,required=True)
    p.add_argument("--out",type=Path,default=Path("results/english-unbiased-v5"))
    p.add_argument("--updates",type=int,default=900)
    p.add_argument("--seeds",type=int,choices=(1,2),default=2)
    args=p.parse_args()
    experiment(source=args.checkpoint_dir,out=args.out,
               updates=args.updates,seeds=SEEDS[:args.seeds])


if __name__=="__main__":
    main()
