"""Instructional copying/role apprenticeship on a frozen-provenance Forge model.

Start from the separate verified compositional candidate, never overwrite it.
Train TWO independent copies on the same prompts and public-language rehearsal;
only their new instruction gold-answer targets differ. Evaluations measure
out-of-vocabulary name copying, selection, giver/recipient roles, and retention.
All labels are deterministic original fixtures; no private or pretrained data.
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

from forge_core.neural.model import ByteTransformer, ModelConfig
from forge_core.neural.preschool_direct import (
    DirectExample, answer_loss, encode, evaluate_direct,
)
from forge_core.neural.preschool_qa import verify_source
from forge_core.neural.preschool_compositional import make_examples as earlier_qa
from forge_core.neural.preschool_instructions import instruction_examples
from forge_core.neural.seed_v07 import heldout, generate_samples
from forge_core.neural.train import make_batch

SKILLS = ("copy", "select", "recipient", "giver", "color", "quantity")
TRAIN_PEOPLE = ("Nora", "Owen", "Ella", "Finn", "Sara", "June",
                "Liam", "Arlo", "Zoe", "Hugo", "Ruby", "Will",
                "Nina", "Noel", "Asha", "Evan", "Paula", "Cody",
                "Greta", "Seth", "Mina", "Otto", "Daisy", "Luca")
EVAL_PEOPLE = ("Faye", "Sage", "Bram", "Amir", "Noemi", "Suri",
               "Zeke", "Dorian", "Juno", "Talia", "Kira", "Flora")
TRAIN_WORDS = ("apple", "bird", "house", "river", "pencil", "ball",
               "garden", "tree", "chair", "book", "lamp", "stone",
               "basket", "flower", "cloud", "moon", "rain", "puddle",
               "pocket", "paint", "boat", "bell", "cookie", "bucket")
EVAL_WORDS = ("violin", "wagon", "scarf", "camera", "mirror", "glove",
              "pumpkin", "candle", "blanket", "ribbon", "kettle", "ladder")
COLORS = ("red", "blue", "green", "yellow", "purple",
          "orange", "pink", "black", "brown", "white")
NUMBERS = ("two", "three", "four", "five", "six",
           "seven", "eight", "nine", "ten", "eleven")


def tasks(split: str, count: int, seed: int = 900) -> list[DirectExample]:
    if split not in ("train", "challenge"):
        raise ValueError("Unknown example split")
    if count < 60 or count > 6000 or count % len(SKILLS):
        raise ValueError("Example count must be 60..6000, divisible by six")
    names = TRAIN_PEOPLE if split == "train" else EVAL_PEOPLE
    words = TRAIN_WORDS if split == "train" else EVAL_WORDS
    rng = random.Random(seed + (0 if split == "train" else 184217))
    results = []
    seen = set()
    for i in range(count):
        skill = SKILLS[i % len(SKILLS)]
        family_size = 4 if split == "train" else 3
        task_index = i // len(SKILLS)
        variant = task_index % family_size
        # Question target and selected list position vary independently of
        # surface-template family: memorizing the phrasing cannot reveal gold.
        t = (task_index // family_size) % 2
        which = (task_index // family_size) % 3
        a, b = rng.sample(names, 2)
        w1, w2, w3 = rng.sample(words, 3)
        color1, color2 = rng.sample(COLORS, 2)
        number1, number2 = rng.sample(NUMBERS, 2)
        if skill == "copy":
            target = a if t == 0 else w1
            if split == "train":
                templates = (
                    f"Instruction: {a} mentioned {w2}. Repeat only {target}.",
                    f"Instruction: {a} saw {w2} near {w3}. Write {target} and stop.",
                    f"Instruction: Between {w2} and {w3}, the word to copy "
                    f"is {target}. Type it.",
                    f"Instruction: {a} read {w2}. Give back the exact word "
                    f"{target}.",
                )
            else:
                templates = (
                    f"Teacher says: echo {target} with no extra words.",
                    f"Copy exactly the word shown here: {target}.",
                    f"Answer with the same text as {target}, nothing else.",
                )
            prompt, answer = templates[variant], target
        elif skill == "select":
            choices = (w1, w2, w3)
            answer = choices[which]
            if split == "train":
                templates = (
                    f"Instruction: From {w1}, {w2}, {w3}, write the "
                    f"{('first','second','third')[which]} word only.",
                    f"Instruction: The words are {w1} / {w2} / {w3}. "
                    f"Copy item number {which+1}.",
                    f"Instruction: Pick position {which+1} in this list: "
                    f"{w1}; {w2}; {w3}.",
                    f"Instruction: Select the {('first','second','third')[which]} "
                    f"from {w1}, {w2}, {w3}.",
                )
            else:
                templates = (
                    f"In the ordered list '{w1}, {w2}, {w3}', give only "
                    f"entry {which+1}.",
                    f"Your task: output the {('first','second','third')[which]} "
                    f"item of {w1}, {w2}, {w3}.",
                    f"Choose the number {which+1} word from these: "
                    f"{w1}, {w2}, {w3}.",
                )
            prompt = templates[variant]
        elif skill in ("recipient", "giver"):
            if split == "train":
                stories = (
                    f"{a} gave the {w1} to {b}.",
                    f"{b} received the {w1} from {a}.",
                    f"The {w1} was passed from {a} to {b}.",
                    f"{a} handed {b} the {w1}.",
                )
                question = (
                    f"Who was given the {w1}?" if skill == "recipient"
                    else f"Who gave away the {w1}?"
                )
                prompt = f"Instruction: Read '{stories[variant]}' {question}"
            else:
                stories = (
                    f"With a smile, {a} offered {b} the {w1}.",
                    f"After accepting the {w1}, {b} thanked {a}.",
                    f"The {w1} was delivered to {b} by {a}.",
                )
                question = (
                    f"Name the person who ended up with the {w1}."
                    if skill == "recipient"
                    else f"Identify the person who supplied the {w1}."
                )
                prompt = f"Read this: '{stories[variant]}' {question}"
            answer = b if skill == "recipient" else a
        elif skill == "color":
            chosen = b if t == 1 else a
            answer = color2 if t == 1 else color1
            if split == "train":
                templates = (
                    f"Instruction: {a} carried a {color1} {w1} and {b} "
                    f"carried a {color2} {w2}. What color did {chosen} have?",
                    f"Instruction: The {w1} owned by {a} was {color1}; "
                    f"the {w2} owned by {b} was {color2}. Say {chosen}'s color.",
                    f"Instruction: {b} had a {color2} {w2}; {a} had a "
                    f"{color1} {w1}. Reply with the color for {chosen}.",
                    f"Instruction: Find the color belonging to {chosen}: "
                    f"{a} has {color1}, and {b} has {color2}.",
                )
            else:
                templates = (
                    f"Look at the note: {a}'s {w1} is {color1} but "
                    f"{b}'s {w2} is {color2}. Give {chosen}'s shade.",
                    f"Teacher asks: {a} picked {color1}, {b} picked "
                    f"{color2}. State the color picked by {chosen}.",
                    f"Read '{b} has {color2}, {a} has {color1}'. "
                    f"Which color goes with {chosen}?",
                )
            prompt = templates[variant]
        else:
            chosen = b if t == 1 else a
            answer = number2 if t == 1 else number1
            if split == "train":
                templates = (
                    f"Instruction: {a} has {number1} stones. {b} has "
                    f"{number2} shells. How many does {chosen} have?",
                    f"Instruction: Count for {a}: {number1}. Count for "
                    f"{b}: {number2}. Reply with {chosen}'s count.",
                    f"Instruction: {b} saved {number2} shells while {a} "
                    f"saved {number1} stones. Say {chosen}'s number.",
                    f"Instruction: From '{a}: {number1}; {b}: {number2}', "
                    f"copy the number belonging to {chosen}.",
                )
            else:
                templates = (
                    f"The numbers are {number1} for {a} and {number2} "
                    f"for {b}. State only the number for {chosen}.",
                    f"Answer this: {b} picked up {number2} things, while "
                    f"{a} picked up {number1}. How many for {chosen}?",
                    f"In the story {a} got {number1} and {b} got "
                    f"{number2}. Give {chosen}'s quantity.",
                )
            prompt = templates[variant]

        prompt = prompt + "\nAnswer:"
        if prompt in seen:
            # Rare accidental duplicates would artificially inflate results.
            # Add an unrelated ordinal only for duplicates instead of reusing
            # a repeated prompt/answer pair as a new training observation.
            prompt = prompt.replace("\nAnswer:",
                f" (practice record {i+1:04d}).\nAnswer:")
        if prompt in seen:
            raise ValueError("Duplicate instruction prompt")
        if answer.casefold() not in prompt.casefold():
            raise ValueError("Gold answer not in supplied instruction")
        seen.add(prompt)
        results.append(DirectExample(
            f"{split}:instruction:{i:05d}", skill, prompt, answer))
    return results


def shuffle_within_skill(rows: list[DirectExample], seed: int):
    rng = random.Random(seed)
    labels = {skill: [x.answer for x in rows if x.skill == skill]
              for skill in SKILLS}
    for skill, values in labels.items():
        original = values[:]
        for _ in range(30):
            rng.shuffle(values)
            if values != original:
                break
        if values == original:
            raise ValueError("Unable to construct shuffled instruction control")
    idx = {skill: 0 for skill in SKILLS}
    result = []
    for item in rows:
        i = idx[item.skill]
        idx[item.skill] += 1
        result.append(DirectExample(item.id, item.skill, item.prompt,
                                    labels[item.skill][i]))
    return result


def experiment(*, source: Path, candidate: Path, out: Path, seed: int = 900,
               stages: tuple[tuple[int,int,float,float],...] = (
                   (120, 250, 0.10, 2e-4),
                   (600, 600, 0.20, 1e-4),
                   (2400, 1400, 0.25, 6e-5),
               )) -> dict:
    if (not stages or len(stages) > 5 or sum(x[1] for x in stages) > 5000 or
        any(n % 6 or n < 60 or n > 6000 or steps < 1 or steps > 2500 or
            not 0 <= mix < 1 or not 0 < lr < 0.01
            for n,steps,mix,lr in stages)):
        raise ValueError("Invalid bounded instruction/copy curriculum")
    torch.set_num_threads(min(2, torch.get_num_threads()))
    pretrain, source_manifest, codec = verify_source(source)
    base_sha = hashlib.sha256((source / "checkpoint.pt").read_bytes()).hexdigest()
    candidate_data = torch.load(candidate / "verified-research.pt",
                                map_location="cpu", weights_only=True)
    if (candidate_data.get("schema") != 4 or
        not candidate_data.get("experimental_only") or
        candidate_data.get("arm") != "verified" or
        candidate_data.get("source_checkpoint_sha256") != base_sha or
        candidate_data.get("configuration") != pretrain["configuration"]):
        raise ValueError("Unapproved initial research candidate or source mismatch")
    cfg = ModelConfig(**pretrain["configuration"])
    base = ByteTransformer(cfg)
    base.load_state_dict(candidate_data["model"], strict=True)
    verified, control = copy.deepcopy(base), copy.deepcopy(base)
    text_train = torch.tensor(
        codec.encode((source / "train.txt").read_text()).ids,dtype=torch.long)
    text_valid = torch.tensor(
        codec.encode((source / "valid.txt").read_text()).ids,dtype=torch.long)
    train_data = tasks("train", stages[-1][0], seed)
    challenge = tasks("challenge", 120, seed+1)
    dev_questions = instruction_examples(233)
    if set(TRAIN_PEOPLE) & set(EVAL_PEOPLE) or set(TRAIN_WORDS) & set(EVAL_WORDS):
        raise ValueError("Instruction challenge leaked entity vocabulary")
    if {x.prompt for x in train_data} & {x.prompt for x in challenge + dev_questions}:
        raise ValueError("Instruction prompt leak")

    def score(model, values):
        return evaluate_direct(model,codec,values)
    old_qa = earlier_qa("challenge", 200, seed=708)
    initial = {
        "challenge": score(base,challenge),
        "previous_50_instructions": score(base,dev_questions),
        "previous_200_qa": score(base,old_qa),
        "heldout_language_loss":round(
            heldout(base,text_valid,windows=64,seq_len=128),6),
    }
    phases=[]
    started=time.monotonic()
    for stage,(n,steps,lm_ratio,lr) in enumerate(stages,1):
        data = train_data[:n]
        null = shuffle_within_skill(data,seed+stage*103)
        if [x.prompt for x in data] != [x.prompt for x in null]:
            raise ValueError("Control prompts differ")
        x_gold=encode(data,codec,cfg.max_seq_len)
        x_null=encode(null,codec,cfg.max_seq_len)
        pools={skill:[i for i,x in enumerate(data) if x.skill==skill]
               for skill in SKILLS}
        rng=random.Random(seed+stage*239)
        schedule=[
            [rng.choice(pools[k]) for k in rng.sample(SKILLS,4)]
            for _ in range(steps)
        ]
        def adapt(model,records,arm):
            optimizer=torch.optim.AdamW(model.parameters(),lr=lr)
            langrng=torch.Generator().manual_seed(seed+stage*809)
            model.train()
            last=0.0
            for t,indices in enumerate(schedule,1):
                task_loss=answer_loss(model,[records[i] for i in indices])
                if lm_ratio:
                    x,y=make_batch(text_train,2,64,langrng)
                    _,language_loss=model(x,y)
                    loss=(1-lm_ratio)*task_loss+lm_ratio*language_loss
                else:
                    loss=task_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                optimizer.step()
                last=float(task_loss.detach())
            return round(last,6)
        gold_train_loss=adapt(verified,x_gold,"verified")
        null_train_loss=adapt(control,x_null,"shuffled")
        # Old measured tasks are regression diagnostics, not untouched tests.
        record={
            "stage":stage, "training_examples":n,"steps_per_arm":steps,
            "language_rehearsal_weight":lm_ratio,
            "verified_last_batch_loss":gold_train_loss,
            "shuffled_last_batch_loss":null_train_loss,
            "training_first_100":score(verified,data[:100]),
            "previous_instruction_50":score(verified,dev_questions),
            "previous_qa_200":score(verified,old_qa[:50]),
            "heldout_language_loss":round(
                heldout(verified,text_valid,windows=64,seq_len=128),6),
        }
        phases.append(record)
        print(json.dumps({
            "stage":stage,
            "train_copy_instruction_accuracy":record["training_first_100"]["accuracy"],
            "previous_instruction_accuracy":record["previous_instruction_50"]["accuracy"],
            "heldout_story_token_loss":record["heldout_language_loss"]
        }),flush=True)
    # New 120-case test is first scored after all stages complete.
    final_gold = score(verified,challenge)
    final_null = score(control,challenge)
    final_old = score(verified,old_qa)
    report={
        "schema":1,"status":"research_only_not_promoted",
        "original_source_run":41572257363,
        "prior_research_run":41649770774,
        "original_source_checkpoint_sha256":base_sha,
        "source_manifest_train_sha256":source_manifest["train_sha256"],
        "initial_research_weights_sha256":hashlib.sha256(
            (candidate/"verified-research.pt").read_bytes()).hexdigest(),
        "seed":seed, "training_examples_final":len(train_data),
        "heldout_new_instructions":len(challenge),
        "same_question_schedule_optimizer_and_language_rehearsal":True,
        "gold_source":"original verified explicit instruction facts, not model self-labels",
        "initial":initial, "stages":phases,
        "final_new_verified":final_gold,
        "final_new_shuffled":final_null,
        "final_previous_50_instructions":score(verified,dev_questions),
        "final_previous_200_qa":final_old,
        "final_heldout_language_loss":round(
            heldout(verified,text_valid,windows=64,seq_len=128),6),
        "final_shuffled_language_loss":round(
            heldout(control,text_valid,windows=64,seq_len=128),6),
        "story_samples":generate_samples(verified,codec,seed,new_tokens=64),
        "elapsed_seconds":round(time.monotonic()-started,2),
        "limitations":[
            "Only internally generated questions with disjoint people/words/phrases",
            "Previously measured 50 and 200 question suites are not fresh independent tests",
            "New 120-case challenge has not been independently annotated by humans",
            "One seed, tiny model; do not extrapolate exponential intelligence growth",
            "No model promotion or Android memory updates are authorized",
        ]
    }
    out.mkdir(parents=True,exist_ok=True)
    (out/"copy-results.json").write_text(
        json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    for arm,model in (("verified",verified),("shuffled",control)):
        torch.save({
            "schema":5,"experimental_only":True,"arm":arm,
            "method":"copy_select_and_role_instructions",
            "origin_checkpoint_sha256":base_sha,
            "parent_research_sha256":report["initial_research_weights_sha256"],
            "configuration":asdict(cfg),"model":model.state_dict(),
        },out/f"{arm}-research.pt")
    print(json.dumps({
        "original_frozen_new":initial["challenge"]["correct"],
        "verified_new":final_gold["correct"],
        "shuffled_new":final_null["correct"],
        "challenge_size":len(challenge),
        "verified_by_skill":final_gold["by_skill"],
        "old_qa_retained":final_old["correct"],
        "language_loss":report["final_heldout_language_loss"],
    },indent=2),flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--parent-candidate",type=Path,required=True)
    p.add_argument("--out",type=Path,default=Path("artifacts/copy-learning"))
    p.add_argument("--seed",type=int,default=900)
    a=p.parse_args()
    experiment(source=a.source,candidate=a.parent_candidate,out=a.out,seed=a.seed)


if __name__=="__main__":
    main()
