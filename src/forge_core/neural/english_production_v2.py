"""Forge English Class v2: produce words and sentences instead of selecting A/B.

The real scratch-trained Forge Seed v0.7 model is adapted in research-only
branches using explicit grammar rules and verified teacher-forced outputs.
Independent grammar reconstructs gold; final evaluation greedily GENERATES
answers without options. Distinct lexical sets and question wording.

Controls: equally exposed staged and mixed lessons, a word-answer shuffled
null, and a matched-update TinyStories-only language rehearsal branch.
No pretrained model is used. No canonical promotion or Android access.
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
from torch.nn import functional as F

from forge_core.neural.model import ByteTransformer, ModelConfig
from forge_core.neural.preschool_qa import verify_source
from forge_core.neural.seed_v07 import heldout
from forge_core.neural.train import make_batch

SKILLS = ("join_parts", "plural_noun", "present_verb", "past_verb",
          "actor_role", "write_sentence", "negate_sentence")
ARMS = ("verified_mixed", "verified_staged", "shuffled_mixed", "text_only")
SEEDS = (8123, 8124)

# Written entirely for this v2 experiment, with no names or target word
# lists copied from the now-used v1 synthetic exam. Lexical transfer is
# tested; however the grammar rule families remain shared and synthetic.
LEX = {
    "train": {
        "people": ("Amina", "Benton", "Camila", "Davin", "Emery", "Flora",
                   "Gideon", "Harper", "Isla", "Jasper"),
        "nouns": ("kite", "daisy", "lamp", "shell", "button",
                  "window", "cookie", "cup", "dish", "cherry"),
        "verbs": ("carry", "paint", "push", "bake", "try",
                  "watch", "help", "move", "dry", "wash"),
        "prefix": ("sun", "moon", "water", "forest", "paper",
                   "honey", "rain", "fire", "garden", "stone"),
        "suffix": ("shine", "light", "fall", "seed", "house",
                   "drop", "walk", "line", "path", "bird"),
    },
    "development": {
        "people": ("Kiran", "Leona", "Milo", "Nessa", "Orion"),
        "nouns": ("piano", "puppy", "brush", "stool", "story"),
        "verbs": ("fix", "glide", "dance", "reach", "reply"),
        "prefix": ("frost", "night", "cloud", "berry", "sand"),
        "suffix": ("ridge", "chant", "bloom", "crest", "pool"),
    },
    "exam": {
        "people": ("Pippa", "Quentin", "Rosa", "Soren", "Talia",
                   "Uma", "Victor", "Wendy", "Yara", "Zeke"),
        "nouns": ("trophy", "apple", "box", "river", "bench",
                  "pillow", "garden", "fox", "baby", "pear"),
        "verbs": ("visit", "race", "hurry", "close", "finish",
                  "mix", "play", "change", "follow", "enjoy"),
        "prefix": ("silver", "morning", "ocean", "dragon", "winter",
                   "meadow", "copper", "maple", "canyon", "dream"),
        "suffix": ("bridge", "field", "ring", "gate", "song",
                   "glow", "trail", "flower", "bank", "shore"),
    },
}

# Explicit prompt style variation; test question texts are not the
# same as lesson instructions. Still synthetically templated.
QUESTIONS = {
    "train": {
        "join_parts": "Join the two word parts and write the result.",
        "plural_noun": "Write the plural form for more than one noun.",
        "present_verb": "Write the verb for one person acting now.",
        "past_verb": "Write the verb for an action yesterday.",
        "actor_role": "Who is the giver or receiver requested in the event?",
        "write_sentence": "Write the full sentence: subject, present verb, object.",
        "negate_sentence": "Write the full negative sentence for this action.",
    },
    "development": {
        "join_parts": "What single word do the pieces make?",
        "plural_noun": "How should this noun be spelled when there are several?",
        "present_verb": "Which verb spelling agrees with one actor?",
        "past_verb": "Give the completed regular past form.",
        "actor_role": "Name the correct participant in the event.",
        "write_sentence": "Put the given parts into a complete English sentence.",
        "negate_sentence": "Express this event as a negative complete sentence.",
    },
    "exam": {
        "join_parts": "Form one written word from these ordered pieces.",
        "plural_noun": "Convert this single noun into its many-things form.",
        "present_verb": "Supply the present-tense action for a single subject.",
        "past_verb": "What regular past-tense verb would fit?",
        "actor_role": "According to the event, provide the requested person's name.",
        "write_sentence": "Produce a complete grammatical sentence from these parts.",
        "negate_sentence": "Rewrite the described action to say it does not happen.",
    },
}


def plural(noun: str) -> str:
    if not re.fullmatch("[a-z]+", noun):
        raise ValueError("Invalid noun")
    if noun.endswith("y") and noun[-2] not in "aeiou":
        return noun[:-1] + "ies"
    if noun.endswith(("s", "sh", "ch", "x", "z")):
        return noun + "es"
    return noun + "s"


def present(verb: str) -> str:
    if not re.fullmatch("[a-z]+", verb):
        raise ValueError("Invalid verb")
    if verb.endswith("y") and verb[-2] not in "aeiou":
        return verb[:-1] + "ies"
    if verb.endswith(("s", "sh", "ch", "x", "z")):
        return verb + "es"
    return verb + "s"


def past(verb: str) -> str:
    if not re.fullmatch("[a-z]+", verb):
        raise ValueError("Invalid verb")
    if verb.endswith("y") and verb[-2] not in "aeiou":
        return verb[:-1] + "ied"
    if verb.endswith("e"):
        return verb + "d"
    return verb + "ed"


def truth(skill: str, source: str) -> str:
    """Parse source and calculate answer without looking at its stored label."""
    if skill == "join_parts":
        m = re.fullmatch(r"parts=([a-z]+)\+([a-z]+)", source)
        if m:
            return "".join(m.groups())
    if skill in ("plural_noun", "present_verb", "past_verb"):
        key = {"plural_noun": "noun", "present_verb": "verb",
               "past_verb": "verb"}[skill]
        m = re.fullmatch(key+r"=([a-z]+);context=([A-Z][a-z]+)", source)
        if m:
            return {"plural_noun": plural, "present_verb": present,
                    "past_verb": past}[skill](m.group(1))
    if skill in ("write_sentence", "negate_sentence"):
        m = re.fullmatch(r"sentence=([A-Z][a-z]+)\|([a-z]+)\|([a-z]+)", source)
        if m:
            who, base, obj = m.groups()
            return (f"{who} {present(base)} the {obj}."
                    if skill == "write_sentence"
                    else f"{who} does not {base} the {obj}.")
    if skill == "actor_role":
        m = re.fullmatch(
            r"event=([A-Z][a-z]+)\|([a-z]+)\|([a-z]+)\|([A-Z][a-z]+);ask=(giver|receiver)",
            source)
        if m:
            giver, verb, obj, receiver, asked = m.groups()
            if giver == receiver:
                raise ValueError("Ambiguous role attribution")
            return giver if asked == "giver" else receiver
    raise ValueError("Unsupported skill or corrupt independent truth")


@dataclass(frozen=True)
class Lesson:
    id: str
    skill: str
    split: str
    source: str
    prompt: str
    answer: str


def check_firewall() -> None:
    for field in ("people", "nouns", "verbs", "prefix", "suffix"):
        seen = set()
        for split in ("train", "development", "exam"):
            values = LEX[split][field]
            if len(set(values)) != len(values) or seen.intersection(values):
                raise ValueError(f"Lexical overlap or duplicate: {field}")
            seen.update(values)


def prompt_for(split: str, skill: str, source: str) -> str:
    return (f"English practice\nExample: {source}\n"
            f"{QUESTIONS[split][skill]}\nAnswer:")


def validate(rows: list[Lesson], split: str | None = None) -> None:
    if not rows or len({x.id for x in rows}) != len(rows):
        raise ValueError("Invalid or duplicate teaching IDs")
    if len({x.prompt for x in rows}) != len(rows):
        raise ValueError("Repeated exact teaching prompts")
    for x in rows:
        if x.skill not in SKILLS or x.split not in LEX:
            raise ValueError("Unknown lesson skill or split")
        if split is not None and x.split != split:
            raise ValueError("Lesson crossed split boundary")
        if x.answer != truth(x.skill, x.source):
            raise ValueError("Wrong supplied answer for verified source")
        if x.prompt != prompt_for(x.split, x.skill, x.source):
            raise ValueError("Unreviewed question wording or data source")
        # Prevent endpoint/answer inclusion: all lesson prompts are
        # source inputs and task descriptions, never solved output lines.
        if "\nAnswer: " in x.prompt:
            raise ValueError("Gold answer leaked into prompt")


def make_lessons(split: str, *, per_skill: int, seed: int) -> list[Lesson]:
    if split not in LEX or per_skill < 4 or per_skill > 100 or per_skill % 2:
        raise ValueError("Unsupported lesson split and count")
    check_firewall()
    rng = random.Random(seed+{"train": 47, "development": 1901, "exam": 8011}[split])
    lex = LEX[split]
    rows = []
    for skill in SKILLS:
        seen = set()
        for i in range(per_skill):
            for attempt in range(2000):
                if skill == "join_parts":
                    source = f"parts={rng.choice(lex['prefix'])}+{rng.choice(lex['suffix'])}"
                elif skill == "plural_noun":
                    source = (f"noun={rng.choice(lex['nouns'])};context="
                              f"{rng.choice(lex['people'])}")
                elif skill == "present_verb":
                    source = (f"verb={rng.choice(lex['verbs'])};context="
                              f"{rng.choice(lex['people'])}")
                elif skill == "past_verb":
                    source = (f"verb={rng.choice(lex['verbs'])};context="
                              f"{rng.choice(lex['people'])}")
                elif skill in ("write_sentence", "negate_sentence"):
                    source = (f"sentence={rng.choice(lex['people'])}|"
                              f"{rng.choice(lex['verbs'])}|{rng.choice(lex['nouns'])}")
                else:
                    giver, receiver = rng.sample(lex["people"], 2)
                    source = (f"event={giver}|give|{rng.choice(lex['nouns'])}|"
                              f"{receiver};ask={'giver' if i%2==0 else 'receiver'}")
                if source not in seen:
                    seen.add(source)
                    rows.append(Lesson(
                        f"english-production-v2:{split}:{skill}:{i}", skill, split,
                        source, prompt_for(split, skill, source),
                        truth(skill, source)))
                    break
            else:
                raise ValueError(f"Not enough unique inputs for {split}/{skill}")
    validate(rows, split)
    return rows


def shuffle_teaching(rows: list[Lesson], seed: int) -> list[Lesson]:
    """Wrong labels with identical questions; shuffle inside skill bins."""
    rng = random.Random(seed)
    answers = {}
    for skill in SKILLS:
        original = [x.answer for x in rows if x.skill == skill]
        if len(set(original)) < 2:
            raise ValueError("Cannot build a meaningful null label control")
        for _ in range(30):
            permuted = original.copy()
            rng.shuffle(permuted)
            if sum(x != y for x, y in zip(original, permuted)) >= len(original)//2:
                break
        else:
            raise ValueError("No sufficiently perturbed shuffle")
        answers[skill] = iter(permuted)
    return [Lesson(x.id, x.skill, x.split, x.source, x.prompt,
                   next(answers[x.skill])) for x in rows]


def schedules(rows: list[Lesson], *, seed: int,
              updates_per_level: int = 80):
    if updates_per_level < 4 or updates_per_level > 200:
        raise ValueError("Outside bounded classroom lesson schedule")
    rng = random.Random(seed)
    buckets = {skill: [i for i, r in enumerate(rows) if r.skill == skill]
               for skill in SKILLS}
    staged = []
    for i, focus in enumerate(SKILLS):
        for n in range(updates_per_level):
            batch = [rng.choice(buckets[focus]) for _ in range(3)]
            earlier = SKILLS[(seed+n) % i] if i else focus
            batch.append(rng.choice(buckets[earlier]))
            staged.append(batch)
    mixed = staged.copy()
    rng.shuffle(mixed)
    if sorted(map(tuple, mixed)) != sorted(map(tuple, staged)):
        raise AssertionError("Mixed arms changed lesson batch exposure")
    return staged, mixed


def encode_samples(rows: list[Lesson], codec, max_context: int):
    """Loss is applied only on answer tokens; no A/B or answer options."""
    items = []
    for x in rows:
        prefix = codec.encode(x.prompt).ids
        response = codec.encode(" " + x.answer + "\n").ids
        if not prefix or not response or len(prefix)+len(response)-1 > max_context:
            raise ValueError(f"Lesson too long for original model: {x.id}")
        tokens = prefix+response[:-1]
        labels = [-100]*(len(prefix)-1)+response
        if len(tokens) != len(labels):
            raise AssertionError("Misaligned autoregressive answer tokens")
        items.append((tokens, labels))
    return items


def train_loss(model: ByteTransformer, batch) -> torch.Tensor:
    length = max(len(x) for x,_ in batch)
    X = torch.zeros((len(batch),length),dtype=torch.long)
    Y = torch.full((len(batch),length),-100,dtype=torch.long)
    for i,(a,b) in enumerate(batch):
        X[i,:len(a)] = torch.tensor(a,dtype=torch.long)
        Y[i,:len(b)] = torch.tensor(b,dtype=torch.long)
    logits,_ = model(X)
    return F.cross_entropy(
        logits.reshape(-1,model.cfg.vocab_size),
        Y.reshape(-1),ignore_index=-100)


def normalize(text: str) -> str:
    return " ".join(text.strip().split()).rstrip(".").casefold()


@torch.no_grad()
def evaluate(model: ByteTransformer, rows: list[Lesson], codec,
             max_new_tokens: int = 18) -> dict:
    if not 2 <= max_new_tokens <= 32:
        raise ValueError("Unbounded inference output")
    model.eval()
    result = {skill: {"correct": 0, "total": 0, "nonempty": 0}
              for skill in SKILLS}
    previews = []
    correct = 0
    for r in rows:
        prefix = codec.encode(r.prompt).ids
        new = []
        state = prefix[:]
        for _ in range(max_new_tokens):
            x = torch.tensor([state[-model.cfg.max_seq_len:]], dtype=torch.long)
            logits,_ = model(x)
            chosen = int(logits[0,-1].argmax())
            state.append(chosen)
            new.append(chosen)
            if "\n" in codec.decode(new):
                break
        raw = codec.decode(new)
        actual = normalize(raw.split("\n",1)[0])
        desired = normalize(r.answer)
        hit = int(actual==desired)
        correct += hit
        item = result[r.skill]
        item["correct"] += hit
        item["total"] += 1
        item["nonempty"] += int(bool(actual))
        if len(previews) < len(SKILLS) and (
                not any(x["skill"] == r.skill for x in previews)):
            previews.append({
                "skill": r.skill, "expected": r.answer,
                "generated": raw[:160], "correct": bool(hit),
            })
    return {"correct": correct, "total":len(rows),
            "exact_accuracy":round(correct/len(rows),4),
            "by_skill": {k:{**v,"accuracy":round(v["correct"]/v["total"],4)}
                         for k,v in result.items()},
            "examples":previews,
            "scoring":"Greedy, open-ended, exact normalized answer; not A/B"}


def comparable_story_loss(model, valid_ids):
    return round(heldout(model,valid_ids,windows=48,seq_len=64),6)


def study(*, folder: Path, out: Path, updates_per_level: int = 80,
          seeds: tuple[int,...] = SEEDS, lr: float = 1e-4,
          rehearsal_weight: float = .25) -> dict:
    if (not 4 <= updates_per_level <= 200 or not 1<=len(seeds)<=3
            or len(set(seeds))!=len(seeds) or
            not 0 < lr <= .001 or not 0 <= rehearsal_weight <= .5):
        raise ValueError("Unregistered research workload")
    torch.set_num_threads(min(2,torch.get_num_threads()))
    chk,manifest,codec=verify_source(folder)
    ck_hash=hashlib.sha256((folder/"checkpoint.pt").read_bytes()).hexdigest()
    text_ids=torch.tensor(codec.encode((folder/"train.txt").read_text()).ids,
                          dtype=torch.long)
    valid_ids=torch.tensor(codec.encode((folder/"valid.txt").read_text()).ids,
                           dtype=torch.long)
    config=ModelConfig(**chk["configuration"])
    baseline=ByteTransformer(config)
    baseline.load_state_dict(chk["model"],strict=True)
    baseline.eval()
    # These source/evaluation examples are fixed before any updates and
    # their gold answers are not consulted during train/dev selection.
    development=make_lessons("development",per_skill=8,seed=33441)
    exam=make_lessons("exam",per_skill=16,seed=99001)
    baseline_story=comparable_story_loss(baseline,valid_ids)
    frozen_dev=evaluate(baseline,development,codec)
    frozen_exam=evaluate(baseline,exam,codec)
    out.mkdir(parents=True,exist_ok=True)
    report=[]
    started=time.monotonic()
    for seed in seeds:
        original=make_lessons("train",per_skill=64,seed=seed)
        all_inputs={x.prompt for x in original}
        if (all_inputs & {x.prompt for x in development+exam}
                or {x.prompt for x in development} & {x.prompt for x in exam}):
            raise ValueError("Training or selection overlaps later questions")
        real_encoded=encode_samples(original,codec,config.max_seq_len)
        wrong=shuffle_teaching(original,seed+401)
        wrong_encoded=encode_samples(wrong,codec,config.max_seq_len)
        if any(a[0][:len(codec.encode(original[i].prompt).ids)]
               != b[0][:len(codec.encode(original[i].prompt).ids)]
               for i,(a,b) in enumerate(zip(real_encoded,wrong_encoded))):
            raise ValueError("Corrupted labels changed original inputs")
        for phase in (development,exam):
            encode_samples(phase,codec,config.max_seq_len)
        staged,mixed=schedules(original,seed=seed,updates_per_level=updates_per_level)
        schedules_for={"verified_mixed":mixed,"verified_staged":staged,
                       "shuffled_mixed":mixed,"text_only":mixed}
        row={"seed":seed,"arms":{}}
        for arm in ARMS:
            model=ByteTransformer(config)
            model.load_state_dict(chk["model"],strict=True)
            optimizer=torch.optim.AdamW(model.parameters(),lr=lr)
            src_rng=torch.Generator().manual_seed(seed+9017)
            best=None
            best_key=(-1,-1,-1)
            curve=[]
            train_examples=wrong_encoded if arm=="shuffled_mixed" else real_encoded
            for step,indexes in enumerate(schedules_for[arm],1):
                model.train()
                text_x,text_y=make_batch(text_ids,2,64,src_rng)
                _,text_loss=model(text_x,text_y)
                if arm=="text_only":
                    loss=text_loss
                    exercise_loss=None
                else:
                    exercise_loss=train_loss(
                        model,[train_examples[idx] for idx in indexes])
                    loss=((1-rehearsal_weight)*exercise_loss
                          +rehearsal_weight*text_loss)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),1.0)
                optimizer.step()
                if step==1 or step % updates_per_level == 0:
                    dev=evaluate(model,development,codec)
                    # Main criterion exact open-ended output, then
                    # performance of weakest individual lesson.
                    poorest=min(x["correct"] for x in dev["by_skill"].values())
                    rank=(dev["correct"],poorest,step)
                    if rank>best_key:
                        best_key=rank
                        best={"step":step,"weights":copy.deepcopy(model.state_dict())}
                    curve.append({
                        "update":step,"dev_exact":dev["correct"],
                        "weakest_skill_correct":poorest,
                        "loss":round(float(loss.detach()),5),
                        "lesson_loss": None if exercise_loss is None else
                            round(float(exercise_loss.detach()),5),
                    })
                    print(json.dumps({"seed":seed,"arm":arm,**curve[-1]}),flush=True)
            if best is None:
                raise AssertionError("Missing development-selected checkpoint")
            last=copy.deepcopy(model.state_dict())
            final_opt=optimizer.state_dict()
            model.load_state_dict(best["weights"])
            scored=evaluate(model,exam,codec)
            scored_dev=evaluate(model,development,codec)
            lm_loss=comparable_story_loss(model,valid_ids)
            torch.save({
                "schema":1,"research_only":True,"arm":arm,"seed":seed,
                "source_checkpoint_sha256":ck_hash,
                "source_tokenizer_sha256":chk["tokenizer_sha256"],
                "configuration":asdict(config),
                "best_dev_step":best["step"],
                "best_weights":best["weights"],
                "last_weights":last,
                "last_optimizer":final_opt,
                "rehearsal_rng":src_rng.get_state(),
                "torch_rng":torch.get_rng_state(),
            },out/f"seed-{seed}-{arm}-research.pt")
            row["arms"][arm]={
                "selected_dev_step":best["step"],
                "development":scored_dev,
                "synthetic_exam":scored,
                "heldout_story_loss":lm_loss,
                "story_loss_ratio_to_frozen":round(lm_loss/baseline_story,5),
                "updates":len(staged),"curve":curve,
            }
        report.append(row)
        (out/f"seed-{seed}-results.json").write_text(
            json.dumps(row,indent=2)+"\n",encoding="utf-8")
    result={
        "schema":1,"experiment":"forge-english-production-v2",
        "status":"research_only_not_promoted",
        "source_run_id":41572257363,"source_checkpoint_sha256":ck_hash,
        "source_tokenizer_sha256":chk["tokenizer_sha256"],
        "public_train_sha256":manifest["train_sha256"],
        "public_valid_sha256":manifest["valid_sha256"],
        "original_model_from_scratch":True,
        "skills":list(SKILLS),
        "updates_per_arm_per_seed":len(staged),
        "seeds":list(seeds),
        "train_items_per_seed":len(original),
        "dev_items":len(development),"exam_items":len(exam),
        "public_rehearsal_fraction":rehearsal_weight,
        "dev_exam_distinct_lexicons_and_wordings":True,
        "final_exam_is_used_after_this_experiment":True,
        "comparisons":list(ARMS),
        "baseline":{"development":frozen_dev,"synthetic_exam":frozen_exam,
                    "heldout_story_loss":baseline_story},
        "results":report,
        "elapsed_seconds":round(time.monotonic()-started,2),
        "promoted":False,
        "limitations":[
            "Synthetic template family overlaps abstract grammar between splits",
            "One exam set used once after dev selection, not an external reading test",
            "Short answers are exact-match generated, not conversational fluency",
            "Simple regular morphology only, no irregular words or speech/audio",
            "No extra language-model pretraining or Android/device memory",
            "Several distinct skills require longer generations and may fail",
            "The control arms differ in supervision and objective by design",
            "Experimental weights are not interchangeable with canonical Forge Seed",
        ],
    }
    (out/"english-production-v2.json").write_text(
        json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({
        "seconds":result["elapsed_seconds"],
        "frozen":frozen_exam["correct"],
        "results":[{
            "seed":x["seed"],"arms":{
                key:{"exact":value["synthetic_exam"]["correct"],
                     "per_skill":{
                         k:v["correct"] for k,v in
                         value["synthetic_exam"]["by_skill"].items()},
                     "story_ratio":value["story_loss_ratio_to_frozen"]}
                for key,value in x["arms"].items()}
        } for x in report],
    },indent=2),flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint-dir",type=Path,required=True)
    p.add_argument("--out",type=Path,default=Path("results/english-production-v2"))
    p.add_argument("--updates-per-level",type=int,default=80)
    p.add_argument("--seeds",type=int,choices=(1,2),default=2)
    a=p.parse_args()
    study(folder=a.checkpoint_dir,out=a.out,
          updates_per_level=a.updates_per_level,seeds=SEEDS[:a.seeds])


if __name__=="__main__":
    main()
