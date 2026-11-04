"""A predeclared, READ-ONLY word-copy challenge for v5 research weights.

No learning: use the eight original v5 model copies selected on v5
development data and score genuinely new, human-readable word sources
plus an edited one-character twin. This does NOT test word meaning.
External concatenation remains separately named as scaffolding.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import time

import torch

from forge_core.neural.english_unbiased_v5 import (
    ALPHABET, ARMS, SEEDS, Task, evaluate, make_tasks,
    prompt_for, source_strings,
)
from forge_core.neural.model import ByteTransformer, ModelConfig
from forge_core.neural.preschool_qa import verify_source

# Fixed *before* final v5 results were available. These are small,
# manually curated common nouns, not model-generated challenge words.
WORDS = (
    "apple", "bread", "cloud", "dream", "earth", "flame",
    "glass", "heart", "island", "jungle", "kitten", "ladder",
    "market", "needle", "orange", "pencil", "quartz", "rocket",
    "silver", "turtle", "umbrella", "violet", "window", "yellow",
    "zipper", "garden", "forest", "bridge", "mirror", "planet",
    "candle", "harvest", "marble", "thunder", "morning", "dolphin",
)
CHALLENGE_SCHEMA = 1


def changed_twin(word: str) -> tuple[str,int]:
    if not re.fullmatch("[a-z]{4,8}",word):
        raise ValueError("Unregistered 4–8-letter English word")
    h = hashlib.sha256(("forge-realword-challenge-v5:"+word).encode()).digest()
    at = h[0] % len(word)
    shift = 1 + (h[1] % 25)
    char = ALPHABET[(ALPHABET.index(word[at]) + shift) % 26]
    twin = word[:at]+char+word[at+1:]
    if len(twin) != len(word) or sum(x!=y for x,y in zip(word,twin)) != 1:
        raise AssertionError("Invalid challenge pair")
    return twin,at+1


def challenge_rows() -> list[Task]:
    if len(set(WORDS)) != len(WORDS):
        raise ValueError("Duplicate original challenge words")
    rows = []
    all_words = set()
    for n,word in enumerate(WORDS):
        twin,index = changed_twin(word)
        if word in all_words or twin in all_words:
            raise ValueError("Overlapping word or edited twin")
        all_words.update((word,twin))
        for variant,text in enumerate((word,twin)):
            for position in range(1,len(text)+1):
                rows.append(Task(
                    id=f"forge-realword-challenge:{n:03}:{variant}:{position}",
                    split="exam", family=word, variant=variant,
                    text=text, changed_position=index, index=position,
                    prompt=prompt_for("exam",text,position),
                    answer=text[position-1],
                ))
    return rows


def check_unseen_challenge(rows: list[Task]) -> None:
    all_words = source_strings(rows)
    previous = [
        make_tasks("development",families=18,seed=7811),
        make_tasks("exam",families=40,seed=10413),
    ]
    previous.extend(make_tasks("train",families=180,seed=seed)
                    for seed in SEEDS)
    for old in previous:
        if all_words & source_strings(old):
            raise ValueError("Challenge source collides with original v5 tasks")
    if len(all_words) != 2*len(WORDS):
        raise ValueError("Challenge whole-word uniqueness failed")


def lineage(record: dict, *, arm: str, seed: int,
            original_hash: str, tokenizer_hash: str,
            config: dict) -> None:
    if (record.get("schema") != 1 or not record.get("research_only")
            or record.get("arm") != arm or record.get("seed") != seed
            or record.get("original_checkpoint_sha256") != original_hash
            or record.get("source_tokenizer_sha256") != tokenizer_hash
            or record.get("configuration") != config
            or "best_weights" not in record):
        raise ValueError("v5 adapted checkpoint has wrong original lineage")


def audit(*, source: Path, trained: Path, out: Path) -> dict:
    torch.set_num_threads(min(2,torch.get_num_threads()))
    original,manifest,codec=verify_source(source)
    source_hash=hashlib.sha256((source/"checkpoint.pt").read_bytes()).hexdigest()
    saved_report_path=trained/"english-unbiased-copy-v5.json"
    if not saved_report_path.exists():
        raise ValueError("Expected completed v5 experiment report missing")
    saved=json.loads(saved_report_path.read_text(encoding="utf-8"))
    if (saved.get("schema") != 1
            or saved.get("promoted") is not False
            or saved.get("source_training_run_id") != 41572257363
            or saved.get("source_checkpoint_sha256") != source_hash
            or saved.get("source_tokenizer_sha256") != original["tokenizer_sha256"]
            or saved.get("original_corpus_train_sha256") != manifest["train_sha256"]
            or saved.get("original_corpus_valid_sha256") != manifest["valid_sha256"]
            or saved.get("arms") != list(ARMS)
            or saved.get("seeds") != list(SEEDS)):
        raise ValueError("Original v5 research report fails provenance checks")
    rows=challenge_rows()
    check_unseen_challenge(rows)
    cfg=ModelConfig(**original["configuration"])
    started=time.monotonic()
    base=ByteTransformer(cfg)
    base.load_state_dict(original["model"],strict=True)
    baseline=evaluate(base,rows,codec)
    results=[]
    for seed in SEEDS:
        values={}
        for arm in ARMS:
            research_file=trained/f"seed-{seed}-{arm}-research.pt"
            if not research_file.exists():
                raise ValueError("Missing approved unpromoted v5 research model")
            checkpoint=torch.load(research_file,map_location="cpu",
                                  weights_only=True)
            lineage(checkpoint,arm=arm,seed=seed,original_hash=source_hash,
                    tokenizer_hash=original["tokenizer_sha256"],
                    config=original["configuration"])
            model=ByteTransformer(cfg)
            model.load_state_dict(checkpoint["best_weights"],strict=True)
            values[arm]=evaluate(model,rows,codec,
                                 hide_source=(arm=="source_masked"))
        results.append({"seed":seed,"arms":values})
        out.mkdir(parents=True,exist_ok=True)
        (out/f"seed-{seed}-word-transfer.json").write_text(
            json.dumps(results[-1],indent=2)+"\n",encoding="utf-8")
    report={
        "schema":CHALLENGE_SCHEMA,
        "experiment":"english-v5-independent-word-source-audit",
        "audit_no_training":True,
        "original_v5_run_id":41886375209,
        "source_v07_run_id":41572257363,
        "original_model_sha256":source_hash,
        "original_tokenizer_sha256":original["tokenizer_sha256"],
        "challenge_words":list(WORDS),
        "edited_twin_count":len(WORDS),
        "all_source_strings_disjoint_from_v5_train_dev_exam":True,
        "one_letter_at_each_source_index":True,
        "challenge_question_count":len(rows),
        "frozen_baseline":baseline,
        "results":results,
        "elapsed_seconds":round(time.monotonic()-started,2),
        "weights_promoted":False,
        "limitations":[
            "Common words may have occurred in original public story pretraining",
            "Altered twin strings are nonwords and source-copying, not semantics",
            "Word list was fixed before v5 results, but authored by an AI assistant",
            "Evaluation uses the same indexing questions as earlier synthetic tests",
            "Each complete word is assembled externally from separate neural letter answers",
            "Two training seeds share the same manually curated challenge words",
            "No new optimizer steps, APK files, personal memory or model promotion",
        ],
    }
    (out/"english-v5-word-source-audit.json").write_text(
        json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({
        "seconds":report["elapsed_seconds"],
        "baseline_correct":baseline["letters_correct"],
        "samples_per_seed":len(rows),
        "results":[{
            "seed":r["seed"],
            "arms":{name:{
                "letters":metrics["letters_correct"],
                "hard_pairs":metrics["paired_both_correct"],
                "externally_assembled":metrics["externally_assembled_exact"],
            } for name,metrics in r["arms"].items()}
        } for r in results],
    },indent=2),flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--trained",type=Path,required=True)
    p.add_argument("--out",type=Path,default=Path("results/english-words-v5"))
    a=p.parse_args()
    audit(source=a.source,trained=a.trained,out=a.out)


if __name__=="__main__":
    main()
