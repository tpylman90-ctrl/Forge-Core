"""Neural source-span pointer research for Forge Teacher v1.

The original scratch-trained Forge language backbone is FROZEN. A new
small trainable attention/pointer head selects one source WORD span
from a question, including previously unseen proper names. Training
uses audited rule-verified Teacher v1 examples, compared against a
matched incorrect-span control, never self-approved teacher guesses.

Copying is a limited capability; this is NOT autonomous reasoning.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import re
import time

import torch
from torch import nn
from torch.nn import functional as F

from forge_core.learning.teacher_v1 import audit_bundle, SKILL_SET
from forge_core.neural.model import ByteTransformer, ModelConfig
from forge_core.neural.preschool_qa import verify_source


_WORD = re.compile(r"[A-Za-z]+")


class WordPointer(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.rank = nn.Linear(width, 1, bias=False)

    def forward(self, query: torch.Tensor, words: torch.Tensor) -> torch.Tensor:
        if query.ndim != 1 or words.ndim != 2 or query.shape[0] != words.shape[1]:
            raise ValueError("Incompatible pointer feature shapes")
        q = self.query(query)
        keys = self.key(words)
        return torch.mv(keys, q) / math.sqrt(query.numel()) + self.rank(words).squeeze(-1)


@torch.no_grad()
def frozen_hidden(backbone: ByteTransformer, token_ids: list[int]) -> torch.Tensor:
    if not 1 <= len(token_ids) <= backbone.cfg.max_seq_len:
        raise ValueError("Source question outside frozen model context")
    ids = torch.tensor([token_ids], dtype=torch.long)
    positions = torch.arange(ids.shape[1], dtype=torch.long)
    z = backbone.token(ids) + backbone.position(positions)[None, :, :]
    for layer in backbone.layers:
        z = layer(z)
    return backbone.final_norm(z)[0].detach()


@dataclass
class Features:
    id: str
    skill: str
    query: torch.Tensor
    words: torch.Tensor
    candidates: tuple[str, ...]
    gold_index: int


@torch.no_grad()
def features_for(backbone: ByteTransformer, codec, row) -> Features:
    encoding = codec.encode(row.prompt)
    hidden = frozen_hidden(backbone, encoding.ids)
    candidates = []
    word_vectors = []
    for match in _WORD.finditer(row.prompt):
        indexes = [i for i, (left, right) in enumerate(encoding.offsets)
                   if right > match.start() and left < match.end()]
        if not indexes:
            continue
        candidates.append(match.group())
        word_vectors.append(hidden[indexes].mean(0))
    if not word_vectors:
        raise ValueError("Prompt has no copyable words")
    correct = [i for i,word in enumerate(candidates)
               if word.casefold() == row.answer.casefold()]
    if not correct:
        raise ValueError("Verified gold word cannot be copied from prompt")
    if len(candidates) < 2:
        raise ValueError("Expected at least one distractor")
    return Features(row.id, row.skill, hidden[-1],
                    torch.stack(word_vectors), tuple(candidates), correct[0])


@torch.no_grad()
def evaluate(head: WordPointer, records: list[Features]) -> dict:
    head.eval()
    per_skill = {skill: {"correct": 0, "total": 0} for skill in SKILL_SET}
    examples = []
    correct = 0
    for item in records:
        predicted_index = int(head(item.query,item.words).argmax())
        prediction = item.candidates[predicted_index]
        match = int(prediction.casefold() ==
                    item.candidates[item.gold_index].casefold())
        correct += match
        per_skill[item.skill]["correct"] += match
        per_skill[item.skill]["total"] += 1
        if len(examples) < 12:
            examples.append({
                "id": item.id, "skill": item.skill,
                "expected": item.candidates[item.gold_index],
                "predicted": prediction, "correct": bool(match),
            })
    return {
        "correct": correct, "total": len(records),
        "accuracy": round(correct/len(records),4),
        "by_skill": {
            skill:{**counts,
                   "accuracy":round(counts["correct"]/counts["total"],4)
                   if counts["total"] else None}
            for skill, counts in sorted(per_skill.items())
        },
        "examples": examples,
        "measure": "neural pointer copies a word from an input prompt",
    }


def experiment(*, source: Path, parent: Path, lessons: Path, out: Path,
               seed: int = 3381, steps: int = 500, lr: float = 0.001) -> dict:
    if not 1 <= seed <= 1_000_000 or not 1 <= steps <= 2000 or not 0 < lr <= 0.01:
        raise ValueError("Out-of-bounds neural pointer research")
    torch.set_num_threads(min(2,torch.get_num_threads()))
    train, dev = audit_bundle(lessons)
    original, manifest, codec = verify_source(source)
    origin_sha = hashlib.sha256((source/"checkpoint.pt").read_bytes()).hexdigest()
    parent_path=parent/"verified-research.pt"
    parent_weights = torch.load(parent_path,map_location="cpu",weights_only=True)
    if (parent_weights.get("schema") != 4
        or not parent_weights.get("experimental_only")
        or parent_weights.get("arm") != "verified"
        or parent_weights.get("source_checkpoint_sha256") != origin_sha
        or parent_weights.get("configuration") != original["configuration"]):
        raise ValueError("Unapproved parent candidate source")
    backbone = ByteTransformer(ModelConfig(**original["configuration"]))
    backbone.load_state_dict(parent_weights["model"],strict=True)
    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad_(False)
    train_features = [features_for(backbone, codec, row) for row in train]
    dev_features = [features_for(backbone, codec, row) for row in dev]
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        head = WordPointer(backbone.cfg.n_embd)
        null = copy.deepcopy(head)
        untrained = copy.deepcopy(head)
    rnd = random.Random(seed + 73)
    wrong_indexes = []
    for item in train_features:
        wrong = rnd.randrange(len(item.candidates)-1)
        if wrong >= item.gold_index:
            wrong += 1
        wrong_indexes.append(wrong)
    train_indices = {
        skill:[i for i, item in enumerate(train_features) if item.skill == skill]
        for skill in sorted(SKILL_SET)
    }
    rng = random.Random(seed + 500)
    skill_order = tuple(sorted(SKILL_SET))
    schedule = [
        [rng.choice(train_indices[skill]) for skill in rng.sample(skill_order,4)]
        for _ in range(steps)
    ]

    def update(pointer, targets, name):
        optimizer = torch.optim.AdamW(pointer.parameters(),lr=lr)
        pointer.train()
        curve=[]
        for step,batch in enumerate(schedule,1):
            losses=[
                F.cross_entropy(pointer(train_features[i].query,
                                        train_features[i].words)[None,:],
                                torch.tensor([targets[i]],dtype=torch.long))
                for i in batch
            ]
            loss=torch.stack(losses).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pointer.parameters(),1.0)
            optimizer.step()
            if step == 1 or step % 100 == 0 or step == steps:
                curve.append({"step":step,
                              "training_loss":round(float(loss.detach()),6)})
        print(json.dumps({"arm":name,"curve":curve}),flush=True)
        return curve

    started=time.monotonic()
    curve_gold=update(head,[x.gold_index for x in train_features],"verified")
    curve_null=update(null,wrong_indexes,"wrong-source-control")
    baseline=evaluate(untrained,dev_features)
    verified=evaluate(head,dev_features)
    control=evaluate(null,dev_features)
    report={
        "schema":1,"research_only":True,"not_promoted":True,
        "original_checkpoint_sha256":origin_sha,
        "parent_research_sha256":hashlib.sha256(parent_path.read_bytes()).hexdigest(),
        "lesson_manifest_sha256":hashlib.sha256(
            (lessons/"manifest.json").read_bytes()).hexdigest(),
        "source_train_sha256":manifest["train_sha256"],
        "original_backbone_frozen":all(not x.requires_grad for x in backbone.parameters()),
        "new_pointer_parameters":sum(x.numel() for x in head.parameters()),
        "all_pointer_weights_random_initialized":True,
        "seed":seed,"steps_per_arm":steps,
        "train_count":len(train_features),"development_count":len(dev_features),
        "untrained":baseline, "verified_pointer":verified,
        "wrong_source_pointer":control,
        "verified_curve":curve_gold,"wrong_control_curve":curve_null,
        "duration_seconds":round(time.monotonic()-started,2),
        "limitations":[
            "Internally authored development questions, not a held-out external exam",
            "Candidate span set includes all words; neural head learns to choose one",
            "Copying source words does not establish event reasoning or grammar understanding",
            "Wrong-source control uses incorrect source spans rather than matched gold frequency",
            "No backbone updates, no external actions, no personal memory and no promotion",
        ],
    }
    out.mkdir(parents=True,exist_ok=True)
    (out/"pointer-results.json").write_text(
        json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    for name,model in (("verified",head),("wrong-source",null)):
        torch.save({
            "schema":1,"experimental_only":True,"arm":name,
            "model":model.state_dict(),"width":backbone.cfg.n_embd,
            "source_checkpoint_sha256":origin_sha,
            "parent_sha256":report["parent_research_sha256"],
        },out/f"{name}-pointer.pt")
    print(json.dumps({
        "development_total":len(dev),
        "untrained":baseline["correct"],
        "verified":verified["correct"],
        "wrong_source_control":control["correct"],
        "pointer_parameters":report["new_pointer_parameters"],
        "backbone_frozen":report["original_backbone_frozen"],
    },indent=2),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--parent",type=Path,required=True)
    parser.add_argument("--lessons",type=Path,required=True)
    parser.add_argument("--out",type=Path,default=Path("artifacts/pointer-v1"))
    parser.add_argument("--steps",type=int,default=500)
    args=parser.parse_args()
    experiment(source=args.source,parent=args.parent,lessons=args.lessons,
               out=args.out,steps=args.steps)


if __name__=="__main__":
    main()
