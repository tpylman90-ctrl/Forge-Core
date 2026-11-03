"""Friday early-language evaluation for Forge Seed v0.7.

Fixed, original held-out role/color/location/number scenarios; paired choice
order. Results are diagnostic (chance baseline ~50%), not comprehension proof.
Long-form prompts require manual review. No personal memories or pretrained
models are used or automatically promoted.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from tokenizers import Tokenizer

from forge_core.neural.model import ByteTransformer, ModelConfig


@dataclass(frozen=True)
class Question:
    skill: str
    story: str
    question: str
    correct: str
    distractor: str


def questions() -> list[Question]:
    items: list[Question] = []
    names = (("Mia", "Ben"), ("Leo", "Ada"), ("Sam", "Max"),
             ("Tia", "Ned"), ("Ivy", "Kim"))
    colors = (("red", "blue"), ("green", "yellow"), ("pink", "white"),
              ("black", "purple"), ("orange", "brown"))
    locations = (("under", "beside"), ("inside", "outside"),
                 ("above", "below"), ("behind", "near"), ("between", "around"))
    for i, ((a,b),(color1,color2),(loc1,loc2)) in enumerate(zip(names,colors,locations)):
        items.append(Question("color",
            f"{a} has a {color1} kite. {b} has a {color2} cup.",
            f"What color is {a}'s kite?", color1, color2))
        items.append(Question("recipient",
            f"{a} handed the toy to {b}.",
            "Who received the toy?", b, a))
        items.append(Question("location",
            f"{a} placed the little ball {loc1} the table.",
            "Where did the ball go?", loc1, loc2))
        digits = (("two","three"), ("four","five"), ("six","seven"),
                  ("eight","nine"), ("one","ten"))
        n1,n2=digits[i]
        items.append(Question("count",
            f"{a} found {n1} small stones. {b} found {n2} small stones.",
            f"How many stones did {a} find?", n1, n2))
    assert len(items)==20
    return items


def build_prompts() -> list[dict]:
    records = []
    for i,item in enumerate(questions()):
        for flip in (0,1):
            left,right = ((item.correct,item.distractor) if flip == 0
                          else (item.distractor,item.correct))
            prompt = (f"Story: {item.story}\nQuestion: {item.question}\n"
                      f"A) {left}\nB) {right}\nAnswer:")
            records.append({
                "id": f"heldout:{item.skill}:{i}:mirror{flip}",
                "skill": item.skill, "prompt": prompt,
                "gold": "A" if flip == 0 else "B",
            })
    if len(records)!=40:
        raise RuntimeError("Benchmark generation error")
    return records


@torch.no_grad()
def forced_choice(model: ByteTransformer, tokenizer: Tokenizer,
                  prompts: list[dict]) -> dict:
    model.eval()
    groups: dict[str,dict] = {}
    ncorrect = 0
    for task in prompts:
        prefix = tokenizer.encode(task["prompt"]).ids
        if len(prefix)>=model.cfg.max_seq_len:
            raise ValueError("Question exceeds model context")
        # Predict the first next token for the two equally short labels.
        # Score a two-way softmax, not unconstrained free text.
        # A and B must differ at their FIRST predicted token. Prefix
        # intentionally ends after ':' and choice labels have no shared
        # leading whitespace token that could make both scores identical.
        label_a = tokenizer.encode("A").ids
        label_b = tokenizer.encode("B").ids
        if not label_a or not label_b or label_a == label_b:
            raise ValueError("Invalid choice tokenization")
        logits,_=model(torch.tensor([prefix],dtype=torch.long))
        probabilities=F.log_softmax(logits[0,-1,:],dim=0)
        score_a=float(probabilities[label_a[0]])
        score_b=float(probabilities[label_b[0]])
        answer="A" if score_a>=score_b else "B"
        correct=int(answer==task["gold"])
        g=groups.setdefault(task["skill"],{"correct":0,"total":0})
        g["correct"]+=correct
        g["total"]+=1
        ncorrect+=correct
    return {
        "correct":ncorrect,"total":len(prompts),
        "accuracy":round(ncorrect/len(prompts),4),
        "by_skill":{name:{"correct":v["correct"],"total":v["total"],
                          "accuracy":round(v["correct"]/v["total"],4)}
                    for name,v in sorted(groups.items())},
        "limit":"A/B next-token forced choice, not free response; first label token only.",
    }


MANUAL_PROMPTS = [
    "Write one complete sentence about a little bird.",
    "Write one complete sentence about a green tree.",
    "Write one complete sentence about a puppy playing.",
    "Write one complete sentence about a rainy day.",
    "Write one complete sentence about a helpful friend.",
    "Write one complete sentence about a red ball.",
    "Write one complete sentence about the warm sun.",
    "Write one complete sentence about a small fish.",
    "Write one complete sentence about an apple.",
    "Write one complete sentence about going home.",
    "Tell a short two-sentence story about a child and a dog.",
    "Tell a short two-sentence story about a rabbit and a bird.",
    "Tell a short two-sentence story about finding a lost toy.",
    "Tell a short two-sentence story about helping a friend.",
    "Tell a short two-sentence story about planting a flower.",
]


@torch.no_grad()
def human_review_prompts(model: ByteTransformer, tokenizer: Tokenizer,
                         seed: int = 925, new_tokens: int = 60) -> list[dict]:
    result=[]
    was_training=model.training
    model.eval()
    with torch.random.fork_rng():
        for i,prompt in enumerate(MANUAL_PROMPTS):
            torch.manual_seed(seed+i)
            prefix = tokenizer.encode(prompt).ids
            result_ids=model.generate(torch.tensor([prefix],dtype=torch.long),
                                      new_tokens=new_tokens,temperature=0.7)
            continuation=tokenizer.decode(
                result_ids[0].tolist()[len(prefix):],skip_special_tokens=True)
            result.append({"prompt":prompt,"response":continuation,
                           "human_grade":"unreviewed"})
    model.train(was_training)
    return result


def evaluate_checkpoint(folder: Path, out: Path) -> dict:
    ckptpath=folder/"checkpoint.pt"
    codecpath=folder/"tokenizer.json"
    manifestpath=folder/"source-manifest.json"
    if not all(p.exists() for p in (ckptpath,codecpath,manifestpath)):
        raise ValueError("Incomplete training checkpoint artifact")
    manifest=json.loads(manifestpath.read_text())
    ckpt=torch.load(ckptpath,map_location="cpu",weights_only=True)
    if ckpt.get("schema")!=3 or ckpt.get("revision")!=manifest.get("revision"):
        raise ValueError("Source revision or checkpoint schema mismatch")
    sha=hashlib.sha256(codecpath.read_bytes()).hexdigest()
    if ckpt.get("tokenizer_sha256")!=sha:
        raise ValueError("Tokenizer differs from checkpoint")
    codec=Tokenizer.from_file(str(codecpath))
    model=ByteTransformer(ModelConfig(**ckpt["configuration"]))
    model.load_state_dict(ckpt["model"])
    model.eval()
    tasks=build_prompts()
    forced=forced_choice(model,codec,tasks)
    review=human_review_prompts(model,codec)
    result={"schema":1,"model_step":ckpt["step"],"parameters":model.num_parameters,
            "benchmarks":{"forced_choice":forced,"manual_generation":review},
            "conditions":{"forced_choice_threshold":"at least 16/20 novel unmirrored (or 32/40 mirrored) correct",
                          "human_sentence_threshold":"8/10 grammatical, complete, relevant",
                          "human_story_threshold":"3/5 coherent two-sentence stories"},
            "certification":"NOT CERTIFIED; direct instruction and prose require human review",
            "limitations":[
                "Template-based story QA, not independent standardized comprehension",
                "A/B first-token scoring is vulnerable to answer-position biases",
                "Mirrored answer choices reduce position bias but not prompt-template leakage",
                "A high score alone cannot establish causal or personal reasoning",
                "No human text-generation grades were performed automatically",
            ]}
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint-dir",type=Path,required=True)
    p.add_argument("--out",type=Path,default=Path("artifacts/preschool/capabilities.json"))
    a=p.parse_args()
    result=evaluate_checkpoint(a.checkpoint_dir,a.out)
    print(json.dumps({k:v for k,v in result.items() if k!="benchmarks"},indent=2))
    print("Forced choice:",json.dumps(result["benchmarks"]["forced_choice"],indent=2))


if __name__=="__main__":
    main()
