"""Forge Copy v6: independently trained character-aligned decoder research.

An original random-initialized trainable auxiliary component, NOT the
canonical Forge Seed language network. It learns attention between source
character slots and an autoregressive output character stream. No index
questions are asked during inference, and EOS must be generated.

Ablations separate correct source copying from format/length shortcuts:
token-only attention, attention plus verified alignment targets, pooled
non-pointer source, source-letters hidden, and shuffled targets. Evaluate
uniform random strings, changed-one-character twins, new longer strings,
and a predeclared set of everyday English words.

Original Forge Seed v0.7 weights and all Android/private data stay untouched.
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

from forge_core.neural.preschool_qa import verify_source

ALPHABET = "abcdefghijklmnopqrstuvwxyz"
EOS, BOS, PAD, MASK = 26, 27, 28, 29
SIZE = 30
MAX_SOURCE = 12
MAX_OUTPUT = 14
ARMS = ("pointer_alignment", "pointer_token_only", "pooled_source",
        "pointer_source_hidden", "pointer_wrong_targets")
SEEDS = (13631, 13632)
WORDS = (
    "banana", "carpet", "melody", "rabbit", "hammer", "puzzle",
    "castle", "lizard", "sunrise", "ticket", "butter", "spoon",
    "flower", "anchor", "camera", "jacket", "napkin", "pepper",
    "valley", "guitar", "beacon", "picnic", "timber", "winter",
    "sunset", "violin", "cabin", "glider", "sailor", "sprout",
    "clover", "funnel", "ripple", "crayon", "falcon", "pillow",
)
# The manually authored words above are illustrative, not a third-party
# independently certified human-authored dataset. Duplicates prohibited.
# Words can appear in the original TinyStories corpus: no semantic claim.


@dataclass(frozen=True)
class Pair:
    original: str
    twin: str
    changed_index: int
    split: str


def sequence_bucket(word: str) -> str:
    if not re.fullmatch("[a-z]{3,12}", word):
        raise ValueError("Source outside registered ASCII lowercase alphabet")
    value = hashlib.sha256(("forge-pointer-v6-families:" + word).encode()).digest()[0] % 12
    if value < 7:
        return "train"
    if value < 9:
        return "dev"
    if value < 11:
        return "exam"
    return "long"


def random_pair(rng: random.Random, minimum: int, maximum: int,
                split: str) -> Pair:
    for _ in range(30000):
        count = rng.randint(minimum, maximum)
        word = "".join(rng.choice(ALPHABET) for _ in range(count))
        index = rng.randrange(count)
        options = [x for x in ALPHABET if x != word[index]]
        different = rng.choice(options)
        twin = word[:index] + different + word[index+1:]
        # Both members of a changed-letter pair MUST map to the SAME
        # partition. Partitioning only the base previously allowed a
        # dev source string to reappear as a train twin.
        if sequence_bucket(word) == split and sequence_bucket(twin) == split:
            return Pair(word, twin, index, split)
    raise ValueError("Failed to generate random source pair")


def generate_pairs(split: str, count: int, seed: int) -> list[Pair]:
    if split not in ("train", "dev", "exam", "long") or not 8 <= count <= 1200:
        raise ValueError("Unregistered data split or sample count")
    lo, hi = (9, 12) if split == "long" else (3, 8)
    rng = random.Random(seed + {"train": 173, "dev": 2203,
                                "exam": 12511, "long": 19031}[split])
    output = []
    seen = set()
    for _ in range(count):
        for attempt in range(30000):
            pair = random_pair(rng, lo, hi, split)
            if pair.original not in seen and pair.twin not in seen:
                seen.add(pair.original)
                seen.add(pair.twin)
                output.append(pair)
                break
        else:
            raise ValueError("Could not create unique source pair")
    validate_pairs(output)
    return output


def word_pairs() -> list[Pair]:
    if len(WORDS) != len(set(WORDS)):
        raise ValueError("Duplicated original word list")
    out = []
    all_words = set()
    for word in WORDS:
        if not re.fullmatch("[a-z]{3,8}", word):
            raise ValueError("Real-word challenge outside known lengths")
        digest = hashlib.sha256(("forge-pointer-v6-natural:" + word).encode()).digest()
        index = digest[0] % len(word)
        offset = 1 + digest[1] % 25
        new = ALPHABET[(ALPHABET.index(word[index]) + offset) % 26]
        twin = word[:index] + new + word[index+1:]
        if word in all_words or twin in all_words:
            raise ValueError("Duplicated or colliding natural-word source")
        all_words.update((word, twin))
        out.append(Pair(word, twin, index, "words"))
    validate_pairs(out)
    return out


def validate_pairs(pairs: list[Pair]) -> None:
    if not pairs:
        raise ValueError("No source pairs")
    seen = set()
    for pair in pairs:
        a,b = pair.original,pair.twin
        if (not re.fullmatch("[a-z]{3,12}",a) or len(a) != len(b)
                or not 0 <= pair.changed_index < len(a)
                or sum(x != y for x,y in zip(a,b)) != 1
                or a[pair.changed_index] == b[pair.changed_index]):
            raise ValueError("Incorrect exact-one-change source pair")
        if pair.split not in ("train", "dev", "exam", "long", "words"):
            raise ValueError("Unrecognized evaluation split")
        if pair.split != "words" and (
                sequence_bucket(a) != pair.split
                or sequence_bucket(b) != pair.split):
            raise ValueError("Base or changed-letter twin crossed split firewall")
        if a in seen or b in seen:
            raise ValueError("Repeated source or twin")
        seen.update((a,b))


def assert_firewall(*sets: list[Pair]) -> None:
    seen = set()
    for pairs in sets:
        local = {w for pair in pairs for w in (pair.original,pair.twin)}
        if local & seen:
            raise ValueError("Train/dev/exam/word source collision")
        seen.update(local)


def expanded(pairs: list[Pair]) -> list[str]:
    return [word for pair in pairs for word in (pair.original,pair.twin)]


def ids(word: str) -> list[int]:
    if not re.fullmatch("[a-z]{3,12}",word):
        raise ValueError("No non-lowercase or oversized source allowed")
    return [ALPHABET.index(c) for c in word]


def batch_tensors(words: list[str], targets: list[str] | None = None,
                  *, mask_source: bool = False, device: str = "cpu"):
    if targets is None:
        targets = words
    if not words or len(words) != len(targets):
        raise ValueError("Mismatched or empty source and teacher labels")
    batch = len(words)
    in_len = max(len(w) for w in words) + 1
    out_len = max(len(t) for t in targets) + 1
    if max(in_len,out_len) > MAX_OUTPUT:
        raise ValueError("Outside bounded model context")
    source = torch.full((batch,in_len),PAD,dtype=torch.long,device=device)
    previous = torch.full((batch,out_len),PAD,dtype=torch.long,device=device)
    gold = torch.full((batch,out_len),-100,dtype=torch.long,device=device)
    # For pointer-alignment teaching, gold attention positions are exact
    # only for true copies. Don't run it with shuffled-target controls.
    alignment = torch.full((batch,out_len),-100,dtype=torch.long,device=device)
    for row,(word,target) in enumerate(zip(words,targets)):
        letters = ids(word)
        teacher = ids(target)
        source[row,:len(word)] = MASK if mask_source else torch.tensor(
            letters,device=device,dtype=torch.long)
        source[row,len(word)] = EOS
        gold[row,:len(target)] = torch.tensor(teacher,device=device)
        gold[row,len(target)] = EOS
        previous[row,0] = BOS
        if len(target):
            previous[row,1:len(target)+1] = torch.tensor(teacher,device=device)
        if word == target:
            alignment[row,:len(word)+1] = torch.arange(
                len(word)+1,device=device)
    return source,previous,gold,alignment


def sinusoidal(length: int, width: int, *, device) -> torch.Tensor:
    """Non-trainable position *features*, not a hardwired source pointer."""
    if length > MAX_OUTPUT or width % 2:
        raise ValueError("Sinusoidal encoding beyond model bounds")
    positions = torch.arange(length,device=device,dtype=torch.float).unsqueeze(1)
    denom = torch.exp(torch.arange(0,width,2,device=device,dtype=torch.float) *
                      (-math.log(10000.)/width))
    phase = positions*denom
    return torch.stack((phase.sin(),phase.cos()),dim=-1).flatten(1)


class NeuralCopyDecoder(nn.Module):
    """Original scratch-trained attention decoder, or pooled ablation.

    No source symbol is directly copied into the output by Python.
    The network emits one of 26 letters or EOS via a learned classifier.
    The decoder receives only its OWN previous outputs at inference.
    """
    def __init__(self, *, width: int = 64, pointer: bool = True):
        super().__init__()
        if not 16 <= width <= 128 or width % 2:
            raise ValueError("Unapproved model width")
        self.width = width
        self.pointer = pointer
        self.embedding = nn.Embedding(SIZE,width)
        self.source_proj = nn.Linear(2*width,width)
        self.query_proj = nn.Linear(2*width,width)
        self.key = nn.Linear(width,width)
        self.value = nn.Linear(width,width)
        self.query = nn.Linear(width,width)
        self.decode = nn.Sequential(
            nn.Linear(2*width,width),nn.GELU(),
            nn.Linear(width,EOS+1),
        )

    def forward(self, source: torch.Tensor, previous: torch.Tensor):
        if source.ndim != 2 or previous.ndim != 2 or (
                source.shape[0] != previous.shape[0]):
            raise ValueError("Decoder expects matching 2D source/previous")
        b,src_len = source.shape
        timesteps = previous.shape[1]
        src_pos = sinusoidal(src_len,self.width,device=source.device)
        query_pos = sinusoidal(timesteps,self.width,device=source.device)
        keys_input = torch.cat((
            self.embedding(source),
            src_pos.unsqueeze(0).expand(b,-1,-1),
        ),dim=-1)
        queries_input = torch.cat((
            self.embedding(previous),
            query_pos.unsqueeze(0).expand(b,-1,-1),
        ),dim=-1)
        encoded = torch.tanh(self.source_proj(keys_input))
        q = torch.tanh(self.query_proj(queries_input))
        present = source != PAD
        if not present.any(dim=1).all():
            raise ValueError("Missing source or EOS marker")
        if self.pointer:
            attention_logits = (
                self.query(q) @ self.key(encoded).transpose(1,2)
                /math.sqrt(self.width))
            attention_logits = attention_logits.masked_fill(
                ~present[:,None,:],-1e9)
            attention = F.softmax(attention_logits,dim=-1)
            context = attention@self.value(encoded)
        else:
            weights = present.to(encoded.dtype)
            pooled = (self.value(encoded)*weights.unsqueeze(-1)).sum(dim=1)
            pooled = pooled / weights.sum(dim=1,keepdim=True)
            context = pooled[:,None,:].expand(-1,timesteps,-1)
            attention = None
        logits = self.decode(torch.cat((q,context),dim=-1))
        return logits,attention


def objective(model: NeuralCopyDecoder, words: list[str],
              targets: list[str] | None = None,
              *, mask_source: bool = False,
              alignment_weight: float = 0.0) -> torch.Tensor:
    if not 0 <= alignment_weight <= 1:
        raise ValueError("Invalid process supervision weight")
    src,prev,gold,aligned = batch_tensors(
        words,targets,mask_source=mask_source)
    logits,attention = model(src,prev)
    token_loss = F.cross_entropy(logits.reshape(-1,EOS+1),
                                 gold.reshape(-1),ignore_index=-100)
    if alignment_weight:
        if attention is None or (aligned != -100).sum() != (gold != -100).sum():
            raise ValueError("Alignment loss requires uncorrupted genuine copies")
        # Gold source indices here are independent formal process labels;
        # they NEVER appear in forward inputs or during inference.
        log_weights = torch.log(attention.clamp_min(1e-9))
        attention_loss = F.nll_loss(
            log_weights.reshape(-1,src.shape[1]),
            aligned.reshape(-1),ignore_index=-100)
        return token_loss + alignment_weight*attention_loss
    return token_loss


def corrupted_map(pairs: list[Pair],seed: int) -> dict[str,str]:
    """Derangement of teacher targets inside same-length groups."""
    rng = random.Random(seed)
    groups: dict[int,list[str]] = {}
    for text in expanded(pairs):
        groups.setdefault(len(text),[]).append(text)
    result = {}
    for size,group in groups.items():
        if len(group) < 4:
            raise ValueError("Not enough matched-length negative labels")
        ordering = group.copy()
        rng.shuffle(ordering)
        shifted = ordering[1:]+ordering[:1]
        for a,b in zip(ordering,shifted):
            if a == b:
                raise AssertionError("Invalid self-copy negative control")
            result[a] = b
    return result


def training_batches(pairs: list[Pair],updates: int,seed: int):
    if not 20 <= updates <= 1600:
        raise ValueError("Training updates outside explicit CPU budget")
    rng = random.Random(seed)
    batches = []
    for _ in range(updates):
        sample = [rng.choice(pairs) for _ in range(4)]
        batches.append([word for pair in sample for word in (
            pair.original,pair.twin)])
    return batches


@torch.no_grad()
def greedy_generate(model: NeuralCopyDecoder, words: list[str],
                    *, mask_source: bool = False) -> list[str]:
    """The model, not a source-index loop, chooses ALL output letters/EOS."""
    model.eval()
    source,_,_,_ = batch_tensors(words,mask_source=mask_source)
    previous = torch.full((len(words),1),BOS,dtype=torch.long)
    sequences = [[] for _ in words]
    finished = torch.zeros(len(words),dtype=torch.bool)
    for step in range(MAX_OUTPUT):
        logits,_ = model(source,previous)
        token = logits[:,-1,:].argmax(dim=-1)
        for i,t in enumerate(token.tolist()):
            if not finished[i]:
                if t == EOS:
                    finished[i] = True
                else:
                    sequences[i].append(t)
        # Finished rows are forced EOS as a bookkeeping input to the
        # next model call; they never contribute output characters.
        token = torch.where(finished,torch.full_like(token,EOS),token)
        previous = torch.cat((previous,token[:,None]),dim=1)
        if finished.all():
            break
    return ["".join(ALPHABET[t] for t in sequence) for sequence in sequences]


def edit_distance(left: str,right: str) -> int:
    line = list(range(len(right)+1))
    for i,char in enumerate(left,1):
        nxt = [i]
        for j,other in enumerate(right,1):
            nxt.append(min(nxt[-1]+1,line[j]+1,line[j-1]+int(char != other)))
        line = nxt
    return line[-1]


@torch.no_grad()
def evaluate(model: NeuralCopyDecoder, pairs: list[Pair],
             *, mask_source: bool = False) -> dict:
    validate_pairs(pairs)
    model.eval()
    words = expanded(pairs)
    guesses = []
    for start in range(0,len(words),64):
        guesses.extend(greedy_generate(model,words[start:start+64],
                                       mask_source=mask_source))
    exact = sum(a==b for a,b in zip(words,guesses))
    ed = sum(edit_distance(a,b) for a,b in zip(words,guesses))
    chars = sum(len(word) for word in words)
    pair_exact = pair_both_changed = pair_changed_response = 0
    for index,p in enumerate(pairs):
        a,b=guesses[2*index:2*index+2]
        pair_exact += int(a==p.original and b==p.twin)
        correct_a = (len(a)>p.changed_index and
                     a[p.changed_index]==p.original[p.changed_index])
        correct_b = (len(b)>p.changed_index and
                     b[p.changed_index]==p.twin[p.changed_index])
        pair_both_changed += int(correct_a and correct_b)
        pair_changed_response += int(a!=b)
    previews = [{
        "original":p.original,"twin":p.twin,
        "generated_original":guesses[2*i],
        "generated_twin":guesses[2*i+1],
        "both_exact":guesses[2*i]==p.original and guesses[2*i+1]==p.twin,
    } for i,p in enumerate(pairs[:8])]
    return {
        "exact":exact,"total":len(words),
        "exact_rate":round(exact/len(words),4),
        "pair_exact":pair_exact,"pair_total":len(pairs),
        "changed_letter_both_correct":pair_both_changed,
        "changed_response":pair_changed_response,
        "character_error_rate":round(ed/max(1,chars),4),
        "total_character_edits":ed,
        "total_target_characters":chars,
        "examples":previews,
        "inference":"Unassisted source-to-whole-output greedy neural decoder with learned EOS",
        "source_letters_hidden":mask_source,
    }


def experiment(*, source: Path, out: Path, updates: int = 800,
               seeds: tuple[int,...] = SEEDS,
               lr: float = .003) -> dict:
    if (not 20 <= updates <= 1600 or not 1 <= len(seeds) <= 2
            or len(set(seeds)) != len(seeds) or not 0 < lr <= .01):
        raise ValueError("Outside preregistered resource boundary")
    # Source reference is verified, but NEVER used to initialize the
    # independently trained auxiliary copy decoder.
    original,manifest,_ = verify_source(source)
    original_sha=hashlib.sha256((source/"checkpoint.pt").read_bytes()).hexdigest()
    torch.set_num_threads(min(2,torch.get_num_threads()))
    dev=generate_pairs("dev",80,10219)
    regular=generate_pairs("exam",128,12347)
    longer=generate_pairs("long",72,17193)
    natural=word_pairs()
    # Preflight ALL scheduled training seeds and all test splits before
    # the first optimizer step. A late split collision must never waste
    # a full seed's CPU training or contaminate its checkpoint.
    source_by_seed={
        seed:generate_pairs("train",512,seed) for seed in seeds
    }
    for training_set in source_by_seed.values():
        assert_firewall(training_set,dev,regular,longer,natural)
    out.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    frozen_results=[]
    for seed in seeds:
        train=source_by_seed[seed]
        shuffle=corrupted_map(train,seed+311)
        batches=training_batches(train,updates,seed=seed+811)
        row={"seed":seed,"arms":{}}
        for arm in ARMS:
            torch.manual_seed(seed+155)
            model=NeuralCopyDecoder(pointer=(arm!="pooled_source"))
            optim=torch.optim.AdamW(model.parameters(),lr=lr)
            best=None
            best_key=(-1,-1,-9999999,-1)
            curve=[]
            for step,words in enumerate(batches,1):
                model.train()
                wrong=[shuffle[x] for x in words] if arm=="pointer_wrong_targets" else None
                loss=objective(
                    model,words,wrong,
                    mask_source=arm=="pointer_source_hidden",
                    alignment_weight=.35 if arm in (
                        "pointer_alignment","pointer_source_hidden") else 0)
                optim.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),1.0)
                optim.step()
                if step==1 or step%200==0 or step==updates:
                    dev_score=evaluate(model,dev,
                                       mask_source=(arm=="pointer_source_hidden"))
                    key=(dev_score["pair_exact"],
                         dev_score["exact"],
                         -dev_score["total_character_edits"],step)
                    if key>best_key:
                        best_key=key
                        best={"update":step,
                              "weights":copy.deepcopy(model.state_dict())}
                    metrics={
                        "step":step,
                        "training_loss":round(float(loss.detach()),5),
                        "dev_exact":dev_score["exact"],
                        "dev_pair_exact":dev_score["pair_exact"],
                        "dev_char_error":dev_score["character_error_rate"],
                    }
                    curve.append(metrics)
                    print(json.dumps({"seed":seed,"arm":arm,**metrics}),flush=True)
            last_weights=copy.deepcopy(model.state_dict())
            model.load_state_dict(best["weights"])
            scores={
                "development":evaluate(
                    model,dev,mask_source=arm=="pointer_source_hidden"),
                "novel_short":evaluate(
                    model,regular,mask_source=arm=="pointer_source_hidden"),
                "unseen_longer":evaluate(
                    model,longer,mask_source=arm=="pointer_source_hidden"),
                "curated_words":evaluate(
                    model,natural,mask_source=arm=="pointer_source_hidden"),
            }
            torch.save({
                "schema":1,"research_only":True,
                "source_checkpoint_sha256":original_sha,
                "original_model_used_for_training":False,
                "arm":arm,"seed":seed,
                "model_initialization":"random-from-scratch",
                "parameter_count":sum(p.numel() for p in model.parameters()),
                "architecture":{"width":model.width,"pointer":model.pointer},
                "best_development_step":best["update"],
                "best_weights":best["weights"],
                "last_weights":last_weights,
                "last_optimizer_state":optim.state_dict(),
                "last_torch_rng_state":torch.get_rng_state(),
            },out/f"seed-{seed}-{arm}-research.pt")
            row["arms"][arm]={
                "parameters":sum(p.numel() for p in model.parameters()),
                "selected_development_step":best["update"],
                "training_curve":curve,
                "evaluation":scores,
            }
        frozen_results.append(row)
        (out/f"seed-{seed}-scores.json").write_text(
            json.dumps(row,indent=2)+"\n",encoding="utf-8")
    final={
        "schema":1,"study":"forge-learned-character-copy-v6",
        "status":"research_only_unpromoted",
        "original_source_run_id":41572257363,
        "original_checkpoint_sha256":original_sha,
        "source_train_sha256":manifest["train_sha256"],
        "source_valid_sha256":manifest["valid_sha256"],
        "original_scratch_language_weights_unchanged":True,
        "module_original_random_initialized":True,
        "training_set_not_pretrained_language_model":True,
        "arms":list(ARMS),"seeds":list(seeds),
        "updates_per_arm_seed":updates,
        "train_families_per_seed":512,
        "development_families":len(dev),
        "novel_short_families":len(regular),
        "unseen_long_families":len(longer),
        "curated_word_families":len(natural),
        "results":frozen_results,
        "elapsed_seconds":round(time.monotonic()-started,2),
        "never_promoted_to_canonical_model_or_android":True,
        "limitations":[
            "The auxiliary neural copy decoder is NOT yet connected to Forge Seed language processing",
            "This is character copying on artificial strings, not grammatical understanding",
            "Source-aligned attention and EOS position add an inductive bias; do not call it unstructured language learning",
            "Even correct copying is a narrow algorithmic skill, not independent intelligence",
            "Longer 9-12 letters and human-readable words are disjoint tests from 3-8-character training",
            "All source strings are lowercase ASCII, not multilingual or speech data",
            "Word list was selected by assistant and might occur in original public stories",
            "Only two model initializations, internally authored synthetic tests",
            "Ablations differ in architecture or training information, not strictly parameter matched",
            "No saved personal memory, no Android modifications, no automatic checkpoint promotion",
        ],
    }
    (out/"english-learned-pointer-v6.json").write_text(
        json.dumps(final,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({
        "seconds":final["elapsed_seconds"],
        "results":[{
            "seed":run["seed"],"arms":{
                name:{
                    split:{
                        "exact":score["exact"],
                        "pair_both":score["pair_exact"],
                        "changed_both":score["changed_letter_both_correct"],
                        "character_error":score["character_error_rate"],
                    } for split,score in val["evaluation"].items()
                } for name,val in run["arms"].items()}
        } for run in frozen_results],
    },indent=2),flush=True)
    return final


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--out",type=Path,default=Path("results/english-pointer-v6"))
    p.add_argument("--updates",type=int,default=800)
    p.add_argument("--seeds",type=int,choices=(1,2),default=2)
    args=p.parse_args()
    experiment(source=args.source,out=args.out,updates=args.updates,
               seeds=SEEDS[:args.seeds])


if __name__=="__main__":
    main()
