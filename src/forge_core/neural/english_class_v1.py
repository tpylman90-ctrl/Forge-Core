"""English Class v1: teach Forge's ORIGINAL scratch-trained language network.

A deliberately simple, deterministic school-style progression:
uppercase letters, vowels, word composition, nouns/verbs, subjects,
and complete grammatical sentences. Three equal-compute arms contrast
ordered lessons, the identical lessons interleaved, and corrupted labels.

All examples are original rule-generated public text, not personal data.
Every answer is independently checked from its source structure. The
existing pinned TinyStories tokenizer and checkpoint are verified and
adapted only into unpromoted research files; language loss is monitored.

A/B forced choice is a PROBE, not a test of fluent open-ended English.
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

SKILLS = ("alphabet_case", "vowels", "word_parts", "nouns_verbs",
          "sentence_subject", "sentence_order")
ARMS = ("english_class", "same_lessons_interleaved", "corrupted_answers")

# The same alphabet must be available in English's train and exam splits.
# Disjoint lexical groups provide an actual unseen-word grammar test.
LEXICON = {
    "train": {
        "people": ("Arlo", "Brenna", "Cora", "Dario", "Etta", "Fenna",
                   "Galen", "Hedra", "Iona", "Joren"),
        "objects": ("lantern", "wagon", "basket", "candle", "pencil",
                    "marble", "bucket", "ribbon", "tunnel", "blanket"),
        "verbs": ("moves", "carries", "holds", "passes", "lifts",
                  "finds", "pushes", "throws", "paints", "opens"),
        "prefixes": ("sun", "rain", "moon", "snow", "wood", "book",
                     "hand", "star", "fish", "light"),
        "suffixes": ("light", "fall", "rise", "box", "house", "path",
                     "stone", "mark", "bird", "side"),
    },
    "development": {
        "people": ("Kyla", "Lennox", "Mavis", "Nolan", "Odessa"),
        "objects": ("compass", "ladder", "flower", "hammer", "pocket"),
        "verbs": ("pulls", "mends", "spins", "folds", "turns"),
        "prefixes": ("pine", "dusk", "wave", "leaf", "hill"),
        "suffixes": ("bank", "gate", "trail", "pool", "crest"),
    },
    "exam": {
        "people": ("Quilla", "Ronan", "Sela", "Tavon", "Ulric",
                   "Vanya", "Willa", "Xander"),
        "objects": ("whistle", "necklace", "mirror", "saddle", "paddle",
                    "suitcase", "feather", "drum"),
        "verbs": ("shakes", "cleans", "fetches", "borrows", "guides",
                  "catches", "repairs", "follows"),
        "prefixes": ("mist", "shore", "dawn", "meadow", "river", "flame",
                     "pearl", "silver"),
        "suffixes": ("walk", "bell", "shell", "bloom", "tower", "bend",
                     "view", "song"),
    },
}
LETTER_ALPHABET = "abcdefghijklmnopqrstuvwxyz"
VOWELS = set("aeiou")

TRAIN_QUESTIONS = {
    "alphabet_case": "Lesson: A capital letter is the uppercase version of a letter. Which is the uppercase form?",
    "vowels": "Lesson: The vowels are a, e, i, o, and u. Which category does the letter belong to?",
    "word_parts": "Lesson: Two word parts can be joined without changing their order. Which joined spelling is correct?",
    "nouns_verbs": "Lesson: A noun names a person or thing; a verb describes an action. Which word is the verb?",
    "sentence_subject": "Lesson: The subject is the one doing the action. Who is the subject?",
    "sentence_order": "Lesson: An English sentence often puts the subject before the verb and object. Which sentence has correct order?",
}
DEV_QUESTIONS = {
    "alphabet_case": "Select the capital version of this lowercase letter.",
    "vowels": "Classify this letter as a vowel or consonant.",
    "word_parts": "Choose the word formed by joining these pieces.",
    "nouns_verbs": "Identify the action word in this sentence.",
    "sentence_subject": "Who performs the action in this sentence?",
    "sentence_order": "Which alternative uses normal subject-verb-object word order?",
}
EXAM_QUESTIONS = {
    "alphabet_case": "If this character were capitalized, which would it be?",
    "vowels": "Which of the two descriptions fits this character?",
    "word_parts": "When the two written units combine, which result follows?",
    "nouns_verbs": "In the example, which choice tells what happened?",
    "sentence_subject": "Name the person responsible for the action.",
    "sentence_order": "Pick the correctly arranged full sentence.",
}


@dataclass(frozen=True)
class Exercise:
    id: str
    split: str
    skill: str
    source: str
    options: tuple[str, str]
    prompt: str
    answer: str


def independent_gold(skill: str, source: str) -> str:
    """Reconstruct gold WITHOUT reading the stored answer or choices."""
    if skill in ("alphabet_case", "vowels"):
        m = re.fullmatch(r"letter=([a-z]);sample_word=([a-z]+)", source)
        if not m:
            raise ValueError("Malformed letter evidence")
        c = m.group(1)
        return c.upper() if skill == "alphabet_case" else (
            "vowel" if c in VOWELS else "consonant")
    if skill == "word_parts":
        m = re.fullmatch(r"parts=([a-z]+)\+([a-z]+)", source)
        if not m:
            raise ValueError("Malformed original word parts")
        return m.group(1) + m.group(2)
    if skill in ("nouns_verbs", "sentence_order"):
        m = re.fullmatch(r"sentence=([A-Z][a-z]+) ([a-z]+) the ([a-z]+)\.", source)
        if not m:
            raise ValueError("Malformed full sentence")
        return m.group(2) if skill == "nouns_verbs" else source[len("sentence="):]
    if skill == "sentence_subject":
        m = re.fullmatch(r"sentence=([A-Z][a-z]+) ([a-z]+) ([A-Z][a-z]+)\.", source)
        if not m:
            raise ValueError("Malformed subject and object sentence")
        if m.group(1) == m.group(3):
            raise ValueError("Ambiguous two-person sentence")
        return m.group(1)
    raise ValueError("Unsupported teaching category")


def validate(rows: list[Exercise], *, allowed_split: str | None = None) -> None:
    if not rows or len({row.id for row in rows}) != len(rows):
        raise ValueError("Duplicate or missing exercise IDs")
    if len({row.prompt for row in rows}) != len(rows):
        raise ValueError("Exact duplicate exercise prompts")
    positions = {skill: [0, 0] for skill in SKILLS}
    for row in rows:
        if row.skill not in SKILLS or row.split not in LEXICON:
            raise ValueError("Invalid skill or split")
        if allowed_split is not None and row.split != allowed_split:
            raise ValueError("Exercise crossed dataset boundary")
        truth = independent_gold(row.skill, row.source)
        if row.options[0] == row.options[1] or row.options.count(truth) != 1:
            raise ValueError("No unique, correct candidate")
        actual = "A" if row.options[0] == truth else "B"
        if row.answer != actual:
            raise ValueError("Stored label contradicts independent grammar oracle")
        if row.split == "train":
            question = TRAIN_QUESTIONS[row.skill]
        elif row.split == "development":
            question = DEV_QUESTIONS[row.skill]
        else:
            question = EXAM_QUESTIONS[row.skill]
        expected = (f"English exercise\nSource: {row.source}\n"
                    f"{question}\nA) {row.options[0]}\n"
                    f"B) {row.options[1]}\nAnswer:")
        if row.prompt != expected:
            raise ValueError("Exercise content or wording was altered")
        positions[row.skill][0 if actual == "A" else 1] += 1
    # A training-only spelling shortcut from always choosing A or B
    # should be impossible by construction in every subject.
    for skill, (a, b) in positions.items():
        if a + b >= 8 and min(a, b) < (a + b) // 4:
            raise ValueError("Answer-position imbalance exceeds reviewed bound")


def check_lexical_firewall() -> None:
    for word_type in ("people", "objects", "verbs", "prefixes", "suffixes"):
        groups = [set(LEXICON[s][word_type]) for s in LEXICON]
        if (groups[0] & groups[1] or groups[0] & groups[2] or
                groups[1] & groups[2]):
            raise ValueError("Training, development, and exam lexica overlap")
    for split, grouped in LEXICON.items():
        if any(len(set(words)) != len(words) for words in grouped.values()):
            raise ValueError(f"Duplicate vocabulary inside {split}")


def make_exercises(split: str, *, per_skill: int = 64,
                   seed: int = 1701) -> list[Exercise]:
    if split not in LEXICON or per_skill < 8 or per_skill > 100 or per_skill % 2:
        raise ValueError("Invalid language lesson split or bounded count")
    check_lexical_firewall()
    rng = random.Random(seed + {
        "train": 31, "development": 10007, "exam": 19111}[split])
    lex = LEXICON[split]
    questions = {"train": TRAIN_QUESTIONS, "development": DEV_QUESTIONS,
                 "exam": EXAM_QUESTIONS}[split]
    rows = []
    for skill in SKILLS:
        used = set()
        for i in range(per_skill):
            # Every adjacent pair reverses A/B positions for two different
            # examples. This gives exact per-skill position balance.
            answer_position = i % 2
            for attempt in range(1000):
                if skill in ("alphabet_case", "vowels"):
                    if skill == "alphabet_case":
                        letter = rng.choice(LETTER_ALPHABET)
                        right = letter.upper()
                        wrong = rng.choice([c for c in LETTER_ALPHABET.upper()
                                            if c != right])
                    else:
                        letter = rng.choice("aeiou" if i % 4 < 2 else
                                            "bcdfghjklmnpqrstvwxyz")
                        right = "vowel" if letter in VOWELS else "consonant"
                        wrong = "consonant" if right == "vowel" else "vowel"
                    source = (f"letter={letter};sample_word="
                              f"{rng.choice(lex['prefixes'])}"
                              f"{rng.choice(lex['suffixes'])}")
                elif skill == "word_parts":
                    first = rng.choice(lex["prefixes"])
                    second = rng.choice(lex["suffixes"])
                    other = rng.choice([c for c in lex["suffixes"] if c != second])
                    source, right, wrong = (
                        f"parts={first}+{second}", first+second, first+other)
                elif skill == "nouns_verbs":
                    name = rng.choice(lex["people"])
                    verb = rng.choice(lex["verbs"])
                    obj = rng.choice(lex["objects"])
                    source = f"sentence={name} {verb} the {obj}."
                    right, wrong = verb, obj
                elif skill == "sentence_subject":
                    first, second = rng.sample(lex["people"], 2)
                    verb = rng.choice(lex["verbs"])
                    source = f"sentence={first} {verb} {second}."
                    right, wrong = first, second
                else:
                    name = rng.choice(lex["people"])
                    verb = rng.choice(lex["verbs"])
                    obj = rng.choice(lex["objects"])
                    source = f"sentence={name} {verb} the {obj}."
                    right = source[len("sentence="):]
                    wrong = f"The {obj} {name} {verb}."
                options = ((right, wrong) if answer_position == 0
                           else (wrong, right))
                prompt = (f"English exercise\nSource: {source}\n"
                          f"{questions[skill]}\nA) {options[0]}\n"
                          f"B) {options[1]}\nAnswer:")
                if prompt not in used:
                    used.add(prompt)
                    rows.append(Exercise(
                        id=f"english-v1:{split}:{skill}:{i}",
                        split=split, skill=skill, source=source,
                        options=options, prompt=prompt,
                        answer=("A" if answer_position == 0 else "B")))
                    break
            else:
                raise ValueError("Unable to generate unique English exercises")
    validate(rows, allowed_split=split)
    return rows


def null_labels(rows: list[Exercise], seed: int) -> list[str]:
    """Wrong teaching, same prompts and exactly preserved A/B distribution."""
    rng = random.Random(seed)
    result = [row.answer for row in rows]
    for skill in SKILLS:
        indexes = [i for i, row in enumerate(rows) if row.skill == skill]
        changed = indexes.copy()
        rng.shuffle(changed)
        labels = [rows[i].answer for i in changed]
        for idx, value in zip(indexes, labels):
            result[idx] = value
    if sum(a != b.answer for a, b in zip(result, rows)) < len(rows) // 4:
        raise ValueError("Shuffled negative control did not corrupt labels")
    return result


def schedules(rows: list[Exercise], *, seed: int,
              updates_per_level: int = 120,
              batch_size: int = 4) -> tuple[list[list[int]], list[list[int]]]:
    """Same *batch multiset* for staged and interleaved, different order."""
    if not 4 <= updates_per_level <= 350 or batch_size != 4:
        raise ValueError("Unreviewed training schedule")
    if not rows or {row.skill for row in rows} != set(SKILLS):
        raise ValueError("Missing curriculum categories")
    rng = random.Random(seed)
    buckets = {skill: [i for i, x in enumerate(rows) if x.skill == skill]
               for skill in SKILLS}
    staged = []
    for level, focus in enumerate(SKILLS):
        for update in range(updates_per_level):
            selected = [rng.choice(buckets[focus]) for _ in range(batch_size)]
            if level:
                # Keep old lessons in use; the comparison gets the exact
                # same examples, only in a different order.
                old_skill = SKILLS[(update + seed) % level]
                selected[-1] = rng.choice(buckets[old_skill])
            staged.append(selected)
    interleaved = list(staged)
    rng.shuffle(interleaved)
    if sorted(map(tuple, staged)) != sorted(map(tuple, interleaved)):
        raise AssertionError("Mismatch in matched-exposure control")
    return staged, interleaved


def encode_exercises(rows, codec, max_context: int) -> list[tuple[list[int], int]]:
    encoded = []
    for row in rows:
        token_ids = codec.encode(row.prompt).ids
        if not 1 <= len(token_ids) <= max_context:
            raise ValueError("English exercise exceeds model's context")
        encoded.append((token_ids, 0 if row.answer == "A" else 1))
    return encoded


def answer_loss(model: ByteTransformer, batch,
                answer_ids: tuple[int, int]) -> torch.Tensor:
    length = max(len(x[0]) for x in batch)
    x = torch.zeros((len(batch), length), dtype=torch.long)
    positions, truth = [], []
    for j, (ids, label) in enumerate(batch):
        x[j, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        positions.append(len(ids) - 1)
        truth.append(label)
    logits, _ = model(x)
    relevant = logits[torch.arange(len(batch)), positions][:, list(answer_ids)]
    return F.cross_entropy(relevant, torch.tensor(truth, dtype=torch.long))


@torch.no_grad()
def score(model: ByteTransformer, examples: list[Exercise], codec,
          answer_ids: tuple[int, int]) -> dict:
    model.eval()
    by_skill = {skill: {"correct": 0, "total": 0, "a_predictions": 0}
                for skill in SKILLS}
    batch_size = 8
    for start in range(0, len(examples), batch_size):
        chunk = examples[start:start+batch_size]
        encoded = encode_exercises(chunk, codec, model.cfg.max_seq_len)
        longest = max(len(ids) for ids, _ in encoded)
        x = torch.zeros((len(chunk), longest), dtype=torch.long)
        positions = []
        for j, (ids, _) in enumerate(encoded):
            x[j, :len(ids)] = torch.tensor(ids, dtype=torch.long)
            positions.append(len(ids) - 1)
        logits, _ = model(x)
        last = logits[torch.arange(len(chunk)), positions][:, list(answer_ids)]
        guesses = last.argmax(dim=-1).tolist()
        for row, pred in zip(chunk, guesses):
            skill = by_skill[row.skill]
            skill["total"] += 1
            skill["correct"] += int(("A" if pred == 0 else "B") == row.answer)
            skill["a_predictions"] += int(pred == 0)
    correct = sum(x["correct"] for x in by_skill.values())
    total = len(examples)
    return {"correct": correct, "total": total,
            "accuracy": round(correct / total, 4),
            "by_skill": {k: {**v, "accuracy": round(v["correct"] /
                                                  max(1, v["total"]), 4)}
                         for k, v in by_skill.items()}}


def comparable_story_loss(model: ByteTransformer, valid_tokens: torch.Tensor) -> float:
    """Same windows and sequence length before and after adaptation.

    Using fewer baseline windows than post-training windows creates a
    non-comparable language retention ratio. Both use 48 fixed positions.
    """
    return round(heldout(model, valid_tokens, windows=48, seq_len=64), 6)


def experiment(*, folder: Path, out: Path, seed: int = 7331,
               updates_per_level: int = 120, lr: float = 1.2e-4,
               rehearsal_weight: float = .20) -> dict:
    if not 4 <= updates_per_level <= 350 or not 0 < lr <= .005:
        raise ValueError("English training exceeds approved CPU bound")
    if not 0 <= rehearsal_weight <= .4:
        raise ValueError("Unreviewed language retention weight")
    torch.set_num_threads(min(2, torch.get_num_threads()))
    checkpoint, manifest, codec = verify_source(folder)
    original_sha = hashlib.sha256((folder/"checkpoint.pt").read_bytes()).hexdigest()
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        original = ByteTransformer(ModelConfig(**checkpoint["configuration"]))
        original.load_state_dict(checkpoint["model"], strict=True)
        models = {arm: copy.deepcopy(original) for arm in ARMS}
    answer_a, answer_b = codec.encode("A").ids, codec.encode("B").ids
    if len(answer_a) != 1 or len(answer_b) != 1 or answer_a[0] == answer_b[0]:
        raise ValueError("A/B answer labels must be distinct atomic BPE tokens")
    answer_ids = answer_a[0], answer_b[0]
    rows_train = make_exercises("train", per_skill=64, seed=seed)
    dev = make_exercises("development", per_skill=12, seed=seed + 11)
    exam = make_exercises("exam", per_skill=16, seed=seed + 97)
    all_rows = rows_train + dev + exam
    if len({row.prompt for row in all_rows}) != len(all_rows):
        raise ValueError("Train/development/exam exercises overlap")
    encoded_train = encode_exercises(rows_train, codec,
                                     original.cfg.max_seq_len)
    for rows in (dev, exam):
        encode_exercises(rows, codec, original.cfg.max_seq_len)
    train_tokens = torch.tensor(
        codec.encode((folder / "train.txt").read_text()).ids, dtype=torch.long)
    valid_tokens = torch.tensor(
        codec.encode((folder / "valid.txt").read_text()).ids, dtype=torch.long)
    staged_order, interleaved_order = schedules(
        rows_train, seed=seed, updates_per_level=updates_per_level)
    labels = null_labels(rows_train, seed=seed + 71)
    corrupted_train = [(encoded_train[i][0], 0 if labels[i] == "A" else 1)
                       for i in range(len(rows_train))]
    frozen_dev = score(original, dev, codec, answer_ids)
    frozen_exam = score(original, exam, codec, answer_ids)
    frozen_text_loss = comparable_story_loss(original, valid_tokens)
    sequences = {
        "english_class": staged_order,
        "same_lessons_interleaved": interleaved_order,
        "corrupted_answers": staged_order,
    }
    series = {}
    started = time.monotonic()
    for arm, model in models.items():
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        # The same public-text rehearsal minibatches are sampled in
        # exactly the same order by all arms.
        lang_rng = torch.Generator().manual_seed(seed + 119)
        curve = []
        best = None
        best_key = (-1, -1, -1)
        material = corrupted_train if arm == "corrupted_answers" else encoded_train
        for n, indices in enumerate(sequences[arm], 1):
            model.train()
            category_loss = answer_loss(
                model, [material[k] for k in indices], answer_ids)
            x, y = make_batch(train_tokens, 2, 64, lang_rng)
            _, language_loss = model(x, y)
            combined = ((1-rehearsal_weight)*category_loss
                        + rehearsal_weight*language_loss)
            optimizer.zero_grad(set_to_none=True)
            combined.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            if n == 1 or n % updates_per_level == 0:
                dev_score = score(model, dev, codec, answer_ids)
                worst = min(v["correct"] for v in dev_score["by_skill"].values())
                chosen = (worst, dev_score["correct"], n)
                if chosen > best_key:
                    best_key = chosen
                    best = {"step": n, "weights": copy.deepcopy(model.state_dict())}
                snapshot = {
                    "update": n,
                    "learning_phase": min(len(SKILLS), 1 + (n - 1) // updates_per_level)
                    if arm != "same_lessons_interleaved" else "mixed",
                    "dev": dev_score,
                    "quiz_loss": round(float(category_loss.detach()), 5),
                    "public_story_rehearsal_loss": round(float(language_loss.detach()), 5),
                }
                curve.append(snapshot)
                print(json.dumps({"arm": arm, "update": n,
                                  "dev": dev_score["accuracy"],
                                  "worst_correct": worst}), flush=True)
        if best is None:
            raise AssertionError("No development-selected model")
        last_weights = copy.deepcopy(model.state_dict())
        final_optimizer = optimizer.state_dict()
        model.load_state_dict(best["weights"])
        final = score(model, exam, codec, answer_ids)
        final_dev = score(model, dev, codec, answer_ids)
        text_loss = comparable_story_loss(model, valid_tokens)
        out.mkdir(parents=True, exist_ok=True)
        torch.save({
            "schema": 1, "research_only": True,
            "arm": arm, "configuration": asdict(model.cfg),
            "source_checkpoint_sha256": original_sha,
            "source_tokenizer_sha256": checkpoint["tokenizer_sha256"],
            "best_dev_step": best["step"],
            "best_weights": best["weights"],
            "last_weights": last_weights,
            "optimizer_last_state": final_optimizer,
            "public_rehearsal_rng_state": lang_rng.get_state(),
            "torch_rng_state": torch.get_rng_state(),
        }, out / f"{arm}-research.pt")
        series[arm] = {
            "best_development_step": best["step"],
            "development": final_dev,
            "sealed_synthetic_exam": final,
            "heldout_story_token_loss": text_loss,
            "retention_ratio_vs_frozen": round(text_loss/frozen_text_loss, 4),
            "history": curve,
        }
    report = {
        "schema": 1, "status": "research_only_unpromoted",
        "source_run_id": 41572257363,
        "source_checkpoint_sha256": original_sha,
        "source_tokenizer_sha256": checkpoint["tokenizer_sha256"],
        "source_corpus_revision": manifest["revision"],
        "train_sha256": manifest["train_sha256"],
        "valid_sha256": manifest["valid_sha256"],
        "exercise_counts": {"training": len(rows_train),
                            "development": len(dev), "exam": len(exam)},
        "exercise_sha256": {
            part: hashlib.sha256("\n".join(r.prompt for r in rows).encode()).hexdigest()
            for part, rows in (("train", rows_train), ("dev", dev), ("exam", exam))
        },
        "skills": list(SKILLS),
        "train_lexicon_disjoint_from_dev_and_exam": True,
        "answer_label_balance_checked": True,
        "matched_sample_multiset_across_arms": True,
        "rehearsal_weight": rehearsal_weight,
        "total_updates_per_arm": len(staged_order),
        "seed": seed,
        "duration_seconds": round(time.monotonic()-started, 2),
        "frozen": {"development": frozen_dev,
                   "synthetic_exam": frozen_exam,
                   "heldout_story_token_loss": frozen_text_loss},
        "arms": series,
        "model_promoted": False,
        "limitations": [
            "Five or six lesson skills do not constitute full English literacy",
            "Alphabet and vowel categories are known; only contexts/wordings transfer",
            "BPE forced-choice A/B tests are not open-ended spelling or conversation",
            "Unseen grammar words were drawn from manually specified synthetic lexica",
            "Training, dev and exam share rule families despite distinct surface phrasings",
            "Original v0.7 story data is used for public-only loss-preserving rehearsal",
            "Single train seed and a small internally authored exam; no external validation",
            "No pretrained language teacher weights, user memory or Android code used",
            "Research checkpoints MUST NOT be promoted without stronger independent tests",
        ],
    }
    (out / "english-class-v1.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "duration_seconds": report["duration_seconds"],
        "frozen": frozen_exam["accuracy"],
        "final": {name: {
            "exam": item["sealed_synthetic_exam"]["accuracy"],
            "dev": item["development"]["accuracy"],
            "retention": item["retention_ratio_vs_frozen"],
        } for name, item in series.items()},
    }, indent=2), flush=True)
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("results/english-class-v1"))
    p.add_argument("--seed", type=int, default=7331)
    p.add_argument("--updates-per-level", type=int, default=120)
    args = p.parse_args()
    experiment(folder=args.checkpoint_dir, out=args.out,
               seed=args.seed, updates_per_level=args.updates_per_level)


if __name__ == "__main__":
    main()
