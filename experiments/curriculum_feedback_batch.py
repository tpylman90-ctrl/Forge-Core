"""Matched answer-only, one-shot structured, and staged structured feedback trials."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import statistics
from pathlib import Path

import torch
from torch import nn

from forge_core.neural.model import ByteTransformer, ModelConfig
from forge_core.neural.preschool_qa import verify_source
from forge_core.neural.train import make_batch

ERRORS = ("missing", "extra", "order", "spacing", "wrong_letter")
TASKS = ("join", "spaced")
CONSONANTS = "bcdfghjklmnpqrstvwxyz"
VOWELS = "aeiou"
RUNS = 5
TRAIN_SEQUENCES_PER_RUN = 60
EXAM_SEQUENCES = 10
UPDATES = 320


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sequence_pool() -> list[str]:
    return [a + b + c for a in CONSONANTS for b in VOWELS
            for c in CONSONANTS if c != a]


def corrupt(source: str, task: str, error: str, rng: random.Random) -> tuple[str, str]:
    gold = source if task == "join" else " ".join(source)
    extra = next(letter for letter in "abcdefghijklmnopqrstuvwxyz" if letter not in source)
    if error == "missing":
        pos = rng.randrange(3)
        compact = source[:pos] + source[pos + 1:]
        attempt = compact if task == "join" else " ".join(compact)
    elif error == "extra":
        compact = source + extra
        attempt = compact if task == "join" else " ".join(compact)
    elif error == "order":
        compact = source[::-1]
        attempt = compact if task == "join" else " ".join(compact)
    elif error == "spacing":
        attempt = " ".join(source) if task == "join" else source
    elif error == "wrong_letter":
        pos = rng.randrange(3)
        substitute = next(letter for letter in "abcdefghijklmnopqrstuvwxyz"
                          if letter not in source)
        compact = source[:pos] + substitute + source[pos + 1:]
        attempt = compact if task == "join" else " ".join(compact)
    else:
        raise ValueError("Unknown error category")
    compact_attempt = "".join(attempt.split())
    valid = {
        "missing": len(compact_attempt) == 2,
        "extra": len(compact_attempt) == 4,
        "order": sorted(compact_attempt) == sorted(source) and compact_attempt != source,
        "spacing": compact_attempt == source and attempt != gold,
        "wrong_letter": (len(compact_attempt) == 3 and
                         sorted(compact_attempt) != sorted(source) and
                         sum(x != y for x, y in zip(compact_attempt, source)) == 1),
    }[error]
    if not valid:
        raise AssertionError(f"Generated attempt does not verify as {error}: {source} / {attempt}")
    return attempt, gold


def build_examples(sequences: list[str], seed: int) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for source in sequences:
        for task in TASKS:
            for error in ERRORS:
                attempt, gold = corrupt(source, task, error, rng)
                out.append({"source": source, "task": task, "error": error,
                            "attempt": attempt, "gold": gold})
    return out


def instruction(task: str) -> str:
    return ("join all letters in order without spaces" if task == "join"
            else "copy all letters in order with one space between letters")


def structured_fields(row: dict) -> dict[str, str]:
    source, attempt = row["source"], row["attempt"]
    compact = "".join(attempt.split())
    error, task = row["error"], row["task"]
    attempt_value = attempt.replace(" ", "_")
    answer_value = row["gold"].replace(" ", "_")
    if error == "missing":
        index = next(i for i, ch in enumerate(source) if ch not in compact)
        position, found, expected, repair = str(index + 1), "none", source[index], "insert"
    elif error == "extra":
        letter = next(ch for ch in compact if compact.count(ch) > source.count(ch))
        position, found, expected, repair = str(compact.index(letter) + 1), letter, "none", "remove"
    elif error == "order":
        index = next(i for i, (got, want) in enumerate(zip(compact, source)) if got != want)
        position, found, expected, repair = str(index + 1), compact[index], source[index], "reorder"
    elif error == "spacing" and task == "join":
        position, found, expected, repair = "all", attempt_value, answer_value, "remove_spaces"
    elif error == "spacing":
        position, found, expected, repair = "all", attempt_value, answer_value, "add_spaces"
    else:
        index = next(i for i, (got, want) in enumerate(zip(compact, source)) if got != want)
        position, found, expected, repair = str(index + 1), compact[index], source[index], "replace"
    return {"answer": answer_value, "error": error, "position": position,
            "found": found, "expected": expected, "repair": repair}


def target(row: dict, method: str, phase: str = "full") -> str:
    if method == "answer_only":
        return f"Correct answer: {row['gold']}.\n"
    fields = structured_fields(row)
    if method == "staged_feedback" and phase == "diagnosis":
        fields = {key: fields[key] for key in ("answer", "error", "position", "found", "expected")}
    return "; ".join(f"{key}={value}" for key, value in fields.items()) + "\n"


def prompt(row: dict, train: bool) -> str:
    if train:
        lead = "Feedback lesson"
        source_label, task_label, attempt_label = "Source letters", "Task", "Proposed answer"
        request = "This response is incorrect. Give helpful feedback and a correction."
    else:
        lead = "Review this attempt"
        source_label, task_label, attempt_label = "Letters given", "Required format", "Attempt"
        request = "The attempt is wrong. Give helpful feedback and a correction."
    return (f"{lead}\n{source_label}: {' '.join(row['source'])}\n"
            f"{task_label}: {instruction(row['task'])}.\n"
            f"{attempt_label}: {row['attempt']}\n{request}\nResponse:")


def verify_example(row: dict) -> None:
    expected = row["source"] if row["task"] == "join" else " ".join(row["source"])
    _, independently_derived = corrupt(row["source"], row["task"], row["error"],
                                         random.Random(19))
    if expected != row["gold"] or independently_derived != row["gold"]:
        raise AssertionError("Training label disagrees with deterministic verifier")
    compact = "".join(row["attempt"].split())
    checks = {
        "missing": len(compact) < len(row["source"]),
        "extra": len(compact) > len(row["source"]),
        "order": sorted(compact) == sorted(row["source"]) and compact != row["source"],
        "spacing": compact == row["source"] and row["attempt"] != row["gold"],
        "wrong_letter": len(compact) == len(row["source"]) and
                        sorted(compact) != sorted(row["source"]),
    }
    if not checks[row["error"]]:
        raise AssertionError("Attempt does not satisfy its claimed error class")


def prepare(source_dir: Path, out: Path) -> dict:
    torch.set_num_threads(min(2, torch.get_num_threads()))
    ckpt, manifest, codec = verify_source(source_dir)
    pool = sequence_pool()
    if len(pool) < RUNS * TRAIN_SEQUENCES_PER_RUN + EXAM_SEQUENCES:
        raise AssertionError("Insufficient unique source sequences")
    random.Random(90817).shuffle(pool)
    group_sets = [pool[i * TRAIN_SEQUENCES_PER_RUN:(i + 1) * TRAIN_SEQUENCES_PER_RUN]
                  for i in range(RUNS)]
    exam_sequences = pool[RUNS * TRAIN_SEQUENCES_PER_RUN:
                          RUNS * TRAIN_SEQUENCES_PER_RUN + EXAM_SEQUENCES]
    if len(set(exam_sequences).intersection(*[set(x) for x in group_sets])):
        raise AssertionError("Exam overlaps training")
    exam = build_examples(exam_sequences, 90819)
    datasets = [build_examples(group, 90821 + i) for i, group in enumerate(group_sets)]
    for group in datasets + [exam]:
        for row in group:
            verify_example(row)
    # Distinct groups and the exam are pairwise disjoint at source-sequence level.
    all_groups = group_sets + [exam_sequences]
    for i, left in enumerate(all_groups):
        for right in all_groups[i + 1:]:
            if set(left) & set(right):
                raise AssertionError("Dataset partitions overlap")

    model = ByteTransformer(ModelConfig(**ckpt["configuration"]))
    model.load_state_dict(ckpt["model"], strict=True)
    exam_metrics = score(model, codec, exam)
    train_ids = torch.tensor(codec.encode((source_dir / "train.txt").read_text()).ids,
                             dtype=torch.long)
    valid_ids = torch.tensor(codec.encode((source_dir / "valid.txt").read_text()).ids,
                             dtype=torch.long)
    before_loss = story_loss(model, valid_ids)

    out.mkdir(parents=True, exist_ok=True)
    (out / "checkpoint.pt").write_bytes((source_dir / "checkpoint.pt").read_bytes())
    for name in ("source-manifest.json", "tokenizer.json", "train.txt", "valid.txt"):
        (out / name).write_bytes((source_dir / name).read_bytes())
    for i, rows in enumerate(datasets, 1):
        save_json(out / f"train-{i:02d}.json", {"examples": rows})
    save_json(out / "exam.json", {"examples": exam})
    result = {
        "schema": 1, "run_count": RUNS, "updates_per_run": UPDATES,
        "methods": ["answer_only", "full_structured", "staged_feedback"],
        "datasets_per_method": RUNS,
        "training_examples_per_dataset": len(datasets[0]),
        "heldout_examples": len(exam),
        "training_sequence_count_each": TRAIN_SEQUENCES_PER_RUN,
        "heldout_sequence_count": EXAM_SEQUENCES,
        "dataset_pairs_disjoint": True,
        "shared_exam_sequence_sha256": digest("\n".join(exam_sequences).encode()),
        "train_sequence_sha256": {
            str(i + 1): digest("\n".join(group).encode()) for i, group in enumerate(group_sets)},
        "base_checkpoint_sha256": digest((source_dir / "checkpoint.pt").read_bytes()),
        "source_train_sha256": manifest["train_sha256"],
        "source_valid_sha256": manifest["valid_sha256"],
        "source_tokenizer_sha256": ckpt["tokenizer_sha256"],
        "starting_checkpoint_step": ckpt["step"],
        "parameter_count": model.num_parameters,
        "baseline": {"exam": exam_metrics, "story_loss": before_loss},
        "research_only": True,
        "limitations": [
            "Synthetic three-letter copying tasks; this is not a general-reasoning benchmark.",
            "Ten held-out source strings create 100 correlated task/error cases.",
            "Structured fields use exact deterministic labels and are scored independently.",
            "Fresh lineage from the public Forge Seed checkpoint; not a continuation of lost Session 4 weights.",
        ],
    }
    save_json(out / "manifest.json", result)
    return result


def encode_rows(rows: list[dict], method: str, codec, cfg, phase: str = "full"):
    batches = []
    for row in rows:
        p = codec.encode(prompt(row, True)).ids
        y = codec.encode(" " + target(row, method, phase)).ids
        if not p or not y or len(p) + len(y) - 1 > cfg.max_seq_len:
            raise ValueError("Training example exceeds model context")
        batches.append((p + y[:-1], [-100] * (len(p) - 1) + y))
    return batches


def task_loss(model, batch) -> torch.Tensor:
    length = max(len(x) for x, _ in batch)
    x = torch.zeros((len(batch), length), dtype=torch.long)
    y = torch.full((len(batch), length), -100, dtype=torch.long)
    for i, (tokens, labels) in enumerate(batch):
        x[i, :len(tokens)] = torch.tensor(tokens, dtype=torch.long)
        y[i, :len(labels)] = torch.tensor(labels, dtype=torch.long)
    logits, _ = model(x)
    return nn.functional.cross_entropy(logits.reshape(-1, model.cfg.vocab_size),
                                       y.reshape(-1), ignore_index=-100)


@torch.no_grad()
def generate(model, codec, text: str, max_tokens: int = 160) -> str:
    state = codec.encode(text).ids
    generated = []
    for _ in range(max_tokens):
        x = torch.tensor([state[-model.cfg.max_seq_len:]], dtype=torch.long)
        logits, _ = model(x)
        token = int(logits[0, -1].argmax())
        state.append(token)
        generated.append(token)
        decoded = codec.decode(generated)
        if "\n" in decoded:
            return decoded.split("\n", 1)[0]
    return codec.decode(generated)


def score(model, codec, rows: list[dict]) -> dict:
    metrics = ("answer", "error", "position", "found", "expected", "repair", "diagnosis", "record")
    totals = {m: 0 for m in metrics}
    by_error = {e: {**{m: 0 for m in metrics}, "total": 0} for e in ERRORS}
    by_task = {t: {**{m: 0 for m in metrics}, "total": 0} for t in TASKS}
    examples = []
    model.eval()
    for row in rows:
        generated = generate(model, codec, prompt(row, False))
        answer_match = re.search(r"Correct answer:\s*([a-z ]+?)(?:\.|$)", generated, re.I)
        answer = " ".join(answer_match.group(1).casefold().split()) if answer_match else ""
        parsed = {k.casefold(): v.strip().casefold() for k, v in re.findall(
            r"(?:^|;\s*)(error|position|found|expected|repair|answer)=([^;\n]*)",
            generated.strip(), re.I)}
        if not answer and "answer" in parsed:
            answer = parsed["answer"].replace("_", " ")
        expected = structured_fields(row)
        values = {key: parsed.get(key, "").replace("_", " ") == expected[key].replace("_", " ")
                  for key in ("error", "position", "found", "expected", "repair")}
        values["answer"] = answer == row["gold"]
        values["diagnosis"] = all(values[key] for key in ("answer", "error", "position", "found", "expected"))
        values["record"] = values["diagnosis"] and values["repair"]
        for key, value in values.items():
            totals[key] += int(value)
            by_error[row["error"]][key] += int(value)
            by_task[row["task"]][key] += int(value)
        by_error[row["error"]]["total"] += 1
        by_task[row["task"]]["total"] += 1
        if len(examples) < 10:
            examples.append({"source": row["source"], "task": row["task"],
                             "error": row["error"], "generated": generated,
                             **{f"{key}_ok": value for key, value in values.items()}})
    n = len(rows)
    return {"total": n, **{k: {"correct": v, "rate": round(v / n, 4)}
                            for k, v in totals.items()},
            "by_error": by_error, "by_task": by_task, "examples": examples,
            "record_scoring": "fixed-schema fields exact-matched independently; diagnosis requires answer+error facts; full record also requires repair"}


@torch.no_grad()
def story_loss(model, tokens: torch.Tensor, windows: int = 32, seq_len: int = 128) -> float:
    gen = torch.Generator().manual_seed(61129)
    starts = torch.randint(0, tokens.numel() - seq_len - 1, (windows,), generator=gen)
    x = torch.stack([tokens[i:i + seq_len] for i in starts])
    y = torch.stack([tokens[i + 1:i + seq_len + 1] for i in starts])
    _, loss = model(x, y)
    return round(float(loss), 6)


def train_one(source_dir: Path, prepared_dir: Path, out: Path,
              dataset_id: int, method: str) -> dict:
    methods = ("answer_only", "full_structured", "staged_feedback")
    if dataset_id not in range(1, RUNS + 1) or method not in methods:
        raise ValueError("Invalid matrix parameters")
    torch.set_num_threads(min(2, torch.get_num_threads()))
    ckpt, _, codec = verify_source(source_dir)
    manifest = json.loads((prepared_dir / "manifest.json").read_text(encoding="utf-8"))
    rows = json.loads((prepared_dir / f"train-{dataset_id:02d}.json").read_text())['examples']
    exam = json.loads((prepared_dir / "exam.json").read_text())['examples']
    for row in rows + exam:
        verify_example(row)
    cfg = ModelConfig(**ckpt["configuration"])
    seed = 10000 + dataset_id
    torch.manual_seed(seed)
    model = ByteTransformer(cfg)
    model.load_state_dict(ckpt["model"], strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00012)
    encoded_full = encode_rows(rows, method, codec, cfg, "full")
    encoded_diagnosis = encode_rows(rows, method, codec, cfg, "diagnosis")
    py_rng = random.Random(202600 + dataset_id)
    text_rng = torch.Generator().manual_seed(207700 + dataset_id)
    train_tokens = torch.tensor(codec.encode((source_dir / "train.txt").read_text()).ids,
                                dtype=torch.long)
    valid_tokens = torch.tensor(codec.encode((source_dir / "valid.txt").read_text()).ids,
                                dtype=torch.long)
    curve = []
    model.train()
    for step in range(1, UPDATES + 1):
        encoded = (encoded_diagnosis if method == "staged_feedback" and step <= UPDATES // 2
                   else encoded_full)
        batch = [encoded[py_rng.randrange(len(encoded))] for _ in range(4)]
        x, y = make_batch(train_tokens, 2, 64, text_rng)
        _, language = model(x, y)
        supervised = task_loss(model, batch)
        loss = 0.8 * supervised + 0.2 * language
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step in (160, 320):
            model.eval()
            point = {"step": step, "heldout": score(model, codec, exam),
                     "story_loss": story_loss(model, valid_tokens)}
            curve.append(point)
            print(json.dumps({"data": dataset_id, "method": method, "step": step,
                              **{key: point["heldout"][key]["correct"] for key in
                                 ("answer", "error", "position", "found",
                                  "expected", "repair", "diagnosis", "record")},
                              "story_loss": point["story_loss"]}), flush=True)
            model.train()
    final = curve[-1]
    out.mkdir(parents=True, exist_ok=True)
    result = {"data_id": dataset_id, "method": method, "updates": UPDATES,
              "training_examples": len(rows), "heldout_examples": len(exam),
              "training_source_hash": manifest["train_sequence_sha256"][str(dataset_id)],
              "shared_exam_hash": manifest["shared_exam_sequence_sha256"],
              "base_checkpoint_hash": manifest["base_checkpoint_sha256"],
              "curve": curve, "final": final, "promoted": False,
              "research_only": True}
    save_json(out / f"run-{dataset_id:02d}-{method}.json", result)
    torch.save({"schema": 1, "research_only": True, "promoted": False,
                "data_id": dataset_id, "method": method, "updates": UPDATES,
                "base_checkpoint_sha256": manifest["base_checkpoint_sha256"],
                "training_source_sha256": result["training_source_hash"],
                "configuration": {k: getattr(cfg, k) for k in
                                  ("vocab_size", "max_seq_len", "n_layer", "n_head", "n_embd", "dropout")},
                "model": model.state_dict()},
               out / f"run-{dataset_id:02d}-{method}-step320.pt")
    return result


def aggregate(results_dir: Path, prepared_dir: Path) -> dict:
    results = [json.loads(p.read_text()) for p in sorted(results_dir.glob("run-*.json"))]
    methods = ("answer_only", "full_structured", "staged_feedback")
    expected = {(i, method) for i in range(1, RUNS + 1) for method in methods}
    got = {(r["data_id"], r["method"]) for r in results}
    if got != expected:
        raise ValueError(f"Expected all 15 run results; missing={sorted(expected-got)} extra={sorted(got-expected)}")
    metrics = ("answer", "error", "position", "found", "expected", "repair", "diagnosis", "record")
    summary = {}
    for method in methods:
        subset = [r for r in results if r["method"] == method]
        summary[method] = {}
        for step in (160, 320):
            arm = [next(x for x in r["curve"] if x["step"] == step) for r in subset]
            summary[method][str(step)] = {
                m: {"mean_correct": round(statistics.mean(x["heldout"][m]["correct"] for x in arm), 3),
                    "sd_correct": round(statistics.stdev(x["heldout"][m]["correct"] for x in arm), 3)}
                for m in metrics}
            summary[method][str(step)]["story_loss_mean"] = round(statistics.mean(x["story_loss"] for x in arm), 6)
            summary[method][str(step)]["story_loss_sd"] = round(statistics.stdev(x["story_loss"] for x in arm), 6)
    paired = {}
    for base, candidate in (("answer_only", "full_structured"),
                            ("answer_only", "staged_feedback"),
                            ("full_structured", "staged_feedback")):
        label = f"{candidate}_minus_{base}"
        paired[label] = {}
        for metric in metrics:
            deltas = []
            for data_id in range(1, RUNS + 1):
                a = next(r for r in results if r["data_id"] == data_id and r["method"] == base)
                b = next(r for r in results if r["data_id"] == data_id and r["method"] == candidate)
                deltas.append(b["final"]["heldout"][metric]["correct"] -
                              a["final"]["heldout"][metric]["correct"])
            paired[label][metric] = {"mean_delta": round(statistics.mean(deltas), 3),
                                     "sd_delta": round(statistics.stdev(deltas), 3),
                                     "candidate_wins": sum(d > 0 for d in deltas),
                                     "ties": sum(d == 0 for d in deltas),
                                     "base_wins": sum(d < 0 for d in deltas),
                                     "per_dataset_delta": deltas}
    subgroup = {}
    for group_key, labels in (("by_error", ERRORS), ("by_task", TASKS)):
        subgroup[group_key] = {}
        for group in labels:
            subgroup[group_key][group] = {}
            for metric in metrics:
                subgroup[group_key][group][metric] = {}
                for method in methods:
                    counts = [next(r for r in results if r["data_id"] == i and r["method"] == method)
                              ["final"]["heldout"][group_key][group][metric] for i in range(1, RUNS + 1)]
                    subgroup[group_key][group][metric][method] = round(statistics.mean(counts), 3)
    prep = json.loads((prepared_dir / "manifest.json").read_text())
    report = {"status": "research_only_no_promotion", "runs_completed": len(results),
              "paired_datasets": RUNS, "updates_per_run": UPDATES,
              "curriculum_transition_step": UPDATES // 2, "methods": list(methods),
              "summary": summary, "paired_comparisons": paired,
              "subgroup_comparison": subgroup,
              "dataset_design": {k: prep[k] for k in (
                  "training_examples_per_dataset", "heldout_examples",
                  "training_sequence_count_each", "heldout_sequence_count",
                  "dataset_pairs_disjoint", "shared_exam_sequence_sha256",
                  "train_sequence_sha256", "base_checkpoint_sha256")},
              "limitations": prep["limitations"] + [
                  "Five paired datasets provide a small robustness check, not a definitive population estimate.",
                  "Fixed-schema scoring tests field recovery, not free-form explanation quality."]}
    save_json(results_dir / "cross-examination.json", report)
    lines = ["# Forge Core — staged structured-feedback comparison", "",
             "Research only; no checkpoint was promoted.", "",
             "Mean ± standard deviation across five runs; held-out sample size is 100.", "",
             "| Method | Step | Exact answer / 100 | Diagnosis record / 100 | Full record / 100 | Story loss |",
             "|---|---:|---:|---:|---:|---:|"]
    for method in methods:
        for step in (160, 320):
            x = summary[method][str(step)]
            lines.append(f"| {method} | {step} | {x['answer']['mean_correct']:.1f} ± {x['answer']['sd_correct']:.1f} | "
                         f"{x['diagnosis']['mean_correct']:.1f} ± {x['diagnosis']['sd_correct']:.1f} | "
                         f"{x['record']['mean_correct']:.1f} ± {x['record']['sd_correct']:.1f} | "
                         f"{x['story_loss_mean']:.6f} ± {x['story_loss_sd']:.6f} |")
    lines += ["", "## Paired 320-update deltas", "",
              "| Comparison | Measure | Mean delta / 100 | Candidate wins | Ties | Base wins |",
              "|---|---|---:|---:|---:|---:|"]
    for comparison, values in paired.items():
        for metric in ("answer", "diagnosis", "record", "repair"):
            x = values[metric]
            lines.append(f"| {comparison} | {metric} | {x['mean_delta']:+.1f} | {x['candidate_wins']} | {x['ties']} | {x['base_wins']} |")
    lines += ["", "## By error type at update 320", "",
              "Mean exact cases per 20.", "",
              "| Error | Answer-only answer | Full structured diagnosis | Staged diagnosis | Full record | Staged full record |",
              "|---|---:|---:|---:|---:|---:|"]
    for group in ERRORS:
        x = subgroup["by_error"][group]
        lines.append(f"| {group} | {x['answer']['answer_only']:.1f} | {x['diagnosis']['full_structured']:.1f} | "
                     f"{x['diagnosis']['staged_feedback']:.1f} | {x['record']['full_structured']:.1f} | "
                     f"{x['record']['staged_feedback']:.1f} |")
    lines += ["", "All methods share the same public Forge Seed checkpoint, data partitions, update budget, optimizer, and sealed exam. "
              "The staged arm learns answer plus diagnosis fields in its first 160 updates, then adds repair for the final 160.", ""]
    (results_dir / "cross-examination.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("prepare", "train", "aggregate"))
    p.add_argument("--source-dir", type=Path, default=Path("original"))
    p.add_argument("--prepared-dir", type=Path, default=Path("prepared"))
    p.add_argument("--out", type=Path, default=Path("results"))
    p.add_argument("--data-id", type=int)
    p.add_argument("--method", choices=("answer_only", "full_structured", "staged_feedback"))
    a = p.parse_args()
    if a.mode == "prepare":
        result = prepare(a.source_dir, a.prepared_dir)
        print(json.dumps({"prepared": True, "base_hash": result["base_checkpoint_sha256"],
                          "exam": result["baseline"]["exam"]}, indent=2), flush=True)
    elif a.mode == "train":
        if a.data_id is None or a.method is None:
            p.error("train mode requires --data-id and --method")
        train_one(a.source_dir, a.prepared_dir, a.out, a.data_id, a.method)
    else:
        result = aggregate(a.out, a.prepared_dir)
        print(json.dumps({"completed": result["runs_completed"],
                          "comparison": result["paired_320_update_comparison"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
