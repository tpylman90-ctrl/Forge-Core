"""Predeclared, nonpersonal semantic-role learning fixtures.

Training, development, and FINAL splits use disjoint names, objects and
sentence-template families. Every example has a counterfactual name swap and
both giver/recipient questions. The gold oracle parses the story independently.
"""
from __future__ import annotations

from dataclasses import dataclass
import random
import re

TRAIN_NAMES = ("Kalen", "Brynna", "Esmir", "Heston", "Vivia", "Corin",
               "Livia", "Tomas", "Mirel", "Perrin", "Zara", "Riva",
               "Jarek", "Ysola", "Faron", "Tamika")
DEV_NAMES = ("Rumi", "Lucia", "Tarek", "Edda", "Simeon", "Farah",
             "Ilona", "Marek", "Dina", "Wesley")
EXAM_NAMES = ("Amaya", "Kito", "Beatrix", "Liora", "Sibel", "Jareth",
              "Osmar", "Adele", "Tamara", "Cedric", "Enzo", "Navia")
TRAIN_OBJECTS = ("parcel", "pennant", "vase", "map", "journal",
                 "coin", "viola", "trophy", "beacon", "medal")
DEV_OBJECTS = ("anchor", "helmet", "banner", "crown", "sketchbook")
EXAM_OBJECTS = ("flute", "thermos", "globe", "drone", "scepter", "teapot")

# Not the prior diagnostic's "handed/received/passive/sent" source templates.
SOURCES = {
    "train": (
        ("offer", "at noon, {giver} offered the {obj} to {recipient}.",
         r"at noon, (?P<giver>[A-Z][a-z]+) offered the (?P<obj>[a-z]+) to (?P<recipient>[A-Z][a-z]+)\."),
        ("accept", "after recess, {recipient} accepted the {obj} from {giver}.",
         r"after recess, (?P<recipient>[A-Z][a-z]+) accepted the (?P<obj>[a-z]+) from (?P<giver>[A-Z][a-z]+)\."),
        ("lend", "in the workshop, {giver} lent {recipient} the {obj}.",
         r"in the workshop, (?P<giver>[A-Z][a-z]+) lent (?P<recipient>[A-Z][a-z]+) the (?P<obj>[a-z]+)\."),
        ("borrow", "by dusk, {recipient} borrowed the {obj} from {giver}.",
         r"by dusk, (?P<recipient>[A-Z][a-z]+) borrowed the (?P<obj>[a-z]+) from (?P<giver>[A-Z][a-z]+)\."),
    ),
    "dev": (
        ("deliver", "near the garden, {giver} delivered a {obj} to {recipient}.",
         r"near the garden, (?P<giver>[A-Z][a-z]+) delivered a (?P<obj>[a-z]+) to (?P<recipient>[A-Z][a-z]+)\."),
        ("collect", "before supper, {recipient} collected the {obj} from {giver}.",
         r"before supper, (?P<recipient>[A-Z][a-z]+) collected the (?P<obj>[a-z]+) from (?P<giver>[A-Z][a-z]+)\."),
    ),
    "exam": (
        ("entrust", "at the dock, {giver} entrusted the {obj} to {recipient}.",
         r"at the dock, (?P<giver>[A-Z][a-z]+) entrusted the (?P<obj>[a-z]+) to (?P<recipient>[A-Z][a-z]+)\."),
        ("courtesy", "while traveling, the {obj} reached {recipient} courtesy of {giver}.",
         r"while traveling, the (?P<obj>[a-z]+) reached (?P<recipient>[A-Z][a-z]+) courtesy of (?P<giver>[A-Z][a-z]+)\."),
    ),
}
QUESTION_VARIANTS = {
    "train": {
        "giver": ("Who gave the {obj}?", "Who was the giver of the {obj}?",
                  "Which person supplied the {obj}?"),
        "recipient": ("Who received the {obj}?", "Who was the recipient of the {obj}?",
                      "Which person took the {obj}?"),
    },
    "dev": {
        "giver": ("Who passed the {obj} along?",),
        "recipient": ("Who acquired the {obj}?",),
    },
    "exam": {
        "giver": ("Which name belongs to the sender of the {obj}?",),
        "recipient": ("Which name belongs to the receiver of the {obj}?",),
    },
}
NAMES = {"train": TRAIN_NAMES, "dev": DEV_NAMES, "exam": EXAM_NAMES}
OBJECTS = {"train": TRAIN_OBJECTS, "dev": DEV_OBJECTS, "exam": EXAM_OBJECTS}
SEEDS = {"train": (7711, 7712), "dev": (8811,), "exam": (9911, 9912, 9913)}
TRIALS = {"train": 10, "dev": 10, "exam": 12}
PROPER = re.compile(r"\b[A-Z][a-z]{2,}\b")


@dataclass(frozen=True)
class RoleRow:
    id: str
    split: str
    template: str
    story: str
    prompt: str
    answer: str
    role: str
    giver: str
    recipient: str
    object_name: str
    seed: int
    trial: int
    swap: int


def source_roles(template: str, story: str) -> tuple[str, str, str]:
    """Independent anchored parse, not dependent on the generated answer."""
    for alternatives in SOURCES.values():
        for name, _, pattern in alternatives:
            if name == template:
                found = re.fullmatch(pattern, story)
                if found is None:
                    raise ValueError("Story fails independent gold grammar")
                data = found.groupdict()
                return data["giver"], data["recipient"], data["obj"]
    raise ValueError("Unknown source template")


def candidate_spans(prompt: str) -> list[tuple[str, int, int]]:
    """Only capitalized entity-like words in the STORY, never gold labels.

    This is an intentionally constrained named-entity detector and must not
    be mistaken for learned language-wide NER.
    """
    if not prompt.startswith("Story: ") or "\nQuestion: " not in prompt:
        raise ValueError("Unknown source/question layout")
    story = prompt[7:prompt.index("\nQuestion: ")]
    spans = [(m.group(), m.start() + 7, m.end() + 7)
             for m in PROPER.finditer(story)]
    if len(spans) != 2 or spans[0][0] == spans[1][0]:
        raise ValueError("Expected exactly two independently detected names")
    return spans


def make_rows(split: str) -> list[RoleRow]:
    if split not in SOURCES:
        raise ValueError("Unknown curriculum split")
    output: list[RoleRow] = []
    seen_prompts: set[str] = set()
    for seed in SEEDS[split]:
        rng = random.Random(seed)
        for name, format_text, _ in SOURCES[split]:
            for trial in range(TRIALS[split]):
                # Reject overlapping *source questions*, not just duplicated IDs.
                # Resampling keeps session/seed identifiers OUT of model inputs
                # and preserves paired questions and counterfactual swaps.
                for attempt in range(128):
                    left, right = rng.sample(NAMES[split], 2)
                    obj = rng.choice(OBJECTS[split])
                    proposed: list[RoleRow] = []
                    for swap in (0, 1):
                        giver, recipient = ((left, right) if swap == 0
                                            else (right, left))
                        story = format_text.format(giver=giver,
                                                   recipient=recipient, obj=obj)
                        for role in ("giver", "recipient"):
                            variants = QUESTION_VARIANTS[split][role]
                            question = variants[(trial + swap) % len(variants)].format(obj=obj)
                            prompt = f"Story: {story}\\nQuestion: {question}\\nAnswer:"
                            proposed.append(RoleRow(
                                id=f"{split}:{seed}:{name}:{trial}:{swap}:{role}",
                                split=split, template=name, story=story,
                                prompt=prompt, answer=giver if role == "giver" else recipient,
                                role=role, giver=giver, recipient=recipient,
                                object_name=obj, seed=seed, trial=trial, swap=swap,
                            ))
                    phrases = [r.prompt for r in proposed]
                    if len(set(phrases)) == len(phrases) and not (set(phrases) & seen_prompts):
                        output.extend(proposed)
                        seen_prompts.update(phrases)
                        break
                else:
                    raise ValueError("Unable to generate unique paired role questions")
    verify_rows(output)
    return output


def verify_disjoint() -> None:
    from forge_core.neural.preschool_copy import (
        TRAIN_PEOPLE, EVAL_PEOPLE, TRAIN_WORDS, EVAL_WORDS)
    from forge_core.neural.pointer_transfer import EXAM_PEOPLE, EXAM_WORDS
    from forge_core.neural.role_binding_diagnostic import PEOPLE as PRIOR_PEOPLE, OBJECTS as PRIOR_OBJECTS
    prior_names = set(TRAIN_PEOPLE) | set(EVAL_PEOPLE) | set(EXAM_PEOPLE) | set(PRIOR_PEOPLE)
    prior_words = set(TRAIN_WORDS) | set(EVAL_WORDS) | set(EXAM_WORDS) | set(PRIOR_OBJECTS)
    for split in NAMES:
        if len(NAMES[split]) != len(set(NAMES[split])) or len(OBJECTS[split]) != len(set(OBJECTS[split])):
            raise ValueError("Duplicate names or objects in split")
        if set(NAMES[split]) & prior_names or set(OBJECTS[split]) & prior_words:
            raise ValueError("Previous experiment vocabulary leaked into new experiment")
        for other in NAMES:
            if split != other and (set(NAMES[split]) & set(NAMES[other])
                                   or set(OBJECTS[split]) & set(OBJECTS[other])):
                raise ValueError("Role training and held-out vocabulary overlap")
    templates = [key for group in SOURCES.values() for key, _, _ in group]
    if len(templates) != len(set(templates)):
        raise ValueError("Overlapping role grammatical templates")


def verify_rows(rows: list[RoleRow]) -> None:
    if not rows or len(set(row.id for row in rows)) != len(rows):
        raise ValueError("Missing or duplicate case IDs")
    if len(set(row.prompt for row in rows)) != len(rows):
        raise ValueError("Overlapping role question prompts")
    for row in rows:
        if row.split not in SOURCES or row.role not in ("giver", "recipient"):
            raise ValueError("Unrecognized role question")
        if row.seed not in SEEDS[row.split] or row.swap not in (0, 1):
            raise ValueError("Unregistered seed or swap")
        giver, recipient, obj = source_roles(row.template, row.story)
        if (giver != row.giver or recipient != row.recipient
                or obj != row.object_name or giver == recipient):
            raise ValueError("Source and independently parsed roles differ")
        should_answer = giver if row.role == "giver" else recipient
        if row.answer != should_answer:
            raise ValueError("Gold answer differs from independently verified truth")
        if row.prompt != ("Story: " + row.story + "\nQuestion: " +
                          row.prompt.split("\nQuestion: ", 1)[-1]):
            raise ValueError("Prompt has unexpected layout")
        question = row.prompt.split("\nQuestion: ", 1)[1]
        legal = [pattern.format(obj=obj) + "\nAnswer:"
                 for pattern in QUESTION_VARIANTS[row.split][row.role]]
        if question not in legal:
            raise ValueError("Question does not match the asserted semantic role")
        spans = candidate_spans(row.prompt)
        if {x[0] for x in spans} != {giver, recipient}:
            raise ValueError("Detected entity slots differ from independently parsed story")


def pairing_key(row: RoleRow) -> tuple[str, int, str, int]:
    return row.split, row.seed, row.template, row.trial
