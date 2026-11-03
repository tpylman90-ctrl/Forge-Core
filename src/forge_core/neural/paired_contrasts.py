"""Verified paired contrasts for a small scratch-trained language student.

Each semantic case produces two states in which the correct answer
changes and BOTH A/B choice orders for each state. The model cannot
succeed by always choosing a letter. Labels come from deterministic
rules, not the model's guesses. All data here is original and nonpersonal.
"""
from __future__ import annotations

import random
from forge_core.neural.preschool_qa import (
    Task, SKILLS, TRAIN_NAMES, TRAIN_OBJECTS, COLORS, PLACES, COUNTS,
    question_prompt,
)


def paired_tasks(question_count: int, seed: int = 410) -> list[Task]:
    if not 32 <= question_count <= 4096 or question_count % 32:
        raise ValueError("Number of questions must be a multiple of 32, from 32 to 4096")
    rng = random.Random(seed)
    results: list[Task] = []
    for i in range(question_count // 4):
        skill = SKILLS[i % 4]
        a, b = rng.sample(TRAIN_NAMES, 2)
        if skill == "color":
            c1, c2 = rng.sample(COLORS, 2)
            thing = rng.choice(TRAIN_OBJECTS)
            story = f"{a} has a {c1} {thing}. {b} has a {c2} {thing}."
            states = [
                (story, f"What color is {a}'s {thing}?", c1, c2),
                (story, f"What color is {b}'s {thing}?", c2, c1),
            ]
        elif skill == "recipient":
            thing = rng.choice(TRAIN_OBJECTS)
            states = [
                (f"{a} handed the {thing} to {b}.",
                 f"Who received the {thing}?", b, a),
                (f"{b} handed the {thing} to {a}.",
                 f"Who received the {thing}?", a, b),
            ]
        elif skill == "location":
            loc1, loc2 = rng.sample(PLACES, 2)
            thing = rng.choice(TRAIN_OBJECTS)
            states = [
                (f"{a} placed the {thing} {loc1} the table.",
                 f"Where did the {thing} go?", loc1, loc2),
                (f"{a} placed the {thing} {loc2} the table.",
                 f"Where did the {thing} go?", loc2, loc1),
            ]
        else:
            n1, n2 = rng.sample(COUNTS, 2)
            story = (f"{a} found {n1} small stones. "
                     f"{b} found {n2} small stones.")
            states = [
                (story, f"How many stones did {a} find?", n1, n2),
                (story, f"How many stones did {b} find?", n2, n1),
            ]
        for state, (story, question, right, wrong) in enumerate(states):
            for mirror in (0, 1):
                prompt, answer = question_prompt(story, question, right,
                                                 wrong, bool(mirror))
                results.append(Task(
                    f"paired:{i:04d}:state{state}:mirror{mirror}",
                    skill, prompt, answer))
    if len(results) != question_count:
        raise RuntimeError("Paired curriculum dimensions corrupted")
    return results
