# ChatGPT adaptive tutor v2

The v2 CPU lane turns the Ask-Why card into a repeatable tutoring exchange:

1. Forge makes a plausible first attempt, including a common mistaken assumption.
2. ChatGPT asks one diagnostic question before explaining.
3. Forge names the mistaken assumption.
4. ChatGPT gives a short explanation, a concrete analogy, and the governing principle.
5. Forge teaches the idea back, retrieves it without the explanation, then solves a changed case.
6. Forge names an observable check and states when the answer depends on current tool/version details.

This matches Forge Core's learning loop: observe, ask why, understand, state a principle, apply, check, and preserve uncertainty. The packs are authored examples from ChatGPT for this Forge Core pilot, released CC0-1.0. Blender and Godot examples tell Forge to verify APIs against the project's installed version.

## Experimental setup

The six training jobs run two seeds per lane: language (43, 44), Blender (731, 732), and Godot (947, 948). Every job restores and verifies the same pinned 107,250-step parent. It records the parent, candidate, lesson data, hashes, and metrics in a 90-day Actions artifact. There is no cross-lane weight sharing or automatic promotion.

Each candidate is scored on held-out tutor-v2, direct-teaching, and Ask-Why v1 text, plus existing lane and shared-language retention checks. This is still next-token text prediction. It does not prove Forge can ask a question on its own, execute Blender/Godot tasks, or carry on a live conversation with ChatGPT. Practical behavior and independent task grading are required before treating a candidate as a capability gain.

## Tutor design choices

- Diagnose the learner's assumption before giving the fix.
- Use one concrete analogy, then state a reusable rule.
- Fade the worked example and ask Forge to retrieve the rule without hints.
- Change the surface details in a transfer problem so copying the example is insufficient.
- Name an observable verification step and distinguish evidence from a plausible explanation.
- Keep all training branches reproducible and isolated; preserve failed candidates for analysis while continuing from the last passing parent.
