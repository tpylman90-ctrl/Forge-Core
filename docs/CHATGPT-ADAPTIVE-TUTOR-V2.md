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

The six-job run uses language seeds (43, 44), Blender seeds (731, 732), and Godot seeds (947, 948), all restoring the same pinned 107,250-step parent. During review, I found the language trainer restored the parent's sampling RNG for both declared seeds, making language 43 and 44 identical; that pair is a reproducibility check, not an independent replication. A separate supplemental workflow now runs language seed 44 with an independent sample/RNG stream while keeping the parent weights and optimizer fixed. Blender and Godot use genuinely distinct lane seeds. Each run records parent, candidate, lesson data, hashes, and metrics in a 90-day Actions artifact. No cross-lane weight sharing or automatic promotion is allowed.

Each candidate is scored on held-out tutor-v2, direct-teaching, and Ask-Why v1 text, plus existing lane and shared-language retention checks. This is still next-token text prediction. It does not prove Forge can ask a question on its own, execute Blender/Godot tasks, or carry on a live conversation with ChatGPT. Practical behavior and independent task grading are required before treating a candidate as a capability gain.

## Tutor design choices

- Diagnose the learner's assumption before giving the fix.
- Use one concrete analogy, then state a reusable rule.
- Fade the worked example and ask Forge to retrieve the rule without hints.
- Change the surface details in a transfer problem so copying the example is insufficient.
- Name an observable verification step and distinguish evidence from a plausible explanation.
- Keep all training branches reproducible and isolated; preserve failed candidates for analysis while continuing from the last passing parent.


## Results

See [the completed v2 study](CHATGPT-ADAPTIVE-TUTOR-V2-RESULTS.md) for all six lane runs and the corrected independent language replication.


## Matched comparison

See [the matched control, direct-v2, and tutor-v2 study](CHATGPT-MATCHED-TUTOR-V2-RESULTS.md). It found clear format-specific text-learning gains, but the tutor arm did not lead the existing lane-test metric.
