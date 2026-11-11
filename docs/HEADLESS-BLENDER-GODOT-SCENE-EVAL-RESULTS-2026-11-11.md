# Headless Blender/Godot scene-spec evaluation — 2026-11-11

## Outcome

The evaluation-only benchmark completed successfully in [GitHub Actions run 43021468048](https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/43021468048). Its [draft release archive](https://github.com/tpylman90-ctrl/Forge-Core/releases/tag/untagged-9fb03899c49fc08c7abb) contains the reports, raw model outputs, archived checkpoints, and engine artifacts.

Each lane first built and inspected a fixed positive-control scene. The Blender and Godot positive controls both passed. Forge's parent and selected-candidate responses in both lanes failed to produce one parseable JSON object matching the small required schema. The evaluator therefore did not invoke either engine on generated output; it never executed model-generated code.

| Lane | Runtime | Parent checkpoint | Candidate checkpoint | Forge spec result |
|---|---|---|---|---|
| Blender | Blender 4.0.2 (Ubuntu runner package) | 95,000 base steps; 1,500 specialist updates; SHA-256 `672db59bfc44d026f26e38d781c84dd24f92882c65b888639ec66a790002ed68` | Identical to parent | 0/2 valid JSON specs |
| Godot | Godot 4.7.2 stable, official Linux build | 95,000 base steps; 1,500 specialist updates; SHA-256 `de7de5052f0a56d821ea3995f1f2e5fbb80d10750e4fb3a564f3c6c4a2a0afdf` | Identical to parent | 0/2 valid JSON specs |

Parent and selected candidate were byte-identical in each lane. Each was sampled with the same fixed prompt, seed, and decoding settings, so the repeated outputs are not independent evidence. Both known-good positive controls passed their engines' structural assertions.

## Interpretation and disposition

This tiny constrained-output check shows that these checkpoints did not yet produce the requested scene specification under this prompt and format. It does not assess broader Blender or Godot knowledge or competence. No training updates occurred, no checkpoint changed, and nothing was promoted to Android.

The first workflow attempts fell during GitHub's Nov. 9, 2026 Actions and Checks API incident (18:12–18:59 UTC). A clean retry after recovery started correctly. Follow-up code fixes aligned generation with the archived BPE tokenizer, added the missing runtime-version helper, and corrected artifact collection. The final workflow and archive both completed successfully.
