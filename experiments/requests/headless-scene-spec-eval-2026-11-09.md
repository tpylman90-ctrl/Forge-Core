# Headless scene-spec evaluation request

Run the constrained, evaluation-only Blender/Godot structural scene benchmark on the archived Blender/Godot specialist parent and selected checkpoints from run 42541109934. This is research only. Do not train or promote weights.


Retry after harness corrections: record engine versions, use instruction-based prompts, and retain nested result files in the comparison bundle.

Retry on 2026-11-11 after GitHub's Nov 9 Actions and Checks API incident was resolved. Use current corrected harness and collect the complete results.

Evaluator fix: generation uses the checkpoint's tokenizer IDs and decode path, with tokenizer SHA and vocabulary compatibility checks. This is required because specialist model IDs are not raw bytes.