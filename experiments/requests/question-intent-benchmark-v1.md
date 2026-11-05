# One-time benchmark generation and integrity gate

Generate and archive Forge question-intent benchmark v1 using only the hand-authored source in src/forge_core/neural/question_intent_benchmark.py.

Run the deterministic size, label balance, wording-family, object-disjointness, and exact-question-overlap checks. Preserve the manifest and split hashes. Do not train a model or use the final split for optimization. The final split remains a research-only internal synthetic benchmark, not an external certification.
