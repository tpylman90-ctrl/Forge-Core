# ChatGPT Ask-Why lane v1

This adds an explicit teaching step to the ChatGPT channel:

1. Forge asks why a step is necessary or why a result occurred.
2. ChatGPT explains in a few concrete steps, uses a familiar analogy, and states what evidence would check the explanation.
3. Forge restates the principle, applies it to a new task, and identifies a verification step.

The workflow compares three matched arms—control, direct ChatGPT teaching, and Ask-Why teaching—for language, Blender, and Godot. It evaluates both held-out direct-lesson and Ask-Why lesson text, while retaining the existing lane and shared-language validation checks. Every branch starts from the same pinned parent for its lane/seed.

This tests whether the explain-and-apply format improves prediction of new examples in that format. It does not establish that the current network can autonomously ask useful questions or carry out the described tools. Candidate weights stay research-only; the lesson dataset and evidence must be reviewed before a separate task-level transfer test.
