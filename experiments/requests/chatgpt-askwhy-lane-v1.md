# ChatGPT Ask-Why lane v1 request

Run `.github/workflows/chatgpt-askwhy-lane-v1.yml` on main.

Compare control, direct ChatGPT lessons, and an explicit Ask-Why curriculum for language, Blender, and Godot. The Ask-Why step must show:
1. Forge asks why it should do or avoid a specific action.
2. ChatGPT explains in short concrete steps and a simple analogy.
3. Forge restates the principle, applies it to the task, and names a check.

Use the same pinned 107,250-step parent, seed, update budget, lane trainer, and retention checks across each lane's three arms. Evaluate every candidate on separate held-out direct and Ask-Why lesson packs as well as the trainer's lane and shared-language measures. Save parents, candidates, data, hashes, and metrics together as Actions artifacts.

This is a one-seed study. Do not promote, merge, or install weights from token-loss results alone. Keep the lanes isolated and use task-level evaluation plus independent seeds before durable checkpoint archival.

Retry note: use this request with the corrected GitHub expression delimiters in the workflow.
