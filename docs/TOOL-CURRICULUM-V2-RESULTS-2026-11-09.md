# Blender/Godot isolated curriculum v2 — 2026-11-09

## What ran

Workflow [42541109934](https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/42541109934) prepared and trained two independent branches. Each lane started from its own verified 1,500-update specialist checkpoint descended from the Forge Seed root. No Blender or Godot weights were mixed with the language branch or with each other.

Each lane used eight original Forge-Core Task/Answer/Why lesson cards (CC0-1.0), its pinned versioned official manual prose (Blender 5.4; Godot 4.7), and a fresh eight-page manual holdout excluded from training and checkpoint selection. Parent and selected checkpoint were scored on the same fresh-document slice and expanded answer-keyed probe. The full parent/candidate checkpoints, optimizer state, tokenizer, source pages, data manifests, evaluations, and hashes are in the [draft research release](https://github.com/tpylman90-ctrl/Forge-Core/releases/tag/untagged-ce7e57566d16d7e69357).

## Results

| Lane | Requested updates | Updates completed before early stop | Parent → selected validation loss | Parent → selected test loss | Fresh document loss, parent → selected | Fresh probe, parent → selected | Selected checkpoint |
|---|---:|---:|---:|---:|---:|---:|---|
| Blender | 750 | 500 | 3.53043514 → 3.53043514 | 3.83430475 → 3.83430475 | 3.81994865 → 3.81994865 | 0/72 → 0/72 | Parent retained: `672db59bfc44d026f26e38d781c84dd24f92882c65b888639ec66a790002ed68` |
| Godot | 750 | 500 | 4.20090926 → 4.20090926 | 4.27263499 → 4.27263499 | 4.11664192 → 4.11664192 | 17/92 → 17/92 | Parent retained: `de7de5052f0a56d821ea3995f1f2e5fbb80d10750e4fb3a564f3c6c4a2a0afdf` |

At both 250-update validation checkpoints, loss was worse than the starting parent. Patience was two evaluations, so both lanes stopped after 500 updates and selected the unchanged parents. Therefore, this run produced no accepted weight improvement. The small forced-choice scores are diagnostics only: they do not establish Blender/Godot competence, and no human grading of generated scripts or scenes was performed.

## Data lineage

- Blender parent: run 42445321887, 1,500 specialist updates; parent SHA-256 `672db59bfc44d026f26e38d781c84dd24f92882c65b888639ec66a790002ed68`.
- Godot parent: run 42445321887, 1,500 specialist updates; parent SHA-256 `de7de5052f0a56d821ea3995f1f2e5fbb80d10750e4fb3a564f3c6c4a2a0afdf`.
- Shared Forge root parent SHA-256: `a7bcdb996e440f44d7496b887b54d19324927b4296fe9ed2f0f0293c88c63a0a`.
- Tokenizer SHA-256: `4ecd7803d4c369e80bda2e74c30cfb769b7e3a942f931df83383eec9f794f04c`.
- Blender: train `c558860e5410e06a2d6d0491597b56147dade3992e5251b4662c4892b640e3a3`; validation `1aa93c321a7e901b8228cc7a58ea658ceb2bcefd1d6715b47ed5a6f15f101213`; test `16ff66b4fa96eef0377c7b8daf7e71775e06b56db285293aabdcf1b38b3e17d5`; lesson cards `cc32cd75a049017cef587cd1c991952c4b883e80d1c20429c7d5c3de875c926f`.
- Godot: train `ab8992fa8f2b4c00ed6e2fd356ffe33f7e90f950d93805e1ab3f6f419e174fec`; validation `914760387ec29a9f3c1e2038ba3d0894e7c41cb943a229e09a407281cedfe994`; test `28f9555cbbe81ccdeebcbf318763aacfbb6dfe55768fcf6d1481b6cd2b1c9b17`; lesson cards `f355120693daaea6c33d325441912d33947b65a00e92937221ad110e6be0630c`.

The Blender lesson set was CC0-1.0 and its manual source is attributed in the archived bundle under the Blender Manual license. Godot documentation source is attributed in the bundle under CC BY 3.0. The eight fresh holdout pages per lane are evaluation-only and do not enter optimizer updates.
