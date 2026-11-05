# Forge question-intent student study v1 — results

**Completed:** November 5, 2026  
**Study run:** [42091686838](https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/42091686838)  
**Artifact:** [forge-question-intent-study-v1](https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/42091686838/artifacts/13444764343) (expires February 3, 2027)

## Frozen source and method

The study used the 28,000-step, 50 MiB / 5 MiB public TinyStories model from run 42085634121. Its 2.22M-parameter language encoder was frozen. The checkpoint file hash was verified unchanged before and after the study: `5b0588374b6e85432fcc518850319f0ace86f746215e097eb0861076deb6a0aa`.

The benchmark had 192 training, 60 development and 90 final questions, balanced across giver, recipient and other-person/action intent. Final wording families and object names were disjoint from training and development. Three matched initialization seeds compared a plain cross-entropy head, cross-entropy plus supervised contrastive loss, and a balanced shuffled-label control. Every head had a 600-update budget; selection used development only.

## Final results

Each result is correct questions out of 90.

| Seed | Plain cross-entropy | Contrastive + CE | Shuffled labels |
| --- | ---: | ---: | ---: |
| 1201 | 56 (62.2%) | 57 (63.3%) | 37 (41.1%) |
| 1202 | 55 (61.1%) | 58 (64.4%) | 16 (17.8%) |
| 1203 | 55 (61.1%) | 55 (61.1%) | 23 (25.6%) |
| **Combined** | **166/270 (61.5%)** | **170/270 (63.0%)** | **76/270 (28.1%)** |

The supervised heads beat their matched shuffled control for every seed. That supports a real effect from the question labels on this internal benchmark. Contrastive training added only **4/270** answers over plain training, a 1.5 percentage-point difference; this is too small to call a meaningful contrastive advantage from one benchmark.

## Role-direction breakdown

| Arm | Giver | Recipient | Other |
| --- | ---: | ---: | ---: |
| Plain cross-entropy | 31/90 (34.4%) | 73/90 (81.1%) | 62/90 (68.9%) |
| Contrastive + CE | 33/90 (36.7%) | 69/90 (76.7%) | 68/90 (75.6%) |

The core failure is directional: both trained heads labeled most unseen giver questions as recipient questions. Only **31–33 of 90** giver questions were correct, while recipient recognition was much stronger. The head learned some question-family signal but did not reliably understand the giver/recipient distinction that blocks the story-role task.

## Decision and next step

**Do not promote or attach the head to story-role inference.** The encoder stayed frozen, no language weights changed, and the head weights are research-only. The final split is now used and must not be reused as an untouched exam.

The next useful lesson should teach role direction with explicitly paired counterfactual questions that differ in who gives and who receives the same item, while including neutral `other` questions. Then test on newly authored, unseen wording families. Keep the test small enough to diagnose direction errors, but do not repeat this final split or simply add more updates to its examples.

This remains a synthetic question-only test. It does not demonstrate that Forge can read a story, bind names to events, or follow natural-language instructions.
