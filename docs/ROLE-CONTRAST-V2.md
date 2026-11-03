# Forge Seed — v2 contrastive roles with a measured CPU budget

## Why another experiment?

The first question-conditioned neural head scored **162/288** on an
internally generated held-out role exam, compared with **148/288**
shuffled-answer control. It answered BOTH "giver" and "recipient"
questions correctly for the same event only **20/144** times.

Forge still has trouble responding differently to two questions about the
same sentence. More unstructured updates alone would risk memorization.

## New preregistered study

The **original scratch-trained v0.7 language network** stays frozen at
its SHA256-verified checkpoint lineage. Newly initialized neural
heads are used for all four research arms.

1. **Paired contrastive training:** Train on groups of four: who gave,
   who received, and the same event with the names reversed. Penalize
   choosing the same person for opposed questions.
2. **Cross-entropy-only:** Same architecture, data exposure, update
   order, and initialization, without the extra paired contrastive term.
3. **Question-ablated contrastive:** Same inputs and pair losses, but
   the query representation and question n-grams are replaced by constants.
   Source-entity features cannot see the later question.
4. **Group-flipped label control:** The entire four-question event
   family has its correct answer position flipped in a reproducible
   random subset of groups. This preserves question/counterfactual
   pairing while removing truthful role supervision.

The head includes original-language-model contextual embeddings and a
**random-initialized trainable character-trigram question encoder**.
Neither this character encoder nor the original Forge weights
derive from another pretrained model. Original Forge weights are frozen.

**Source-name extraction** uses a capitalization heuristic from the
story, independent of gold. We do not claim it has learned general named
entity recognition.

## Data and evaluation firewall

- Training uses the existing 320 publicly synthetic role examples;
  development uses 80 examples with separate names and syntax.
- Final v2 synthetic exam has **360 new questions**, 3 new
  grammatical forms, 3 fixed seeds (10901–10903), 10 original
  events per form/seed, 2 name swaps, 2 role questions.
- Exam names, objects and templates are disjoint from all prior
  Forge role corpora; the exact previous v1 288-case exam is never
  used as the new untouched set.
- Final gold comes from an independent anchored event parser.
  The synthetic exam is not human-authored natural language and is
  not an external certification.
- **Early stopping and best-weight selection use only the development
  set. The new synthetic exam is scored once after training.**
- Run 3 randomly initialized copies (7382–7384), then report
  per-seed mean and spread instead of selecting a lucky seed.
- Key metrics: independent exact-name answers; both-questions
  correct for an event; both name-swapped scenarios correct;
  how often the same event generates a changed answer to a
  changed role question.
- All candidates remain research-only; no automatic promotion,
  no personal data or Android observation, and no licensing shortcut.

## Compute plan

**Single standard GitHub-hosted Ubuntu CPU runner**, not parallel workers.
Default one-time experiment has up to **36 minutes of TRAINING time**,
with a **48-minute job ceiling** to allow installs, validation, artifact
downloads, precomputation, report saving and cleanup.
Maximum 50,000 optimizer updates per arm per initialization.
Training stops when time OR the update cap is reached.
Selected development scores and throughput are logged every 500 rounds.

Run progress and incomplete recovery snapshots include model,
optimizer and random-sampler states; automatic cross-run recovery
has **not** yet been implemented, and should not be claimed.
Artifacts retain experimental snapshots/metrics for 30 days.
One concurrency group serializes runs without cancelling an earlier run.
No auto-repeat schedule and no continuous stress-test runner.

The Python report records elapsed seconds, updates per second, peak
resident memory, all controls, frozen-model lineage, input hashes and
which seeds completed. GitHub Actions also exposes job duration.

### Actual service constraints

GitHub docs say standard hosted Actions runners are free for public
repositories, while individual ordinary job duration is limited to
6 hours. Neither fact promises a guaranteed number of training hours
**per day**, and excessive automated use is not permitted.
Our 36-minute ceiling is a *research budget*, not a discovered quota.

Sources:
- https://docs.github.com/en/actions/concepts/billing-and-usage
- https://docs.github.com/en/actions/reference/limits
- https://docs.github.com/en/site-policy/acceptable-use-policies/github-acceptable-use-policies

We should improve **validated learning per CPU-minute**, not consume
compute on arbitrary cycles.

## Execution

Workflow: .github/workflows/role-contrast-v2.yml.

Use the explicit one-off request in
experiments/requests/role-contrast-v2-long-001.md, or manually launch
the Actions workflow with a 1–36-minute budget.

**Promotion gate:** No new model weights may replace canonical Forge
Seed until multiple replications show clear gains over the
question-ablated, cross-entropy and shuffled controls, pass the
both-correct role test, and survive independent human-authored
questions and language-retention evaluation.
