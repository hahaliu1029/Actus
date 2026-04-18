# Adversarial suites — taxonomy

Split by design doc §621 into five suites:
`ambiguous | sarcasm | temporary | contradictions | testing`.

| Suite | N | Semantics | Current sample origin |
|---|---|---|---|
| `temporary.jsonl` | 5 | 临时任务 — task-local directive (这次/今天/刚才) dressed as a permanent rule | a1-a5 (direct match) |
| `testing.jsonl` | 3 | 会话内试探 — exploratory/hypothetical/subjunctive ("if", "假设", "要是") | a6-a8 (direct match) |
| `contradictions.jsonl` | 3 | 矛盾规则 — retraction/cancel of a prior memory without replacement | a11-a13 (direct match) |
| `ambiguous.jsonl` | 4 | 模糊偏好 — nuanced/borderline/chatter (contrastive framing, off-topic personal) | a17-a20 (direct match) |
| `sarcasm.jsonl` | 5 | 反讽 — **proxy grouping** (see note) | a9-a10 (agent-mimicry) + a14-a16 (injection) |

## `sarcasm.jsonl` is a proxy grouping (gap vs design)

Genuine irony/sarcasm samples aren't in the current 20-row set. The five
rows under `sarcasm.jsonl` are `agent-mimicry` and `prompt-injection`
attempts that share a thematic pattern with sarcasm: **the surface form
looks positive/keepable, but the actual intent is inverted**. In sarcasm
the speaker says "great, another bug" to mean "not great"; in
agent-mimicry the text looks like a user statement but is the agent's
own echo; in injection the text asks for memory save but the payload
aims to subvert the agent.

All three are the same failure mode for the gate: don't be fooled by
surface alignment. Treating them as one suite is an honest approximation
for M2 PR-5; genuine sarcasm samples will land in the gap #2 dataset
expansion pass where proper per-category coverage is the explicit goal.

## Scaling target (deferred to gap #2)

Design doc target: **20-30 samples per suite, per-suite Wilson CI lower
≥ 0.80**. Current N (3-5 per suite) is too small for a meaningful
Wilson CI gate — `test_gate_adversarial_resistance` parametrizes by
suite but applies lenient point-estimate bars until the dataset grows.

## Adding new samples

1. Write rows in JSON Lines format (one object per line).
2. `id` prefix convention: keep `a*` namespace to preserve cross-suite
   uniqueness. Start new rows at `a21` upward.
3. Required fields: `id`, `text`, `expected_verdict`, `expected_category`.
   Optional: `notes` (highly recommended — the rationale is what makes
   future re-labelling defensible).
4. Drop the new file into the appropriate suite path; no registry
   update needed — `paths.py::synthetic_adversarial_suite_path()`
   resolves by name.
