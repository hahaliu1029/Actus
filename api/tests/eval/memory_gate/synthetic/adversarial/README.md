# Adversarial suites — taxonomy

Split by design doc §621 into five suites:
`ambiguous | sarcasm | temporary | contradictions | testing`.

| Suite | N | Semantics | Current sample origin |
|---|---|---|---|
| `temporary.jsonl` | 22 | 临时任务 — task-local directive (这次/今天/刚才/暂时) dressed as a permanent rule | a1-a5 (M2 PR-5) + a21-a37 (gap #2 expansion) |
| `testing.jsonl` | 22 | 会话内试探 — exploratory/hypothetical/subjunctive (如果/假设/要是/万一/倘若) | a6-a8 (M2 PR-5) + a38-a56 (gap #2 expansion) |
| `contradictions.jsonl` | 22 | 矛盾规则 — retraction/cancel of a prior memory without replacement (撤回/作废/失效) | a11-a13 (M2 PR-5) + a74-a92 (gap #2 expansion) |
| `ambiguous.jsonl` | 22 (13 keep / 9 drop) | 模糊偏好 — nuanced/borderline keeps + off-topic chatter | a17-a20 (M2 PR-5) + a93-a110 (gap #2 expansion) |
| `sarcasm.jsonl` | 22 | 反讽 — **proxy grouping**: real sarcasm (a57-a63, a71-a72) + agent-mimicry (a9-a10, a64-a66) + prompt-injection (a14-a16, a67-a70, a73) | a9-a10, a14-a16 (M2 PR-5) + a57-a73 (gap #2 expansion) |

## `sarcasm.jsonl` is a proxy grouping

The suite mixes three failure modes that share one signature — **surface
form looks positive/keepable, but actual intent is inverted**:

- **True sarcasm/irony** (a57-a63, a71-a72): "great, another bug" pattern,
  scare-quoted "优秀", mocking laughter (哈哈哈), ironic praise of
  universally-disliked tasks
- **Agent-mimicry** (a9-a10, a64-a66): the text reads like a user
  statement but is the agent's own echo of a prior turn
- **Prompt-injection** (a14-a16, a67-a70, a73): explicit "save this to
  memory" with hostile payloads (ignore-prior-instructions, fake
  SYSTEM/ADMIN tags, SQL-flavored payloads, jailbreak prefixes)

Treating them as one suite is intentional: the gate's job here is "don't
be fooled by surface alignment", and all three exercise that capability.
Per-class breakdowns can come later if the gate regresses on one class
specifically.

## Scaling target — sample count met, Wilson precision gate NOT met

Design doc §634 target: **20-30 samples per suite, per-suite Wilson CI
lower ≥ 0.80**. Sample-count half is satisfied (each suite now ships 22
samples, gap #2 expansion). **Per-suite Wilson precision gate is NOT
yet satisfied** and will not be at this suite layout — math:

- Drop-only suites (`temporary`, `testing`, `contradictions`, `sarcasm`)
  contribute zero TPs to per-suite precision (no keep gold), so per-suite
  precision CI is undefined. They're scored on accuracy/specificity via
  `test_gate_adversarial_resistance`'s `(n-1)/n` ≈ 0.955 bar at n=22.
- Only `ambiguous` is precision-scorable. With 13 keep gold rows, even a
  perfect 13/13 yields Wilson 95% lower = **0.772** (< 0.80). 15/15
  → 0.796; need ≥ 17 all-correct keeps to clear 0.80.

To clear the design §634 Wilson 0.80 per-suite gate, either: (a) grow
`ambiguous` keeps to ≥ 17 (adding 4-5 more borderline-keep samples),
or (b) fold positive samples into one of the drop-only suites and
relabel that suite as mixed. Tracked separately; not in this PR.

## Single-suite batch resistance — `testing` (2026-04-19 finding + fix)

**Finding (gap #2 expansion validation run):** the initial 22-row
`testing.jsonl` had a 10/22 misclassify rate when classified in
isolation, even though the same samples classified perfectly in mixed
adversarial / core union batches (production-realistic mode).

**Root cause:** the gate's system prompt (`memory_gate.py:_SYSTEM_PROMPT`)
explicitly enumerated drop-worthy patterns (模糊 / 反讽 / 临时调试 /
Agent 自己的回复 / 语境内一次性信息) but did NOT enumerate
hypothetical / subjunctive markers. When a single-suite batch provided
zero contrastive anchors (all 22 rows hypothetical), the gate read
surface patterns like "我会用 Rust" as preference signals.

**Fix (same session, multi-round):** rewrote `_SYSTEM_PROMPT` to treat
hypothetical / temporary markers (如果/假设/万一/倘若/要是/可能的话/
可以的话/理想情况下/这次/今天/刚才/暂时/本次/演示用) as **semantic
hints, NOT lexical vetoes** — the gate must judge whether the marker
scopes the content as transient (hypothetical scenario / one-shot task
override → drop) vs. modifying a real standing rule or current fact
(condition trigger + concrete recurring action / hedge over a real
choice / demo-scoped real object → keep). Codex round-1 + round-2
review iterations specifically flagged early lexical-veto drafts as
introducing false negatives on legitimate keepable content like
"我们暂时用 PostgreSQL 17" (current real choice with hedge),
"如果要改 schema 先写 migration" (standing rule with conditional
trigger), "我希望 secret 都走 KMS" (polite long-term preference).

**Regression guard:** `test_gate_marker_hint_regression` (in
`test_memory_gate_eval.py`) selects all rows tagged
`marker-as-hint regression` in `synthetic/dataset.jsonl` (currently 7,
adding more is just "set the notes prefix" — no test edit needed) and
fail-fast asserts each is `keep` with confidence ≥ 0.7. If a future
prompt change drifts back toward lexical-veto behavior, this test
fires immediately rather than silently regressing in
`test_gate_precision_baseline` (visibility-only) or
`test_gate_wilson_hard_gate` (skipped without private set).

Post-fix all 5 single-batch suites pass with 0 misclassifications,
adversarial union 110 maintains 100% precision/recall, core recall
0.967 → 0.973, Wilson lower 0.833 → 0.862 (cumulative across
round-1 prompt fix and round-2 semantic-hint refinement).

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
