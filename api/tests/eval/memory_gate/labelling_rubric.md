# Memory Gate Labelling Rubric (M2)

**Rubric version:** `v1-m2` — frozen for the M2 ship. See `RUBRIC_VERSION`
in `tests/eval/memory_gate/__init__.py` for the programmatic constant;
when semantics change here, bump both and re-tag the dataset.

**Previous version:** `v0-m1` (M1 initial ship, 25-sample synthetic dataset).
Semantics carried forward unchanged; M2 adds adversarial suites, private
set conventions, and the annotator-drift control set.

## Goal

For each dialog chunk, decide whether it's worth persisting as long-term
agent memory. The gate's job is to filter auto-flush candidates; users
explicitly saving via `memory_save` bypass this rubric.

## Verdict schema

| verdict | Use when |
|---|---|
| `keep` | The content is something the user expressed as a preference, rule, or verifiable fact that **will be useful to recall in a future session**. |
| `drop` | Everything else: transient debug output, session-local context, sarcasm, the agent's own replies, a task description that ends when the task ends. |

## Category (required even when dropping — schema stability)

| category | Semantic | Examples |
|---|---|---|
| `user` | Profile, preference, identity, working habit | "I prefer Go, 10 years experience"; "respond in Chinese"; "I'm a data scientist" |
| `rule` | Behavioral constraint or permanent rule | "never commit without asking"; "stop mocking the database in tests"; "always log in structured JSON" |
| `fact` | Verifiable external fact or project decision | "DB is PostgreSQL 17"; "staging API is at foo.internal/v2"; "payment integration lives in `services/pay/`" |

**If verdict=drop, category is a tie-breaker fallback — pick whichever is
closest semantically; most commonly `user` as a neutral default.** This
keeps the output schema uniform without influencing gate behavior (drop
means drop regardless of category).

## Confidence (0.0 – 1.0)

This is the labeller's self-reported confidence in their verdict. **Not**
an informational-value score. A label with `verdict=keep, confidence=0.5`
means "I think this is keep but I'm not sure" — the threshold
(`memory_gate_threshold`, default 0.7) filters these out; raising the
threshold trades recall for precision.

Reference anchors:
- `0.9+` — unambiguous "memory_save"-grade statements ("我用中文", "以后别
  mock 数据库", "DB 是 PostgreSQL 17")
- `0.7` — typical boundary: clear intent + no contradictory signal in the
  surrounding dialog
- `0.5` — borderline; could be either verdict under a slightly different
  interpretation
- `0.3 or less` — lean strongly toward dropping; any downstream threshold
  above this kills the chunk

## Conservative defaults

When in doubt, **prefer `drop`**. The cost of dropping a useful memory
is recoverable (user can re-assert via `memory_save` or manual create);
the cost of keeping junk is persistent pollution that shows up in prompt
injection and retrieval ranking.

Specifically drop these even if they sound "memory-like":
- Reiterations of the current task ("I want to finish this PR today")
- Sarcasm, hypotheticals, or quotes from the agent itself
- Clarifications on the current task's parameters ("no, use port 8080")
- Status updates ("restarted the server")
- Error messages, tool outputs, debug printouts

Keep these even if terse:
- Identity / role statements
- Permanent rules or prohibitions
- Stable project facts (tech stack, endpoint URLs, compliance constraints)

## Dataset layout (M2)

Two datasets, both consumed by `tune_threshold.py` and the eval pytest suite.

### Public (synthetic), checked into the repo

Path: `api/tests/eval/memory_gate/synthetic/dataset.jsonl`.

Scale target: **100 samples** at M2 ship (25 today, carried over from M1).
Manually crafted + paraphrased so the content is safe to publish.

### Private (real), gitignored

Path default: `~/.actus/eval/memory_gate/real/dataset.jsonl`.
Override: set `ACTUS_EVAL_DATA_DIR` env var, in which case the resolved
path is `${ACTUS_EVAL_DATA_DIR}/memory_gate/real/dataset.jsonl`.

Scale target: **≥ 200 samples** at M2 acceptance. Scraped from the solo
author's real dev sessions, passed through `detect-secrets` + a manual
pass to redact anything PII-adjacent. Only stored locally; not re-hosted.

The resolver and the `private_available()` check live in
`tests/eval/memory_gate/paths.py`. Contributors / CI machines without
the private dataset populated see `private_available() is False` and
run public-only.

## Dataset file format

One JSON object per line:

```json
{"id": "s1", "text": "用户：我平时用 Go，大约 10 年了", "expected_verdict": "keep", "expected_category": "user", "notes": "clear profile statement"}
```

Required keys: `id`, `text`, `expected_verdict`, `expected_category`.
Optional: `notes` (why this label, for the solo author's review).

## Annotator drift — control set (M2)

M2 pulls **30 stable samples** from the private set into
`${ACTUS_EVAL_DATA_DIR}/memory_gate/real/control_set.jsonl` — same
schema as the dataset file. The solo author re-labels this set
**weekly** (fresh judgement, no reference to previous labels). Cohen's
kappa between consecutive weeks is the drift signal:

| kappa | Action |
|---|---|
| ≥ 0.7 | Stable — rubric unchanged. |
| 0.4 – 0.7 | Borderline. Pick the most-drifting sample and add it to the rubric with explicit commentary. |
| < 0.4 | Rubric too ambiguous — tighten definitions or split into more categories. |

`stats.cohens_kappa()` computes the value; the weekly check runs under
`test_annotator_consistency` (landed in a later PR; PR-1 only ships the
helper + the layout conventions).

## M2 precision gate

M2 acceptance criterion:

> precision point estimate and **Wilson 95% CI lower bound** on the
> union (public + private) dataset at `threshold=0.7`, with the lower
> bound ≥ 0.70 as the hard gate.

Use `stats.wilson_ci_lower(tp, tp+fp)` in test assertions.
`tune_threshold.py --sweep 0.5:0.9:0.1 --llm chat_llm` sweeps over
candidate thresholds and flags the best-recall threshold that clears
the gate.
