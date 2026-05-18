"""Phase 1 minimal subagent: summary join prompt + deterministic validator.

Citation marker format: [[Cn:<child_id_prefix_8>]] where n is 1-indexed
position in completed_children and <child_id_prefix_8> is the first 8
characters of the child session id. This format prevents LLM fabrication
(can't invent matching child_id prefix) and is regex-validatable.

Validator runs AFTER LLM produces summary. On first failure, caller
re-prompts with errors as feedback (1× retry). 2nd failure → yield
JoinedSummaryEvent with validation_warnings populated.
"""
from __future__ import annotations

import re
from typing import Any

SUMMARY_PROMPT_TEMPLATE = """你是研究综合 agent。下面是 N 个独立子调研的结果。

任务：把它们整合成一份压缩、结构化、可审计的 summary 给主 agent 使用。

# Output Rubric (MUST FOLLOW)

1. **结构必须有**：
   - 背景（≤ 50 字）
   - 关键发现（每条 ≤ 30 字，标 [[Cn:<child_id_prefix_8>]] 标注来源）
   - 各子结论（每个 child 一段，≤ 100 字）
   - 整合判断（≤ 100 字）
   - 未解问题 / 矛盾（如有）
   - dropped_children（如有 failed/timed_out 子；说明它们错过了什么）

2. **每条主张必须 cite child id**：用 `[[C1:abcdef01]]` `[[C2:12345678]]` 标注源
   - 反例: ❌ "Anthropic 用 orchestrator-worker"
   - 正例: ✅ "Anthropic 用 orchestrator-worker [[C1:abcdef01]]"

3. **Self-check requirements**:
   - 是否每个 completed child 都被引用至少一次？若有 child 未被引用，**必须**在 "dropped_children" 里说明为什么不可用
   - 是否检测到 children 之间矛盾？若有 → **必须**列在"未解问题"
   - 是否避免了复述每条原文？→ 提炼共识 + 分歧，不抄写

4. **Length cap**: 总长 ≤ 1200 字符（≈ 800 中文字）。超长不收。

5. **Failure mode（如所有 children 都 failed/timed_out）**:
   - summary 仍要产出，开头说明 "所有子任务失败"
   - 列出每个 child 的 outcome + error_summary
   - 不要凭空编造研究结果

各子调研结果如下：

{children_section}

# 现在按 rubric 输出 summary：
"""


def build_summary_prompt(
    prompts: list[str],
    completed_children: list[Any],
    dropped_children: list[dict],
) -> str:
    """Construct the summary join prompt with structured citation hints.

    completed_children: list of objects with .child_id, .final_answer
    """
    lines: list[str] = []
    for idx, child in enumerate(completed_children, start=1):
        prefix = child.child_id[:8]
        prompt_text = prompts[idx - 1] if idx - 1 < len(prompts) else "(missing)"
        lines.append(
            f"[[C{idx}:{prefix}]] Q: {prompt_text}\n"
            f"A: {child.final_answer or '(empty)'}"
        )
    if dropped_children:
        lines.append("\nDropped children:")
        for d in dropped_children:
            cid = d.get("child_id", "?")[:8] if d.get("child_id") else "?"
            lines.append(
                f"  - {cid} outcome={d.get('outcome', '?')} "
                f"error={d.get('error_summary', '')}"
            )
    children_section = "\n\n".join(lines) if lines else "(no completed children)"
    return SUMMARY_PROMPT_TEMPLATE.format(children_section=children_section)


_MARKER_RE = re.compile(r"\[\[C(\d+):([A-Za-z0-9_-]{8})\]\]")
_VALID_DROPPED_OUTCOMES = {"failed", "timed_out", "cancelled", "waiting"}


def validate_joined_summary(
    summary_text: str,
    completed_children: list[Any],
    dropped_children: list[dict],
) -> tuple[bool, list[str]]:
    """Deterministic validation of LLM-produced summary.

    Returns (ok, errors). On (False, [...]) caller should re-prompt LLM
    with errors as feedback (1× retry); 2nd failure → yield event with
    validation_warnings = errors.
    """
    errors: list[str] = []

    expected_markers = {
        f"[[C{i+1}:{c.child_id[:8]}]]": c.child_id
        for i, c in enumerate(completed_children)
    }

    for marker, child_id in expected_markers.items():
        if marker not in summary_text:
            errors.append(
                f"child {child_id} not cited (expected marker {marker})"
            )

    found = set(_MARKER_RE.findall(summary_text))
    for n_str, prefix in found:
        marker = f"[[C{n_str}:{prefix}]]"
        if marker not in expected_markers:
            errors.append(f"fabricated citation marker: {marker}")

    if len(summary_text) > 1200:
        errors.append(f"summary exceeds 1200 char cap ({len(summary_text)})")

    for d in dropped_children:
        outcome = d.get("outcome")
        if outcome not in _VALID_DROPPED_OUTCOMES:
            errors.append(f"invalid dropped outcome: {outcome}")

    return (len(errors) == 0, errors)
