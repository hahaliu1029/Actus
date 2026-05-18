"""Preflight UX classifier for subagent research probe.

Two-layer filter (UX gate, not security gate):
1. Static keyword filter: blocks obvious coding / write side-effect prompts.
2. Batched LLM classifier: one call yields yes/no for N prompts.

SECURITY: this is a UX gate. Actual security is runtime tool_filter
(static allowlist) + PE-0 PermissionEngine (dynamic policy). Even if
user clicks "override" past the classifier, tool_filter still prevents
write tools from being invoked.

Fail-open on LLM error: classifier is best-effort UX hint; downstream
runtime guards remain authoritative.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)


# ASCII keywords matched as whole words (regex \b...\b) to avoid false
# positives like `fix` matching `prefix` / `fixture` or `rm` matching `arm`.
# CJK has no whitespace word boundaries; we keep CJK as substring but narrow
# the set to phrases unlikely to appear in legitimate research prompts.
# Multi-word ASCII phrases (e.g. "create file") are matched as substrings.
_ASCII_WORD_BLOCK = {
    "fix", "refactor", "save", "publish", "deploy", "delete", "rm",
}
_ASCII_PHRASE_BLOCK = {
    "create file", "write file", "drop table",
}
# CJK intent verbs and short phrases. We intentionally retain the
# strong-signal verbs `修改 / 发布 / 部署 / 删除` even though they can
# false-positive on prompts like "研究发布日期"; the classifier is a UX
# gate with fail-open semantics (rejected prompts return a clear reason
# the user can rephrase), and runtime tool_filter is the actual security
# gate. `实现` was deliberately removed (too broad — false-positives on
# "研究实现原理").
_CJK_SUBSTRING_BLOCK = {
    "重构", "写代码", "修改", "提交", "发布", "部署", "删除",
}
_BLOCK_KEYWORDS = (
    _ASCII_WORD_BLOCK | _ASCII_PHRASE_BLOCK | _CJK_SUBSTRING_BLOCK
)
_ASCII_WORD_RE = {
    kw: re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE)
    for kw in _ASCII_WORD_BLOCK
}


@dataclass(frozen=True)
class ClassifierResult:
    approved: bool
    reason: str


class SubagentResearchClassifier:
    """Preflight classifier for breadth-research prompts."""

    def __init__(self, llm: BaseChatModel) -> None:
        self._llm = llm

    def _static_block(self, prompt: str) -> str | None:
        """Return reason if prompt matches a static block keyword, else None.

        ASCII word-class keywords match on regex word boundaries so `fix` no
        longer trips on `prefix`/`fixture`. ASCII phrases and narrowed CJK
        terms match as substrings (CJK has no whitespace word boundaries).
        """
        for kw, pat in _ASCII_WORD_RE.items():
            if pat.search(prompt):
                return f"matched static block keyword: {kw}"
        lower = prompt.lower()
        for kw in _ASCII_PHRASE_BLOCK:
            if kw in lower:
                return f"matched static block keyword: {kw}"
        for kw in _CJK_SUBSTRING_BLOCK:
            if kw in prompt:
                return f"matched static block keyword: {kw}"
        return None

    def _build_batch_prompt(self, prompts: list[str]) -> str:
        lines = [
            "Classify each of the following prompts as either 'yes' (independent",
            "breadth research, no coding/writes/shared-state needed) or 'no'",
            "(coding / writes / shared-state required). Output exactly N lines",
            "in format: `<n>. <yes|no> - <one-line reason>`.",
            "",
        ]
        for i, p in enumerate(prompts, start=1):
            lines.append(f"{i}. {p}")
        return "\n".join(lines)

    def _parse_batch_response(
        self, response_text: str, n: int
    ) -> list[ClassifierResult]:
        """Parse N lines of `<idx>. <yes|no> - reason`.

        Robust to LLM formatting quirks: missing lines default to approved=True.
        """
        results: list[ClassifierResult | None] = [None] * n
        for line in response_text.strip().splitlines():
            m = re.match(r"^\s*(\d+)\.\s*(yes|no)\b\s*-?\s*(.*)$", line, re.IGNORECASE)
            if not m:
                continue
            idx = int(m.group(1)) - 1
            verdict = m.group(2).lower() == "yes"
            reason = m.group(3).strip() or ("approved" if verdict else "rejected")
            if 0 <= idx < n:
                results[idx] = ClassifierResult(approved=verdict, reason=reason)
        return [
            r or ClassifierResult(approved=True, reason="missing_classifier_line_fail_open")
            for r in results
        ]

    async def classify_batch(self, prompts: list[str]) -> list[ClassifierResult]:
        """Classify N prompts; return per-prompt result.

        Static rules first; LLM call only for prompts that pass static.
        Batched LLM call for cost efficiency.
        """
        results: list[ClassifierResult | None] = [None] * len(prompts)
        llm_indices: list[int] = []
        llm_prompts: list[str] = []

        for i, p in enumerate(prompts):
            reason = self._static_block(p)
            if reason:
                results[i] = ClassifierResult(approved=False, reason=reason)
            else:
                llm_indices.append(i)
                llm_prompts.append(p)

        if llm_prompts:
            try:
                batch_prompt = self._build_batch_prompt(llm_prompts)
                response = await self._llm.ainvoke(batch_prompt)
                response_text = getattr(response, "content", str(response))
                llm_results = self._parse_batch_response(
                    response_text, len(llm_prompts)
                )
                for idx, res in zip(llm_indices, llm_results):
                    results[idx] = res
            except Exception as e:
                logger.warning(
                    "subagent classifier LLM error (fail-open): %s", e
                )
                for idx in llm_indices:
                    results[idx] = ClassifierResult(
                        approved=True,
                        reason=f"classifier_error_fail_open: {e}",
                    )

        return [r for r in results if r is not None]
