"""Gradual compaction: Level 2 (LLM summary) + Level 3 (hard truncation).

Spec: docs/superpowers/specs/2026-04-01-gradual-compaction-design.md
"""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)

from app.domain.services.graphs.context_assembler import group_messages
from app.domain.services.graphs.token_estimator import TokenEstimator

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)

# ── Summary markers ──────────────────────────────────────────────────────────

SUMMARY_START = "<!-- CONVERSATION_SUMMARY -->"
SUMMARY_END = "<!-- /CONVERSATION_SUMMARY -->"
SUMMARY_PATTERN = re.compile(
    re.escape(SUMMARY_START) + r".*?" + re.escape(SUMMARY_END),
    re.DOTALL,
)

# ── Identifier extraction patterns ──────────────────────────────────────────

IDENTIFIER_PATTERNS = [
    re.compile(r'(?:/[\w.-]+){2,}(?:\.\w+)'),        # file paths (2+ segments + ext)
    re.compile(r'https?://\S{10,}'),                   # URLs (10+ chars)
    re.compile(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', re.IGNORECASE),
]

# ── Compaction ID derivation [R4-P1-1] ──────────────────────────────────────


def compute_messages_input_hash(messages: list[BaseMessage]) -> str:
    """Stable order-preserving hash of the input message list.

    [CXR1-P2-9] Serialize as a JSON array of per-message dicts (length-safe;
    no manual delimiter that content could collide with). The previous
    pipe/null-byte separator design was vulnerable to content containing
    those sentinels.

    Used as the strongest "this exact moment in graph state" signal so the
    derived compaction_id is retry-idempotent (same input → same id) and
    distinct across graph-state-advanced compactions (different input → different id).
    """
    serialized = []
    for m in messages:
        # m.content can be str OR list[dict] (multimodal). For the dict form
        # we let json.dumps handle nested structures with sort_keys for stability.
        if isinstance(m.content, str):
            content_repr: object = m.content
        else:
            content_repr = m.content  # let json.dumps recurse with sort_keys
        serialized.append({
            "kind": type(m).__name__,
            "content": content_repr,
            "tool_calls": getattr(m, "tool_calls", None) or [],
            "tool_call_id": getattr(m, "tool_call_id", "") or "",
        })
    blob = json.dumps(
        serialized,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def derive_compaction_id(
    session_id: str,
    messages_input_hash: str,
    tokens_before_total: int,
    tokens_after_total: int,
    messages_removed_total: int,
    summary: str,
) -> str:
    """Content-derived idempotency key. See spec § Schema, derivation rule [R4-P1-1]."""
    payload = (
        f"{session_id}:{messages_input_hash}:{tokens_before_total}:"
        f"{tokens_after_total}:{messages_removed_total}:{summary[:64]}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# ── Module-level helper functions ────────────────────────────────────────────

def extract_identifiers(text: str) -> list[str]:
    """Extract deduplicated file paths, URLs, and UUIDs from text."""
    seen: dict[str, None] = {}  # ordered dict trick for deduplication preserving order
    for pattern in IDENTIFIER_PATTERNS:
        for match in pattern.finditer(text):
            value = match.group(0)
            if value not in seen:
                seen[value] = None
    return list(seen.keys())


def extract_summary_section(content: str) -> str:
    """Extract existing summary text from SystemMessage content.

    Returns the text between SUMMARY markers, or "" if not found.
    """
    match = SUMMARY_PATTERN.search(content)
    if not match:
        return ""
    block = match.group(0)
    # Strip the outer markers to return the inner content
    inner = block[len(SUMMARY_START):-len(SUMMARY_END)].strip()
    return inner


def inject_summary_section(
    content: str,
    summary: str,
    messages_removed: int,
    tokens_freed: int,
) -> str:
    """Inject or replace summary section in SystemMessage content."""
    new_block = (
        f"{SUMMARY_START}\n"
        f"## 对话历史摘要（自动生成）\n\n"
        f"{summary}\n\n"
        f"_压缩了 {messages_removed} 条消息，释放约 {tokens_freed} tokens_\n"
        f"{SUMMARY_END}"
    )

    if SUMMARY_PATTERN.search(content):
        return SUMMARY_PATTERN.sub(new_block, content)
    else:
        return content + f"\n\n{new_block}"


@dataclass(frozen=True)
class CompactionResult:
    """Immutable result of a compaction operation."""

    messages: tuple[BaseMessage, ...]
    level_applied: int          # 0=none, 2=summary, 3=hard truncation
    tokens_before: int
    tokens_after: int
    summary_injected: bool
    messages_removed: int
    usage_ratio_after: float    # tokens_after / context_window
    # B6 additions [R2-P2-8 + R4-P1-1]
    summary_text: str | None = None
    operations: list[dict] = field(default_factory=list)
    compaction_id: str | None = None


# ── GradualCompactor ──────────────────────────────────────────────────────────

_HARD_COMPACT_KEEP = 19  # number of non-system messages to keep in Level 3


class GradualCompactor:
    """Two-level context compressor: Level 2 (LLM summary) + Level 3 (hard truncation).

    Stateless: all state flows through method arguments and return values.
    The sole public entry point is ``try_compact()``.
    """

    SUMMARY_PROMPT = """请将以下对话历史压缩为简洁的摘要。

要求：
1. 保留所有关键标识符（文件路径、URL、UUID、API 端点、变量名）
2. 保留：决策记录、待办事项、约束条件、用户未回复的请求
3. 省略：工具调用的详细输出、重复的中间步骤、已完成且不影响后续的操作细节
4. 摘要长度不超过 {max_chars} 字符

{existing_summary_section}

必须保留的标识符：
{extracted_identifiers}

待压缩的对话：
{messages_to_summarize}
"""

    def __init__(
        self,
        token_estimator: TokenEstimator,
        soft_trigger_ratio: float = 0.85,
        hard_trigger_ratio: float = 0.95,
        target_ratio: float = 0.65,
        summary_max_chars: int = 16_000,
        token_safety_factor: float = 1.15,
    ) -> None:
        self._estimator = token_estimator
        self._soft = soft_trigger_ratio
        self._hard = hard_trigger_ratio
        self._target = target_ratio
        self._summary_max_chars = summary_max_chars
        self._safety = token_safety_factor

    # ── public entry ─────────────────────────────────────────────────────────

    async def try_compact(
        self,
        messages: list[BaseMessage],
        context_window: int,
        summary_llm: "BaseChatModel | None",
        *,
        config: dict | None = None,
    ) -> CompactionResult:
        """Route to the appropriate compaction level based on current token usage.

        Decision flow:
        1. Estimate tokens × safety_factor → usage_ratio
        2. < soft_trigger_ratio  → no compaction (level=0)
        3. >= soft and < hard and summary_llm available → Level 2 (_soft_compact)
           • post-verify: if still > target_ratio + 0.03 → escalate to Level 3
        4. >= hard or summary_llm unavailable → Level 3 (_hard_compact)
        """
        raw_tokens = self._estimator.estimate_messages(messages)
        tokens_before = int(raw_tokens * self._safety)
        usage_ratio = tokens_before / context_window if context_window > 0 else 0.0

        logger.debug(
            "try_compact: raw=%d safety_factor=%.2f tokens=%d window=%d ratio=%.3f",
            raw_tokens,
            self._safety,
            tokens_before,
            context_window,
            usage_ratio,
        )

        # No compaction needed
        if usage_ratio < self._soft:
            tokens_after = tokens_before
            return CompactionResult(
                messages=tuple(messages),
                level_applied=0,
                tokens_before=tokens_before,
                tokens_after=tokens_after,
                summary_injected=False,
                messages_removed=0,
                usage_ratio_after=usage_ratio,
            )

        # Level 2: LLM summary (only when summary_llm available and below hard threshold)
        if usage_ratio < self._hard and summary_llm is not None:
            result = await self._soft_compact(
                messages=messages,
                context_window=context_window,
                summary_llm=summary_llm,
                tokens_before=tokens_before,
                config=config,
            )
            # Post-verify: if still above target + 3% tolerance → escalate
            post_verify_threshold = self._target + 0.03
            if result.usage_ratio_after > post_verify_threshold:
                logger.info(
                    "Level 2 post-verify failed (ratio=%.3f > %.3f), escalating to Level 3",
                    result.usage_ratio_after,
                    post_verify_threshold,
                )
                return self._hard_compact(
                    messages=result.messages,
                    context_window=context_window,
                    tokens_before=result.tokens_after,
                    prior_operations=copy.deepcopy(result.operations),
                    prior_summary_text=result.summary_text,
                )
            return result

        # Level 3: hard truncation
        return self._hard_compact(
            messages=messages,
            context_window=context_window,
            tokens_before=tokens_before,
        )

    # ── Level 3: hard truncation ──────────────────────────────────────────────

    def _hard_compact(
        self,
        messages: list[BaseMessage],
        context_window: int,
        tokens_before: int,
        *,
        prior_operations: list[dict] | None = None,
        prior_summary_text: str | None = None,
    ) -> CompactionResult:
        """Keep SystemMessage + last 19 messages, inject truncation marker.

        If there are 20 or fewer messages total (sys + ≤19 others), no messages
        are removed — only the truncation marker is injected.
        """
        if not messages:
            hard_op = {
                "kind": "hard_truncate",
                "tokens_before": tokens_before,
                "tokens_after": 0,
                "messages_removed": 0,
                "messages_kept": 0,
            }
            operations = [*(prior_operations or []), hard_op]
            if prior_operations:
                # [CXR1-P1-1] Roll up totals across the escalation chain
                rollup_tokens_before = prior_operations[0].get("tokens_before", tokens_before)
                rollup_messages_removed = sum(
                    op.get("messages_removed", op.get("messages_summarized", 0))
                    for op in operations
                )
            else:
                rollup_tokens_before = tokens_before
                rollup_messages_removed = 0
            return CompactionResult(
                messages=(),
                level_applied=3,
                tokens_before=rollup_tokens_before,
                tokens_after=0,
                summary_injected=False,
                messages_removed=rollup_messages_removed,
                usage_ratio_after=0.0,
                summary_text=prior_summary_text,  # preserve LLM summary on escalation
                operations=operations,
            )

        sys_msg = messages[0]
        rest = messages[1:]

        removed_count = max(0, len(rest) - _HARD_COMPACT_KEEP)
        kept = rest[-_HARD_COMPACT_KEEP:] if removed_count > 0 else rest

        # Inject truncation marker into SystemMessage
        new_sys_content = self._inject_truncation_marker(
            original_content=sys_msg.content if isinstance(sys_msg.content, str) else str(sys_msg.content),
            removed_count=removed_count,
        )
        new_sys = SystemMessage(content=new_sys_content)

        result_messages = [new_sys] + list(kept)
        tokens_after = self._estimator.estimate_messages(result_messages)
        usage_ratio_after = tokens_after / context_window if context_window > 0 else 0.0

        logger.info(
            "_hard_compact: removed=%d, kept=%d, tokens %d→%d (ratio=%.3f)",
            removed_count,
            len(kept),
            tokens_before,
            tokens_after,
            usage_ratio_after,
        )

        hard_op = {
            "kind": "hard_truncate",
            "tokens_before": tokens_before,
            "tokens_after": tokens_after,
            "messages_removed": removed_count,
            "messages_kept": min(_HARD_COMPACT_KEEP, len(rest)),
        }
        operations = [*(prior_operations or []), hard_op]

        if prior_operations:
            # [CXR1-P1-1] Roll up totals across the escalation chain
            rollup_tokens_before = prior_operations[0].get("tokens_before", tokens_before)
            rollup_messages_removed = sum(
                op.get("messages_removed", op.get("messages_summarized", 0))
                for op in operations
            )
        else:
            rollup_tokens_before = tokens_before
            rollup_messages_removed = removed_count

        return CompactionResult(
            messages=tuple(result_messages),
            level_applied=3,
            tokens_before=rollup_tokens_before,   # [CXR1-P1-1] not raw arg
            tokens_after=tokens_after,
            summary_injected=removed_count > 0,
            messages_removed=rollup_messages_removed,  # [CXR1-P1-1] sum across chain
            usage_ratio_after=usage_ratio_after,
            summary_text=prior_summary_text,  # preserve LLM summary on escalation
            operations=operations,
        )

    def _inject_truncation_marker(self, original_content: str, removed_count: int) -> str:
        """Inject a truncation notice inside the <!-- CONVERSATION_SUMMARY --> block.

        If the block already exists: append the marker inside it (replacing the
        block wholesale so the marker appears at the end of the existing content).
        If not: create a new block.
        """
        marker_line = f"_硬截断：移除了 {removed_count} 条历史消息_"

        existing_match = SUMMARY_PATTERN.search(original_content)
        if existing_match:
            block_body = existing_match.group(0)
            inner = block_body[len(SUMMARY_START):-len(SUMMARY_END)].strip()
            # Always reconstruct with canonical heading to prevent structure drift
            new_block = (
                f"{SUMMARY_START}\n"
                f"## 对话历史摘要（自动生成）\n\n"
                f"{inner}\n\n"
                f"{marker_line}\n"
                f"{SUMMARY_END}"
            )
            return SUMMARY_PATTERN.sub(new_block, original_content)
        else:
            new_block = (
                f"\n\n{SUMMARY_START}\n"
                f"## 对话历史摘要（自动生成）\n\n"
                f"{marker_line}\n"
                f"{SUMMARY_END}"
            )
            return original_content + new_block

    # ── Level 2: LLM summary ─────────────────────────────────────────────────

    async def _soft_compact(
        self,
        messages: list[BaseMessage],
        context_window: int,
        summary_llm: "BaseChatModel",
        tokens_before: int,
        *,
        config: dict | None = None,
    ) -> CompactionResult:
        """Level 2: LLM summary compaction.

        Algorithm:
        1. Calculate tokens_to_free = tokens_before - (context_window * target_ratio)
        2. Group non-system messages and select oldest groups until enough tokens freed
        3. Extract identifiers from selected messages
        4. Build prompt including existing summary (if any) and call summary_llm
        5. Inject summary into SystemMessage, return new message list
        6. On LLM exception, fall back to _hard_compact
        """
        sys_msg = messages[0]
        rest = messages[1:]

        tokens_to_free = tokens_before - int(context_window * self._target)

        # Group non-system messages and select oldest groups to summarize
        groups = group_messages(rest)
        selected_groups: list = []
        accumulated_tokens = 0

        for group in groups:
            group_tokens = sum(self._estimator.estimate_message(m) for m in group.messages)
            selected_groups.append(group)
            accumulated_tokens += group_tokens
            if accumulated_tokens >= tokens_to_free:
                break

        # Nothing to summarize
        if not selected_groups:
            tokens_after = tokens_before
            return CompactionResult(
                messages=tuple(messages),
                level_applied=0,
                tokens_before=tokens_before,
                tokens_after=tokens_after,
                summary_injected=False,
                messages_removed=0,
                usage_ratio_after=tokens_after / context_window if context_window > 0 else 0.0,
            )

        # Flatten selected groups into messages to summarize
        messages_to_summarize: list[BaseMessage] = []
        for group in selected_groups:
            messages_to_summarize.extend(group.messages)

        # Remaining messages (not summarized).
        # Boundary is the total message count across all selected groups.
        # Safe because group_messages() preserves order and covers all messages.
        selected_count = sum(len(g.messages) for g in selected_groups)
        remaining_messages = rest[selected_count:]

        # Extract identifiers from selected messages
        all_text = "\n".join(
            m.content if isinstance(m.content, str) else str(m.content)
            for m in messages_to_summarize
        )
        identifiers = extract_identifiers(all_text)
        identifiers_str = "\n".join(identifiers) if identifiers else "（无）"

        # Build existing summary section context
        sys_content = sys_msg.content if isinstance(sys_msg.content, str) else str(sys_msg.content)
        existing_summary = extract_summary_section(sys_content)
        existing_summary_section = (
            f"已有摘要（请在新摘要中整合）：\n{existing_summary}"
            if existing_summary
            else ""
        )

        # Serialize messages to summarize
        serialized_msgs = "\n\n".join(
            f"[{type(m).__name__}]: {m.content if isinstance(m.content, str) else str(m.content)}"
            for m in messages_to_summarize
        )

        prompt = self.SUMMARY_PROMPT.format(
            max_chars=self._summary_max_chars,
            existing_summary_section=existing_summary_section,
            extracted_identifiers=identifiers_str,
            messages_to_summarize=serialized_msgs,
        )

        # Invoke LLM, fall back to hard compact on failure.
        # B4 M0: thread caller's config (callbacks + metadata) so
        # graph-external compaction LLM call lands in the cost ledger.
        ainvoke_kwargs: dict[str, object] = {}
        if config is not None:
            ainvoke_kwargs["config"] = config
        try:
            llm_response = await summary_llm.ainvoke(prompt, **ainvoke_kwargs)
            summary_text = llm_response.content if isinstance(llm_response.content, str) else str(llm_response.content)
        except Exception as exc:
            logger.warning("_soft_compact: LLM invocation failed (%s), falling back to Level 3", exc)
            return self._hard_compact(
                messages=messages,
                context_window=context_window,
                tokens_before=tokens_before,
            )

        # Hard-truncate LLM output to summary_max_chars
        summary_text = summary_text[: self._summary_max_chars]

        tokens_freed = accumulated_tokens
        new_sys_content = inject_summary_section(
            content=sys_content,
            summary=summary_text,
            messages_removed=selected_count,
            tokens_freed=tokens_freed,
        )
        new_sys = SystemMessage(content=new_sys_content)

        result_messages = [new_sys] + list(remaining_messages)
        tokens_after = self._estimator.estimate_messages(result_messages)
        usage_ratio_after = tokens_after / context_window if context_window > 0 else 0.0

        logger.info(
            "_soft_compact: summarized=%d msgs, freed≈%d tokens, tokens %d→%d (ratio=%.3f)",
            selected_count,
            tokens_freed,
            tokens_before,
            tokens_after,
            usage_ratio_after,
        )

        soft_op = {
            "kind": "llm_summary",
            "summary_chars": len(summary_text),
            "identifiers_preserved_count": len(identifiers),
            "messages_summarized": selected_count,
            "tokens_before": tokens_before,
            "tokens_after": tokens_after,
        }
        return CompactionResult(
            messages=tuple(result_messages),
            level_applied=2,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            summary_injected=True,
            messages_removed=selected_count,
            usage_ratio_after=usage_ratio_after,
            summary_text=summary_text,
            operations=[soft_op],
        )
