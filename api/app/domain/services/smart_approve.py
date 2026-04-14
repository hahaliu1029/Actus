from __future__ import annotations
import logging
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)

_PROMPT = """You are a security reviewer for an AI agent's tool execution.
Evaluate whether this tool call is safe to execute.

Tool: {tool_name}
Arguments: {tool_args}
Risk Level: {risk_level}
Matched Dangerous Patterns: {matched_patterns}
Current Task Context: {task_context}

Rules:
- APPROVE if the command is clearly safe and aligned with the task
- DENY if the command could genuinely damage the system or data
- ESCALATE if you're uncertain

Respond with exactly one word: APPROVE, DENY, or ESCALATE."""


class SmartApprove:
    """LLM-assisted risk evaluation for tool calls."""

    def __init__(self, llm: BaseChatModel):
        self._llm = llm

    async def evaluate(self, tool_name: str, tool_args: dict[str, Any],
                       risk_level: str, matched_patterns: list[str],
                       task_context: str) -> str:
        """Returns 'approve', 'deny', or 'escalate'."""
        try:
            prompt = _PROMPT.format(
                tool_name=tool_name, tool_args=tool_args,
                risk_level=risk_level,
                matched_patterns=matched_patterns or "none",
                task_context=task_context or "not available",
            )
            response = await self._llm.ainvoke(prompt)
            decision = response.content.strip().upper()
            if decision in ("APPROVE", "DENY", "ESCALATE"):
                return decision.lower()
            logger.warning("SmartApprove unexpected response: %s, escalating", decision)
            return "escalate"
        except Exception:
            logger.exception("SmartApprove failed, escalating")
            return "escalate"
