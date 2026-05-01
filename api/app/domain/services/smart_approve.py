from __future__ import annotations
import logging
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)


# Decision-recorder port — Callable[(name, *, outcome, reason, attrs), None]
# matching ``infrastructure.observability.decision_trace.record_decision``.
# SmartApprove holds it as an opaque callable so the domain layer does NOT
# import the OTel-backed helper directly (PR-S3-2 reviewer round-2 P3
# fix). Composition (``application/composition/graph_assembly.py``)
# injects the OTel impl; tests pass either the real impl, a fake, or
# ``None`` (no observability).
DecisionRecorder = Callable[..., None]


def _safe_record(
    recorder: DecisionRecorder | None,
    name: str,
    *,
    outcome: str,
    reason: str | None = None,
    attrs: dict[str, Any] | None = None,
) -> None:
    """Call ``recorder`` swallowing any exception.

    PR-S3-2 reviewer round-3 P3: the port contract is *"observability
    failure cannot taint decision"*. The injected recorder runs inside
    SmartApprove's outer ``try/except`` — without per-call wrap, a
    raising recorder on the APPROVE/DENY path would (a) escape the
    happy path, (b) get caught by the outer except, (c) cause the
    except branch to call the **same failing recorder** again, (d)
    propagate the second exception out of ``evaluate()``. The
    OTel-backed production recorder is already best-effort, so today's
    production is unaffected; this hardens the port surface so test
    fakes and future custom impls cannot taint agent decisions either.
    """
    if recorder is None:
        return
    try:
        recorder(name, outcome=outcome, reason=reason, attrs=attrs)
    except Exception:
        logger.debug(
            "decision recorder raised for name=%r outcome=%r; swallowing",
            name,
            outcome,
            exc_info=True,
        )

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

    def __init__(
        self,
        llm: BaseChatModel,
        *,
        decision_recorder: DecisionRecorder | None = None,
    ) -> None:
        self._llm = llm
        # Optional decision-trace hook (PR-S3-2). When ``None``, no
        # observability emission — the domain layer stays free of
        # OTel coupling. Composition layer injects the real
        # ``record_decision`` callable; tests pass either a fake, the
        # real one, or leave it ``None`` to suppress emits.
        self._recorder = decision_recorder

    async def evaluate(self, tool_name: str, tool_args: dict[str, Any],
                       risk_level: str, matched_patterns: list[str],
                       task_context: str) -> str:
        """Returns 'approve', 'deny', or 'escalate'.

        B5 PR-S3-2: when a ``decision_recorder`` is injected, each
        outcome path records a ``decision.smart_approve`` event with
        ``decision_outcome`` and (for non-happy paths) a short
        ``decision_reason`` (``"unexpected_response"`` / ``"llm_error"``).
        Tool args are NEVER recorded — only ``tool_name`` rides as a
        canonical attribute for join. The recorder is called inside
        the existing ``try/except`` so any observability failure
        cannot taint the agent decision; the recorder itself MUST be
        best-effort (the OTel-backed ``record_decision`` is —
        ``infrastructure/observability/decision_trace.py``).
        """
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
                outcome = decision.lower()
                _safe_record(
                    self._recorder,
                    "smart_approve",
                    outcome=outcome,
                    attrs={"tool_name": tool_name},
                )
                return outcome
            logger.warning("SmartApprove unexpected response: %s, escalating", decision)
            _safe_record(
                self._recorder,
                "smart_approve",
                outcome="escalate",
                reason="unexpected_response",
                attrs={"tool_name": tool_name},
            )
            return "escalate"
        except Exception:
            logger.exception("SmartApprove failed, escalating")
            _safe_record(
                self._recorder,
                "smart_approve",
                outcome="escalate",
                reason="llm_error",
                attrs={"tool_name": tool_name},
            )
            return "escalate"
