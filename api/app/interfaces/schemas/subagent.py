"""Phase 1 minimal subagent research feature: request/response schemas.

Probe events extend BaseEvent with `type` field (NOT `event_type`) so
EventMapper (api/app/interfaces/schemas/event.py:613) can route them
via the standard event dispatch path. `id` field inherited from BaseEvent
satisfies CS3 invariant (ServerSentEvent.id == event.id).
"""
from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field

from app.domain.models.event import BaseEvent


class ChildOutcome(str, Enum):
    """Per spec § Child Outcome Enum.

    WAITING is defensive — should NOT occur with tool_filter enforcement
    (no tool can trigger ToolConfirmationEvent when tool_filter restricts
    to read-only allowlist). If detected, treat as FAILED upstream.
    """
    COMPLETED = "completed"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    WAITING = "waiting"
    CANCELLED = "cancelled"


class ResearchSubagentRequest(BaseModel):
    """POST /sessions/{parent_id}/subagents/research request body.

    Field constraints per spec Open Q #2: hard cap max_children=3.
    """
    prompts: list[str] = Field(min_length=1, max_length=3)
    max_children: int = Field(default=3, ge=1, le=3)


class ChildStartedEvent(BaseEvent):
    """Emitted once per child session when probe service begins fan-out."""
    type: Literal["child_started"] = "child_started"
    probe_run_id: str
    child_session_id: str
    prompt: str


class ChildDoneEvent(BaseEvent):
    """Emitted as each child completes (via asyncio.as_completed)."""
    type: Literal["child_done"] = "child_done"
    probe_run_id: str
    child_session_id: str
    outcome: ChildOutcome
    final_answer: Optional[str] = None
    transcript_tokens: int
    error_summary: Optional[str] = None


class DroppedChild(BaseModel):
    child_id: str
    outcome: str
    error_summary: Optional[str] = None


class JoinedSummaryEvent(BaseEvent):
    """Final probe event: integrated summary + multi-metric."""
    type: Literal["joined_summary"] = "joined_summary"
    probe_run_id: str
    summary: str
    summary_tokens: int
    completed_children: list[str]
    dropped_children: list[DroppedChild] = Field(default_factory=list)
    metrics: dict
    validation_warnings: list[str] = Field(default_factory=list)
