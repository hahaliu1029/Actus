"""C2 v1 ChildAgentTaskRunnerFactory (spec §5.3 + §8.5.1).

PR-2 skeleton: surface interface only. PR-4 wires actual construction
with terminal_envelope_publisher_disabled=True + cancel_event injection.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import asyncio

    from app.domain.services.agent_task_runner import AgentTaskRunner
    from app.domain.services.permission.child_permission_context import (
        ChildPermissionContext,
    )


class ChildAgentTaskRunnerFactory:
    """Builds restricted AgentTaskRunner for coordinator_step children."""

    def __init__(self) -> None:
        pass

    async def build(
        self,
        *,
        child_session_id: str,
        child_permission_context: "ChildPermissionContext",
        tool_filter_preset: str,
        cancel_event: "asyncio.Event",
    ) -> "AgentTaskRunner":
        """[C2 PR-4 stub] Wires the actual AgentTaskRunner construction in PR-4."""
        raise NotImplementedError("PR-4 territory")
