"""[child SPAWN_ACK fix] Coordinator child runners must receive the lifespan
``SupervisorRegistry`` singleton through ``ChildRunnerSharedDeps``.

Live bug (2026-07-13): the child build chain (main.py ``_resolve_child_runner_deps``
→ ``_make_shared_child_runner_builder._build`` → ``AgentTaskRunner``) never
threaded ``supervisor_registry``, so every mailbox-plane child hit
``agent_task_runner._maybe_emit_spawn_ack_and_heartbeat``'s ``registry is None``
branch and SKIPPED SPAWN_ACK + heartbeat entirely ("child-side mailbox
publisher: registry is None … skipping SPAWN_ACK/heartbeat" warning) — the
parent-side supervisor then only sees the child via orphan_reconcile fallback.

Harness mirrors ``tests/app/test_d1a_root_child_same_sentinel.py``: REAL
builder + fake deps, identity assertions on the built runner.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from app.domain.models.app_config import (
    A2AConfig,
    AgentConfig,
    MCPConfig,
    ToolRuntimeConfig,
)
from app.interfaces.service_dependencies import (
    ChildRunnerSharedDeps,
    _make_shared_child_runner_builder,
)


def _child_deps(**overrides) -> ChildRunnerSharedDeps:
    base = dict(
        uow_factory=lambda: MagicMock(),
        llm=MagicMock(),
        agent_config=AgentConfig(),
        mcp_config=MCPConfig(),
        a2a_config=A2AConfig(),
        file_storage=MagicMock(),
        search_engine=MagicMock(),
        checkpointer_pool=MagicMock(),
        execution_supervisor=MagicMock(),
        tool_runtime=ToolRuntimeConfig(),
    )
    base.update(overrides)
    return ChildRunnerSharedDeps(**base)


def _build_child_runner(deps: ChildRunnerSharedDeps):
    builder = _make_shared_child_runner_builder(resolve_child_runner_deps=lambda: deps)
    return builder(
        session_id="c1",
        tool_filter=frozenset({"file_write"}),
        mailbox_publisher=MagicMock(),
        terminal_envelope_publisher_disabled=True,
        sandbox=MagicMock(),
        browser=MagicMock(),
        user_id="u1",
        cost_callback_handler=MagicMock(),
    )


def test_child_runner_receives_supervisor_registry_by_identity() -> None:
    """deps.supervisor_registry → runner._supervisor_registry (same instance),
    so the child-side SPAWN_ACK/heartbeat path stops skipping."""
    sentinel = MagicMock(name="SupervisorRegistry")
    runner = _build_child_runner(_child_deps(supervisor_registry=sentinel))
    assert runner._supervisor_registry is sentinel


def test_child_runner_registry_defaults_none_for_legacy_deps() -> None:
    """Un-populated deps (legacy/test constructions) keep today's behavior:
    no registry, child never invents one."""
    runner = _build_child_runner(_child_deps())
    assert runner._supervisor_registry is None
