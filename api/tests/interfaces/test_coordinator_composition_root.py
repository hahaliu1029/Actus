"""C2 finish-core F1.7 — shared child-runner builder + composition-root rewiring.

The composition root no longer injects the *bare* ``AgentTaskRunner`` class into
``ChildAgentTaskRunnerFactory(runner_class=...)``. Instead it injects a
``ChildRunnerBuilder`` callable produced by ``_make_shared_child_runner_builder``
that:

* resolves the process-shared deps LAZILY (post-lifespan) from the live
  ``AgentService`` via the ``resolve_child_runner_deps`` closure, and
* derives a child-scoped ``agent_config`` whose ``tool_confirmation.enabled``
  is forced ``False`` (§5.1.7 — a lease-bound child must never block on human
  confirmation), without mutating the parent config, and
* passes ``coord_deps=None`` so the child is NOT itself a nested coordinator.
"""

from __future__ import annotations

from unittest.mock import MagicMock


def test_shared_runner_builder_constructs_full_child_runner_with_confirmation_off():
    """The shared runner builder resolves the process-shared deps LAZILY and
    builds a real AgentTaskRunner (11 required ctor args satisfied) with a
    child-scoped agent_config whose tool_confirmation.enabled is False (§5.1.7)."""
    from app.interfaces.service_dependencies import (
        _make_shared_child_runner_builder, ChildRunnerSharedDeps,
    )
    from app.domain.models.app_config import (
        AgentConfig, MCPConfig, A2AConfig, ToolRuntimeConfig,
    )

    parent_agent_config = AgentConfig()
    assert parent_agent_config.tool_confirmation.enabled is True

    # B1 spec R2#1: root's ToolRuntimeConfig threads through the child builder.
    sentinel_tool_runtime = ToolRuntimeConfig(tool_running_events_enabled=True)
    deps = ChildRunnerSharedDeps(
        uow_factory=lambda: MagicMock(), llm=MagicMock(),
        agent_config=parent_agent_config, mcp_config=MCPConfig(), a2a_config=A2AConfig(),
        file_storage=MagicMock(), search_engine=MagicMock(),
        checkpointer_pool=MagicMock(), execution_supervisor=MagicMock(),
        tool_runtime=sentinel_tool_runtime,
    )
    builder = _make_shared_child_runner_builder(resolve_child_runner_deps=lambda: deps)
    runner = builder(
        session_id="c1", tool_filter=frozenset({"file_write"}),
        mailbox_publisher=MagicMock(), terminal_envelope_publisher_disabled=True,
        sandbox=MagicMock(), browser=MagicMock(), user_id="u1",
        cost_callback_handler=MagicMock(),
    )
    assert runner._agent_config.tool_confirmation.enabled is False
    assert parent_agent_config.tool_confirmation.enabled is True  # parent untouched
    assert runner._coord_deps_for_planner is None  # child is NOT a nested coordinator
    # B1 spec R2#1: child runner received the exact ToolRuntimeConfig from deps
    # (agent_task_runner stores tool_runtime or ToolRuntimeConfig(); a non-None
    # sentinel is kept by identity, so child/root flags can't drift).
    assert runner._tool_runtime is sentinel_tool_runtime


def test_shared_runner_builder_threads_session_state_machine():
    """A4-1 §6: the child runner builder threads the SSM from ChildRunnerSharedDeps
    into the constructed AgentTaskRunner."""
    from app.interfaces.service_dependencies import (
        _make_shared_child_runner_builder, ChildRunnerSharedDeps,
    )
    from app.domain.models.app_config import (
        AgentConfig, MCPConfig, A2AConfig, ToolRuntimeConfig,
    )
    from app.domain.services.session.default_state_machine import (
        DefaultSessionStateMachine,
    )

    ssm = DefaultSessionStateMachine(uow_factory=lambda: None)
    sentinel_tool_runtime = ToolRuntimeConfig(tool_concurrency_enabled=True)
    deps = ChildRunnerSharedDeps(
        uow_factory=lambda: MagicMock(), llm=MagicMock(),
        agent_config=AgentConfig(), mcp_config=MCPConfig(), a2a_config=A2AConfig(),
        file_storage=MagicMock(), search_engine=MagicMock(),
        checkpointer_pool=MagicMock(), execution_supervisor=MagicMock(),
        session_state_machine=ssm,
        tool_runtime=sentinel_tool_runtime,
    )
    builder = _make_shared_child_runner_builder(resolve_child_runner_deps=lambda: deps)
    runner = builder(
        session_id="c1", tool_filter=frozenset({"file_write"}),
        mailbox_publisher=MagicMock(), terminal_envelope_publisher_disabled=True,
        sandbox=MagicMock(), browser=MagicMock(), user_id="u1",
        cost_callback_handler=MagicMock(),
    )
    assert runner._session_state_machine is ssm
    assert runner._tool_runtime is sentinel_tool_runtime  # B1 spec R2#1
