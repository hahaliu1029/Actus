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
    from app.domain.models.app_config import AgentConfig, MCPConfig, A2AConfig

    parent_agent_config = AgentConfig()
    assert parent_agent_config.tool_confirmation.enabled is True

    deps = ChildRunnerSharedDeps(
        uow_factory=lambda: MagicMock(), llm=MagicMock(),
        agent_config=parent_agent_config, mcp_config=MCPConfig(), a2a_config=A2AConfig(),
        file_storage=MagicMock(), search_engine=MagicMock(),
        checkpointer_pool=MagicMock(), execution_supervisor=MagicMock(),
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
