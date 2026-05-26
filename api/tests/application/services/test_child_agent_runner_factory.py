"""C2 PR-4 Task 4.6 — ChildAgentTaskRunnerFactory full-build tests.

Spec ref: §5.3 + §8.5.1.

The factory wraps ``AgentTaskRunner`` construction for coordinator-step
children. It:
- Resolves the tool filter from ``tool_filter_preset`` via
  ``resolve_preset`` (raises ValueError on unknown name — fail closed).
- Sets ``terminal_envelope_publisher_disabled=True`` iff
  ``tool_filter_preset == "coordinator_step"`` (the only preset that
  routes through ``CoordinatorChildRunner._finalize_*`` as the sole
  terminal envelope publisher per spec §8.5.1 r6 P0-1).
- Returns a ``BuiltChildRunner`` wrapper carrying the inner
  ``AgentTaskRunner`` + the ``cancel_event`` + ``child_permission_context``
  so ``CoordinatorChildRunner`` (Task 4.7) can thread cancel_event into
  ``config["configurable"]["cancel_event"]`` at invocation time.

The wrapper is needed because the live ``AgentTaskRunner`` ctor (~40 kwargs)
does NOT accept ``cancel_event`` / ``child_permission_context``. Those are
runtime concerns plumbed via graph config, NOT runner-construction concerns.
Wrapping keeps the AgentTaskRunner surface untouched.
"""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from app.application.services.child_agent_runner_factory import (
    BuiltChildRunner,
    ChildAgentTaskRunnerFactory,
)


pytestmark = pytest.mark.anyio


def _mk_cctx() -> object:
    """Return a synthetic ChildPermissionContext sentinel. The factory does
    NOT introspect it; it just carries it to the wrapper for downstream."""
    cctx = MagicMock(name="ChildPermissionContext")
    cctx.child_session_id = "c1"
    cctx.parent_session_id = "p1"
    cctx.coordinator_run_id = "r1"
    cctx.work_unit_id = "wu1"
    return cctx


async def test_build_coordinator_step_disables_terminal_publisher() -> None:
    """[§8.5.1 r6 P0-1] tool_filter_preset='coordinator_step' →
    terminal_envelope_publisher_disabled=True. CoordinatorChildRunner is
    the sole terminal publisher for this preset."""
    runner_class = MagicMock(name="AgentTaskRunner")
    publisher = MagicMock(name="MailboxPublisher")
    factory = ChildAgentTaskRunnerFactory(
        runner_class=runner_class, mailbox_publisher=publisher,
    )
    ce = asyncio.Event()
    built = await factory.build(
        child_session_id="c1",
        child_permission_context=_mk_cctx(),
        tool_filter_preset="coordinator_step",
        cancel_event=ce,
    )
    assert isinstance(built, BuiltChildRunner)
    runner_class.assert_called_once()
    kw = runner_class.call_args.kwargs
    assert kw["terminal_envelope_publisher_disabled"] is True
    assert kw["session_id"] == "c1"
    assert kw["mailbox_publisher"] is publisher


async def test_build_subagent_research_keeps_default_publisher() -> None:
    """[§8.5.1 r6 P0-1 negative] tool_filter_preset='subagent_research' →
    disabled=False (default behavior, AgentTaskRunner publishes its own
    terminal envelope). Only coordinator_step gets the gate."""
    runner_class = MagicMock(name="AgentTaskRunner")
    publisher = MagicMock(name="MailboxPublisher")
    factory = ChildAgentTaskRunnerFactory(
        runner_class=runner_class, mailbox_publisher=publisher,
    )
    ce = asyncio.Event()
    await factory.build(
        child_session_id="c1",
        child_permission_context=_mk_cctx(),
        tool_filter_preset="subagent_research",
        cancel_event=ce,
    )
    kw = runner_class.call_args.kwargs
    assert kw["terminal_envelope_publisher_disabled"] is False


async def test_build_resolves_tool_filter_from_preset() -> None:
    """[§5.3] resolved tool_filter (frozenset) is forwarded to runner ctor
    as ``tool_filter``. Verifies that the factory does NOT pass the raw
    preset name string by mistake."""
    runner_class = MagicMock(name="AgentTaskRunner")
    factory = ChildAgentTaskRunnerFactory(
        runner_class=runner_class, mailbox_publisher=MagicMock(),
    )
    await factory.build(
        child_session_id="c1",
        child_permission_context=_mk_cctx(),
        tool_filter_preset="coordinator_step",
        cancel_event=asyncio.Event(),
    )
    kw = runner_class.call_args.kwargs
    tf = kw["tool_filter"]
    assert isinstance(tf, frozenset)
    assert tf, "tool_filter must be a non-empty allowlist for coordinator_step"


async def test_build_unknown_preset_raises_value_error() -> None:
    """[§5.3 fail-closed] resolve_preset raises on unknown preset; the
    factory MUST propagate, not swallow. Silently defaulting to None would
    open a least-privilege escape (no tool_filter == full allowlist)."""
    runner_class = MagicMock(name="AgentTaskRunner")
    factory = ChildAgentTaskRunnerFactory(
        runner_class=runner_class, mailbox_publisher=MagicMock(),
    )
    with pytest.raises(ValueError):
        await factory.build(
            child_session_id="c1",
            child_permission_context=_mk_cctx(),
            tool_filter_preset="totally_made_up_preset",
            cancel_event=asyncio.Event(),
        )


async def test_built_wrapper_carries_runtime_deps() -> None:
    """[§8.4 + §5.3] cancel_event + child_permission_context flow through
    the wrapper, NOT through AgentTaskRunner.__init__. CoordinatorChildRunner
    consumes them to assemble config + run finalizers."""
    runner_class = MagicMock(name="AgentTaskRunner")
    runner_instance = MagicMock(name="runner-inst")
    runner_class.return_value = runner_instance
    factory = ChildAgentTaskRunnerFactory(
        runner_class=runner_class, mailbox_publisher=MagicMock(),
    )
    ce = asyncio.Event()
    cctx = _mk_cctx()
    built = await factory.build(
        child_session_id="c1",
        child_permission_context=cctx,
        tool_filter_preset="coordinator_step",
        cancel_event=ce,
    )
    assert built.runner is runner_instance
    assert built.cancel_event is ce
    assert built.child_permission_context is cctx
    assert built.terminal_envelope_publisher_disabled is True

    kw = runner_class.call_args.kwargs
    assert "cancel_event" not in kw
    assert "child_permission_context" not in kw


def test_factory_ctor_stores_deps() -> None:
    """Smoke: ctor stores runner_class + mailbox_publisher as private fields
    so repeated build() calls reuse the same dependencies."""
    runner_class = MagicMock()
    publisher = MagicMock()
    factory = ChildAgentTaskRunnerFactory(
        runner_class=runner_class, mailbox_publisher=publisher,
    )
    assert getattr(factory, "_runner_class") is runner_class
    assert getattr(factory, "_mailbox_publisher") is publisher
