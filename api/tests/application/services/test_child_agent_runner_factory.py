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

from app.application.services.agent_task_runner_invoke_adapter import (
    AgentTaskRunnerInvokeAdapter,
)
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
        task_cls=MagicMock(),
    )
    ce = asyncio.Event()
    built = await factory.build(
        child_session_id="c1",
        child_permission_context=_mk_cctx(),
        tool_filter_preset="coordinator_step",
        cancel_event=ce,
        sandbox=MagicMock(),
        browser=MagicMock(),
        user_id="u1",
        cost_callback_handler=MagicMock(),
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
        task_cls=MagicMock(),
    )
    ce = asyncio.Event()
    await factory.build(
        child_session_id="c1",
        child_permission_context=_mk_cctx(),
        tool_filter_preset="subagent_research",
        cancel_event=ce,
        sandbox=MagicMock(),
        browser=MagicMock(),
        user_id="u1",
        cost_callback_handler=MagicMock(),
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
        task_cls=MagicMock(),
    )
    await factory.build(
        child_session_id="c1",
        child_permission_context=_mk_cctx(),
        tool_filter_preset="coordinator_step",
        cancel_event=asyncio.Event(),
        sandbox=MagicMock(),
        browser=MagicMock(),
        user_id="u1",
        cost_callback_handler=MagicMock(),
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
        task_cls=MagicMock(),
    )
    with pytest.raises(ValueError):
        await factory.build(
            child_session_id="c1",
            child_permission_context=_mk_cctx(),
            tool_filter_preset="totally_made_up_preset",
            cancel_event=asyncio.Event(),
            sandbox=MagicMock(),
            browser=MagicMock(),
            user_id="u1",
            cost_callback_handler=MagicMock(),
        )


async def test_built_wrapper_carries_runtime_deps() -> None:
    """[§8.4 + §5.3] cancel_event + child_permission_context flow through
    the wrapper, NOT through AgentTaskRunner.__init__. CoordinatorChildRunner
    consumes them to assemble config + run finalizers."""
    runner_class = MagicMock(name="AgentTaskRunner")
    runner_instance = MagicMock(name="runner-inst")
    runner_instance.set_coordinator_cancel_event = MagicMock()
    runner_class.return_value = runner_instance
    factory = ChildAgentTaskRunnerFactory(
        runner_class=runner_class, mailbox_publisher=MagicMock(),
        task_cls=MagicMock(),
    )
    ce = asyncio.Event()
    cctx = _mk_cctx()
    built = await factory.build(
        child_session_id="c1",
        child_permission_context=cctx,
        tool_filter_preset="coordinator_step",
        cancel_event=ce,
        sandbox=MagicMock(),
        browser=MagicMock(),
        user_id="u1",
        cost_callback_handler=MagicMock(),
    )
    # F1.3: build wraps the raw runner in the invoke-adapter; built.runner is
    # the adapter, not the raw runner instance. The raw runner is reachable as
    # the adapter's ._runner and received the cancel-event wiring.
    assert isinstance(built.runner, AgentTaskRunnerInvokeAdapter)
    assert built.runner._runner is runner_instance
    runner_instance.set_coordinator_cancel_event.assert_called_once_with(ce)
    assert built.cancel_event is ce
    assert built.child_permission_context is cctx
    assert built.terminal_envelope_publisher_disabled is True

    kw = runner_class.call_args.kwargs
    assert "cancel_event" not in kw
    assert "child_permission_context" not in kw


async def test_build_threads_coordinator_metrics_recorder_to_adapter() -> None:
    """[C2b rollout WS1b Task 2.3] build(..., coordinator_metrics_recorder=rec)
    passes the recorder into the adapter ctor so _drain can record tool_calls."""
    runner_class = MagicMock(name="AgentTaskRunner")
    runner_instance = MagicMock(name="runner-inst")
    runner_instance.set_coordinator_cancel_event = MagicMock()
    runner_class.return_value = runner_instance
    factory = ChildAgentTaskRunnerFactory(
        runner_class=runner_class, mailbox_publisher=MagicMock(),
        task_cls=MagicMock(),
    )
    rec = MagicMock(name="CoordinatorMetricsRecorder")
    built = await factory.build(
        child_session_id="c1",
        child_permission_context=_mk_cctx(),
        tool_filter_preset="coordinator_step",
        cancel_event=asyncio.Event(),
        sandbox=MagicMock(),
        browser=MagicMock(),
        user_id="u1",
        cost_callback_handler=MagicMock(),
        coordinator_metrics_recorder=rec,
    )
    assert isinstance(built.runner, AgentTaskRunnerInvokeAdapter)
    assert built.runner._coordinator_metrics_recorder is rec


def test_factory_ctor_stores_deps() -> None:
    """Smoke: ctor stores runner_class + mailbox_publisher as private fields
    so repeated build() calls reuse the same dependencies."""
    runner_class = MagicMock()
    publisher = MagicMock()
    task_cls = MagicMock()
    factory = ChildAgentTaskRunnerFactory(
        runner_class=runner_class, mailbox_publisher=publisher,
        task_cls=task_cls,
    )
    assert getattr(factory, "_runner_class") is runner_class
    assert getattr(factory, "_mailbox_publisher") is publisher
    assert getattr(factory, "_task_cls") is task_cls


async def test_build_forwards_per_child_deps_and_wraps_in_adapter():
    import asyncio
    from unittest.mock import MagicMock
    from app.application.services.child_agent_runner_factory import (
        ChildAgentTaskRunnerFactory,
    )
    from app.application.services.agent_task_runner_invoke_adapter import (
        AgentTaskRunnerInvokeAdapter,
    )

    raw_runner = MagicMock()
    raw_runner.set_coordinator_cancel_event = MagicMock()
    builder = MagicMock(return_value=raw_runner)  # the shared runner builder
    fake_task_cls = MagicMock()
    factory = ChildAgentTaskRunnerFactory(
        runner_class=builder, mailbox_publisher=MagicMock(), task_cls=fake_task_cls,
    )
    ce = asyncio.Event()
    sandbox = MagicMock()
    browser = MagicMock()
    cost_handler = MagicMock()
    built = await factory.build(
        child_session_id="c1",
        child_permission_context=_mk_cctx(),
        tool_filter_preset="coordinator_step",
        cancel_event=ce,
        sandbox=sandbox,
        browser=browser,
        user_id="u1",
        cost_callback_handler=cost_handler,
    )
    # builder received the per-child deps
    kw = builder.call_args.kwargs
    assert kw["sandbox"] is sandbox
    assert kw["browser"] is browser
    assert kw["user_id"] == "u1"
    assert kw["cost_callback_handler"] is cost_handler
    assert kw["terminal_envelope_publisher_disabled"] is True
    # returned runner is the invoke-adapter, not the raw runner
    assert isinstance(built.runner, AgentTaskRunnerInvokeAdapter)
    raw_runner.set_coordinator_cancel_event.assert_called_once_with(ce)
