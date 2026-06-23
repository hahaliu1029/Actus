"""C2-full S2 Task 5.3 — bind-time shell-union in ChildAgentTaskRunnerFactory.

Spec ref: §3.5 (bind-time dual-loosening).

When a coordinator child runs shell-mode (master flag
``is_coordinator_shell_mode_enabled()`` ON **AND** its
``child_permission_context.shell_mode`` True), ``build`` unions the 5
raw-shell tools into the resolved ``coordinator_step`` allowlist so
``llm.bind_tools`` exposes them. Flag-off (the default) OR shell_mode-off ⇒
byte-for-byte unchanged behavior (base preset only).

This is a lockstep invariant with the runtime gate un-block
(child_scope_gate): the 5-tool set here must match the gate's
``SHELL_HARD_BLOCKED_NAMES``.
"""
import asyncio
from unittest.mock import MagicMock

import pytest

from app.application.services import child_agent_runner_factory as factory_mod
from app.application.services.child_agent_runner_factory import (
    ChildAgentTaskRunnerFactory,
)
from app.domain.models.tool_filter_presets import COORDINATOR_STEP_PRESET
from app.domain.services.permission.child_permission_context import (
    ChildBudget, ChildPermissionContext, SpawnManifest,
)
from app.domain.services.tool_filter_presets import (
    COORDINATOR_STEP_BASE_ALLOWED_TOOLS,
    SUBAGENT_RESEARCH_ALLOWED_TOOLS,
)

anyio_backend = "asyncio"
pytestmark = pytest.mark.anyio

_SHELL_FIVE = frozenset({
    "shell_execute", "shell_wait_process", "shell_kill_process",
    "shell_write_input", "shell_read_output",
})


class _CaptureRunner:
    last_kwargs: dict = {}

    def __init__(self, **kwargs):
        type(self).last_kwargs = kwargs


class _FakeTask:
    def __init__(self, *a, **k):
        pass


def _cpc(*, shell_mode: bool) -> ChildPermissionContext:
    return ChildPermissionContext(
        parent_session_id="p1", child_session_id="c1", coordinator_run_id="r1",
        work_unit_id="wu1",
        spawn_manifest=SpawnManifest(
            allowed_tools=frozenset({"file_read"}), path_leases=(),
            runtime_caps=frozenset(), tree_leases=(), shell_mode=shell_mode,
        ),
        session_mode_revision=1,
        budget=ChildBudget(max_tool_calls=10, max_token_cost_usd=1.0,
                           max_wallclock_seconds=600),
        shell_mode=shell_mode,
    )


def _factory() -> ChildAgentTaskRunnerFactory:
    return ChildAgentTaskRunnerFactory(
        runner_class=_CaptureRunner,
        mailbox_publisher=MagicMock(),
        task_cls=_FakeTask,
    )


async def _build(*, shell_mode: bool):
    f = _factory()
    return await f.build(
        child_session_id="c1",
        child_permission_context=_cpc(shell_mode=shell_mode),
        tool_filter_preset=COORDINATOR_STEP_PRESET,
        cancel_event=asyncio.Event(),
        sandbox=MagicMock(),
        browser=MagicMock(),
        user_id="u1",
        cost_callback_handler=MagicMock(),
    )


async def test_flag_off_no_shell_union(monkeypatch):
    monkeypatch.setattr(factory_mod, "is_coordinator_shell_mode_enabled",
                        lambda: False)
    await _build(shell_mode=True)
    bound = _CaptureRunner.last_kwargs["tool_filter"]
    assert bound == COORDINATOR_STEP_BASE_ALLOWED_TOOLS
    assert not (_SHELL_FIVE & bound)


async def test_shell_mode_off_no_shell_union(monkeypatch):
    monkeypatch.setattr(factory_mod, "is_coordinator_shell_mode_enabled",
                        lambda: True)
    await _build(shell_mode=False)
    bound = _CaptureRunner.last_kwargs["tool_filter"]
    assert bound == COORDINATOR_STEP_BASE_ALLOWED_TOOLS


async def test_flag_on_and_shell_mode_unions_shell(monkeypatch):
    monkeypatch.setattr(factory_mod, "is_coordinator_shell_mode_enabled",
                        lambda: True)
    await _build(shell_mode=True)
    bound = _CaptureRunner.last_kwargs["tool_filter"]
    assert _SHELL_FIVE <= bound
    assert COORDINATOR_STEP_BASE_ALLOWED_TOOLS <= bound
    # union ONLY adds the 5 shell tools, nothing else.
    assert bound == COORDINATOR_STEP_BASE_ALLOWED_TOOLS | _SHELL_FIVE


async def test_non_coordinator_preset_no_shell_union_even_flag_on(monkeypatch):
    # [codex PR-5 R1 P1] The shell-mode widen is SCOPED to the coordinator_step
    # preset. A non-coordinator preset (subagent_research) with shell_mode=True
    # AND the flag ON must NOT gain the exec/control shell union — self-defends
    # the reusable factory boundary. subagent_research carries shell_read_output
    # by its own design but NONE of the 4 exec/control shell tools, so the
    # equality below FAILS if the preset gate is removed (union would add them).
    monkeypatch.setattr(factory_mod, "is_coordinator_shell_mode_enabled",
                        lambda: True)
    f = _factory()
    await f.build(
        child_session_id="c1",
        child_permission_context=_cpc(shell_mode=True),
        tool_filter_preset="subagent_research",
        cancel_event=asyncio.Event(),
        sandbox=MagicMock(),
        browser=MagicMock(),
        user_id="u1",
        cost_callback_handler=MagicMock(),
    )
    bound = _CaptureRunner.last_kwargs["tool_filter"]
    assert bound == SUBAGENT_RESEARCH_ALLOWED_TOOLS  # byte-for-byte: no union
    assert "shell_execute" not in bound
    assert "shell_kill_process" not in bound


def test_shell_set_lockstep_gate_factory_resolver():
    # [codex PR-5 R1 P2] Pin the 5-tool shell set across ALL THREE sites so a
    # future edit to one cannot silently drift: factory union == gate un-block
    # == resolver canonical "shell" category; and the set is a subset of the
    # children hard-block. The other tests duplicate the local _SHELL_FIVE
    # literal; THIS test ties that literal to the live production sources.
    from app.domain.services.permission.child_scope_gate import (
        HARD_BLOCKED_FOR_CHILDREN,
        SHELL_HARD_BLOCKED_NAMES,
    )
    from app.domain.services.tools.tool_source_resolver import (
        _CANONICAL_TOOL_IDENTITIES,
    )

    resolver_shell = frozenset(
        name
        for name, (_src, cat) in _CANONICAL_TOOL_IDENTITIES.items()
        if cat == "shell"
    )
    assert factory_mod._SHELL_MODE_UNION_TOOLS == _SHELL_FIVE
    assert factory_mod._SHELL_MODE_UNION_TOOLS == SHELL_HARD_BLOCKED_NAMES
    assert factory_mod._SHELL_MODE_UNION_TOOLS == resolver_shell
    assert factory_mod._SHELL_MODE_UNION_TOOLS <= HARD_BLOCKED_FOR_CHILDREN
