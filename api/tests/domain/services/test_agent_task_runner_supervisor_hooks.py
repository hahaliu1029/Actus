"""C3 PR-3c — Unit tests for ``AgentTaskRunner`` mailbox supervisor hooks
(plan §6.5 / agent_task_runner.py lines 3305-3383).

Targets three small but load-bearing helper methods added in PR-3c:

* ``_is_root_session`` — single-shot UoW lookup of ``Session.worker_type``
  with lazy caching + DB-failure-safe ``False`` fallback.
* ``_maybe_spawn_mailbox_supervisor`` — best-effort spawn hook, called
  right after the runner flips the session to RUNNING. Multiple guards:
  registry-injection / feature-flag / root-session-only.
* ``_maybe_stop_mailbox_supervisor`` — best-effort stop hook, called
  from terminal-status paths. Only gated by registry-injection +
  root-session — the flag is NOT re-checked so a flag-flip during a
  session still cleans up the slot that was spawned earlier.

Pure unit tests — we instantiate ``AgentTaskRunner.__new__`` and inject
just the attrs each helper reads, so we don't need a real LLM, sandbox,
event bus, or DB. This keeps the test focused on the PR-3c surface
without coupling to the wider runner constructor (which takes 20+
dependencies and is exercised by the integration suite).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from app.domain.services.agent_task_runner import AgentTaskRunner


# ---- Stubs ------------------------------------------------------------------


class _StubSessionRepo:
    def __init__(self, session_obj: Any) -> None:
        self.session_obj = session_obj
        self.calls: list[str] = []

    async def get_by_id(self, session_id: str) -> Any:
        self.calls.append(session_id)
        return self.session_obj


class _StubUow:
    def __init__(self, session_repo: _StubSessionRepo) -> None:
        self.session = session_repo

    async def __aenter__(self) -> "_StubUow":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None


def _uow_factory(session_repo: _StubSessionRepo):
    def _factory() -> _StubUow:
        return _StubUow(session_repo)
    return _factory


def _failing_uow_factory():
    """A UoW factory that raises on entry — exercises the DB-down branch
    of ``_is_root_session``."""
    class _ExplodingUow:
        async def __aenter__(self) -> "_ExplodingUow":
            raise RuntimeError("synthetic DB outage")

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

    return lambda: _ExplodingUow()


class _StubRegistry:
    def __init__(self) -> None:
        self.spawn_calls: list[str] = []
        self.stop_calls: list[str] = []
        self.spawn_raises: bool = False
        self.stop_raises: bool = False

    async def spawn(self, root_id: str) -> None:
        self.spawn_calls.append(root_id)
        if self.spawn_raises:
            raise RuntimeError("synthetic spawn failure")

    async def stop(self, root_id: str) -> None:
        self.stop_calls.append(root_id)
        if self.stop_raises:
            raise RuntimeError("synthetic stop failure")

    async def health_check(self) -> dict[str, str]:
        return {}


class _NoSession:
    """Sentinel — pass to ``session_obj_override`` to simulate a session
    row that was deleted between RUNNING-transition and the helper call."""


def _make_runner(
    *,
    session_id: str = "sid-1",
    worker_type: str = "root",
    supervisor_registry: _StubRegistry | None = None,
    mailbox_supervisor_enabled: bool = True,
    cached_is_root: bool | None = None,
    uow_factory: Any = None,
    session_obj_override: Any = None,
) -> AgentTaskRunner:
    """Construct a bare runner with just the attrs PR-3c hooks read.

    We bypass ``__init__`` because it requires a 20-dep dependency
    graph; the hooks under test only touch a small surface. The trade-off
    is acknowledged: changes to the helper bodies that read NEW attrs
    will require updating this fixture. The PR-3c hooks are small +
    explicit so the risk is bounded.
    """
    runner = AgentTaskRunner.__new__(AgentTaskRunner)
    runner._session_id = session_id  # type: ignore[attr-defined]
    runner._supervisor_registry = supervisor_registry  # type: ignore[attr-defined]
    runner._mailbox_supervisor_enabled = mailbox_supervisor_enabled  # type: ignore[attr-defined]
    runner._cached_is_root_session = cached_is_root  # type: ignore[attr-defined]
    runner._supervisor_spawned = False  # type: ignore[attr-defined]

    session_obj: Any
    if session_obj_override is _NoSession:
        session_obj = None
    elif session_obj_override is not None:
        session_obj = session_obj_override
    else:
        # Tiny mock with the worker_type attribute the helper reads.
        session_obj = MagicMock()
        session_obj.worker_type = worker_type

    if uow_factory is None:
        runner._uow_factory = _uow_factory(_StubSessionRepo(session_obj))  # type: ignore[attr-defined]
    else:
        runner._uow_factory = uow_factory  # type: ignore[attr-defined]
    return runner


# ---- _is_root_session -------------------------------------------------------


@pytest.mark.anyio
class TestIsRootSession:
    async def test_returns_true_for_root_worker_type(self) -> None:
        runner = _make_runner(worker_type="root")
        assert await runner._is_root_session() is True
        # Cached after first lookup.
        assert runner._cached_is_root_session is True

    async def test_returns_false_for_subagent_worker_type(self) -> None:
        runner = _make_runner(worker_type="subagent")
        assert await runner._is_root_session() is False
        assert runner._cached_is_root_session is False

    async def test_cached_short_circuits_uow_lookup(self) -> None:
        repo = _StubSessionRepo(MagicMock(worker_type="root"))
        runner = _make_runner(
            uow_factory=_uow_factory(repo),
            cached_is_root=True,
        )
        result = await runner._is_root_session()
        assert result is True
        # Cache hit → no UoW lookup ran.
        assert repo.calls == []

    async def test_missing_session_row_treated_as_non_root(self) -> None:
        """If the session row vanished between RUNNING transition and the
        hook firing, the helper returns False — safer than spawning a
        supervisor for an unknown root."""
        runner = _make_runner(session_obj_override=_NoSession)
        assert await runner._is_root_session() is False

    async def test_db_outage_treated_as_non_root(self) -> None:
        """UoW factory raising → ``False`` + warning logged.

        Behavior contract from agent_task_runner.py:3313 — supervisor
        wiring is best-effort plumbing, never a precondition.
        """
        runner = _make_runner(uow_factory=_failing_uow_factory())
        assert await runner._is_root_session() is False
        assert runner._cached_is_root_session is False


# ---- _maybe_spawn_mailbox_supervisor ----------------------------------------


@pytest.mark.anyio
class TestMaybeSpawnSupervisor:
    async def test_spawns_on_root_when_enabled(self) -> None:
        reg = _StubRegistry()
        runner = _make_runner(
            session_id="root-1",
            worker_type="root",
            supervisor_registry=reg,
            mailbox_supervisor_enabled=True,
        )
        await runner._maybe_spawn_mailbox_supervisor()
        assert reg.spawn_calls == ["root-1"]
        assert runner._supervisor_spawned is True

    async def test_noop_when_registry_none(self) -> None:
        runner = _make_runner(supervisor_registry=None)
        await runner._maybe_spawn_mailbox_supervisor()
        # _is_root_session was NEVER queried — the helper short-circuits
        # on the registry-missing gate first.
        assert runner._cached_is_root_session is None

    async def test_noop_when_flag_disabled(self) -> None:
        reg = _StubRegistry()
        runner = _make_runner(
            supervisor_registry=reg, mailbox_supervisor_enabled=False
        )
        await runner._maybe_spawn_mailbox_supervisor()
        assert reg.spawn_calls == []
        # Flag-gate short-circuit also avoids the UoW lookup.
        assert runner._cached_is_root_session is None

    async def test_noop_for_subagent_session(self) -> None:
        reg = _StubRegistry()
        runner = _make_runner(
            session_id="child-1",
            worker_type="subagent",
            supervisor_registry=reg,
            mailbox_supervisor_enabled=True,
        )
        await runner._maybe_spawn_mailbox_supervisor()
        assert reg.spawn_calls == []
        assert runner._supervisor_spawned is False

    async def test_spawn_failure_is_swallowed(self) -> None:
        """Spawn failure must NOT propagate — the agent loop runs
        without a supervisor; reconcile_orphans retries on next pod."""
        reg = _StubRegistry()
        reg.spawn_raises = True
        runner = _make_runner(
            session_id="root-x",
            worker_type="root",
            supervisor_registry=reg,
            mailbox_supervisor_enabled=True,
        )
        # Must not raise.
        await runner._maybe_spawn_mailbox_supervisor()
        assert reg.spawn_calls == ["root-x"]
        # _supervisor_spawned stays False because the exception fired
        # before the flag was set.
        assert runner._supervisor_spawned is False


# ---- _maybe_stop_mailbox_supervisor -----------------------------------------


@pytest.mark.anyio
class TestMaybeStopSupervisor:
    async def test_stops_on_root_unconditionally(self) -> None:
        """Plan §6.5: stop hook does NOT re-check the feature flag — a
        flag flip during a session still cleans up the slot."""
        reg = _StubRegistry()
        runner = _make_runner(
            session_id="root-1",
            worker_type="root",
            supervisor_registry=reg,
            mailbox_supervisor_enabled=False,  # flag flipped off mid-session
        )
        await runner._maybe_stop_mailbox_supervisor()
        assert reg.stop_calls == ["root-1"]

    async def test_noop_when_registry_none(self) -> None:
        runner = _make_runner(supervisor_registry=None)
        await runner._maybe_stop_mailbox_supervisor()
        # No crash; nothing else to assert.

    async def test_noop_for_subagent_session(self) -> None:
        reg = _StubRegistry()
        runner = _make_runner(
            worker_type="subagent",
            supervisor_registry=reg,
        )
        await runner._maybe_stop_mailbox_supervisor()
        assert reg.stop_calls == []

    async def test_stop_failure_is_swallowed(self) -> None:
        reg = _StubRegistry()
        reg.stop_raises = True
        runner = _make_runner(
            session_id="root-y",
            worker_type="root",
            supervisor_registry=reg,
        )
        # Must not raise.
        await runner._maybe_stop_mailbox_supervisor()
        assert reg.stop_calls == ["root-y"]


# ---- Hook call-site enforcement --------------------------------------------
#
# codex r2 [HIGH TEST] — integration tests for the lifecycle wiring are
# auto-skipped in PR-3c pending PR-4 harness fixtures, so the production
# call sites (``invoke`` spawning, ``_set_terminal_status_with_notifications``
# stopping) had no CI gate. These behavioral checks AST-scan the runner
# source at module import time so a future PR that accidentally drops the
# spawn/stop hook call from those methods fails fast — without needing the
# full DB+Redis stack.
#
# AST scan vs. mock-and-call: the runner's ``invoke`` is 600+ lines of
# orchestration, exercising it end-to-end would require the 20-arg ctor
# wiring. The hook contract is "the call SITE exists", not "the call
# happens to fire under condition X" — that latter is what the unit tests
# above already cover via direct helper invocation. So a structural CI
# gate is the right tool: it pins the wiring even if PR-4 reshuffles
# orchestration logic.


import ast
import pathlib


_RUNNER_SOURCE = pathlib.Path(
    __file__
).parent.parent.parent.parent.parent / "api/app/domain/services/agent_task_runner.py"
# Fall back to a cwd-anchored discovery if the parents-walk is wrong (e.g.
# running under pytest-cov rewrite); resolve via the imported class.
if not _RUNNER_SOURCE.exists():
    import inspect

    _RUNNER_SOURCE = pathlib.Path(inspect.getsourcefile(AgentTaskRunner) or "")


def _find_method(tree: ast.Module, class_name: str, method_name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if (
                    isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and item.name == method_name
                ):
                    return item
    raise LookupError(f"{class_name}.{method_name} not found")


def _find_nested(parent: ast.AST, name: str) -> ast.AsyncFunctionDef | ast.FunctionDef:
    """Locate an inner function ``name`` defined anywhere under ``parent``.

    Used to find ``_terminal_op`` inside ``_set_terminal_status``.
    """
    for node in ast.walk(parent):
        if (
            isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef))
            and node.name == name
        ):
            return node
    raise LookupError(f"nested function {name!r} not found")


def _calls(method: ast.AST, attr_name: str) -> list[ast.Call]:
    """Return all AST Call nodes that look like ``self.<attr_name>(...)``."""
    calls = []
    for node in ast.walk(method):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == attr_name:
                if isinstance(func.value, ast.Name) and func.value.id == "self":
                    calls.append(node)
    return calls


def _lineno_of_first_call(method: ast.AST, attr_name: str) -> int | None:
    matches = _calls(method, attr_name)
    return matches[0].lineno if matches else None


def _lineno_of_first_update_to_running(method: ast.AST) -> int | None:
    """Find the lineno of the first ``self._uow.session.update_status(...,
    SessionStatus.RUNNING)`` call (or equivalent), so we can assert the
    spawn hook fires AFTER the runner has flipped the session to RUNNING.

    Match heuristic: any Call whose func attr is ``update_status`` AND
    whose arglist contains an ``ast.Attribute`` with ``attr == "RUNNING"``.
    Robust against future signature reshuffles that keep the same surface.
    """
    for node in ast.walk(method):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Attribute) and f.attr == "update_status":
                for arg in node.args:
                    if isinstance(arg, ast.Attribute) and arg.attr == "RUNNING":
                        return node.lineno
    return None


class TestHookCallSiteEnforcement:
    """CI-enforced AST checks: production call sites MUST keep invoking
    the PR-3c hooks in the right order + the right shield envelope.
    Drops a clear failure if PR-4 / future PRs reshuffle orchestration
    without preserving the lifecycle contract.
    """

    def test_invoke_calls_maybe_spawn_mailbox_supervisor(self) -> None:
        tree = ast.parse(_RUNNER_SOURCE.read_text())
        invoke = _find_method(tree, "AgentTaskRunner", "invoke")
        calls = _calls(invoke, "_maybe_spawn_mailbox_supervisor")
        assert calls, (
            "AgentTaskRunner.invoke must call self._maybe_spawn_mailbox_supervisor() "
            "after RUNNING transition — PR-3c lifecycle wiring contract"
        )

    def test_invoke_spawns_after_running_transition(self) -> None:
        """codex r3 [HIGH TEST] — the spawn must come AFTER the RUNNING
        status flip; otherwise the supervisor might process envelopes
        for a session that's still PENDING in DB.
        """
        tree = ast.parse(_RUNNER_SOURCE.read_text())
        invoke = _find_method(tree, "AgentTaskRunner", "invoke")
        running_line = _lineno_of_first_update_to_running(invoke)
        spawn_line = _lineno_of_first_call(invoke, "_maybe_spawn_mailbox_supervisor")
        assert running_line is not None, (
            "AgentTaskRunner.invoke must call update_status(..., SessionStatus.RUNNING)"
        )
        assert spawn_line is not None
        assert spawn_line > running_line, (
            f"_maybe_spawn_mailbox_supervisor() must come AFTER the RUNNING "
            f"status flip (running_line={running_line} spawn_line={spawn_line}) — "
            f"PR-3c plan §6.5 ordering invariant"
        )

    def test_stop_hook_lives_inside_terminal_shielded_task(self) -> None:
        """codex r3 [HIGH ARCH+TEST] — the stop hook MUST be invoked
        from inside ``_set_terminal_status._terminal_op`` (the body that
        runs inside the ``asyncio.shield``-wrapped terminal_task). A
        plain ``await self._maybe_stop_mailbox_supervisor()`` in
        ``_set_terminal_status_with_notifications`` (post R2 placement)
        was vulnerable to outer-cancel leaks at the shield seam.
        """
        tree = ast.parse(_RUNNER_SOURCE.read_text())
        terminal_status = _find_method(
            tree, "AgentTaskRunner", "_set_terminal_status"
        )
        terminal_op = _find_nested(terminal_status, "_terminal_op")
        assert _calls(terminal_op, "_maybe_stop_mailbox_supervisor"), (
            "_set_terminal_status._terminal_op (the body wrapped in "
            "asyncio.shield) must call self._maybe_stop_mailbox_supervisor() "
            "so outer cancellation cannot leak the per-pod MailboxSupervisor "
            "task — PR-3c codex r3 [HIGH ARCH] invariant"
        )
