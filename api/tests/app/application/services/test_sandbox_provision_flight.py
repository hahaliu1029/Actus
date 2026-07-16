"""Sandbox Provision Mode — flight/reconcile test facade + Task 1/2/3 tests.

This module hosts the shared ``_Fakes`` facade + ``svc_and_fakes`` fixture that
the whole Sandbox Provision Mode epic's flight / reconcile / interaction tests
reuse. The facade is defined here (NOT imported from an existing test) — see
the task brief: no ``svc_and_fakes`` exists in the current suite.

Async runner note: this project ships **pytest-anyio**, not pytest-asyncio
(see ``tests/conftest.py`` — "@pytest.mark.anyio, NOT @pytest.mark.asyncio").
So tests are marked via the module-level ``pytestmark = pytest.mark.anyio`` and
a local ``anyio_backend`` fixture, mirroring
``test_sandbox_lifecycle_hardening.py``. (The brief's illustrative snippets
write ``@pytest.mark.asyncio``; under this repo that marker would leave the
coroutine un-awaited, so the module-level ``anyio`` marker is authoritative and
the per-method ``asyncio`` markers are intentionally omitted.)

Task 3 extension: ``FakeUoW`` / ``FakeSandboxCls`` / ``FakeSandbox`` /
``FakeRegistry`` grow the gates the ``bind_new`` flight tests need (create /
register / read-commit / nth-save / transition-commit / dispose hangs, plus
BaseException + cancel injection). The cancel-swallow commit sub-window fakes
mirror ``DBUnitOfWork.__aexit__`` (``db_uow.py:71``) EXACTLY: they catch
``CancelledError`` in the commit window, log, and do NOT ``uncancel()`` so the
task stays ``cancelling() > 0`` and the service's
``_raise_if_read_swallowed_cancel`` guard can detect it.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Optional

import pytest

from app.application.services.sandbox_lifecycle_service import SandboxLifecycleService
from app.application.services.sandbox_provision_flight import ProvisionFlightTable
from app.domain.errors.sandbox_lifecycle import (
    SandboxBindingMissing,
    SandboxLifecycleError,
    SandboxProvisionInvalidated,
    SessionFinalizedError,
)
from app.domain.models.session import (
    DestroyReason,
    SandboxBinding,
    SandboxBindingState,
    Session,
    SessionStatus,
)

pytestmark = pytest.mark.anyio

logger = logging.getLogger(__name__)

# Shorthand aliases (mirror sandbox_lifecycle_service module-level aliases).
UNBOUND = SandboxBindingState.UNBOUND
CREATING = SandboxBindingState.CREATING
ACTIVE = SandboxBindingState.ACTIVE
DESTROYED = SandboxBindingState.DESTROYED

# Task 4: real-enum-backed reason constants (brief §冻结决策1 — do NOT hard-code
# member names by guessing; import the enum so the mapping stays locked to it).
DELETE_REASON = DestroyReason.SESSION_DELETE        # delete-class → "delete" → tombstones
NON_DELETE_REASON = DestroyReason.WATCHDOG_TIMEOUT  # non-delete → "destroy" → no tombstone

# String → state map for the ``hang_on_transition_commit`` gate.
_STATE_MAP = {
    "unbound": UNBOUND,
    "creating": CREATING,
    "active": ACTIVE,
    "destroyed": DESTROYED,
}

# SPM Task 6: sentinel distinguishing "``get_strict_returns`` never armed"
# (fall through to the ``get``-equivalent live-sandbox default) from an explicit
# ``get_strict_returns = None`` (arm the definite-NotFound / terminal case).
_UNSET = object()


@pytest.fixture
def anyio_backend():
    return "asyncio"


# ── Lightweight in-process fakes ──────────────────────────────────────────────


class FakeSandbox:
    """Minimal fake sandbox instance.

    Reads injection knobs (``ensure_raises`` / ``dispose_gate``) off its owning
    ``FakeSandboxCls`` so a test can arm them before ``create`` runs.
    """

    def __init__(self, owner: "FakeSandboxCls | None" = None, sid: str = "sbx-1") -> None:
        self.destroyed = False
        self._owner = owner
        self._id = sid

    @property
    def id(self) -> str:
        return self._id

    async def ensure_sandbox(self) -> None:
        owner = self._owner
        if owner is not None and owner.ensure_gate is not None:
            owner.ensure_reached.set()
            await owner.ensure_gate.wait()
        if owner is not None and owner.ensure_raises is not None:
            raise owner.ensure_raises

    async def destroy(self) -> bool:
        owner = self._owner
        if owner is not None and owner.dispose_gate is not None:
            owner.dispose_reached.set()
            await owner.dispose_gate.wait()
        self.destroyed = True
        return True


class FakeSandboxCls:
    """Instance-based stand-in for the ``Sandbox`` *type* passed to the service.

    The service does ``await self._sandbox_cls.create(...)`` and, on the
    rehydrate path, ``await self._sandbox_cls.get(id)``. Per-instance recorders
    hold call logs + injection gates so no class-level state leaks between tests.
    """

    def __init__(self) -> None:
        self.create_calls: list[dict] = []
        self.created: list[FakeSandbox] = []
        # Injection gates (armed by tests):
        self.create_gate: Optional[asyncio.Event] = None   # pause create until set
        self.create_reached = asyncio.Event()              # signaled when create entered
        self.ensure_raises: Optional[BaseException] = None  # ensure_sandbox raises this
        self.ensure_gate: Optional[asyncio.Event] = None   # pause ensure_sandbox until set
        self.ensure_reached = asyncio.Event()              # signaled when ensure_sandbox entered
        self.dispose_gate: Optional[asyncio.Event] = None  # pause destroy until set
        self.dispose_reached = asyncio.Event()             # signaled when destroy entered
        self.container_dead = False                        # get() returns None (orphan)
        self.get_calls = 0                                 # rehydrate lookups
        # SPM Task 6: get_strict + label-sweep primitives (mirror Task 5's real
        # DockerSandbox classmethods). ``get_strict`` is async; the two sweep
        # primitives are SYNC (the service wraps them in ``asyncio.to_thread``).
        self.get_strict_calls = 0                          # get_strict lookups
        self.get_strict_returns: object = _UNSET           # armed → get_strict returns this
        self.get_strict_raises: Optional[BaseException] = None  # armed → get_strict raises
        self.managed: list[dict] = []                      # list_managed_containers rows
        self.removed: list[str] = []                       # remove_container call log
        self.list_raises: Optional[BaseException] = None   # armed → list_managed raises

    async def create(
        self,
        user_id: Optional[str] = None,
        *,
        session_id: Optional[str] = None,
        attempt: Optional[str] = None,
        runtime_policy=None,
    ) -> FakeSandbox:
        self.create_calls.append(
            {
                "user_id": user_id,
                "session_id": session_id,
                "attempt": attempt,
                "runtime_policy": runtime_policy,
            }
        )
        sbx = FakeSandbox(owner=self, sid=f"sbx-{len(self.created) + 1}")
        self.created.append(sbx)
        if self.create_gate is not None:
            self.create_reached.set()
            await self.create_gate.wait()
        return sbx

    async def get(self, id: str) -> Optional[FakeSandbox]:
        self.get_calls += 1
        if self.container_dead:
            return None
        return FakeSandbox(owner=self, sid=id)

    async def get_strict(self, id: str) -> Optional[FakeSandbox]:
        """Task 5/6 strict variant: distinguishes daemon-unreachable (raises
        ``SandboxDaemonUnreachable``) from terminal/gone (returns ``None``).

        Honors ``get_strict_raises`` / ``get_strict_returns``; when neither is
        armed it mirrors ``get`` (live sandbox unless ``container_dead``) so a
        future live-rehydrate test needs no special casing.
        """
        self.get_strict_calls += 1
        if self.get_strict_raises is not None:
            raise self.get_strict_raises
        if self.get_strict_returns is not _UNSET:
            return self.get_strict_returns  # type: ignore[return-value]
        if self.container_dead:
            return None
        return FakeSandbox(owner=self, sid=id)

    def list_managed_containers(self) -> list[dict]:
        """SYNC (mirrors the real classmethod — the service wraps it in
        ``asyncio.to_thread``). Honors ``list_raises`` for the fail-safe test."""
        if self.list_raises is not None:
            raise self.list_raises
        return self.managed

    def remove_container(self, name: str) -> None:
        """SYNC — records the removal so the label-sweep tests can assert it."""
        self.removed.append(name)


class FakeSessionRepo:
    """In-memory session repository backed by a shared ``sessions`` dict."""

    def __init__(self, sessions: dict[str, Session], events: list, owner: "FakeUoW") -> None:
        self._sessions = sessions
        self._events = events
        self._owner = owner

    async def get_by_id(self, session_id: str) -> Optional[Session]:
        owner = self._owner
        if owner.hang_on_get_by_id is not None:
            owner.get_by_id_reached.set()
            await owner.hang_on_get_by_id.wait()  # blocks until cancel (never set)
        session = self._sessions.get(session_id)
        # DB-rollback fidelity (FIX-H support): snapshot the PRE-transition binding
        # on the active UoW frame so ``__aexit__`` can restore it if this ctx exits
        # with an exception (e.g. a transition's commit/save raises). A real DB
        # rolls a failed commit back → the persisted row is untouched, so a failed
        # CREATING transition leaves the binding UNBOUND. The old fake left the
        # in-place ``session.sandbox_binding`` mutation persisted, which would have
        # masked the phantom UNBOUND→UNBOUND rollback FIX-H targets.
        if session is not None and owner._ctx_stack:
            owner._ctx_stack[-1].setdefault(
                "rollback_binding", (session, session.sandbox_binding)
            )
        return session

    async def get_all(self) -> list[Session]:
        # SPM Task 6: reconcile_orphans + the label-sweep both scan all sessions.
        # FIX-F(b): optional fault injection. ``get_all_raise_on_call`` (1-based)
        # scopes the raise to a single call so reconcile's own top-level get_all
        # can succeed while ONLY the sweep's later get_all fails (proving the
        # sweep's internal fail-safe, not an early reconcile bail-out).
        owner = self._owner
        owner._get_all_calls += 1
        if owner.get_all_raises is not None and (
            owner.get_all_raise_on_call is None
            or owner._get_all_calls == owner.get_all_raise_on_call
        ):
            raise owner.get_all_raises
        return list(self._sessions.values())

    async def save(self, session: Session) -> None:
        self._sessions[session.id] = session
        state = session.sandbox_binding.state
        if state == ACTIVE:
            # Signal "reached the ACTIVE commit / register boundary" so the
            # cancel-after-active-commit test can time its cancel precisely.
            self._owner.register_reached.set()
        await self._owner._maybe_hang_on_save(state)

    async def add_event(self, session_id: str, event) -> None:
        self._events.append(event)


class FakeAuditRepo:
    """In-memory ``sandbox_lifecycle_log`` capturing ``create`` kwargs as rows."""

    def __init__(self, audit_rows: list[dict]) -> None:
        self._rows = audit_rows

    async def create(self, **kwargs) -> None:
        self._rows.append(dict(kwargs))


class FakeUoW:
    """Fake unit-of-work exposing ``session`` + ``sandbox_lifecycle_log`` repos.

    A single instance is reused across ``_transition`` / read calls (via the
    ``uow_factory`` lambda) so binding state persists between transitions.

    Cancel-swallow fidelity: ``__aexit__`` mirrors ``DBUnitOfWork.__aexit__``
    (``db_uow.py:71``) — a cancel landing in the commit sub-window is caught,
    logged, and NOT ``uncancel()``-ed, leaving ``cancelling() > 0``.
    """

    def __init__(
        self,
        sessions: dict[str, Session],
        events: list,
        audit_rows: list[dict],
    ) -> None:
        self.sessions = sessions
        self.session = FakeSessionRepo(sessions, events, owner=self)
        self.sandbox_lifecycle_log = FakeAuditRepo(audit_rows)

        # ── read-await hang (cancel propagates naturally, NOT swallowed) ──
        self.hang_on_get_by_id: Optional[asyncio.Event] = None
        self.get_by_id_reached = asyncio.Event()

        # ── read-commit hang (cancel-swallow sub-window) ──
        self.hang_on_read_commit: Optional[asyncio.Event] = None
        self.read_commit_reached = asyncio.Event()

        # ── transition-commit hang (target-state gated, cancel-swallow) ──
        self.hang_on_transition_commit: Optional[str] = None
        self.transition_commit_reached = asyncio.Event()
        self._transition_hold = asyncio.Event()  # never set → blocks until cancel

        self.swallow_commit_cancel = False  # mirror db_uow.py:71 swallow-no-uncancel

        # ── register boundary (ACTIVE commit) ──
        self.register_reached = asyncio.Event()

        # ── label-sweep get_all fault injection (FIX-F b) ──
        self.get_all_raises: Optional[BaseException] = None
        self.get_all_raise_on_call: Optional[int] = None  # 1-based; None → every call
        self._get_all_calls = 0

        # ── nth-save hang (+ optional fail-on-release) ──
        self._hang_save_n: Optional[int] = None
        self._fail_on_release: Optional[BaseException] = None
        self._release_event = asyncio.Event()
        self.hang_reached = asyncio.Event()
        self._save_count = 0

        # per-context frames (stack) for read-vs-transition detection
        self._ctx_stack: list[dict] = []

    # ── test-facing arming helpers ──

    def hang_on_nth_save(self, n: int) -> None:
        self._hang_save_n = n

    def release_hang(self) -> None:
        self._release_event.set()

    def hang_then_fail_on_transition(self, window: str, exc: BaseException) -> None:
        self._hang_save_n = {"creating": 1, "active": 2}[window]
        self._fail_on_release = exc

    # ── internal hang machinery ──

    async def _maybe_hang_on_save(self, saved_state) -> None:
        self._save_count += 1
        if self._ctx_stack:
            self._ctx_stack[-1]["saved"] = True
            self._ctx_stack[-1]["state"] = saved_state
        if self._hang_save_n is not None and self._save_count == self._hang_save_n:
            self.hang_reached.set()
            await self._release_event.wait()
            if self._fail_on_release is not None:
                raise self._fail_on_release

    async def __aenter__(self) -> "FakeUoW":
        self._ctx_stack.append({"saved": False, "state": None})
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        frame = self._ctx_stack.pop() if self._ctx_stack else {"saved": False, "state": None}
        if exc_type is not None:
            # rollback path — no commit-window hang; propagate the inbound exc.
            # DB-rollback fidelity (FIX-H support): restore the pre-transition
            # binding snapshot so a transition that mutated the shared session
            # object in-memory before its save raised is reverted (mirrors a real
            # DB, where a failed commit leaves the persisted row unchanged).
            rb = frame.get("rollback_binding")
            if rb is not None:
                sess, prev_binding = rb
                sess.sandbox_binding = prev_binding
            return False
        saved = frame["saved"]
        state = frame["state"]
        # read-only commit sub-window (no save this ctx)
        if not saved and self.hang_on_read_commit is not None:
            self.read_commit_reached.set()
            try:
                await self.hang_on_read_commit.wait()
            except asyncio.CancelledError:
                if self.swallow_commit_cancel:
                    logger.warning("fake UoW read-commit cancel swallowed (no uncancel)")
                    return False  # swallow, do NOT uncancel → cancelling() stays > 0
                raise
        # transition commit sub-window (target-state gated)
        if saved and self.hang_on_transition_commit is not None:
            want = _STATE_MAP.get(self.hang_on_transition_commit)
            if want is not None and state == want:
                self.transition_commit_reached.set()
                try:
                    await self._transition_hold.wait()
                except asyncio.CancelledError:
                    if self.swallow_commit_cancel:
                        logger.warning(
                            "fake UoW transition-commit cancel swallowed (no uncancel)"
                        )
                        return False
                    raise
        return False


class _FakeHandle:
    """Trivial SandboxHandle stand-in — only ``release`` is exercised."""

    def __init__(self, registry: "FakeRegistry") -> None:
        self._registry = registry

    def release(self) -> None:
        self._registry._open = max(0, self._registry._open - 1)


class FakeRegistry:
    """Injected in place of the service's internal ``SandboxRegistry`` so tests
    can count acquires / open handles, force a registry miss, and observe the
    register "hook"."""

    def __init__(self) -> None:
        self._sandboxes: dict[str, object] = {}
        self._generations: dict[str, int] = {}
        self._open = 0
        self.acquire_handle_calls = 0
        self.miss = False
        self.hooks_ran: list = []

    def get_live_event_sink(self, session_id: str):
        return None

    def get_sandbox(self, session_id: str):
        if self.miss:
            return None
        return self._sandboxes.get(session_id)

    def get_generation(self, session_id: str):
        return self._generations.get(session_id)

    def update_generation(self, session_id: str, generation: int) -> None:
        self._generations[session_id] = generation

    def register(self, session_id: str, sandbox, generation: int) -> None:
        self._sandboxes[session_id] = sandbox
        self._generations[session_id] = generation
        self.hooks_ran.append(("register", session_id))

    def acquire_handle(self, session_id: str) -> _FakeHandle:
        self.acquire_handle_calls += 1
        self._open += 1
        return _FakeHandle(self)

    # ── destroy / reconcile teardown (SPM Task 6) ──
    # No-op stand-ins so the reconcile "container alive" branch can run to
    # completion (register → cancel_and_drain → destroy_infra → remove →
    # DESTROYED). The real registry drains in-flight handles + rms the
    # container; the flight tests never needed these because they only hit
    # destroy()'s early-return branches.

    async def cancel_and_drain(self, session_id: str, timeout: Optional[float] = None) -> None:
        return None

    async def destroy_infra(self, session_id: str) -> None:
        return None

    def remove(self, session_id: str) -> None:
        self._sandboxes.pop(session_id, None)
        self._generations.pop(session_id, None)

    # ── test-facing seeding ──

    def seed_sandbox(self, session_id: str, sandbox=None) -> None:
        self._sandboxes[session_id] = sandbox if sandbox is not None else FakeSandbox()
        self._generations.setdefault(session_id, 1)

    @property
    def open_handles(self) -> int:
        return self._open


# ── Facade ────────────────────────────────────────────────────────────────────


@dataclass
class _Fakes:
    """Test-facing handle over the fakes wired into a ``SandboxLifecycleService``."""

    session_id: str
    uow: FakeUoW
    sandbox_cls: FakeSandboxCls
    registry: FakeRegistry
    events: list  # every add_event-captured event
    audit_rows: list[dict]  # every sandbox_lifecycle_log.create row
    # cancel-after-active-commit gate object (created by the test; the fake only
    # consumes ``register_reached``, so this is an inert holder — see the test).
    register_gate: Optional[asyncio.Event] = None

    @property
    def register_reached(self) -> asyncio.Event:
        return self.uow.register_reached

    @property
    def registry_open_handles(self) -> int:
        return self.registry.open_handles

    @property
    def rehydrate_calls(self) -> int:
        return self.sandbox_cls.get_calls

    @property
    def hooks_ran(self) -> list:
        return self.registry.hooks_ran

    def binding_state(self, sid: str) -> SandboxBindingState:
        """Current persisted binding state of session ``sid``."""
        return self.uow.sessions[sid].sandbox_binding.state

    def binding_generation(self, sid: str) -> int:
        return self.uow.sessions[sid].sandbox_binding.generation

    def last_event_reason(self) -> Optional[str]:
        """``reason`` of the most recent ``sandbox_state_changed`` event."""
        changed = [e for e in self.events if e.type == "sandbox_state_changed"]
        return changed[-1].reason if changed else None

    def make_binding(
        self,
        *,
        state: SandboxBindingState,
        sandbox_id: Optional[str] = None,
    ) -> None:
        """Preset the fixture session's binding row to ``state`` / ``sandbox_id``."""
        session = self.uow.sessions[self.session_id]
        session.sandbox_binding = SandboxBinding(state=state, id=sandbox_id)

    def seed_active_binding(self, sid: str) -> None:
        """Preset session ``sid`` to an ACTIVE binding with a sandbox id so
        ``acquire`` reaches the ACTIVE branch (registry-hit or rehydrate)."""
        session = self.uow.sessions[sid]
        session.sandbox_binding = SandboxBinding(
            state=ACTIVE, id="sbx-seed", generation=1
        )


def _unbound_session(sid: str, user_id: Optional[str] = None) -> Session:
    return Session(
        id=sid,
        user_id=user_id,
        status=SessionStatus.PENDING,
        sandbox_binding=SandboxBinding(state=UNBOUND),
    )


@pytest.fixture
def svc_and_fakes():
    """Return ``(SandboxLifecycleService, _Fakes)``.

    The service is built with ``FakeSandboxCls`` + a ``FakeUoW`` factory; its
    internal registry is replaced with a ``FakeRegistry`` so the flight tests can
    count acquires / open handles and drive the register hook.
    """
    session_id = "sess-1"
    sessions: dict[str, Session] = {session_id: _unbound_session(session_id)}
    events: list = []
    audit_rows: list[dict] = []

    uow = FakeUoW(sessions, events, audit_rows)
    sandbox_cls = FakeSandboxCls()
    registry = FakeRegistry()
    svc = SandboxLifecycleService(
        sandbox_cls=sandbox_cls,
        uow_factory=lambda: uow,
    )
    svc._registry = registry  # type: ignore[assignment]
    fakes = _Fakes(
        session_id=session_id,
        uow=uow,
        sandbox_cls=sandbox_cls,
        registry=registry,
        events=events,
        audit_rows=audit_rows,
    )
    return svc, fakes


# ── Concurrent A/B factory (Task 3 concurrent test) ──────────────────────────


class _FactorySessionRepo:
    def __init__(self, factory: "_Factory") -> None:
        self._f = factory

    async def get_by_id(self, session_id: str):
        return self._f.sessions.get(session_id)

    async def save(self, session: Session) -> None:
        self._f.sessions[session.id] = session
        if self._f._uow._ctx_stack:
            self._f._uow._ctx_stack[-1]["sid"] = session.id
            self._f._uow._ctx_stack[-1]["state"] = session.sandbox_binding.state

    async def add_event(self, session_id: str, event) -> None:
        self._f.events.append(event)


class _FactoryUoW:
    """Multiplexing UoW: routes the CREATING-commit hang to the per-session
    gate so two concurrent bind_new calls block on independent gates. Frames are
    popped BEFORE hanging to keep the shared stack uncorrupted under A/B
    interleaving (each transition runs enter→save→exit without yielding until the
    commit hang)."""

    def __init__(self, factory: "_Factory") -> None:
        self._f = factory
        self.session = _FactorySessionRepo(factory)
        self.sandbox_lifecycle_log = FakeAuditRepo(factory.audit_rows)
        self._ctx_stack: list[dict] = []

    async def __aenter__(self) -> "_FactoryUoW":
        self._ctx_stack.append({"sid": None, "state": None})
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        frame = self._ctx_stack.pop() if self._ctx_stack else {"sid": None, "state": None}
        if exc_type is not None:
            return False
        sid = frame["sid"]
        state = frame["state"]
        if state == CREATING and sid in self._f._sess_ctl:
            ctl = self._f._sess_ctl[sid]
            ctl.creating_commit_reached.set()
            await ctl.creating_commit_gate.wait()
        return False


@dataclass
class _FactorySession:
    session_id: str
    _factory: "_Factory"
    creating_commit_gate: asyncio.Event = field(default_factory=asyncio.Event)
    creating_commit_reached: asyncio.Event = field(default_factory=asyncio.Event)

    @property
    def binding_state(self) -> SandboxBindingState:
        return self._factory.sessions[self.session_id].sandbox_binding.state


class _Factory:
    """Same app-singleton service, per-session CREATING-commit gates."""

    def __init__(self) -> None:
        self.sessions: dict[str, Session] = {}
        self.events: list = []
        self.audit_rows: list[dict] = []
        self._sess_ctl: dict[str, _FactorySession] = {}
        self._uow = _FactoryUoW(self)
        self.service = SandboxLifecycleService(
            sandbox_cls=FakeSandboxCls(),
            uow_factory=lambda: self._uow,
        )
        self.service._registry = FakeRegistry()  # type: ignore[assignment]

    def session(self, sid: str) -> _FactorySession:
        if sid not in self._sess_ctl:
            self.sessions[sid] = _unbound_session(sid)
            self._sess_ctl[sid] = _FactorySession(session_id=sid, _factory=self)
        return self._sess_ctl[sid]


@pytest.fixture
def svc_and_fakes_factory():
    return _Factory()


# ── Task 1 tests ──────────────────────────────────────────────────────────────


class TestTransitionEventReason:
    async def test_event_reason_overrides_event_payload_only(self, svc_and_fakes):
        svc, fakes = svc_and_fakes  # binding starts UNBOUND
        await svc._transition(fakes.session_id, target=CREATING)
        await svc._transition(
            fakes.session_id, target=UNBOUND, event_reason="provision_failed"
        )
        evts = [e for e in fakes.events if e.type == "sandbox_state_changed"]
        assert evts[-1].reason == "provision_failed"  # event carries the override
        assert fakes.audit_rows[-1]["reason"] is None  # audit keeps legacy derivation

    async def test_default_none_keeps_legacy_derivation(self, svc_and_fakes):
        svc, fakes = svc_and_fakes
        await svc._transition(fakes.session_id, target=CREATING)
        evts = [e for e in fakes.events if e.type == "sandbox_state_changed"]
        assert evts[-1].reason is None


# ── Task 2 tests ──────────────────────────────────────────────────────────────


class TestProvisionFlightTable:
    """``ProvisionFlightTable`` — pure in-memory flight registry + deletion
    tombstones (spec §5.2c, DD-17). All methods are synchronous, so these are
    plain (non-async) test methods; the module-level ``pytest.mark.anyio`` marker
    is a no-op for sync tests (anyio only intercepts coroutine test functions).
    """

    def test_begin_get_finish_roundtrip(self):
        t = ProvisionFlightTable()
        f = t.begin("s1")
        assert t.get("s1") is f and len(f.attempt) == 32
        t.finish("s1")
        assert t.get("s1") is None
        t.finish("s1")  # 幂等

    def test_begin_twice_raises(self):
        t = ProvisionFlightTable()
        t.begin("s1")
        with pytest.raises(RuntimeError):
            t.begin("s1")

    def test_invalidate_active_flight_marks(self):
        t = ProvisionFlightTable()
        f = t.begin("s1")
        assert t.invalidate("s1", "destroy") is True
        assert f.invalidated == "destroy"
        assert not t.is_tombstoned("s1")  # destroy 不落 tombstone

    def test_invalidate_delete_with_active_flight_marks_AND_tombstones(self):
        """codex planR1#1：delete 恒 tombstone——覆盖 destroy 返回后行硬删前的二次 bind_new 窗口"""
        t = ProvisionFlightTable()
        f = t.begin("s1")
        assert t.invalidate("s1", "delete") is True
        assert f.invalidated == "delete"
        assert t.is_tombstoned("s1")

    def test_invalidate_absent_delete_registers_tombstone(self):
        t = ProvisionFlightTable()
        assert t.invalidate("s1", "delete") is False
        assert t.is_tombstoned("s1")

    def test_invalidate_absent_destroy_is_noop(self):
        t = ProvisionFlightTable()
        assert t.invalidate("s1", "destroy") is False
        assert not t.is_tombstoned("s1")

    def test_delete_is_sticky_against_later_quiesce(self):
        """FIX-J: a later ``quiesce`` must NOT downgrade an earlier ``delete``
        mark. The tombstone still blocks rebinding, but
        ``SandboxProvisionInvalidated.invalidation_outcome`` would otherwise
        misreport a *deleted* session as merely quiesced. Both calls still
        return True (an active flight was marked)."""
        t = ProvisionFlightTable()
        f = t.begin("s1")
        assert t.invalidate("s1", "delete") is True
        assert t.invalidate("s1", "quiesce") is True   # downgrade ignored
        assert f.invalidated == "delete"

    def test_quiesce_upgrades_to_delete(self):
        """FIX-J (other direction): upgrades TO ``delete`` are always honored —
        a delete arriving after a quiesce overrides it AND tombstones."""
        t = ProvisionFlightTable()
        f = t.begin("s1")
        assert t.invalidate("s1", "quiesce") is True
        assert t.invalidate("s1", "delete") is True    # upgrade honored
        assert f.invalidated == "delete"
        assert t.is_tombstoned("s1")

    def test_tombstone_fifo_cap(self):
        t = ProvisionFlightTable()
        for i in range(10_001):
            t.invalidate(f"s{i}", "delete")
        assert not t.is_tombstoned("s0")       # 最老者被逐出
        assert t.is_tombstoned("s10000")


# ── Task 3 tests ──────────────────────────────────────────────────────────────


class TestBindNewFlight:
    async def test_cancel_during_create_rolls_back_unbound(self, svc_and_fakes):
        """场景②：create await 期取消 → UNBOUND + provision_cancelled + late disposer 清容器.

        Synchronization robustness (implementation judgment call): the frozen
        brief illustrated this with a bare ``await asyncio.sleep(0)`` to "enter
        CREATING", but the mandated ``ensure_future``+``shield`` CREATING task
        adds scheduling hops so a single yield lands the cancel at the CREATING
        shield (before ``create`` ever runs → no container to late-dispose). We
        wait on the fake's ``create_reached`` event instead — same assertions,
        reliable timing (cancel lands squarely in the create shield window)."""
        svc, fakes = svc_and_fakes
        fakes.sandbox_cls.create_gate = asyncio.Event()      # create 挂起直到 set
        task = asyncio.create_task(svc.bind_new(fakes.session_id))
        await asyncio.wait_for(fakes.sandbox_cls.create_reached.wait(), timeout=2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert fakes.binding_state(fakes.session_id) == UNBOUND
        assert fakes.last_event_reason() == "provision_cancelled"
        fakes.sandbox_cls.create_gate.set()                   # create 迟到完成
        await asyncio.sleep(0.01)                             # late disposer 跑完
        assert fakes.sandbox_cls.created[-1].destroyed         # 场景⑥ late dispose

    async def test_shutdown_drains_pending_late_disposer(self, svc_and_fakes):
        """FIX-G: ``shutdown`` must give an in-flight late disposer a bounded
        window to finish, so a container born after a cancel isn't orphaned when
        the loop tears down. Park create, cancel the bind (late disposer spawned,
        now awaiting the still-gated create), release ``create_gate``, then
        ``await svc.shutdown()`` — by the time it returns the container has been
        destroyed (the drain waited for the disposer). Before the fix ``shutdown``
        only reset the registry/locks and returned without yielding, so the
        disposer was still pending → the container leaked."""
        svc, fakes = svc_and_fakes
        fakes.sandbox_cls.create_gate = asyncio.Event()       # create 挂起直到 set
        task = asyncio.create_task(svc.bind_new(fakes.session_id))
        await asyncio.wait_for(fakes.sandbox_cls.create_reached.wait(), timeout=2)
        task.cancel()                                         # create shield 取消 → 生成 late disposer
        with pytest.raises(asyncio.CancelledError):
            await task
        assert svc._late_dispose_tasks                        # 一个 disposer 仍 pending（等待 create）
        fakes.sandbox_cls.create_gate.set()                   # 放行 shielded create → disposer 可推进
        await svc.shutdown()                                  # drain 等 disposer 落定
        assert fakes.sandbox_cls.created[-1].destroyed        # shutdown 返回时容器已 dispose

    async def test_cancel_at_read_await_clean_abort(self, svc_and_fakes):
        """取消落在前置 read ``get_by_id`` 的 await 上 → 正常传播 → 在任何
        flight/CREATING/容器之前 clean abort（零 flight、零 transition、零 container create）。"""
        svc, fakes = svc_and_fakes
        fakes.uow.hang_on_get_by_id = asyncio.Event()     # get_by_id await 挂起
        task = asyncio.create_task(svc.bind_new(fakes.session_id))
        await asyncio.wait_for(fakes.uow.get_by_id_reached.wait(), timeout=2)
        task.cancel()                                     # 落在 read await → 正常传播（未被吞）
        with pytest.raises(asyncio.CancelledError):
            await task
        assert svc._flights.get(fakes.session_id) is None            # 零 flight（begin 前已 abort）
        assert fakes.binding_state(fakes.session_id) == UNBOUND      # 零 transition
        assert fakes.sandbox_cls.created == []                       # 零 container create

    async def test_cancel_at_read_commit_window_clean_abort(self, svc_and_fakes):
        """取消落在 read UoW ``__aexit__`` 的空只读 commit 子窗口 → 被 UoW **吞掉**
        （fake 镜像 ``db_uow.py:71`` 只 log 不 ``uncancel()``）→ task 仍 ``cancelling()>0`` →
        bind_new 的 read-commit 守卫显式 honor → clean abort（零 flight、零 transition/audit、零 container）。"""
        svc, fakes = svc_and_fakes
        fakes.uow.hang_on_read_commit = asyncio.Event()   # __aexit__ commit 挂起直到取消到达
        fakes.uow.swallow_commit_cancel = True            # 捕获 CancelledError 只 log、**不** uncancel
        task = asyncio.create_task(svc.bind_new(fakes.session_id))
        await asyncio.wait_for(fakes.uow.read_commit_reached.wait(), timeout=2)
        task.cancel()                                     # 落在 read-commit → 被 UoW 吞（task 仍 cancelling()>0）
        with pytest.raises(asyncio.CancelledError):       # 守卫重新 honor 被吞的取消
            await task
        assert svc._flights.get(fakes.session_id) is None            # 零 flight（守卫在 begin 前 abort）
        assert fakes.binding_state(fakes.session_id) == UNBOUND      # 零 transition
        assert fakes.audit_rows == []                                # 零 audit 行
        assert fakes.sandbox_cls.created == []                       # 零 container create

    async def test_delete_during_read_window_clean_abort_no_compensation(self, svc_and_fakes):
        """FIX-A (P1-1): a delete landing in the read-UoW window — between the
        entry tombstone check and ``flight.begin()`` — must abort CLEANLY. The
        post-begin tombstone re-check is hoisted BEFORE the ``try:``, so on a
        tombstone hit it explicitly finishes the flight and raises
        ``SessionFinalizedError`` without ever entering the compensation path →
        ZERO transition, ZERO audit row, ZERO container, flight cleared.

        Before the fix the re-check sat INSIDE the ``try:``; its raise triggered
        the ``except BaseException`` compensation which fired
        ``_transition(target=UNBOUND, event_reason="provision_failed")`` on a
        still-UNBOUND binding → a spurious UNBOUND→UNBOUND ``sandbox_state_changed``
        event (wrong reason) + audit row on a session being deleted."""
        svc, fakes = svc_and_fakes
        fakes.uow.hang_on_get_by_id = asyncio.Event()     # park bind_new in the read
        task = asyncio.create_task(svc.bind_new(fakes.session_id))
        await asyncio.wait_for(fakes.uow.get_by_id_reached.wait(), timeout=2)
        # A delete lands during the read window: it registers a tombstone, but no
        # flight exists yet (begin() runs only after the read) → invalidate == False.
        assert svc._flights.invalidate(fakes.session_id, "delete") is False
        fakes.uow.hang_on_get_by_id.set()                 # release the read → re-check fires
        with pytest.raises(SessionFinalizedError):
            await task
        # Clean-abort invariant: zero transition, zero audit, zero container, flight gone.
        assert [e for e in fakes.events if e.type == "sandbox_state_changed"] == []
        assert fakes.audit_rows == []
        assert fakes.sandbox_cls.created == []
        assert svc._flights.get(fakes.session_id) is None

    @pytest.mark.parametrize("registry_hit", [True, False])
    async def test_cancel_at_acquire_read_commit_clean_abort(self, svc_and_fakes, registry_hit):
        """``_acquire_locked`` 同型 read-commit 窗口——provisioner.get() 先调 acquire()。
        取消落 ``_acquire_locked`` 只读 UoW 的 commit 子窗口被吞（cancelling()>0）→ 同一
        helper ``_raise_if_read_swallowed_cancel()`` 在状态分支前 honor → clean abort：零
        registry acquire、零 rehydrate、零 hooks。registry-hit 与 rehydrate 两路径均覆盖。"""
        svc, fakes = svc_and_fakes
        fakes.seed_active_binding(fakes.session_id)               # 预置 ACTIVE binding
        if registry_hit:
            fakes.registry.seed_sandbox(fakes.session_id)         # registry 命中路径
        fakes.uow.hang_on_read_commit = asyncio.Event()          # _acquire_locked 读 UoW commit 挂起
        fakes.uow.swallow_commit_cancel = True                   # 镜像 db_uow.py:71 吞取消不 uncancel
        task = asyncio.create_task(svc.acquire(fakes.session_id))
        await asyncio.wait_for(fakes.uow.read_commit_reached.wait(), timeout=2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert fakes.registry.acquire_handle_calls == 0          # 零 registry acquire
        assert fakes.rehydrate_calls == 0                        # 零 rehydrate
        assert fakes.hooks_ran == []                             # 零 hooks

    async def test_cancel_at_orphan_transition_records_cancelled(self, svc_and_fakes):
        """registry-miss + 容器已死 → ``_rehydrate_or_mark_orphan`` 的 ``_transition(DESTROYED)``
        commit 窗口取消被吞（cancelling()>0）→ orphan guard 重抛 CancelledError（非落
        SessionFinalizedError→provisioner 误记 failed）。**DESTROYED commit 保留、不断言 UNBOUND**。"""
        svc, fakes = svc_and_fakes
        fakes.seed_active_binding(fakes.session_id)              # ACTIVE binding，registry miss
        fakes.registry.miss = True
        fakes.sandbox_cls.container_dead = True                  # 探测容器已死 → orphan 路径
        fakes.uow.hang_on_transition_commit = "destroyed"       # DESTROYED transition commit 挂起
        fakes.uow.swallow_commit_cancel = True                  # 镜像 db_uow.py:71 吞取消不 uncancel
        task = asyncio.create_task(svc.acquire(fakes.session_id))
        await asyncio.wait_for(fakes.uow.transition_commit_reached.wait(), timeout=2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert fakes.binding_state(fakes.session_id) == DESTROYED   # DESTROYED commit 保留（不 UNBOUND）
        assert fakes.hooks_ran == []                               # 零 hooks

    async def test_cancel_during_container_create_rolls_back_creating(self, svc_and_fakes):
        """取消发生在 CREATING commit 之后、ACTIVE commit 之前（容器创建 await 期）→
        active_committed=False → 合法 CREATING→UNBOUND + dispose。（sync via create_reached
        for the same scheduling reason as test_cancel_during_create.)"""
        svc, fakes = svc_and_fakes
        fakes.sandbox_cls.create_gate = asyncio.Event()   # create 挂起 → 取消落在 create await
        task = asyncio.create_task(svc.bind_new(fakes.session_id))
        await asyncio.wait_for(fakes.sandbox_cls.create_reached.wait(), timeout=2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert fakes.binding_state(fakes.session_id) == UNBOUND       # 合法 CREATING→UNBOUND
        assert fakes.registry_open_handles == 0
        fakes.sandbox_cls.create_gate.set()               # create 迟到完成 → late disposer
        await asyncio.sleep(0.01)
        assert fakes.sandbox_cls.created[-1].destroyed               # 容器被清（late dispose）

    async def test_cancel_after_active_commit_leaves_active_for_rehydrate(self, svc_and_fakes):
        """取消落在 ACTIVE commit **之后**、register 前 → active_committed=True →
        **不做非法 ACTIVE→UNBOUND、不 dispose**——binding 合法 ACTIVE、容器有效但 registry 未注册
        → 既有 c-1 lazy-rehydrate 兜底（与 always parity）。"""
        svc, fakes = svc_and_fakes
        fakes.register_gate = asyncio.Event()             # inert holder (see _Fakes docstring)
        task = asyncio.create_task(svc.bind_new(fakes.session_id))
        await asyncio.wait_for(fakes.register_reached.wait(), timeout=2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert fakes.binding_state(fakes.session_id) == ACTIVE        # 合法 ACTIVE 保留（无非法边）
        assert fakes.binding_generation(fakes.session_id) == 1        # generation 不被违约重置
        assert not fakes.sandbox_cls.created[-1].destroyed           # 容器有效不 dispose（lazy-rehydrate）

    async def test_cancel_during_CREATING_commit_awaits_then_rolls_back(self, svc_and_fakes):
        """CREATING transition commit 进行中取消 → await creating_task 到 done → 合法 CREATING→UNBOUND
        （或 UNBOUND 良性）；无容器（create 在 CREATING 后）→ 无 dispose 竞态。"""
        svc, fakes = svc_and_fakes
        fakes.uow.hang_on_nth_save(1)                     # #1 = CREATING transition 的 save
        task = asyncio.create_task(svc.bind_new(fakes.session_id))
        await asyncio.wait_for(fakes.uow.hang_reached.wait(), timeout=2)
        task.cancel()
        fakes.uow.release_hang()                          # 释放 → creating_task 完成落库（DB=CREATING）
        with pytest.raises(asyncio.CancelledError):
            await task
        assert fakes.binding_state(fakes.session_id) == UNBOUND       # await-to-done 后合法回滚
        assert fakes.registry_open_handles == 0

    async def test_cancel_during_ACTIVE_commit_determinate_committed(self, svc_and_fakes):
        """ACTIVE transition commit 进行中取消 + commit **确实落库** → await active_task 到 done →
        active_committed=True → 合法 ACTIVE 保留（**非** 非法 ACTIVE→UNBOUND）、gen=1、容器不 dispose。"""
        svc, fakes = svc_and_fakes
        fakes.uow.hang_on_nth_save(2)                     # #2 = ACTIVE transition 的 save
        task = asyncio.create_task(svc.bind_new(fakes.session_id))
        await asyncio.wait_for(fakes.uow.hang_reached.wait(), timeout=2)
        task.cancel()
        fakes.uow.release_hang()                          # 释放 → ACTIVE commit 确实落库（committed）
        with pytest.raises(asyncio.CancelledError):
            await task
        assert fakes.binding_state(fakes.session_id) == ACTIVE        # 确定 committed → 合法 ACTIVE 保留
        assert fakes.binding_generation(fakes.session_id) == 1
        assert not fakes.sandbox_cls.created[-1].destroyed           # 不 dispose（lazy-rehydrate）
        assert fakes.registry_open_handles == 0                       # register 未跑（外层已取消）

    async def test_concurrent_two_session_cancel_no_pending_clobber(self, svc_and_fakes_factory):
        """并发 A/B session 各卡在 **CREATING transition commit gate**；交错释放，证明各自
        只等待自己的 transition（pending_transition 是调用栈局部变量，非实例属性）。"""
        svc = svc_and_fakes_factory.service
        a, b = svc_and_fakes_factory.session("A"), svc_and_fakes_factory.session("B")
        ta = asyncio.create_task(svc.bind_new(a.session_id))
        tb = asyncio.create_task(svc.bind_new(b.session_id))
        await asyncio.wait_for(a.creating_commit_reached.wait(), timeout=2)   # A 卡 CREATING commit
        await asyncio.wait_for(b.creating_commit_reached.wait(), timeout=2)   # B 也卡（并发）
        ta.cancel(); tb.cancel()
        # 真分阶段释放：先释放 A、断言 A 完成而 B 仍 pending、再释放 B。
        a.creating_commit_gate.set()
        with pytest.raises(asyncio.CancelledError):
            await ta                                                 # A 独立完成（据自己的 pending 回滚）
        assert a.binding_state == UNBOUND
        assert not tb.done()                                         # B 仍 pending——证明 A 没等 B（无串扰）
        b.creating_commit_gate.set()
        with pytest.raises(asyncio.CancelledError):
            await tb
        assert b.binding_state == UNBOUND

    @pytest.mark.parametrize("window", ["creating", "active"])
    async def test_cancel_then_transition_fails_still_compensates(self, svc_and_fakes, window):
        """外层取消后，pending transition **以 RuntimeError（commit-error）结束**——
        ``_await_to_done``（asyncio.wait 不 shield）**不让 inner 异常越过补偿**：仍执行
        CREATING→UNBOUND + dispose，最终重抛原**取消**（非 inner RuntimeError 覆盖）。"""
        svc, fakes = svc_and_fakes
        fakes.uow.hang_then_fail_on_transition(window, exc=RuntimeError("commit boom"))
        task = asyncio.create_task(svc.bind_new(fakes.session_id))
        await asyncio.wait_for(fakes.uow.hang_reached.wait(), timeout=2)
        task.cancel()
        fakes.uow.release_hang()                          # transition 释放后以 RuntimeError 结束
        with pytest.raises(asyncio.CancelledError):       # 重抛的是原取消，非 RuntimeError
            await task
        assert fakes.binding_state(fakes.session_id) == UNBOUND       # 补偿未被 inner 异常越过

    async def test_creating_commit_failure_skips_redundant_rollback(self, svc_and_fakes):
        """FIX-H: a DIRECT (no outer cancel) CREATING-transition commit failure →
        the transition's UoW rolls back → DB is provably still UNBOUND. The
        compensation must therefore SKIP the redundant ``_transition(UNBOUND)``
        that would emit a phantom UNBOUND→UNBOUND ``sandbox_state_changed`` event +
        audit row. Binding stays UNBOUND, ZERO events, ZERO audit rows; the
        original ``RuntimeError`` still propagates.

        Before the fix: one phantom UNBOUND→UNBOUND event + one audit row. This is
        the deterministic-failure sibling of
        ``test_cancel_then_transition_fails_still_compensates`` — there an OUTER
        cancel + a committed/ambiguous window keeps the unconditional rollback;
        here the CREATING commit itself failed, so the rollback is provably
        redundant."""
        svc, fakes = svc_and_fakes
        # Arm the CREATING transition (save #1) to fail; release the hang up-front
        # so it fails deterministically with NO outer cancel.
        fakes.uow.hang_then_fail_on_transition("creating", exc=RuntimeError("commit boom"))
        fakes.uow.release_hang()
        with pytest.raises(RuntimeError):
            await svc.bind_new(fakes.session_id)
        assert fakes.binding_state(fakes.session_id) == UNBOUND
        assert [e for e in fakes.events if e.type == "sandbox_state_changed"] == []
        assert fakes.audit_rows == []
        assert fakes.sandbox_cls.created == []                     # never reached create

    async def test_repeated_cancel_during_compensation_still_completes(self, svc_and_fakes):
        """补偿（dispose+rollback）打包 comp_task + await-to-determinacy——补偿进行中**再次取消**
        不跳过 rollback、不后台泄漏：comp_task 被 await 到 done 才重抛，最终 UNBOUND + 容器已 dispose。"""
        svc, fakes = svc_and_fakes
        fakes.sandbox_cls.ensure_raises = RuntimeError("dead")   # 进补偿路径（create OK + ensure fail）
        fakes.sandbox_cls.dispose_gate = asyncio.Event()         # dispose 挂起 → 补偿期可再取消
        task = asyncio.create_task(svc.bind_new(fakes.session_id))
        await asyncio.wait_for(fakes.sandbox_cls.dispose_reached.wait(), timeout=2)
        task.cancel(); await asyncio.sleep(0); task.cancel()      # 补偿进行中重复取消
        fakes.sandbox_cls.dispose_gate.set()                     # 释放 dispose → comp_task 完成
        with pytest.raises((asyncio.CancelledError, RuntimeError)):
            await task
        assert fakes.binding_state(fakes.session_id) == UNBOUND   # rollback 未被跳过
        assert fakes.sandbox_cls.created[-1].destroyed           # 容器已 dispose（补偿完整落定）

    async def test_ensure_fail_disposes_container(self, svc_and_fakes):
        """场景⑦（lifecycle 半）：create OK + ensure 失败 → 容器清理 + UNBOUND + provision_failed"""
        svc, fakes = svc_and_fakes
        fakes.sandbox_cls.ensure_raises = RuntimeError("supervisord dead")
        with pytest.raises(RuntimeError):
            await svc.bind_new(fakes.session_id)
        assert fakes.binding_state(fakes.session_id) == UNBOUND
        assert fakes.sandbox_cls.created[-1].destroyed
        assert fakes.last_event_reason() == "provision_failed"

    async def test_tombstoned_session_refuses_bind_new(self, svc_and_fakes):
        """场景⑤前半：tombstone 入口检查"""
        svc, fakes = svc_and_fakes
        svc._flights.invalidate(fakes.session_id, "delete")
        with pytest.raises(SessionFinalizedError):
            await svc.bind_new(fakes.session_id)
        assert fakes.sandbox_cls.created == []                 # 零容器

    async def test_create_passes_session_and_attempt_kwargs(self, svc_and_fakes):
        svc, fakes = svc_and_fakes
        await svc.bind_new(fakes.session_id)
        call = fakes.sandbox_cls.create_calls[-1]
        assert call["session_id"] == fakes.session_id
        assert len(call["attempt"]) == 32

    async def test_happy_path_flight_cleared(self, svc_and_fakes):
        svc, fakes = svc_and_fakes
        await svc.bind_new(fakes.session_id)
        assert svc._flights.get(fakes.session_id) is None
        assert fakes.binding_state(fakes.session_id) == ACTIVE

    async def test_invalidated_flight_disposes_and_raises(self, svc_and_fakes):
        """CAS coverage (additive): a concurrent invalidate on the in-flight
        flight after create returns → dispose container + SandboxProvisionInvalidated,
        binding rolled back to UNBOUND with provision_cancelled, no illegal edge."""
        svc, fakes = svc_and_fakes
        fakes.sandbox_cls.create_gate = asyncio.Event()
        task = asyncio.create_task(svc.bind_new(fakes.session_id))
        await asyncio.wait_for(fakes.sandbox_cls.create_reached.wait(), timeout=2)
        svc._flights.invalidate(fakes.session_id, "destroy")   # CAS-1 window arm
        fakes.sandbox_cls.create_gate.set()                    # create returns → CAS-1 fires
        with pytest.raises(SandboxProvisionInvalidated):
            await task
        assert fakes.binding_state(fakes.session_id) == UNBOUND
        assert fakes.last_event_reason() == "provision_cancelled"
        assert fakes.sandbox_cls.created[-1].destroyed         # CAS dispose ran

    async def test_invalidated_during_ensure_sandbox_disposes_at_cas2(self, svc_and_fakes):
        """CAS-2 coverage (dedicated): an invalidate arriving DURING ``ensure_sandbox``
        — after ``create`` returned (so CAS-1 already passed) and before the ACTIVE
        transition — fires the **CAS-2** check specifically → dispose container +
        ``SandboxProvisionInvalidated``, binding rolled back to UNBOUND with
        ``provision_cancelled``. Gating ``ensure_sandbox`` (not ``create``) is what
        makes CAS-1 provably inert here: the flight is still un-invalidated when
        ``create`` returns, so only the post-``ensure_sandbox`` CAS-2 window can catch
        it. Mirrors the CAS-1 test (``test_invalidated_flight_disposes_and_raises``)."""
        svc, fakes = svc_and_fakes
        fakes.sandbox_cls.ensure_gate = asyncio.Event()        # ensure_sandbox 挂起直到 set
        task = asyncio.create_task(svc.bind_new(fakes.session_id))
        # create already returned (flight NOT invalidated → CAS-1 passed); we are now
        # suspended inside ensure_sandbox, strictly between CAS-1 and CAS-2.
        await asyncio.wait_for(fakes.sandbox_cls.ensure_reached.wait(), timeout=2)
        svc._flights.invalidate(fakes.session_id, "destroy")   # arm CAS-2 mid-ensure
        fakes.sandbox_cls.ensure_gate.set()                    # ensure returns → CAS-2 fires
        with pytest.raises(SandboxProvisionInvalidated):
            await task
        assert fakes.binding_state(fakes.session_id) == UNBOUND
        assert fakes.last_event_reason() == "provision_cancelled"
        assert fakes.sandbox_cls.created[-1].destroyed         # CAS-2 dispose ran


# ── Task 4 tests ──────────────────────────────────────────────────────────────


class TestDestroyFlightWiring:
    """Task 4: ``destroy``/``suspend`` intent-before-lock flight invalidation +
    delete-class tombstone registration + removal of the dead ``destroy``-CREATING
    branch (spec §5.2c, DD-17; INV-SPM-2 ``always``-mode hardening).

    Async methods carry NO ``@pytest.mark.asyncio``: the module-level
    ``pytest.mark.anyio`` marker (see file header) is authoritative. The brief's
    illustrative bare ``asyncio.sleep(0)`` "reach CREATING" sync point is replaced
    with the fake's ``create_reached`` gate for reliable scheduling under the
    ``ensure_future``+``shield`` CREATING task (identical rationale to
    ``TestBindNewFlight``); ``sleep(0)`` is kept only to let the *pre-lock*
    ``invalidate`` in ``destroy`` run while ``bind_new`` still holds the lock.
    """

    def test_flight_outcome_for_maps_only_delete_reasons(self):
        """Requirement #1: import the REAL ``DestroyReason`` enum and lock the
        mapping — a member whose name OR value contains 'delete' → 'delete';
        every other member → 'destroy'. Iterating the enum keeps this honest if
        new members are added later."""
        from app.application.services.sandbox_lifecycle_service import (
            _flight_outcome_for,
        )

        for reason in DestroyReason:
            expected = (
                "delete"
                if "delete" in reason.name.lower() or "delete" in reason.value.lower()
                else "destroy"
            )
            assert _flight_outcome_for(reason) == expected, reason
        # explicit anchors (brief §冻结决策1)
        assert _flight_outcome_for(DestroyReason.SESSION_DELETE) == "delete"
        assert _flight_outcome_for(DestroyReason.WATCHDOG_TIMEOUT) == "destroy"
        assert _flight_outcome_for(DestroyReason.FORCE_TERMINATE) == "destroy"

    async def test_delete_during_create_aborts_provision(self, svc_and_fakes):
        """场景④: bind_new stuck in ``create``; a delete-class ``destroy`` writes
        its intent BEFORE taking the lock → the in-flight provision's CAS-1 check
        aborts (dispose + rollback UNBOUND), and the delete tombstone refuses a
        second bind_new even before the session row is hard-deleted."""
        svc, fakes = svc_and_fakes
        fakes.sandbox_cls.create_gate = asyncio.Event()
        bind_task = asyncio.create_task(svc.bind_new(fakes.session_id))
        # reliable CREATING + active-flight sync point (see class docstring):
        await asyncio.wait_for(fakes.sandbox_cls.create_reached.wait(), timeout=2)
        destroy_task = asyncio.create_task(
            svc.destroy(fakes.session_id, reason=DELETE_REASON)  # real enum
        )
        await asyncio.sleep(0)  # let destroy run its pre-lock invalidate
        # intent-before-lock: signal + tombstone landed while bind_new holds lock
        assert svc._flights.get(fakes.session_id).invalidated == "delete"
        assert svc._flights.is_tombstoned(fakes.session_id)
        fakes.sandbox_cls.create_gate.set()
        with pytest.raises(SandboxProvisionInvalidated):
            await bind_task
        with pytest.raises(SandboxBindingMissing):  # destroy then acquires lock → UNBOUND
            await destroy_task
        assert fakes.sandbox_cls.created[-1].destroyed          # CAS-1 dispose
        assert fakes.binding_state(fakes.session_id) == UNBOUND
        with pytest.raises(SessionFinalizedError):  # delete恒 tombstone → 二次 bind_new 拒绝
            await svc.bind_new(fakes.session_id)

    async def test_absent_flight_delete_tombstones_then_bind_refused(self, svc_and_fakes):
        """场景⑤ full: UNBOUND ``destroy(delete)`` → ``SandboxBindingMissing`` while
        STILL registering the tombstone (invalidate('delete') always tombstones),
        so a late bind_new is refused; no container is ever created."""
        svc, fakes = svc_and_fakes  # UNBOUND, no flight
        with pytest.raises(SandboxBindingMissing):
            await svc.destroy(fakes.session_id, reason=DELETE_REASON)
        assert svc._flights.is_tombstoned(fakes.session_id)  # tombstoned despite raise
        with pytest.raises(SessionFinalizedError):
            await svc.bind_new(fakes.session_id)
        assert fakes.sandbox_cls.created == []

    async def test_user_stop_destroy_does_not_tombstone(self, svc_and_fakes):
        """A non-delete ``destroy`` on UNBOUND raises ``SandboxBindingMissing`` but
        does NOT tombstone — the session can still be bound afterwards."""
        svc, fakes = svc_and_fakes
        with pytest.raises(SandboxBindingMissing):
            await svc.destroy(fakes.session_id, reason=NON_DELETE_REASON)
        assert not svc._flights.is_tombstoned(fakes.session_id)
        await svc.bind_new(fakes.session_id)  # succeeds — no tombstone
        assert fakes.binding_state(fakes.session_id) == ACTIVE

    async def test_suspend_signals_quiesce_to_active_flight(self, svc_and_fakes):
        """Requirement #2: ``suspend`` invalidates an in-flight provision with
        'quiesce' BEFORE its lock; 'quiesce' never tombstones (the session may be
        resumed / re-bound later). The post-lock reject is the existing contract."""
        svc, fakes = svc_and_fakes
        flight = svc._flights.begin(fakes.session_id)  # simulate an in-flight provision
        with pytest.raises(SandboxLifecycleError):     # UNBOUND → suspend rejects post-lock
            await svc.suspend(fakes.session_id)
        assert flight.invalidated == "quiesce"         # pre-lock signal fired
        assert not svc._flights.is_tombstoned(fakes.session_id)

    async def test_suspend_signals_quiesce_before_lock(self, svc_and_fakes):
        """Requirement #2 (rigorous ordering): prove ``suspend`` fires its 'quiesce'
        invalidate STRICTLY BEFORE contending for the per-session lock. Park a real
        ``bind_new`` inside the held lock (create gate), spawn ``suspend``, yield one
        turn: the flight is already 'quiesce'-marked WHILE bind_new still holds the
        lock — impossible unless invalidate precedes lock acquisition (if it were
        inside the lock, suspend would block at the lock BEFORE reaching invalidate and
        the flight would stay un-marked). Mirrors ``test_delete_during_create_aborts_provision``."""
        svc, fakes = svc_and_fakes
        fakes.sandbox_cls.create_gate = asyncio.Event()
        bind_task = asyncio.create_task(svc.bind_new(fakes.session_id))
        # bind_new now holds the per-session lock, parked in create (flight begun):
        await asyncio.wait_for(fakes.sandbox_cls.create_reached.wait(), timeout=2)
        suspend_task = asyncio.create_task(svc.suspend(fakes.session_id))
        await asyncio.sleep(0)  # let suspend run its pre-lock invalidate (it then blocks on the lock)
        # intent-before-lock: quiesce signal landed while bind_new STILL holds the lock.
        flight = svc._flights.get(fakes.session_id)
        assert flight is not None and flight.invalidated == "quiesce"
        assert not svc._flights.is_tombstoned(fakes.session_id)  # quiesce never tombstones
        assert not suspend_task.done()                           # suspend still blocked on the lock
        fakes.sandbox_cls.create_gate.set()  # release bind_new → CAS aborts the provision
        with pytest.raises(SandboxProvisionInvalidated):
            await bind_task
        # suspend then acquires the lock; binding is UNBOUND (rolled back) → rejects.
        with pytest.raises(SandboxLifecycleError):
            await suspend_task
        assert fakes.binding_state(fakes.session_id) == UNBOUND

    def test_destroy_creating_dead_branch_removed(self):
        import inspect

        src = inspect.getsource(SandboxLifecycleService.destroy)
        assert "binding.state == CREATING" not in src
