"""C3 PR-4.5 — AgentTaskRunner child publisher path (spec §9.1 + §11.1).

codex r10 [R10-2, HIGH TEST] — the integration smoke test
``tests/integration/test_mailbox_e2e.py`` covers only the parent-side
publisher (SubagentResearchService → Redis Stream). The full T10
lifecycle (SPAWN_REQUEST → SPAWN_ACK → PROGRESS_UPDATE → RESULT_READY
→ destroy(SUBAGENT_TERMINAL_RESULT)) requires a real AgentTaskRunner
mailbox-plane child, which needs a real LLM + sandbox harness.

These unit tests pin the AgentTaskRunner child-publisher contracts
without that harness:

* ``_is_mailbox_plane_child`` re-reads the DB row on each call (no
  caching of the predicate — §11.6 rollback can flip it mid-run).
* ``_maybe_spawn_child_publisher`` ensures the supervisor exists,
  emits SPAWN_ACK, and starts the heartbeat task.
* ``_maybe_stop_child_publisher`` cleans up heartbeat
  unconditionally and emits RESULT_READY or CANCEL_ACK depending on
  the terminal_reason mapping.

The runner is bypassed via ``__new__`` so the heavy constructor wiring
(LLM, sandbox, MCP, A2A, etc.) is not exercised; only the publisher
attrs are stubbed in.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from app.domain.models.mailbox_envelope import (
    MailboxEnvelopeType,
    ProducerRole,
    ResultReadyOutcome,
)
from app.domain.models.session import Session, SessionStatus
from app.domain.services.agent_task_runner import AgentTaskRunner


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class _CapturingPublisher:
    def __init__(self) -> None:
        self.published: list = []

    async def publish(self, envelope) -> None:
        self.published.append(envelope)


class _FakeRegistry:
    def __init__(self) -> None:
        self.spawn_calls: list[str] = []

    async def spawn(self, root_id: str) -> None:
        self.spawn_calls.append(root_id)

    async def stop(self, root_id: str) -> None:
        del root_id


class _FakeSessionRepo:
    def __init__(self, session: Session | None) -> None:
        self._session = session

    async def get_by_id(self, sid: str) -> Session | None:
        del sid
        return self._session


class _FakeUoW:
    def __init__(self, session: Session | None) -> None:
        self._session = session

    async def __aenter__(self) -> "_FakeUoW":
        return self

    async def __aexit__(self, *args: Any) -> None:
        del args

    @property
    def session(self) -> _FakeSessionRepo:
        return _FakeSessionRepo(self._session)


def _build_runner(
    *,
    session_id: str,
    session_row: Session | None,
    publisher: Any,
    registry: Any | None = None,
) -> AgentTaskRunner:
    runner = AgentTaskRunner.__new__(AgentTaskRunner)
    runner._session_id = session_id
    runner._mailbox_publisher = publisher
    runner._supervisor_registry = registry
    runner._uow_factory = lambda: _FakeUoW(session_row)
    runner._heartbeat_task = None
    runner._heartbeat_handle = None
    runner._cached_session_for_publisher = None
    runner._spawn_correlation_id = f"spawn:{session_id}"
    runner._external_terminal_owner = False
    runner._external_heartbeat_owner = False
    return runner


def _mailbox_child_row(session_id: str) -> Session:
    return Session(
        id=session_id,
        user_id="u1",
        worker_type="subagent",
        subagent_control_plane="mailbox",
        parent_session_id="root-1",
        status=SessionStatus.RUNNING,
    )


def _legacy_child_row(session_id: str) -> Session:
    return Session(
        id=session_id,
        user_id="u1",
        worker_type="subagent",
        subagent_control_plane="legacy",
        parent_session_id="root-1",
        status=SessionStatus.RUNNING,
    )


# ── _is_mailbox_plane_child ──────────────────────────────────────────


@pytest.mark.anyio
async def test_is_mailbox_plane_child_true_for_mailbox_subagent() -> None:
    runner = _build_runner(
        session_id="c1",
        session_row=_mailbox_child_row("c1"),
        publisher=_CapturingPublisher(),
    )
    assert await runner._is_mailbox_plane_child() is True


@pytest.mark.anyio
async def test_is_mailbox_plane_child_false_for_legacy_subagent() -> None:
    runner = _build_runner(
        session_id="c1",
        session_row=_legacy_child_row("c1"),
        publisher=_CapturingPublisher(),
    )
    assert await runner._is_mailbox_plane_child() is False


@pytest.mark.anyio
async def test_is_mailbox_plane_child_not_cached_so_rollback_observed() -> None:
    """codex r3 [R3-5, HIGH ARCH] — predicate must re-read the row on
    every call so a §11.6 rollback (mailbox → legacy) mid-run is
    immediately visible to the terminal publish gate.
    """
    holder: dict[str, Session | None] = {"row": _mailbox_child_row("c1")}

    class _MutableUoW:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            del args

        @property
        def session(self) -> _FakeSessionRepo:
            return _FakeSessionRepo(holder["row"])

    runner = AgentTaskRunner.__new__(AgentTaskRunner)
    runner._session_id = "c1"
    runner._mailbox_publisher = _CapturingPublisher()
    runner._supervisor_registry = None
    runner._uow_factory = lambda: _MutableUoW()
    runner._heartbeat_task = None
    runner._heartbeat_handle = None
    runner._cached_session_for_publisher = None
    runner._spawn_correlation_id = "spawn:c1"

    assert await runner._is_mailbox_plane_child() is True
    # Simulate §11.6 rollback.
    holder["row"] = _legacy_child_row("c1")
    assert await runner._is_mailbox_plane_child() is False


# ── _maybe_spawn_child_publisher ─────────────────────────────────────


@pytest.mark.anyio
async def test_spawn_publisher_emits_spawn_ack_and_starts_heartbeat() -> None:
    pub = _CapturingPublisher()
    registry = _FakeRegistry()
    runner = _build_runner(
        session_id="c1",
        session_row=_mailbox_child_row("c1"),
        publisher=pub,
        registry=registry,
    )

    await runner._maybe_spawn_child_publisher()
    try:
        # Supervisor ensure ran for the parent root.
        assert registry.spawn_calls == ["root-1"]
        # SPAWN_ACK envelope is in the publisher.
        ack_envs = [
            e for e in pub.published
            if e.type == MailboxEnvelopeType.SPAWN_ACK
        ]
        assert len(ack_envs) == 1, f"expected exactly 1 SPAWN_ACK; got {pub.published}"
        ack = ack_envs[0]
        assert ack.parent_session_id == "root-1"
        assert ack.child_session_id == "c1"
        assert ack.producer_role == ProducerRole.CHILD_AGENT
        assert ack.payload["accepted"] is True
        assert ack.payload["sandbox_ready"] is True
        # Heartbeat task is alive.
        assert runner._heartbeat_handle is not None
        assert not runner._heartbeat_handle.done()
    finally:
        await runner._cleanup_heartbeat_task()


@pytest.mark.anyio
async def test_external_heartbeat_owner_keeps_spawn_ack_without_inner_heartbeat() -> None:
    pub = _CapturingPublisher()
    registry = _FakeRegistry()
    runner = _build_runner(
        session_id="c1",
        session_row=_mailbox_child_row("c1"),
        publisher=pub,
        registry=registry,
    )
    runner._external_heartbeat_owner = True

    await runner._maybe_spawn_child_publisher()

    assert registry.spawn_calls == ["root-1"]
    assert [e.type for e in pub.published] == [MailboxEnvelopeType.SPAWN_ACK]
    assert runner._heartbeat_task is None
    assert runner._heartbeat_handle is None


@pytest.mark.anyio
async def test_external_terminal_owner_never_publishes_inner_terminal() -> None:
    pub = _CapturingPublisher()
    runner = _build_runner(
        session_id="c1",
        session_row=_mailbox_child_row("c1"),
        publisher=pub,
        registry=_FakeRegistry(),
    )
    runner._external_terminal_owner = True

    await runner._maybe_stop_child_publisher(
        SessionStatus.COMPLETED, terminal_reason="natural"
    )

    assert pub.published == []


@pytest.mark.parametrize(
    ("external_terminal_owner", "external_heartbeat_owner"),
    [(False, False), (False, True), (True, False), (True, True)],
)
@pytest.mark.anyio
async def test_independent_ownership_quadrants_control_inner_wire_lifecycle(
    external_terminal_owner: bool,
    external_heartbeat_owner: bool,
) -> None:
    """SPAWN_ACK remains inner-owned in all quadrants; heartbeat and terminal
    envelope ownership are independently controlled by their respective flag.
    """
    pub = _CapturingPublisher()
    runner = _build_runner(
        session_id="quadrant-child",
        session_row=_mailbox_child_row("quadrant-child"),
        publisher=pub,
        registry=_FakeRegistry(),
    )
    runner._external_terminal_owner = external_terminal_owner
    runner._external_heartbeat_owner = external_heartbeat_owner

    await runner._maybe_spawn_child_publisher()
    await asyncio.sleep(0)
    await runner._maybe_stop_child_publisher(
        SessionStatus.COMPLETED, terminal_reason="natural"
    )

    types = [envelope.type for envelope in pub.published]
    assert types.count(MailboxEnvelopeType.SPAWN_ACK) == 1
    assert (MailboxEnvelopeType.PROGRESS_UPDATE in types) is (
        not external_heartbeat_owner
    )
    assert (MailboxEnvelopeType.RESULT_READY in types) is (
        not external_terminal_owner
    )
    assert runner._heartbeat_task is None
    assert runner._heartbeat_handle is None


@pytest.mark.anyio
async def test_spawn_publisher_noop_for_legacy_child() -> None:
    pub = _CapturingPublisher()
    registry = _FakeRegistry()
    runner = _build_runner(
        session_id="c1",
        session_row=_legacy_child_row("c1"),
        publisher=pub,
        registry=registry,
    )

    await runner._maybe_spawn_child_publisher()
    assert pub.published == []
    assert registry.spawn_calls == []
    assert runner._heartbeat_handle is None


@pytest.mark.anyio
async def test_child_side_spawn_failure_continues_to_publish() -> None:
    """codex r18 [R18-2, HIGH ARCH] — child-side ``registry.spawn``
    is redundant best-effort (parent already prewires). When it
    raises (e.g. transient registry error on a supervisor that's
    actually healthy), the publisher MUST continue — fail-closing
    here disables a working publisher and breaks the resume path.
    Earlier rounds wrongly early-returned on this exception.
    """

    class _RaisingRegistry:
        async def spawn(self, root_id: str) -> None:
            raise RuntimeError("transient registry error")

        async def stop(self, root_id: str) -> None:
            del root_id

    pub = _CapturingPublisher()
    runner = _build_runner(
        session_id="c1",
        session_row=_mailbox_child_row("c1"),
        publisher=pub,
        registry=_RaisingRegistry(),
    )

    await runner._maybe_spawn_child_publisher()
    try:
        # Despite spawn raising, SPAWN_ACK was still published and
        # the heartbeat task started.
        ack_envs = [
            e for e in pub.published if e.type == MailboxEnvelopeType.SPAWN_ACK
        ]
        assert len(ack_envs) == 1
        assert runner._heartbeat_handle is not None
    finally:
        await runner._cleanup_heartbeat_task()


@pytest.mark.anyio
async def test_child_side_no_registry_skips_publish() -> None:
    """codex r13 [R13-4] / r18 [R18-2] — missing registry means we
    genuinely have no way to confirm a consumer exists, so the
    publisher must skip (the orphan reconcile path is the fallback
    for any in-flight envelopes)."""
    pub = _CapturingPublisher()
    runner = _build_runner(
        session_id="c1",
        session_row=_mailbox_child_row("c1"),
        publisher=pub,
        registry=None,
    )

    await runner._maybe_spawn_child_publisher()
    assert pub.published == []
    assert runner._heartbeat_handle is None


@pytest.mark.anyio
async def test_spawn_ack_publish_failure_does_not_block_heartbeat() -> None:
    """codex r3 [R3-4, HIGH CONTRACT] — a transient SPAWN_ACK publish
    failure must NOT abort heartbeat start; the heartbeat is the only
    liveness signal the supervisor has after a missed SPAWN_ACK.
    """

    class _FlakyAckPublisher(_CapturingPublisher):
        async def publish(self, envelope) -> None:
            if envelope.type == MailboxEnvelopeType.SPAWN_ACK:
                raise RuntimeError("transient publish failure")
            self.published.append(envelope)

    pub = _FlakyAckPublisher()
    runner = _build_runner(
        session_id="c1",
        session_row=_mailbox_child_row("c1"),
        publisher=pub,
        registry=_FakeRegistry(),
    )

    await runner._maybe_spawn_child_publisher()
    try:
        # Heartbeat task started despite ACK failure.
        assert runner._heartbeat_handle is not None
        assert not runner._heartbeat_handle.done()
    finally:
        await runner._cleanup_heartbeat_task()


# ── _maybe_stop_child_publisher terminal mapping ─────────────────────


@pytest.mark.anyio
async def test_terminal_natural_completion_emits_result_ready_success() -> None:
    pub = _CapturingPublisher()
    runner = _build_runner(
        session_id="c1",
        session_row=_mailbox_child_row("c1"),
        publisher=pub,
        registry=_FakeRegistry(),
    )

    await runner._maybe_stop_child_publisher(
        SessionStatus.COMPLETED, terminal_reason="natural"
    )

    terminal = [e for e in pub.published if e.type == MailboxEnvelopeType.RESULT_READY]
    assert len(terminal) == 1
    assert terminal[0].payload["outcome"] == ResultReadyOutcome.SUCCESS.value


@pytest.mark.anyio
async def test_terminal_runner_error_emits_result_ready_failed() -> None:
    """codex r4 [R4-4] / r21 [R21-1, HIGH CONTRACT] — runner exception
    path sets ``_runner_exception_terminal=True`` so the mailbox audit
    reports FAILED. The domain-model ``terminal_reason`` Literal only
    accepts a fixed set of values (``"runner_error"`` is NOT one of
    them); the flag-based override keeps the wire-side outcome
    correct without violating the domain contract.
    """
    pub = _CapturingPublisher()
    runner = _build_runner(
        session_id="c1",
        session_row=_mailbox_child_row("c1"),
        publisher=pub,
        registry=_FakeRegistry(),
    )
    runner._runner_exception_terminal = True

    await runner._maybe_stop_child_publisher(
        SessionStatus.COMPLETED, terminal_reason=None
    )

    terminal = [e for e in pub.published if e.type == MailboxEnvelopeType.RESULT_READY]
    assert len(terminal) == 1
    assert terminal[0].payload["outcome"] == ResultReadyOutcome.FAILED.value


@pytest.mark.anyio
async def test_terminal_user_cancel_emits_cancel_ack_cancelled() -> None:
    pub = _CapturingPublisher()
    runner = _build_runner(
        session_id="c1",
        session_row=_mailbox_child_row("c1"),
        publisher=pub,
        registry=_FakeRegistry(),
    )

    await runner._maybe_stop_child_publisher(
        SessionStatus.COMPLETED, terminal_reason="user_cancel"
    )

    terminal = [e for e in pub.published if e.type == MailboxEnvelopeType.CANCEL_ACK]
    assert len(terminal) == 1
    assert terminal[0].payload["final_state"] == "cancelled"


@pytest.mark.anyio
async def test_terminal_timed_out_emits_result_ready_failed() -> None:
    pub = _CapturingPublisher()
    runner = _build_runner(
        session_id="c1",
        session_row=_mailbox_child_row("c1"),
        publisher=pub,
        registry=_FakeRegistry(),
    )

    await runner._maybe_stop_child_publisher(
        SessionStatus.TIMED_OUT, terminal_reason="watchdog_timeout"
    )

    terminal = [e for e in pub.published if e.type == MailboxEnvelopeType.RESULT_READY]
    assert len(terminal) == 1
    assert terminal[0].payload["outcome"] == ResultReadyOutcome.FAILED.value


@pytest.mark.anyio
async def test_full_lifecycle_spawn_heartbeat_terminal_sequence() -> None:
    """codex r17–r23 [HIGH TEST recurring] — substitute for the
    deferred T10 full E2E. Drives ``_maybe_spawn_child_publisher``
    followed by enough wall-clock time for at least one heartbeat
    PROGRESS_UPDATE, then ``_maybe_stop_child_publisher`` with a
    natural-completion terminal, and asserts the canonical envelope
    sequence:

      SPAWN_ACK → ≥1 PROGRESS_UPDATE(HEARTBEAT) → RESULT_READY(SUCCESS)

    With NO CANCEL_ACK on the success path. This is the wire-level
    contract the supervisor consumer relies on; it pins
    ``_pr4_5_agent_service_callback`` and ``CancelAckHandler``
    invariants by composition.
    """
    pub = _CapturingPublisher()
    registry = _FakeRegistry()
    runner = _build_runner(
        session_id="c1",
        session_row=_mailbox_child_row("c1"),
        publisher=pub,
        registry=registry,
    )
    # Shrink heartbeat interval so the test finishes inside a
    # reasonable wall-clock budget.
    from app.domain.services.child_heartbeat_task import ChildHeartbeatTask
    orig_init = ChildHeartbeatTask.__init__

    def _fast_init(self, publisher, parent_session_id, child_session_id, **kwargs):
        kwargs["interval_seconds"] = 0.03
        orig_init(self, publisher, parent_session_id, child_session_id, **kwargs)

    ChildHeartbeatTask.__init__ = _fast_init  # type: ignore[method-assign]
    try:
        await runner._maybe_spawn_child_publisher()
        # Let the heartbeat task fire at least once.
        await asyncio.sleep(0.10)
        await runner._maybe_stop_child_publisher(
            SessionStatus.COMPLETED, terminal_reason="natural"
        )
    finally:
        ChildHeartbeatTask.__init__ = orig_init  # type: ignore[method-assign]

    seen_types = [e.type for e in pub.published]

    # 1. SPAWN_ACK is the first envelope.
    assert seen_types[0] == MailboxEnvelopeType.SPAWN_ACK, (
        f"first envelope must be SPAWN_ACK; got {seen_types}"
    )
    # 2. At least one PROGRESS_UPDATE(HEARTBEAT) between SPAWN_ACK
    #    and the terminal.
    heartbeats = [
        e for e in pub.published
        if e.type == MailboxEnvelopeType.PROGRESS_UPDATE
        and e.payload.get("kind") == "heartbeat"
    ]
    assert len(heartbeats) >= 1, (
        f"expected ≥1 heartbeat between spawn and terminal; got {seen_types}"
    )
    # 3. RESULT_READY(SUCCESS) is the final envelope.
    terminal = [
        e for e in pub.published if e.type == MailboxEnvelopeType.RESULT_READY
    ]
    assert len(terminal) == 1, f"expected exactly 1 RESULT_READY; got {seen_types}"
    assert terminal[0].payload["outcome"] == ResultReadyOutcome.SUCCESS.value
    # 4. NO CANCEL_ACK on the success path (would indicate a
    #    spurious supervisor-terminate marker leaked from another
    #    test or a contract drift).
    assert not any(
        e.type == MailboxEnvelopeType.CANCEL_ACK for e in pub.published
    ), f"success path must not emit CANCEL_ACK; got {seen_types}"


@pytest.mark.anyio
async def test_terminal_skipped_when_no_registry() -> None:
    """codex r23 [R23-2, MEDIUM ARCH] — terminal publish must check
    ``_supervisor_registry`` is wired, same as the spawn path. If
    registry is None we have no confirmed consumer, so publishing
    RESULT_READY/CANCEL_ACK would just bloat the stream until the
    next pod-start orphan_reconcile sweep.
    """
    pub = _CapturingPublisher()
    runner = _build_runner(
        session_id="c1",
        session_row=_mailbox_child_row("c1"),
        publisher=pub,
        registry=None,
    )
    # Don't run spawn (which would skip due to missing registry too);
    # exercise the terminal gate directly.
    await runner._maybe_stop_child_publisher(
        SessionStatus.COMPLETED, terminal_reason="natural"
    )
    assert pub.published == []


@pytest.mark.anyio
async def test_terminal_cleans_up_heartbeat_unconditionally() -> None:
    """codex r4 [R4-2, HIGH PERF] / r5 [R5-2] — heartbeat must be
    stopped on terminal entry even when the predicate says we're no
    longer mailbox-plane (e.g. §11.6 mid-run rollback). Otherwise the
    asyncio task outlives the runner and keeps emitting PROGRESS_UPDATE.
    """
    pub = _CapturingPublisher()
    runner = _build_runner(
        session_id="c1",
        session_row=_legacy_child_row("c1"),  # currently legacy
        publisher=pub,
    )

    started_event = asyncio.Event()
    stopped_event = asyncio.Event()

    class _FakeHeartbeat:
        async def stop(self) -> None:
            stopped_event.set()

    async def _run_until_stop() -> None:
        started_event.set()
        try:
            await asyncio.wait_for(stopped_event.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            return

    runner._heartbeat_task = _FakeHeartbeat()
    runner._heartbeat_handle = asyncio.create_task(_run_until_stop())
    await started_event.wait()

    # Row is legacy; publisher gate skips RESULT_READY/CANCEL_ACK but
    # heartbeat cleanup MUST still run.
    await runner._maybe_stop_child_publisher(
        SessionStatus.COMPLETED, terminal_reason="natural"
    )

    assert stopped_event.is_set()
    # No terminal envelope on stream (legacy row).
    assert pub.published == []
    # Handle nulled out for re-entry safety.
    assert runner._heartbeat_handle is None
