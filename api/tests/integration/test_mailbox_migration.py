"""C3 PR-5 — control-plane default + rollback NULL-coalesce migration tests.

Spec sections covered:
- §11.1 — legacy ``suspend()`` removal is exercised indirectly via the
  publisher-path E2E (``test_mailbox_e2e.py``); not duplicated here.
- §11.2 — new subagent sessions default to
  ``subagent_control_plane='mailbox'`` once
  ``MAILBOX_SUPERVISOR_ENABLED=true``; root sessions never carry a
  control plane.
- §11.6 — rollback runbook: after the operator runs
  ``UPDATE sessions SET subagent_control_plane='legacy' WHERE ... AND
  subagent_control_plane='mailbox'``, the supervisor's rollback-stop
  check treats NULL rows (pre-C3 backfill) and explicit ``'legacy'``
  rows as both canonical legacy via
  ``(c.subagent_control_plane or 'legacy') == 'legacy'``.

Requires live PostgreSQL. Marked ``@pytest.mark.integration`` so the
unit gate (``-m 'not integration'``) skips the suite.
"""

from __future__ import annotations

import asyncio
import uuid as _uuid

import pytest

pytestmark = [pytest.mark.anyio, pytest.mark.integration]


# ──────────────────────────────────────────────────────────────────────
# §11.2 — default control_plane behavior
# ──────────────────────────────────────────────────────────────────────


async def test_pre_pr5_subagent_rows_with_null_plane_remain_legacy(
    db_session, sample_user
):
    """Spec §11.2 + migration backfill — NULL is canonical legacy.

    A subagent row whose ``subagent_control_plane`` was never set
    (pre-C3 backfill state) must persist NULL on disk; the supervisor's
    rollback stop check treats NULL identically to ``'legacy'`` via
    the ``(c.subagent_control_plane or 'legacy')`` coalesce (covered
    by ``test_rollback_check_stops_supervisor_when_all_children_legacy_or_null``).
    """
    from sqlalchemy import select

    from app.domain.models.session import SessionStatus
    from app.infrastructure.models.session import SessionModel

    # codex r4 [HIGH TEST] — subagent rows require parent_session_id
    # (CHECK constraint ``ck_sessions_worker_type_parent_invariant``
    # at api/alembic/versions/c1a_session_tree_expand.py:133). Build
    # a synthetic root first so the subagent row has a parent FK.
    root_sid = f"sess-pr5mig-root-{_uuid.uuid4().hex[:12]}"
    db_session.add(
        SessionModel(
            id=root_sid,
            user_id=sample_user.id,
            status=SessionStatus.RUNNING.value,
            title="pre-PR5 root for NULL-plane subagent",
            task_id=root_sid,
            execution_mode="foreground",
            execution_phase="running",
            retry_budget_remaining=3,
            was_background=False,
            worker_type="root",
        )
    )
    sid = f"sess-pr5mig-{_uuid.uuid4().hex[:12]}"
    orm = SessionModel(
        id=sid,
        user_id=sample_user.id,
        parent_session_id=root_sid,
        status=SessionStatus.RUNNING.value,
        title="pre-PR5 subagent with NULL plane",
        task_id=sid,
        execution_mode="foreground",
        execution_phase="running",
        retry_budget_remaining=3,
        was_background=False,
        worker_type="subagent",
        # subagent_control_plane intentionally unset — pre-C3 baseline.
    )
    db_session.add(orm)
    await db_session.flush()

    refreshed = await db_session.execute(
        select(SessionModel).where(SessionModel.id == sid)
    )
    row = refreshed.scalar_one()
    assert row.worker_type == "subagent"
    assert row.subagent_control_plane is None, (
        "pre-PR5 subagent row must persist NULL plane — backfill semantics"
    )


async def test_create_session_with_parent_defaults_subagent_to_mailbox(
    db_session, sample_user
):
    """Spec §11.2 — after PR-5 flips ``MAILBOX_SUPERVISOR_ENABLED=true``,
    new subagent children created via ``SessionService`` default to
    ``subagent_control_plane='mailbox'``.

    Driven through ``SessionService.create_session_with_parent`` so the
    three-tier flag precedence (settings stub > flag_reader > cached
    ``get_settings``) is exercised; the test pins the flag via the
    ``settings`` stub (tier 1).
    """
    from app.application.services.session_service import SessionService
    from app.domain.models.session import SessionStatus
    from app.infrastructure.models.session import SessionModel
    from app.infrastructure.repositories.db_session_repository import (
        DBSessionRepository,
    )

    class _StubSettings:
        mailbox_supervisor_enabled = True

    # Root session (worker_type='root' — no control_plane).
    root_sid = f"sess-pr5root-{_uuid.uuid4().hex[:12]}"
    db_session.add(
        SessionModel(
            id=root_sid,
            user_id=sample_user.id,
            status=SessionStatus.RUNNING.value,
            title="pr5 root",
            task_id=root_sid,
            execution_mode="foreground",
            execution_phase="running",
            retry_budget_remaining=3,
            was_background=False,
            worker_type="root",
        )
    )
    await db_session.flush()

    # SessionService writes via UoW; we need a UoW backed by our test
    # db_session so the assertion can read back the row.
    class _TestUow:
        def __init__(self, db_session) -> None:
            self.db_session = db_session
            self.session = DBSessionRepository(db_session=db_session)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return None

    class _TestUowFactory:
        def __init__(self, db_session) -> None:
            self._db_session = db_session

        def __call__(self):
            return _TestUow(self._db_session)

    svc = SessionService(
        uow_factory=_TestUowFactory(db_session),  # type: ignore[arg-type]
        settings=_StubSettings(),  # type: ignore[arg-type]
    )
    # codex r1 [F3, MEDIUM TEST] — ``create_session_with_parent`` raises
    # ``ValueError`` when ``tool_filter_preset`` is ``None`` (T12 / Phase 1
    # PR-X contract at session_service.py:154). Use ``'subagent_research'``
    # — the canonical preset for subagent children that this PR's
    # default-mailbox flow targets.
    child = await svc.create_session_with_parent(
        parent_session_id=root_sid,
        user_id=sample_user.id,
        title="pr5 child",
        tool_filter_preset="subagent_research",
    )

    assert child.worker_type == "subagent"
    assert child.subagent_control_plane == "mailbox", (
        "PR-5 default: subagent children default to mailbox when flag is True"
    )


async def test_root_sessions_have_null_control_plane(
    db_session, sample_user
):
    """Spec §11.2 — root sessions never carry ``subagent_control_plane``."""
    from sqlalchemy import select

    from app.domain.models.session import SessionStatus
    from app.infrastructure.models.session import SessionModel

    sid = f"sess-pr5rootonly-{_uuid.uuid4().hex[:12]}"
    orm = SessionModel(
        id=sid,
        user_id=sample_user.id,
        status=SessionStatus.RUNNING.value,
        title="pure root",
        task_id=sid,
        execution_mode="foreground",
        execution_phase="running",
        retry_budget_remaining=3,
        was_background=False,
        worker_type="root",
    )
    db_session.add(orm)
    await db_session.flush()

    refreshed = await db_session.execute(
        select(SessionModel).where(SessionModel.id == sid)
    )
    row = refreshed.scalar_one()
    assert row.worker_type == "root"
    assert row.subagent_control_plane is None


# ──────────────────────────────────────────────────────────────────────
# §11.6 — rollback NULL-coalesce stop check
# ──────────────────────────────────────────────────────────────────────


def _build_test_supervisor_context(
    root_sid: str,
    session_repo,
    stop_callback,
):
    """Construct a synthetic SupervisorContext sufficient for the
    rollback-stop check (no Redis / publisher / audit calls hit on
    this code path). Returns a built MailboxSupervisor."""
    from app.application.services.mailbox_supervisor import (
        MailboxSupervisor,
        SupervisorContext,
    )

    class _StubAuditRepo:
        async def insert_event(self, *a, **kw):
            return None

        async def find_event(self, *a, **kw):
            return None

    class _StubPublisher:
        async def publish(self, *a, **kw):
            return None

    class _StubSandboxLifecycle:
        async def destroy(self, *a, **kw):
            return None

    class _StubTelemetry:
        async def emit(self, *a, **kw):
            return None

    async def _stub_agent_callback(envelope):
        return None

    ctx = SupervisorContext(
        root_session_id=root_sid,
        pod_id="test-pod",
        instance_id="testinst",
        redis=None,  # type: ignore[arg-type] — rollback check never touches redis
        audit_repo=_StubAuditRepo(),  # type: ignore[arg-type]
        publisher=_StubPublisher(),  # type: ignore[arg-type]
        sandbox_lifecycle=_StubSandboxLifecycle(),  # type: ignore[arg-type]
        agent_service_callback=_stub_agent_callback,
        telemetry=_StubTelemetry(),  # type: ignore[arg-type]
        session_repo=session_repo,  # type: ignore[arg-type]
        stop_self_callback=stop_callback,
    )
    return MailboxSupervisor(ctx)


async def test_rollback_check_stops_supervisor_when_all_children_legacy_or_null(
    db_session, sample_user, session_repo
):
    """Spec §11.6 + R1 P2.2 — rollback stop check NULL-coalesces.

    Construct a root with two subagent children: one with
    ``subagent_control_plane=NULL`` (pre-C3 baseline) and one with
    ``'legacy'`` (post-rollback row). Both should be treated as legacy
    by the ``(c.subagent_control_plane or 'legacy') == 'legacy'``
    coalesce, so the rollback stop check fires ``stop_self_callback``.
    """
    from app.domain.models.session import SessionStatus
    from app.infrastructure.models.session import SessionModel

    root_sid = f"sess-pr5rb-{_uuid.uuid4().hex[:12]}"
    db_session.add(
        SessionModel(
            id=root_sid,
            user_id=sample_user.id,
            status=SessionStatus.RUNNING.value,
            title="pr5 rollback root",
            task_id=root_sid,
            execution_mode="foreground",
            execution_phase="running",
            retry_budget_remaining=3,
            was_background=False,
            worker_type="root",
        )
    )
    sub_null_sid = f"sess-pr5rb-null-{_uuid.uuid4().hex[:12]}"
    db_session.add(
        SessionModel(
            id=sub_null_sid,
            user_id=sample_user.id,
            parent_session_id=root_sid,
            status=SessionStatus.RUNNING.value,
            title="pr5 rollback NULL subagent",
            task_id=sub_null_sid,
            execution_mode="foreground",
            execution_phase="running",
            retry_budget_remaining=3,
            was_background=False,
            worker_type="subagent",
            subagent_control_plane=None,
        )
    )
    sub_legacy_sid = f"sess-pr5rb-legacy-{_uuid.uuid4().hex[:12]}"
    db_session.add(
        SessionModel(
            id=sub_legacy_sid,
            user_id=sample_user.id,
            parent_session_id=root_sid,
            status=SessionStatus.RUNNING.value,
            title="pr5 rollback legacy subagent",
            task_id=sub_legacy_sid,
            execution_mode="foreground",
            execution_phase="running",
            retry_budget_remaining=3,
            was_background=False,
            worker_type="subagent",
            subagent_control_plane="legacy",
        )
    )
    await db_session.flush()

    stopped: list[str] = []

    async def _stop_cb() -> None:
        stopped.append(root_sid)

    sup = _build_test_supervisor_context(root_sid, session_repo, _stop_cb)
    await sup._check_should_stop_for_rollback()
    # codex r3 [F3, HIGH TEST] — the supervisor schedules the
    # stop_self_callback via ``asyncio.create_task`` (self-cancel
    # safety: awaiting registry.stop from inside the supervisor's own
    # run task would deadlock). The fire-and-forget task doesn't run
    # until the current task yields, so the test must yield before
    # asserting on the callback's side effect.
    await asyncio.sleep(0)

    assert stopped == [root_sid], (
        "rollback stop check must fire stop_self_callback when all "
        "subagent children are legacy-or-NULL (R1 P2.2 NULL-coalesce)"
    )
    assert sup._stopping.is_set(), (
        "rollback stop must ALSO set _stopping eagerly so the run "
        "loop exits even if cancel propagation is delayed"
    )


async def test_rollback_check_does_not_stop_when_a_child_is_still_mailbox(
    db_session, sample_user, session_repo
):
    """Negative case — at least one mailbox child means the supervisor
    still has work to do, so the rollback check is a no-op. Guards
    against accidental loosening of the all-legacy predicate (e.g.
    ``all`` → ``any`` typo)."""
    from app.domain.models.session import SessionStatus
    from app.infrastructure.models.session import SessionModel

    root_sid = f"sess-pr5rb2-{_uuid.uuid4().hex[:12]}"
    db_session.add(
        SessionModel(
            id=root_sid,
            user_id=sample_user.id,
            status=SessionStatus.RUNNING.value,
            title="pr5 rollback no-stop root",
            task_id=root_sid,
            execution_mode="foreground",
            execution_phase="running",
            retry_budget_remaining=3,
            was_background=False,
            worker_type="root",
        )
    )
    sub_mailbox_sid = f"sess-pr5rb2-mb-{_uuid.uuid4().hex[:12]}"
    db_session.add(
        SessionModel(
            id=sub_mailbox_sid,
            user_id=sample_user.id,
            parent_session_id=root_sid,
            status=SessionStatus.RUNNING.value,
            title="pr5 rollback no-stop mailbox subagent",
            task_id=sub_mailbox_sid,
            execution_mode="foreground",
            execution_phase="running",
            retry_budget_remaining=3,
            was_background=False,
            worker_type="subagent",
            subagent_control_plane="mailbox",
        )
    )
    sub_legacy_sid = f"sess-pr5rb2-lg-{_uuid.uuid4().hex[:12]}"
    db_session.add(
        SessionModel(
            id=sub_legacy_sid,
            user_id=sample_user.id,
            parent_session_id=root_sid,
            status=SessionStatus.RUNNING.value,
            title="pr5 rollback no-stop legacy subagent",
            task_id=sub_legacy_sid,
            execution_mode="foreground",
            execution_phase="running",
            retry_budget_remaining=3,
            was_background=False,
            worker_type="subagent",
            subagent_control_plane="legacy",
        )
    )
    await db_session.flush()

    stopped: list[str] = []

    async def _stop_cb() -> None:
        stopped.append(root_sid)

    sup = _build_test_supervisor_context(root_sid, session_repo, _stop_cb)
    await sup._check_should_stop_for_rollback()
    # codex r3 [F3, HIGH TEST] — yield even on the negative case so
    # that any unintentionally scheduled task (regression: predicate
    # loosened to ``any`` instead of ``all``) gets a chance to run
    # before we assert ``stopped == []``.
    await asyncio.sleep(0)

    assert stopped == [], (
        "rollback stop check must NOT fire when at least one subagent "
        "child is still mailbox-plane"
    )
    assert not sup._stopping.is_set(), (
        "live-mailbox no-stop guard: _stopping must also remain unset "
        "(belt-and-suspenders against a regression that signals stop "
        "via _stopping while skipping the callback)"
    )


async def test_rollback_check_ignores_terminal_mailbox_children(
    db_session, sample_user, session_repo
):
    """codex r6 [MEDIUM TEST] — terminal mailbox rows must NOT block
    rollback exit.

    Spec §11.6 rollback SQL leaves terminal rows on ``'mailbox'``
    because their destroy already ran (``completed``/``timed_out``
    are not in the SQL WHERE clause). The supervisor's predicate
    must EXCLUDE those rows from the all-legacy check, otherwise a
    completed mailbox child would block supervisor exit forever
    after rollback.

    Setup: one ``completed`` mailbox child + one ``timed_out`` mailbox
    child + one ``running`` legacy child + one ``running`` NULL child.
    Both terminal rows must be excluded (otherwise a regression that
    only excludes ``completed`` would slip through); the two live
    rows both coalesce to legacy; supervisor stops.
    """
    from app.domain.models.session import SessionStatus
    from app.infrastructure.models.session import SessionModel

    root_sid = f"sess-pr5rb3-{_uuid.uuid4().hex[:12]}"
    db_session.add(
        SessionModel(
            id=root_sid,
            user_id=sample_user.id,
            status=SessionStatus.RUNNING.value,
            title="pr5 rollback terminal-excluded root",
            task_id=root_sid,
            execution_mode="foreground",
            execution_phase="running",
            retry_budget_remaining=3,
            was_background=False,
            worker_type="root",
        )
    )
    # codex r7 [MEDIUM TEST] — cover BOTH terminal statuses so a
    # regression that only excludes ``completed`` (and forgets
    # ``timed_out``) would be caught here, not just in the separate
    # all-terminal no-stop test.
    sub_completed_sid = f"sess-pr5rb3-completed-{_uuid.uuid4().hex[:12]}"
    db_session.add(
        SessionModel(
            id=sub_completed_sid,
            user_id=sample_user.id,
            parent_session_id=root_sid,
            status=SessionStatus.COMPLETED.value,
            title="pr5 rollback completed mailbox subagent",
            task_id=sub_completed_sid,
            execution_mode="foreground",
            execution_phase="completed",
            retry_budget_remaining=3,
            was_background=False,
            worker_type="subagent",
            subagent_control_plane="mailbox",
        )
    )
    sub_timedout_sid = f"sess-pr5rb3-timedout-{_uuid.uuid4().hex[:12]}"
    db_session.add(
        SessionModel(
            id=sub_timedout_sid,
            user_id=sample_user.id,
            parent_session_id=root_sid,
            status=SessionStatus.TIMED_OUT.value,
            title="pr5 rollback timed_out mailbox subagent",
            task_id=sub_timedout_sid,
            execution_mode="foreground",
            execution_phase="timed_out",
            retry_budget_remaining=3,
            was_background=False,
            worker_type="subagent",
            subagent_control_plane="mailbox",
        )
    )
    # Live legacy subagent (post-rollback flip).
    sub_legacy_sid = f"sess-pr5rb3-lg-{_uuid.uuid4().hex[:12]}"
    db_session.add(
        SessionModel(
            id=sub_legacy_sid,
            user_id=sample_user.id,
            parent_session_id=root_sid,
            status=SessionStatus.RUNNING.value,
            title="pr5 rollback running legacy subagent",
            task_id=sub_legacy_sid,
            execution_mode="foreground",
            execution_phase="running",
            retry_budget_remaining=3,
            was_background=False,
            worker_type="subagent",
            subagent_control_plane="legacy",
        )
    )
    # Live NULL subagent (pre-C3 backfill).
    sub_null_sid = f"sess-pr5rb3-null-{_uuid.uuid4().hex[:12]}"
    db_session.add(
        SessionModel(
            id=sub_null_sid,
            user_id=sample_user.id,
            parent_session_id=root_sid,
            status=SessionStatus.RUNNING.value,
            title="pr5 rollback running NULL subagent",
            task_id=sub_null_sid,
            execution_mode="foreground",
            execution_phase="running",
            retry_budget_remaining=3,
            was_background=False,
            worker_type="subagent",
            subagent_control_plane=None,
        )
    )
    await db_session.flush()

    stopped: list[str] = []

    async def _stop_cb() -> None:
        stopped.append(root_sid)

    sup = _build_test_supervisor_context(root_sid, session_repo, _stop_cb)
    await sup._check_should_stop_for_rollback()
    await asyncio.sleep(0)

    assert stopped == [root_sid], (
        "rollback stop check must EXCLUDE terminal mailbox rows from the "
        "all-legacy predicate (their destroy already ran); only the two "
        "live legacy/NULL rows count, and both are legacy → stop"
    )
    assert sup._stopping.is_set(), (
        "terminal-excluded stop must ALSO set _stopping eagerly"
    )


async def test_rollback_check_does_not_stop_when_all_subagents_terminal(
    db_session, sample_user, session_repo
):
    """codex r5/r6 [HIGH CONTRACT regression guard] — the supervisor
    must NOT stop when all subagent children are terminal but no
    rollback has happened.

    The runner's terminal sequence at
    ``agent_task_runner._terminal_op`` commits the DB status BEFORE
    publishing the terminal envelope. A supervisor that stopped on
    "all DB terminal" could miss the envelope and lose the
    ``destroy()`` call. R5 reverted the all-terminal stop branch;
    this test pins the invariant so future refactors don't reopen
    the race.

    Setup: a root with one or more terminal mailbox subagent
    children, NO live subagents, NO rollback SQL applied. The
    rollback check must early-return without stopping.
    """
    from app.domain.models.session import SessionStatus
    from app.infrastructure.models.session import SessionModel

    root_sid = f"sess-pr5rb4-{_uuid.uuid4().hex[:12]}"
    db_session.add(
        SessionModel(
            id=root_sid,
            user_id=sample_user.id,
            status=SessionStatus.RUNNING.value,
            title="pr5 rollback all-terminal-no-stop root",
            task_id=root_sid,
            execution_mode="foreground",
            execution_phase="running",
            retry_budget_remaining=3,
            was_background=False,
            worker_type="root",
        )
    )
    sub_completed_sid = f"sess-pr5rb4-c-{_uuid.uuid4().hex[:12]}"
    db_session.add(
        SessionModel(
            id=sub_completed_sid,
            user_id=sample_user.id,
            parent_session_id=root_sid,
            status=SessionStatus.COMPLETED.value,
            title="pr5 rollback completed mailbox subagent",
            task_id=sub_completed_sid,
            execution_mode="foreground",
            execution_phase="completed",
            retry_budget_remaining=3,
            was_background=False,
            worker_type="subagent",
            subagent_control_plane="mailbox",
        )
    )
    sub_timedout_sid = f"sess-pr5rb4-t-{_uuid.uuid4().hex[:12]}"
    db_session.add(
        SessionModel(
            id=sub_timedout_sid,
            user_id=sample_user.id,
            parent_session_id=root_sid,
            status=SessionStatus.TIMED_OUT.value,
            title="pr5 rollback timed_out mailbox subagent",
            task_id=sub_timedout_sid,
            execution_mode="foreground",
            execution_phase="timed_out",
            retry_budget_remaining=3,
            was_background=False,
            worker_type="subagent",
            subagent_control_plane="mailbox",
        )
    )
    await db_session.flush()

    stopped: list[str] = []

    async def _stop_cb() -> None:
        stopped.append(root_sid)

    sup = _build_test_supervisor_context(root_sid, session_repo, _stop_cb)
    await sup._check_should_stop_for_rollback()
    await asyncio.sleep(0)

    assert stopped == [], (
        "rollback stop check must NOT fire when all subagents are terminal "
        "without a rollback — runner's DB-commit precedes envelope publish, "
        "so stopping here could lose destroy() (R5 race-fix regression guard)"
    )
    # codex r7 [MEDIUM TEST] — the real run loop exits on
    # ``_stopping.is_set()``, not just the callback being called. Pin
    # that too so a regression that sets ``_stopping`` without
    # scheduling the callback can't slip through.
    assert not sup._stopping.is_set(), (
        "all-terminal no-stop guard: rollback check must NOT set "
        "_stopping either — only rollback (live-children-all-legacy) "
        "should signal supervisor exit"
    )
