"""C3 PR-4.5 — R1 P0.2 contract test.

When ``SubagentResearchService.run_research`` is called against a root that
predates the supervisor (long-running pre-PR-5 root), the service MUST
call ``supervisor_registry.spawn(root_id)`` BEFORE publishing any
SPAWN_REQUEST envelope. Otherwise, the envelopes go into a Redis Stream
that has no consumer group, leaving them stranded until the next pod
restart sweeps via ``reconcile_orphans``.

This is a direct unit test of ``_ensure_supervisor_and_publish_spawns``
ordering: we record the call sequence on a fake registry + publisher and
assert spawn is observed before any publish call. Full E2E coverage via
the HTTP endpoint lives in ``tests/integration/test_mailbox_e2e.py``.
"""

from __future__ import annotations

from typing import Any

import pytest

from app.application.services.subagent_research_service import (
    SubagentResearchService,
)
from app.domain.models.mailbox_envelope import MailboxEnvelopeType
from app.domain.models.session import Session


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class _OrderRecorder:
    def __init__(self) -> None:
        self.events: list[str] = []


class _FakeRegistry:
    def __init__(self, recorder: _OrderRecorder) -> None:
        self._recorder = recorder
        self.spawn_calls: list[str] = []

    async def spawn(self, root_id: str) -> None:
        self._recorder.events.append(f"spawn:{root_id}")
        self.spawn_calls.append(root_id)

    async def stop(self, root_id: str) -> None:
        del root_id


class _FakePublisher:
    def __init__(self, recorder: _OrderRecorder) -> None:
        self._recorder = recorder
        self.published = []

    async def publish(self, envelope) -> None:
        self._recorder.events.append(f"publish:{envelope.type.value}")
        self.published.append(envelope)


def _build_service(
    *,
    registry: Any | None,
    publisher: Any | None,
) -> SubagentResearchService:
    """Bypass full constructor wiring via ``__new__`` and set just the
    attributes ``_ensure_supervisor_and_publish_spawns`` reads.

    The helper does not touch the heavyweight collaborators
    (session_service / agent_service / supervisor / token_estimator /
    summary_llm / classifier / sandbox_lifecycle_service / quota_service),
    so ``__new__`` + targeted setattr is the minimal isolation surface.
    """
    svc = SubagentResearchService.__new__(SubagentResearchService)
    svc._supervisor_registry = registry  # type: ignore[attr-defined]
    svc._mailbox_publisher = publisher  # type: ignore[attr-defined]
    return svc


def _child(plane: str | None) -> Session:
    return Session(
        worker_type="subagent",
        subagent_control_plane=plane,  # type: ignore[arg-type]
        parent_session_id="root-1",
        tool_filter_preset="subagent_research",
    )


@pytest.mark.anyio
async def test_supervisor_spawn_before_publish_when_mailbox_child_present() -> None:
    rec = _OrderRecorder()
    registry = _FakeRegistry(rec)
    publisher = _FakePublisher(rec)
    svc = _build_service(registry=registry, publisher=publisher)
    svc._handoff_published_child_ids = set()  # type: ignore[attr-defined]

    await svc._ensure_supervisor_and_publish_spawns(
        parent_id="root-1",
        children_with_prompts=[(_child("mailbox"), "p1"), (_child("mailbox"), "p2")],
    )

    # Exactly one spawn, both publishes.
    assert registry.spawn_calls == ["root-1"]
    assert len(publisher.published) == 2

    # Ordering invariant: spawn must precede every publish.
    spawn_idx = rec.events.index("spawn:root-1")
    publish_indices = [i for i, e in enumerate(rec.events) if e.startswith("publish:")]
    assert all(idx > spawn_idx for idx in publish_indices), (
        f"spawn must precede every publish; got events={rec.events}"
    )
    # And every publish is a SPAWN_REQUEST.
    for env in publisher.published:
        assert env.type == MailboxEnvelopeType.SPAWN_REQUEST
        assert env.parent_session_id == "root-1"


@pytest.mark.anyio
async def test_no_publish_when_all_children_legacy() -> None:
    """Pre-PR-5 every child has ``subagent_control_plane='legacy'`` — the
    publisher path must remain a no-op so no envelopes appear.

    Spawn is also skipped: there is no mailbox plane consumer to spin up.
    """
    rec = _OrderRecorder()
    registry = _FakeRegistry(rec)
    publisher = _FakePublisher(rec)
    svc = _build_service(registry=registry, publisher=publisher)
    svc._handoff_published_child_ids = set()  # type: ignore[attr-defined]

    await svc._ensure_supervisor_and_publish_spawns(
        parent_id="root-1",
        children_with_prompts=[(_child("legacy"), "p1"), (_child(None), "p2")],
    )

    assert registry.spawn_calls == []
    assert publisher.published == []


@pytest.mark.anyio
async def test_spawn_failure_skips_publish_and_rolls_back_to_legacy() -> None:
    """codex r4 [R4-1, HIGH ARCH] + r7 [R7-1] — when supervisor spawn
    fails AND we have no other way to confirm a supervisor exists, the
    publish MUST be skipped, the child's handoff MUST NOT be recorded,
    AND the child's DB row MUST be rolled back to legacy so the runner
    takes the legacy path. The finally block in ``run_research`` then
    runs legacy suspend.
    """
    rec = _OrderRecorder()

    class _RaisingRegistry(_FakeRegistry):
        async def spawn(self, root_id: str) -> None:
            self._recorder.events.append(f"spawn-fail:{root_id}")
            raise RuntimeError("registry transient failure")

    registry = _RaisingRegistry(rec)
    publisher = _FakePublisher(rec)
    svc = _build_service(registry=registry, publisher=publisher)
    svc._handoff_published_child_ids = set()  # type: ignore[attr-defined]

    rollback_ids: list[str] = []

    async def _spy_rollback(child_id: str, **kwargs) -> bool:
        del kwargs  # accepts expected_parent_id
        rollback_ids.append(child_id)
        return True

    svc._rollback_child_to_legacy = _spy_rollback  # type: ignore[assignment]

    target = _child("mailbox")
    child_pair = [(target, "test-prompt")]
    await svc._ensure_supervisor_and_publish_spawns(
        parent_id="root-1",
        children_with_prompts=child_pair,
    )

    # Publish must NOT have happened (no supervisor to consume).
    assert publisher.published == []
    # Handoff must NOT be recorded — finally falls back to legacy suspend.
    assert svc._handoff_published_child_ids == set()
    # codex r7 [R7-1] — rollback fired so the child runner reads legacy.
    assert rollback_ids == [target.id]


@pytest.mark.anyio
async def test_publish_failure_logged_not_raised() -> None:
    """Best-effort: a publisher exception on one child does NOT abort the
    loop over the other children. The legacy suspend fallback (line ~466)
    will reap the failed-publish child later."""
    rec = _OrderRecorder()
    registry = _FakeRegistry(rec)

    class _FlakyPublisher(_FakePublisher):
        def __init__(self, recorder: _OrderRecorder) -> None:
            super().__init__(recorder)
            self._n = 0

        async def publish(self, envelope) -> None:
            self._n += 1
            if self._n == 1:
                raise RuntimeError("publisher transient failure")
            self._recorder.events.append(f"publish:{envelope.type.value}")
            self.published.append(envelope)

    publisher = _FlakyPublisher(rec)
    svc = _build_service(registry=registry, publisher=publisher)
    # codex r3 [R3-3, HIGH ARCH] — init the published-set the helper
    # writes into (production path sets this from run_research; here
    # we exercise the helper directly).
    svc._handoff_published_child_ids = set()  # type: ignore[attr-defined]

    children_pair = [(_child("mailbox"), "p1"), (_child("mailbox"), "p2")]
    await svc._ensure_supervisor_and_publish_spawns(
        parent_id="root-1",
        children_with_prompts=children_pair,
    )

    # 2 publish attempts, 1 succeeded, 1 raised but did not propagate.
    assert len(publisher.published) == 1
    # codex r3 [R3-3] — only the successful publish is in the set; the
    # failed one is left out so the caller's finally-block reaps it
    # via legacy suspend.
    assert len(svc._handoff_published_child_ids) == 1
    failed_id = children_pair[0][0].id
    succeeded_id = children_pair[1][0].id
    assert succeeded_id in svc._handoff_published_child_ids
    assert failed_id not in svc._handoff_published_child_ids


@pytest.mark.anyio
async def test_cross_root_child_refused() -> None:
    """codex r1 [R1-7, MEDIUM SEC] — publisher must refuse children
    whose ``parent_session_id`` does not equal the request's
    ``parent_id``. Defense in depth: an upstream bug that hands us a
    foreign child would otherwise let us publish destroy authority on
    the wrong root's stream."""
    rec = _OrderRecorder()
    registry = _FakeRegistry(rec)
    publisher = _FakePublisher(rec)
    svc = _build_service(registry=registry, publisher=publisher)

    # Two children: one belongs to root-1 (legit), one belongs to a
    # foreign root (must be refused).
    legit = _child("mailbox")  # parent_session_id="root-1"
    foreign = Session(
        worker_type="subagent",
        subagent_control_plane="mailbox",  # type: ignore[arg-type]
        parent_session_id="root-other",
        tool_filter_preset="subagent_research",
    )

    # codex r22 [R22-3, MEDIUM TEST] — also assert rollback fires for
    # cross-root child + do-not-start marker is set on failure.
    rollback_ids: list[str] = []

    async def _spy_rollback(child_id: str, **kwargs) -> bool:
        del kwargs
        rollback_ids.append(child_id)
        # Cross-root rollback returns False (rowcount=0 in real DB
        # because the parent_id WHERE guard filters us out).
        return False

    svc._rollback_child_to_legacy = _spy_rollback  # type: ignore[assignment]

    await svc._ensure_supervisor_and_publish_spawns(
        parent_id="root-1",
        children_with_prompts=[(legit, "p-legit"), (foreign, "p-foreign")],
    )

    assert len(publisher.published) == 1
    assert publisher.published[0].parent_session_id == "root-1"
    assert publisher.published[0].child_session_id == legit.id
    # Foreign child must have been rolled back (attempted) AND
    # marked as do-not-start so run_research skips its runner.
    assert foreign.id in rollback_ids
    assert foreign.id in svc._failed_rollback_child_ids
    assert legit.id not in svc._failed_rollback_child_ids


@pytest.mark.anyio
async def test_non_subagent_child_refused() -> None:
    """Same defense — refuse non-subagent rows. The DB CHECK constraint
    keeps ``subagent_control_plane`` NULL for roots; even if a stray
    row sneaks through, publisher must reject."""
    rec = _OrderRecorder()
    registry = _FakeRegistry(rec)
    publisher = _FakePublisher(rec)
    svc = _build_service(registry=registry, publisher=publisher)

    weird = Session(
        worker_type="root",  # root with mailbox plane — bogus
        subagent_control_plane="mailbox",  # type: ignore[arg-type]
        parent_session_id=None,
        tool_filter_preset=None,
    )
    await svc._ensure_supervisor_and_publish_spawns(
        parent_id="root-1",
        children_with_prompts=[(weird, "p-weird")],
    )

    assert publisher.published == []


@pytest.mark.anyio
async def test_publish_failure_rolls_back_child_to_legacy() -> None:
    """codex r6 [R6-1, CRITICAL ARCH] + r12 [R12-1, HIGH ARCH] — when
    SPAWN_REQUEST publish fails, the child's DB row must be rolled
    back to ``subagent_control_plane='legacy'`` so the child's runner
    reads legacy and does NOT publish SPAWN_ACK/RESULT_READY into a
    stream with no supervisor.

    Also asserts the rollback-success bool contract: a True return
    leaves the child OUT of ``_failed_rollback_child_ids`` (so caller
    starts the runner via the legacy path). The False-return path is
    pinned in ``test_failed_rollback_marks_child_do_not_start``.
    """
    rec = _OrderRecorder()
    registry = _FakeRegistry(rec)

    class _AlwaysFailingPublisher(_FakePublisher):
        async def publish(self, envelope) -> None:
            raise RuntimeError("transient publisher failure")

    publisher = _AlwaysFailingPublisher(rec)
    svc = _build_service(registry=registry, publisher=publisher)
    svc._handoff_published_child_ids = set()  # type: ignore[attr-defined]
    svc._failed_rollback_child_ids = set()  # type: ignore[attr-defined]

    rollback_ids: list[str] = []

    async def _spy_rollback_ok(child_id: str, **kwargs) -> bool:
        del kwargs  # accepts expected_parent_id
        rollback_ids.append(child_id)
        return True

    svc._rollback_child_to_legacy = _spy_rollback_ok  # type: ignore[assignment]

    target = _child("mailbox")
    await svc._ensure_supervisor_and_publish_spawns(
        parent_id="root-1",
        children_with_prompts=[(target, "p")],
    )

    # Publish raised; child must NOT be in handoff set; rollback fired.
    assert publisher.published == []
    assert svc._handoff_published_child_ids == set()
    assert rollback_ids == [target.id]
    # codex r12 [R12-4] — successful rollback must NOT mark child as
    # do-not-start; the legacy path takes over via parent finally.
    assert target.id not in svc._failed_rollback_child_ids


@pytest.mark.anyio
async def test_failed_rollback_marks_child_do_not_start() -> None:
    """codex r11 [R11-2] + r12 [R12-1] — when
    ``_rollback_child_to_legacy`` returns False (DB UPDATE failed),
    the child MUST land in ``_failed_rollback_child_ids`` so
    ``run_research`` skips ``_consume_child`` task creation and
    prevents the parallel-writer race."""
    rec = _OrderRecorder()
    registry = _FakeRegistry(rec)

    class _AlwaysFailingPublisher(_FakePublisher):
        async def publish(self, envelope) -> None:
            raise RuntimeError("transient publisher failure")

    publisher = _AlwaysFailingPublisher(rec)
    svc = _build_service(registry=registry, publisher=publisher)
    svc._handoff_published_child_ids = set()  # type: ignore[attr-defined]
    svc._failed_rollback_child_ids = set()  # type: ignore[attr-defined]

    rollback_ids: list[str] = []

    async def _spy_rollback_fails(child_id: str, **kwargs) -> bool:
        del kwargs  # accepts expected_parent_id
        rollback_ids.append(child_id)
        return False

    svc._rollback_child_to_legacy = _spy_rollback_fails  # type: ignore[assignment]

    target = _child("mailbox")
    await svc._ensure_supervisor_and_publish_spawns(
        parent_id="root-1",
        children_with_prompts=[(target, "p")],
    )

    assert rollback_ids == [target.id]
    assert target.id in svc._failed_rollback_child_ids


@pytest.mark.anyio
async def test_cap_task_prompt_respects_byte_budget() -> None:
    """codex r6 [R6-4, MEDIUM CONTRACT] — truncated prompts must
    include the ellipsis suffix WITHIN the
    ``_SPAWN_REQUEST_PROMPT_MAX_BYTES`` budget, not on top of it.
    """
    from app.application.services.subagent_research_service import (
        SubagentResearchService,
    )
    max_bytes = SubagentResearchService._SPAWN_REQUEST_PROMPT_MAX_BYTES
    too_long = "A" * (max_bytes * 2)
    capped = SubagentResearchService._cap_task_prompt(too_long)
    encoded = capped.encode("utf-8")
    assert len(encoded) <= max_bytes, (
        f"capped prompt is {len(encoded)} bytes; must be ≤ {max_bytes}"
    )
    assert capped.endswith("…")

    # Short prompts pass through unchanged.
    short = "fits"
    assert SubagentResearchService._cap_task_prompt(short) == short


@pytest.mark.anyio
async def test_helper_noop_when_wiring_absent() -> None:
    """Tests / pre-PR-5 deployments may construct SubagentResearchService
    without ``supervisor_registry`` AND ``mailbox_publisher``. Helper must
    return without raising."""
    svc = _build_service(registry=None, publisher=None)
    await svc._ensure_supervisor_and_publish_spawns(
        parent_id="root-1",
        children_with_prompts=[(_child("mailbox"), "p")],
    )  # no AttributeError, no exception
