"""C3 PR-4.5 — T10 publisher-path smoke (spec §11.1).

**Scope (intentionally narrow):** verify the **publisher half** of the
pipeline: POST to ``/api/sessions/{root}/subagents/research`` with the
mailbox flag enabled puts a ``SPAWN_REQUEST`` envelope on
``actus:child:{root}:mailbox`` and triggers
``SupervisorRegistry.spawn(root_id)`` before the publish.

**Deferred (codex r1–r6 [HIGH TEST] — known scope gap):** the full
spec/plan T10 pipeline — ``SPAWN_REQUEST → SPAWN_ACK →
PROGRESS_UPDATE → RESULT_READY → supervisor destroy →
sandbox_destroyed_at IS NOT NULL → no legacy suspend`` — needs a
running AgentTaskRunner mailbox-plane child, which in turn needs a
real LLM (or a fully-stubbed flow) AND a fake sandbox AND the
SupervisorRegistry consumer loop active. That harness isn't built
yet; the publisher unit tests
(``tests/app/application/services/test_subagent_research_supervisor_prewire.py``)
+ runner publisher unit tests
(``tests/domain/services/test_child_heartbeat_task.py``,
``tests/app/interfaces/test_pr4_5_agent_service_callback.py``)
cover the individual contracts; a full pipeline harness is
deferred to a follow-up PR.

codex r1 [R1-8] — earlier rounds used
``model_copy(update={"subagent_control_plane": "mailbox"})`` which only
flipped the in-memory child; the DB row stayed legacy. Round 2 persists
via direct UPDATE before the publisher path runs and decodes the
envelope from the ``envelope`` field (Redis Stream's canonical payload)
rather than the flat top-level fields (which are observational only).

Requires live PostgreSQL + Redis. Marked ``@pytest.mark.integration`` so
the local CI gate can scope to ``-m "not integration"`` when DB/Redis
aren't available.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.anyio, pytest.mark.integration]


@pytest.fixture(autouse=True)
def _restore_dependency_overrides(app):
    snapshot = dict(app.dependency_overrides)
    try:
        yield
    finally:
        app.dependency_overrides.clear()
        app.dependency_overrides.update(snapshot)


async def test_mailbox_plane_research_emits_spawn_request_envelope(
    asgi_client,
    sample_user_token,
    sample_session,
    redis_client,
    db_session,
    monkeypatch,
):
    """POST /subagents/research with mailbox-plane child → SPAWN_REQUEST
    envelope persisted on the per-root mailbox stream, child row in DB
    carries ``subagent_control_plane='mailbox'``.

    The child agent's heavy execution path (LLM + sandbox) is short-
    circuited via a stub ``_consume_child`` that returns immediately.
    The control_plane override is applied at the DB layer via a wrapper
    on ``create_session_with_parent`` so the change persists (the
    runner reads it back on resume / orphan reconcile paths).
    """
    from sqlalchemy import select, update

    from app.application.services.session_service import SessionService
    from app.application.services.subagent_research_service import (
        ChildResult,
        SubagentResearchService,
    )
    from app.domain.services.graphs.token_estimator import TokenEstimator
    from app.domain.services.subagent_research_classifier import ClassifierResult
    from app.infrastructure.external.mailbox.redis_mailbox_publisher import (
        RedisMailboxPublisher,
    )
    from app.infrastructure.models.session import SessionModel
    from app.infrastructure.repositories.db_session_repository import (
        DBSessionRepository,
    )
    from app.interfaces.schemas.subagent import ChildOutcome
    from app.interfaces.service_dependencies import (
        get_subagent_research_service,
    )

    class _SameSessionUow:
        def __init__(self):
            self.db_session = db_session
            self.session = DBSessionRepository(db_session=db_session)

        async def __aenter__(self):
            return self

        async def __aexit__(self, _exc_type, _exc_val, _exc_tb):
            return False

    def _same_session_uow_factory():
        return _SameSessionUow()

    class _Classifier:
        async def classify_batch(self, prompts):
            return [ClassifierResult(approved=True, reason="test") for _ in prompts]

    class _Quota:
        renew_interval_seconds = 60.0

        async def acquire(self, _user_id, _probe_run_id):
            return True

        async def renew(self, _user_id, _probe_run_id):
            return True

        async def release(self, _user_id, _probe_run_id):
            return True

    class _Registry:
        async def spawn(self, _parent_id):
            return None

    svc = SubagentResearchService(
        session_service=SessionService(uow_factory=_same_session_uow_factory),
        agent_service=object(),
        execution_supervisor=object(),
        token_estimator=TokenEstimator(strategy="char"),
        summary_llm=object(),
        classifier=_Classifier(),
        sandbox_lifecycle_service=object(),
        quota_service=_Quota(),
        supervisor_registry=_Registry(),
        mailbox_publisher=RedisMailboxPublisher(redis_client.client),
    )

    async def _stub_consume(child_session_id: str, user_id: str, prompt: str):
        del user_id
        return ChildResult(
            child_id=child_session_id,
            prompt=prompt,
            outcome=ChildOutcome.COMPLETED,
            final_answer="stub",
            transcript_tokens=0,
            error_summary=None,
        )

    async def _stub_summary(**_kwargs):
        return "stub summary", []

    svc._consume_child = _stub_consume  # type: ignore[assignment]
    svc._do_summary_join_with_retry = _stub_summary  # type: ignore[assignment]

    from fastapi import FastAPI
    app_instance: FastAPI = asgi_client._transport.app  # type: ignore[attr-defined]
    app_instance.dependency_overrides[get_subagent_research_service] = lambda: svc

    # codex r1 [R1-8] — persist control_plane='mailbox' on the child
    # DB row so the runner-side reads see the same value as the
    # publisher (avoiding the in-memory-only model_copy that earlier
    # rounds used). Wrap ``create_session_with_parent`` to issue an
    # UPDATE right after save, then return a model_copy reflecting the
    # new persisted value.
    orig_create = SessionService.create_session_with_parent

    async def _create_and_persist_mailbox_child(self, *args, **kwargs):  # noqa: ANN001
        child = await orig_create(self, *args, **kwargs)
        async with self._uow_factory() as uow:
            await uow.db_session.execute(
                update(SessionModel)
                .where(SessionModel.id == child.id)
                .values(subagent_control_plane="mailbox")
            )
            await uow.db_session.commit()
        return child.model_copy(update={"subagent_control_plane": "mailbox"})

    monkeypatch.setattr(
        SessionService,
        "create_session_with_parent",
        _create_and_persist_mailbox_child,
    )

    response = await asgi_client.post(
        f"/api/sessions/{sample_session.id}/subagents/research",
        json={"prompts": ["short research task"], "max_children": 1},
        headers={"Authorization": f"Bearer {sample_user_token}"},
    )

    assert response.status_code in (200, 202), response.text

    # Verify the SPAWN_REQUEST envelope landed on the mailbox stream for
    # this root. ``redis_client`` is the integration fixture giving the
    # ``redis.asyncio.Redis`` wrapper; ``.client`` is the raw connection.
    stream_key = f"actus:child:{sample_session.id}:mailbox"
    n = await redis_client.client.xlen(stream_key)
    assert n >= 1, (
        f"expected ≥1 envelope on {stream_key} after mailbox-plane research POST; got {n}"
    )

    entries = await redis_client.client.xrange(stream_key, count=1)
    assert entries, "stream was empty after publish"
    _, fields = entries[0]
    type_val = fields.get(b"type") or fields.get("type")
    assert type_val in (b"SPAWN_REQUEST", "SPAWN_REQUEST")

    # codex r2 [R2-5, MEDIUM TEST] — RedisMailboxPublisher writes the
    # full envelope as JSON under the ``envelope`` field; the flat
    # ``child_session_id`` field is NOT part of the wire format. Parse
    # the envelope JSON to extract child_session_id correctly.
    import json
    env_blob = fields.get(b"envelope") or fields.get("envelope")
    assert env_blob, "envelope field missing on stream entry"
    env_dict = json.loads(env_blob)
    child_id = env_dict["child_session_id"]
    assert child_id, "envelope.child_session_id must be set"

    row = (
        await db_session.execute(
            select(SessionModel).where(SessionModel.id == child_id)
        )
    ).scalar_one_or_none()
    assert row is not None, f"child session row {child_id} missing"
    assert row.subagent_control_plane == "mailbox", (
        "child row must persist subagent_control_plane='mailbox' so the "
        "runner reads it back on resume / orphan reconcile."
    )

    # codex r2 [R2-6, MEDIUM CONTRACT] — SPAWN_REQUEST.task_prompt
    # carries the actual prompt (redacted to a length cap on the
    # publisher side). The test prompt fits well under the cap, so it
    # arrives verbatim.
    payload = env_dict["payload"]
    assert payload["agent_kind"] == "research"
    assert payload["task_prompt"] == "short research task"
