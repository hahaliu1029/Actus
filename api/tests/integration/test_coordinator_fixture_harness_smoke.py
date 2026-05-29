"""[PR-9b-C / Task C9] Fixture harness smoke + INV-A7 from PR-9b-A.

Asserts:
- INV-C1: ``async_session`` yields the production session **factory** (a callable),
  and ``async with async_session() as s:`` opens a real, executing scoped session.
- INV-C2: the priced fake worker LLM surfaces a priced identity
  (``provider_id='openai_official'`` / ``model='gpt-4o-mini'``) AND a populated
  ``usage_metadata`` on its emitted ``AIMessage`` — the precondition for the
  cost ledger to record ``cost_status='actual'`` (NOT ``'unknown'``).
- INV-C3: the per-test ``minio_real`` bucket + ``sandbox_real`` container are
  created and destroyed with no residue (list-before == list-after).
- INV-C4: ``coordinator_truncation`` wipes the coordinator tables (sessions empty);
  AND, end-to-end, driving ``CostCallbackHandler`` with the priced fake LLM's
  ``AIMessage`` lands a real ``cost_records`` row with ``total_usd > 0`` and
  ``cost_status='actual'`` — validating the C6 ``provider_id='openai_official'``
  pricing chain (priced AIMessage -> CostCallbackHandler -> static pricing ->
  cost_records) the whole harness exists to exercise.
- INV-A7: with the coordinator flag ON and the 18 cfg keys wired by PR-9b-A,
  a real flag-on dispatch through the planner_react flow does NOT raise
  ``KeyError`` on any ``cfg[...]`` read inside ``parallel_execution_subgraph``.

⚠️ This is an INTEGRATION smoke: it needs a live Postgres + Redis + MinIO +
Docker sandbox (per fixture). It is validated by CI; it CANNOT run on a host
without those services (CLAUDE.md). The module imports + collects cleanly with
no infra; the fixture bodies only touch infra when a test actually requests them.

Async convention: ``pytest.mark.anyio`` + plain ``async def`` test functions
(``pytest_asyncio`` is NOT installed; the ``anyio_backend`` fixture in
``tests/integration/conftest.py`` drives the loop). This mirrors the sibling
``test_fixture_mock_llm_priced.py`` (the C6 analog) and the coordinator E2E
files, all of which use the anyio marker and no per-test ``@pytest.mark.asyncio``.
"""
from __future__ import annotations

import asyncio

import pytest
from sqlalchemy import text


pytestmark = [
    pytest.mark.integration,
    pytest.mark.coordinator_recovery,
    pytest.mark.anyio,
]


# ── INV-C1 ───────────────────────────────────────────────────────────────────


async def test_async_session_yields_factory(async_session):
    """INV-C1 — ``async_session`` is the session FACTORY itself, not a single
    pre-opened session. ``async with async_session() as s:`` must open a real,
    executing scoped session.
    """
    assert callable(async_session), (
        "INV-C1 — async_session must be the production sessionmaker (a callable "
        "factory), so each `async with async_session() as s:` opens a fresh "
        "scoped session."
    )
    async with async_session() as s:
        res = await s.execute(text("SELECT 1"))
        assert res.scalar() == 1


# ── INV-C2 ───────────────────────────────────────────────────────────────────


async def test_fixture_mock_llm_3_workers_priced(fixture_mock_llm_3_workers):
    """INV-C2 — the 3-worker fixture is a list of priced fakes; each carries a
    priced identity (``provider_id='openai_official'`` — NOT the heuristic
    ``'openai'`` that misses the pricing table) and a populated
    ``usage_metadata`` on its scripted ``AIMessage``.

    NOTE: ``fixture_mock_llm_3_workers`` is a ``list`` per the plan; ``len(...) == 3``
    and indexing ``[0]`` is plan-consistent. (The pre-existing E2E tests call
    ``.setup_responses(...)`` on this list — a documented C↔D contract gap that
    is a D-phase concern, NOT this smoke's.)
    """
    from langchain_core.messages import AIMessage, HumanMessage

    assert len(fixture_mock_llm_3_workers) == 3
    for llm in fixture_mock_llm_3_workers:
        params = llm._identifying_params
        assert params["provider_id"] == "openai_official", params
        assert params["model"] == "gpt-4o-mini", params
        # ``PricedFakeListChatModel(responses=[...])`` stores raw *strings* in
        # ``.responses``; the priced ``AIMessage`` (with ``usage_metadata``) is
        # produced by ``_generate`` / ``_agenerate``. Drive ``ainvoke`` to get
        # the real emitted message instead of indexing ``.responses`` (a str).
        msg = await llm.ainvoke([HumanMessage(content="x")])
        assert isinstance(msg, AIMessage), msg
        assert msg.usage_metadata is not None, (
            "INV-C2 — worker AIMessage must carry usage_metadata so the cost "
            "ledger can price it"
        )
        assert msg.usage_metadata["input_tokens"] > 0


# ── INV-C4 (truncation correctness) ──────────────────────────────────────────


async def test_truncation_clears_coordinator_tables(
    async_session, fresh_test_user,
):
    """INV-C4 — ``_truncate_coordinator_tables`` actually clears EVERY table in
    ``_COORDINATOR_TRUNCATE_TABLES``, and the table list is complete.

    This is deliberately NON-vacuous and does NOT depend on the
    ``coordinator_truncation`` fixture's autouse timing (it neither requests
    that fixture nor relies on a clean-DB precondition). Instead it:

      1. SEEDS at least one COMMITTED row into EACH of the five coordinator
         tables (``sessions`` + ``cost_records`` via the ``fresh_test_user`` FK
         parent; the three audit tables have no FKs).
      2. Asserts every table is NON-empty (the seed actually committed).
      3. Calls the SAME ``_truncate_coordinator_tables`` helper the
         ``coordinator_truncation`` fixture uses (single source of truth).
      4. Asserts EVERY table in ``_COORDINATOR_TRUNCATE_TABLES`` is now empty.

    Step 4 proves truncation clears the footprint AND that the table list is
    complete: if a future edit drops a table from ``_COORDINATOR_TRUNCATE_TABLES``
    (so it's seeded-and-asserted-nonempty in step 2 but never truncated), this
    test fails — restoring the INV-C4 table-list regression protection that the
    prior clean-DB-only assertion silently voided.

    Marked ``[integration, coordinator_recovery, anyio]`` (module pytestmark) —
    pg-only (``users`` + coordinator tables). Stays in the CI recovery pass; NO
    ``sandbox`` marker (no MinIO/Docker needed).
    """
    from uuid import uuid4

    from tests.integration.coordinator_fixtures import (
        _COORDINATOR_TRUNCATE_TABLES,
        _truncate_coordinator_tables,
    )

    # ``fresh_test_user`` yields a committed UserModel (FK parent for
    # sessions.user_id + cost_records.user_id). Use ``.id``.
    user_id = str(fresh_test_user.id)
    tag = uuid4().hex[:8]
    session_id = f"c9-trunc-{tag}"
    run_id = f"run-{tag}"

    # ── Step 1: seed one COMMITTED row into EACH coordinator table. ──────────
    # Column requirements grep-verified against the ORM models
    # (app/infrastructure/models/{session,cost_record_orm,coordinator_apply_audit,
    #  coordinator_result_envelope_store,mailbox_envelope_audit}.py):
    #   - sessions:        id + user_id FK; other NOT NULLs carry server_defaults.
    #   - cost_records:    session_id + user_id FK + NOT NULL run_id / node_name /
    #                      model / provider / pricing_version / cost_status
    #                      (token + total_usd cols server_default 0).
    #   - coordinator_apply_audit:            no FK; NOT NULL coordinator_run_id /
    #                      parent_session_id / status / started_at.
    #   - coordinator_result_envelope_store:  no FK; NOT NULL coordinator_run_id /
    #                      work_unit_id / child_session_id / envelope_type /
    #                      payload (received_at server_default NOW()).
    #   - mailbox_envelope_audit:             no FK; PK (parent_session_id,
    #                      envelope_id) + NOT NULL child_session_id / type /
    #                      producer_role (received_at + reclaim_count default).
    async with async_session() as s:
        await s.execute(
            text(
                "INSERT INTO sessions(id, user_id, created_at, updated_at) "
                "VALUES (:sid, :uid, NOW(), NOW())"
            ),
            {"sid": session_id, "uid": user_id},
        )
        await s.execute(
            text(
                "INSERT INTO cost_records("
                "  id, session_id, user_id, run_id, node_name, model, provider, "
                "  pricing_version, cost_status"
                ") VALUES ("
                "  :id, :sid, :uid, :run_id, 'smoke_node', 'gpt-4o-mini', "
                "  'openai_official', 'test-pricing', 'actual'"
                ")"
            ),
            {
                "id": f"cost-{tag}",
                "sid": session_id,
                "uid": user_id,
                "run_id": run_id,
            },
        )
        await s.execute(
            text(
                "INSERT INTO coordinator_apply_audit("
                "  coordinator_run_id, parent_session_id, status, started_at"
                ") VALUES (:run_id, :sid, 'in_progress', NOW())"
            ),
            {"run_id": run_id, "sid": session_id},
        )
        await s.execute(
            text(
                "INSERT INTO coordinator_result_envelope_store("
                "  coordinator_run_id, work_unit_id, child_session_id, "
                "  envelope_type, payload"
                ") VALUES (:run_id, :wu, :child, 'RESULT_READY', "
                "  CAST(:payload AS JSONB))"
            ),
            {
                "run_id": run_id,
                "wu": f"wu-{tag}",
                "child": f"child-{tag}",
                "payload": "{}",
            },
        )
        await s.execute(
            text(
                "INSERT INTO mailbox_envelope_audit("
                "  parent_session_id, envelope_id, child_session_id, type, "
                "  producer_role"
                ") VALUES (:sid, :env, :child, 'RESULT_READY', 'child')"
            ),
            {
                "sid": session_id,
                "env": f"env-{tag}",
                "child": f"child-{tag}",
            },
        )
        await s.commit()

    # ── Step 2: assert every seeded table is NON-empty. ──────────────────────
    async with async_session() as s:
        for table in _COORDINATOR_TRUNCATE_TABLES:
            res = await s.execute(text(f"SELECT COUNT(*) FROM {table}"))  # noqa: S608 — table name from a trusted module constant
            count = res.scalar()
            assert count >= 1, (
                f"INV-C4 precondition — seed must commit at least one row into "
                f"{table!r} so truncation has something to clear; got {count}. "
                f"A vacuous (0-row) precondition would make the post-truncate "
                f"assertion prove nothing."
            )

    # ── Step 3: truncate via the SAME helper the fixture uses. ───────────────
    await _truncate_coordinator_tables(async_session)

    # ── Step 4: assert EVERY expected table is now empty. ────────────────────
    # codex R3-F3 (MEDIUM): the post-truncate empty-assert iterates a HARDCODED
    # literal — NOT ``_COORDINATOR_TRUNCATE_TABLES`` (the constant Step 3 just
    # truncated). The decoupling is INTENTIONAL regression protection: if a
    # future edit drops a table from ``_COORDINATOR_TRUNCATE_TABLES``, truncation
    # will skip it BUT this assertion still checks it, so the seeded row survives
    # and the test FAILS — exactly the "dropped table breaks the test" guarantee.
    # Iterating the same constant the truncate uses would make that guarantee
    # circular (drop a table → skip truncating it AND stop asserting it → silent
    # leak passes green). Keep this literal in sync with the 5 coordinator tables
    # the harness owns; adding a table here without adding it to the constant
    # makes the test fail loudly (the deliberate forcing function).
    _EXPECTED_EMPTY_AFTER_TRUNCATE = (
        "sessions",
        "cost_records",
        "coordinator_apply_audit",
        "coordinator_result_envelope_store",
        "mailbox_envelope_audit",
    )
    # codex R4-F2 (MEDIUM) — close the CASCADE blind-spot: the per-table
    # COUNT==0 loop below alone can't catch dropping e.g. ``cost_records`` from
    # ``_COORDINATOR_TRUNCATE_TABLES``, because ``TRUNCATE sessions ... CASCADE``
    # clears the FK-child anyway. Assert DIRECT membership-equivalence so ANY
    # drift between the harness-owned truncate constant and this expected set
    # fails the test regardless of CASCADE side effects.
    from tests.integration.coordinator_fixtures import (
        _COORDINATOR_TRUNCATE_TABLES,
    )
    assert set(_EXPECTED_EMPTY_AFTER_TRUNCATE) == set(_COORDINATOR_TRUNCATE_TABLES), (
        "INV-C4 table-list drift: expected "
        f"{sorted(_EXPECTED_EMPTY_AFTER_TRUNCATE)} != truncate constant "
        f"{sorted(_COORDINATOR_TRUNCATE_TABLES)} — keep both in sync."
    )
    async with async_session() as s:
        for table in _EXPECTED_EMPTY_AFTER_TRUNCATE:
            res = await s.execute(text(f"SELECT COUNT(*) FROM {table}"))  # noqa: S608 — table name from a trusted hardcoded literal
            count = res.scalar()
            assert count == 0, (
                f"INV-C4 — _truncate_coordinator_tables must clear {table!r}, "
                f"but it still has {count} row(s). Either truncation skipped this "
                f"table or it was dropped from _COORDINATOR_TRUNCATE_TABLES."
            )


# ── INV-C3 (smoke + leak check) ──────────────────────────────────────────────


# marked sandbox to exclude from the pg+redis CI recovery pass; needs MinIO —
# see PR-9b-C CI-infra note (provisioning MinIO/Docker in ci.yml is a separate
# decision).
@pytest.mark.sandbox
async def test_minio_real_bucket_present_during_test(minio_real):
    """INV-C3 (smoke) — the per-test ``minio_real`` bucket exists and is named
    ``actus-test-<...>`` while the test runs.
    """
    assert minio_real.bucket_name.startswith("actus-test-"), minio_real.bucket_name


# marked sandbox to exclude from the pg+redis CI recovery pass; needs MinIO —
# see PR-9b-C CI-infra note (provisioning MinIO/Docker in ci.yml is a separate
# decision).
@pytest.mark.sandbox
async def test_minio_bucket_list_unchanged_across_test():
    """INV-C3 — a bucket created + torn down inside one test leaves the set of
    ``actus-test-*`` buckets unchanged (no leak).

    Drives the raw ``minio.Minio`` SDK directly — consistent with the
    ``minio_real`` fixture (Task C4): production storage adapters do NOT expose
    bucket lifecycle methods, and the fixture constructs the client from
    ``core.config.get_settings()`` (``minio_endpoint`` / ``minio_access_key`` /
    ``minio_secret_key`` / ``minio_secure`` / ``minio_region``), NOT raw env
    vars. We mirror that construction here so the smoke uses the same auth
    surface the fixture uses.
    """
    import uuid as _uuid

    from minio import Minio
    from minio.deleteobjects import DeleteObject

    from core.config import get_settings

    settings = get_settings()

    http_client = None
    if settings.minio_secure:
        import urllib3

        http_client = urllib3.PoolManager(cert_reqs="CERT_NONE")

    client = Minio(
        endpoint=settings.minio_endpoint,
        access_key=settings.minio_access_key,
        secret_key=settings.minio_secret_key,
        secure=settings.minio_secure,
        region=getattr(settings, "minio_region", None),
        http_client=http_client,
    )

    def _list_actus_test_buckets() -> set[str]:
        return {
            b.name
            for b in client.list_buckets()
            if b.name.startswith("actus-test-")
        }

    before = await asyncio.to_thread(_list_actus_test_buckets)

    bucket_name = f"actus-test-{_uuid.uuid4().hex[:8]}"
    await asyncio.to_thread(client.make_bucket, bucket_name)
    try:
        pass  # exercise body — fixture-equivalent lifecycle
    finally:
        # Recursive delete: list + bulk-remove objects, then the bucket.
        objects = await asyncio.to_thread(
            lambda: list(client.list_objects(bucket_name, recursive=True))
        )
        delete_list = [DeleteObject(o.object_name) for o in objects]
        if delete_list:
            await asyncio.to_thread(
                lambda: list(client.remove_objects(bucket_name, delete_list))
            )
        await asyncio.to_thread(client.remove_bucket, bucket_name)

    after = await asyncio.to_thread(_list_actus_test_buckets)
    assert after == before, (
        f"INV-C3 violated — MinIO bucket leaked across test: "
        f"new={after - before} removed={before - after}"
    )


# marked sandbox to exclude from the pg+redis CI recovery pass; needs Docker —
# see PR-9b-C CI-infra note (provisioning MinIO/Docker in ci.yml is a separate
# decision).
@pytest.mark.sandbox
async def test_sandbox_container_list_unchanged_across_test():
    """INV-C3 — a ``DockerSandbox`` created + destroyed inside one test leaves no
    container residue.

    Construction is ``await DockerSandbox.create(user_id=...)`` (the classmethod
    factory — grep-verified at
    ``app/infrastructure/external/sandbox/docker_sandbox.py``), matching how the
    ``sandbox_real`` fixture (Task C5) builds it. Containers are filtered by
    image-tag prefix (the sandbox image is built locally per docker-compose;
    re-grep the image name if compose changes it).
    """
    import docker as _docker

    from app.infrastructure.external.sandbox.docker_sandbox import DockerSandbox

    client = _docker.from_env()
    # Sandbox image is built locally (docker-compose `sandbox-image`). Re-grep
    # the actual tag if compose renames it.
    SANDBOX_IMAGE_PREFIX = "actus-sandbox"

    def _list_sandbox_containers() -> set[str]:
        return {
            c.id
            for c in client.containers.list(all=True)
            if any(
                SANDBOX_IMAGE_PREFIX in (t or "") for t in (c.image.tags or [])
            )
        }

    before = await asyncio.to_thread(_list_sandbox_containers)
    sandbox = await DockerSandbox.create(user_id=f"smoke-{_uuid4hex()}")
    try:
        pass
    finally:
        await sandbox.destroy()
    after = await asyncio.to_thread(_list_sandbox_containers)
    assert after == before, (
        f"INV-C3 violated — sandbox container leaked: "
        f"new={after - before} removed={before - after}"
    )


def _uuid4hex() -> str:
    import uuid as _uuid

    return _uuid.uuid4().hex[:8]


# ── INV-C4 (env activation) ──────────────────────────────────────────────────


async def test_env_flag_activates_is_coordinator_enabled(env_with_coordinator_flag_on):
    """INV-C4 (env activation) — ``env_with_coordinator_flag_on`` flips
    ``is_coordinator_enabled()`` to ``True``.

    ``coordinator_feature_flag.is_coordinator_enabled`` reads the env var every
    call (no cache — grep-verified), so a single ``monkeypatch.setenv`` suffices.
    """
    from app.domain.services.coordinator_feature_flag import is_coordinator_enabled

    assert is_coordinator_enabled() is True


# ── INV-C4 (load-bearing: priced chain -> real cost_records row) ─────────────


async def test_real_cost_records_total_usd_positive(
    async_session, fresh_test_user, fixture_mock_llm_3_workers,
):
    """INV-C4 — drive ``CostCallbackHandler`` with a priced ``AIMessage`` from the
    fake worker LLM and assert a ``cost_records`` row with ``total_usd > 0`` and
    ``cost_status='actual'`` lands in the ledger.

    This validates the priced ``provider_id='openai_official'`` chain
    end-to-end: priced AIMessage -> CostCallbackHandler.on_chat_model_start (reads
    ``provider_id`` from ``invocation_params``) -> on_llm_end (prices via static
    pricing) -> persister -> cost_records. A ``cost_status='unknown'`` here would
    mean the pricing lookup missed (the heuristic ``'openai'`` bug).
    """
    from uuid import uuid4

    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_core.outputs import ChatGeneration, LLMResult

    from app.domain.models.cost_record import CostRecord
    from app.domain.services.cost_callback_handler import CostCallbackHandler
    from app.infrastructure.repositories.db_cost_record_repository import (
        DbCostRecordRepository,
    )

    # ``fresh_test_user`` yields a committed UserModel (exposes ``.id``) — NOT a
    # bare id string. Use ``.id`` for the FK value. (Grep-verified against the
    # C8 fixture body.)
    user_id = str(fresh_test_user.id)

    # Seed a session row to satisfy the cost_records.session_id FK. The sessions
    # table's NOT NULL columns all carry server_defaults (grep-verified against
    # the ORM model), so a raw INSERT of just id/user_id/created_at/updated_at
    # is sufficient — the DB fills the rest.
    session_id = f"c9-smoke-{uuid4().hex[:8]}"
    async with async_session() as s:
        await s.execute(
            text(
                "INSERT INTO sessions(id, user_id, created_at, updated_at) "
                "VALUES (:sid, :uid, NOW(), NOW())"
            ),
            {"sid": session_id, "uid": user_id},
        )
        await s.commit()

    # ``CostCallbackHandler.__init__(*, session_id, user_id, persister)`` — the
    # persister is a CALLABLE taking a CostRecord, NOT a repository instance
    # (grep-verified, cost_callback_handler.py). Production wires a closure over
    # a session-factory; we mirror that here.
    async def _persister(record: CostRecord) -> None:
        async with async_session() as inner:
            repo = DbCostRecordRepository(inner)  # ctor takes AsyncSession
            await repo.insert(record)  # method name grep-verified
            await inner.commit()

    handler = CostCallbackHandler(
        session_id=session_id,
        user_id=user_id,
        persister=_persister,
    )

    # Entry path: on_chat_model_start + on_llm_end are async (grep-verified). The
    # priced identity (provider_id/model) is read from ``invocation_params``.
    llm = fixture_mock_llm_3_workers[0]
    # ``.responses[0]`` is a raw *str* (FakeListChatModel stores strings); the
    # priced ``AIMessage`` with ``usage_metadata`` is produced by ``_generate``.
    # Drive ``ainvoke`` to obtain the real message the cost handler must price.
    priced_msg = await llm.ainvoke([HumanMessage(content="x")])
    assert isinstance(priced_msg, AIMessage), priced_msg
    run_id = uuid4()
    await handler.on_chat_model_start(
        serialized={},
        messages=[[]],
        run_id=run_id,
        invocation_params=llm._identifying_params,
    )
    llm_result = LLMResult(
        generations=[[ChatGeneration(message=priced_msg)]],
        llm_output={"model_name": "gpt-4o-mini"},
    )
    await handler.on_llm_end(llm_result, run_id=run_id)
    await handler.flush_pending()

    async with async_session() as s:
        res = await s.execute(
            text(
                "SELECT COUNT(*), COALESCE(SUM(total_usd), 0), MAX(cost_status) "
                "FROM cost_records WHERE session_id = :sid AND user_id = :uid"
            ),
            {"sid": session_id, "uid": user_id},
        )
        row = res.one()
        row_count, total_usd, cost_status = int(row[0]), float(row[1]), row[2]

    assert row_count >= 1, (
        f"INV-C4 — seeded session must have at least one cost_records row after "
        f"on_llm_end + flush; got row_count={row_count}. "
        f"Hint: check that the persister closure actually committed."
    )
    assert total_usd > 0.0, (
        f"INV-C4 — fake LLM must produce a priced cost_records row "
        f"(total_usd={total_usd}, cost_status={cost_status}). "
        f"Hint: check that fixture_mock_llm_3_workers sets "
        f"provider_id='openai_official' (NOT 'openai')."
    )
    # CostStatus enum values (grep-verified, cost_record.py):
    # "actual" | "estimated" | "partial" | "unknown". A priced AIMessage with
    # full usage_metadata + matched pricing produces "actual".
    assert cost_status == "actual", (
        f"INV-C4 — priced LLM must produce cost_status='actual', got "
        f"{cost_status!r}; 'unknown' means the pricing lookup missed."
    )


# ── INV-A7 (flag-on dispatch does not raise KeyError on cfg keys) ────────────


async def test_flag_on_dispatch_does_not_raise_keyerror():
    """INV-A7 — flag-on dispatch must NOT raise ``KeyError`` on any coordinator
    ``cfg[...]`` key inside ``parallel_execution_subgraph``.

    HONEST SKIP (C↔D gap) — see report F2. Genuinely exercising INV-A7 requires
    driving the dispatch with a fake planner whose ``with_structured_output``
    emits a ``PlanResponse`` carrying ``parallel_work_units`` (so the coordinator
    branch is actually entered). But:

    - ``async_client`` wraps the REAL ``app.main:app`` and runs its REAL
      lifespan, which builds the ``AgentService`` singleton with the
      config-derived LLM (``_ConfigSnapshot.llm`` via ``_build_llm`` in
      ``app/interfaces/service_dependencies.py``). That LLM is baked into the
      app.state singleton BEFORE any test body runs.
    - ``fixture_mock_llm_parallel_planner`` is NOT wired into that app — there is
      no DI seam on ``async_client`` to swap the planner LLM. Overriding
      ``get_agent_service`` would mean rebuilding the entire heavy AgentService
      (uow_factory / sandbox / coord_deps / …) with the fake LLM — which IS the
      D-phase LLM-injection harness (the skipped ``test_coordinator_e2e_*.py``
      tests depend on the same not-yet-built ``fixture_mock_llm_3_workers``
      ``.setup_responses(...)`` harness API; see the C↔D CONTRACT GAP note in
      ``coordinator_fixtures.py``).
    - With the real config LLM (no API key in the pg+redis-only CI recovery
      pass), the dispatch never reaches the coordinator branch, so any
      terminal ``error`` would pass the old assertion VACUOUSLY — proving
      nothing about cfg-key wiring.

    Rather than green-pass on an arbitrary error, skip honestly until the
    D-phase harness lands. INV-A7's cfg-key wiring is independently covered by
    the PR-9b-A unit/composition gates (the 18 cfg keys in
    ``PlannerReActFlow._build_config`` + ``test_build_supervisor_registry_pr4_gate``
    composition wiring).
    """
    pytest.skip(
        "INV-A7 genuine flag-on dispatch needs the D-phase LLM-injection harness "
        "(fixture_mock_llm_parallel_planner not yet wired into app.main); see "
        "C↔D gap. Asserting on an arbitrary terminal error would pass vacuously."
    )
