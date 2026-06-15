"""[PR-9b-C] Coordinator E2E fixture harness. INV-C1..C5 asserted by test_coordinator_fixture_harness_smoke.py (Task C9)."""

from __future__ import annotations

import asyncio
import logging
import uuid as _uuid
from typing import Any, List, Mapping, Optional

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda

logger = logging.getLogger(__name__)


# ── C6 priced fake LLM ──────────────────────────────────────────────────────
#
# The stock ``langchain_core.language_models.fake_chat_models.FakeListChatModel``
# is a ``SimpleChatModel`` whose ``_call`` returns a bare ``str``; the wrapper
# AIMessage carries **no** ``usage_metadata`` and ``_identifying_params`` only
# yields ``{'_type': 'fake-list-chat-model'}``.  B4's ``CostCallbackHandler``
# (api/app/domain/services/cost_callback_handler.py:237-246) reads ``model`` and
# ``provider_id`` from ``invocation_params`` (populated from
# ``_identifying_params`` via ``BaseChatModel._get_invocation_params``); a stock
# FakeListChatModel therefore stamps ``model='unknown'`` + heuristic provider,
# which misses the ``openai_official`` pricing table (api/app/domain/services/
# pricing/static_pricing.py:57) -> ``cost_status=UNKNOWN``.
#
# ``PricedFakeListChatModel`` overrides both:
#   1. ``_identifying_params`` -> ``{'model': model, 'provider_id': provider_id}``
#      so the cost handler's authoritative-provider branch fires.
#   2. ``_generate`` -> attach a real ``usage_metadata`` block to the returned
#      AIMessage so the handler records non-zero token counts (B4 INV: no silent
#      zero-cost fallback).
#
# ``PricedPlannerFakeListChatModel`` (defined below) further overrides
# ``with_structured_output`` because stock ``FakeListChatModel`` raises
# ``NotImplementedError`` there and the real planner silently degrades on that
# failure (see that class's docstring).
#
# We subclass rather than attribute-set because ``BaseChatModel`` is a Pydantic
# model and ``_identifying_params`` is a read-only property — attribute
# assignment is rejected by Pydantic v2.


class PricedFakeListChatModel(FakeListChatModel):
    """FakeListChatModel that surfaces a priced identity + usage_metadata.

    Cycles through ``responses`` (str) like the parent, but each emitted
    AIMessage carries ``usage_metadata`` and the model advertises
    ``model='gpt-4o-mini'`` / ``provider_id='openai_official'`` so the
    coordinator cost rollup attributes real cost (INV-C2).
    """

    # Pydantic fields (the parent is a Pydantic BaseChatModel).
    priced_model: str = "gpt-4o-mini"
    provider_id: str = "openai_official"
    input_tokens: int = 10
    output_tokens: int = 5

    @property
    def _identifying_params(self) -> Mapping[str, Any]:  # type: ignore[override]
        # Merged into invocation_params by BaseChatModel._get_invocation_params;
        # CostCallbackHandler reads ``model`` + ``provider_id`` from there.
        return {"model": self.priced_model, "provider_id": self.provider_id}

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        # Mirror FakeListChatModel cycling, but build a full AIMessage with
        # usage_metadata instead of relying on SimpleChatModel's str wrapper.
        response = self.responses[self.i]
        if self.i < len(self.responses) - 1:
            self.i += 1
        else:
            self.i = 0
        message = AIMessage(
            content=response,
            usage_metadata={
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "total_tokens": self.input_tokens + self.output_tokens,
            },
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        return self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def _build_parallel_plan() -> "Any":
    """Build a validated ``PlanResponse`` with one 3-WRITE parallel step.

    Source of truth for the planner flavor.  Shape mirrors
    ``app/domain/models/llm_responses.py::PlanResponse`` + ``StepDef`` +
    ``ParallelWorkUnitGroupRequest`` (``work_units: list[WorkUnitRequest]``;
    each WRITE unit needs a non-empty ``proposed_paths`` per
    ``WorkUnitRequest`` validator).  Validating here (``model_validate``) makes
    a schema drift fail loudly at fixture-build time rather than vacuously.
    """
    from app.domain.models.llm_responses import PlanResponse

    raw = {
        "message": "patch 3 files in parallel",
        "goal": "patch a/b/c.py",
        "title": "coordinator parallel write",
        "language": "en",
        "steps": [
            {
                "id": "step_fixture_001",
                "description": "patch 3 files in parallel",
                "parallel_work_units": {
                    "work_units": [
                        {
                            "objective": "rewrite /workspace/a.py",
                            "phase": "write",
                            "allowed_tools": ["file_read", "file_write"],
                            "proposed_paths": [
                                {"path": "/workspace/a.py", "op": "modify"}
                            ],
                        },
                        {
                            "objective": "rewrite /workspace/b.py",
                            "phase": "write",
                            "allowed_tools": ["file_read", "file_write"],
                            "proposed_paths": [
                                {"path": "/workspace/b.py", "op": "modify"}
                            ],
                        },
                        {
                            "objective": "rewrite /workspace/c.py",
                            "phase": "write",
                            "allowed_tools": ["file_read", "file_write"],
                            "proposed_paths": [
                                {"path": "/workspace/c.py", "op": "modify"}
                            ],
                        },
                    ]
                },
            }
        ],
    }
    return PlanResponse.model_validate(raw)


class PricedPlannerFakeListChatModel(PricedFakeListChatModel):
    """Priced fake planner that ALSO satisfies ``with_structured_output``.

    The real planner (``graphs/main_graph.py`` planner_node ~line 616 and
    ``flows/planner_react.py:1157``) calls
    ``self._llm.with_structured_output(PlanResponse).ainvoke(messages)`` inside
    a ``try/except Exception`` that **silently falls back to a single-step plan
    with NO parallel_work_units** on any error.  Stock
    ``FakeListChatModel.with_structured_output`` raises ``NotImplementedError``
    (verified), so a plain priced fake would never drive the 3-worker dispatch
    — C9/D would pass vacuously (plan C6 note, line 4048, flagged this).

    This subclass overrides ``with_structured_output`` to return a
    ``RunnableLambda`` that emits the validated ``PlanResponse`` instance
    (``_build_parallel_plan``).  Both call sites use **async** ``ainvoke``;
    ``RunnableLambda`` wraps the sync fn for async automatically, so a single
    lambda serves both ``invoke`` and ``ainvoke``.  Priced identity +
    usage_metadata stay intact on the base ``_generate`` path (the base
    AIMessage content is the plan's JSON) so any cost-handler assertion on the
    raw model still sees ``openai_official`` + non-zero tokens.
    """

    def with_structured_output(  # type: ignore[override]
        self,
        schema: Any = None,
        **kwargs: Any,
    ) -> Runnable:
        plan = _build_parallel_plan()
        return RunnableLambda(lambda _messages: plan)


class PricedRoutingFakeChatModel(PricedFakeListChatModel):
    """[C2 finish-core §5.5 G5] ONE fake injected as _ConfigSnapshot.llm, shared
    by the parent planner + every child's planner + every child's ReAct.

    - with_structured_output(PlanResponse): routes BY CONTEXT — a parent call
      (no selector line in messages) → the parallel plan; a child call (messages
      carry the registered selector, e.g. the objective line) → a NON-parallel
      single-step plan so the child proceeds into ReAct (not back into the
      coordinator branch). PlanUpdateResponse → keep-complete (no steps);
      ConversationSummaryResponse → empty.
    - bind_tools(...): routes child ReAct calls by the selector to a per-selector
      deque of scripted AIMessages (call-1 tool calls, call-2 final). Per-selector
      asyncio.Lock so concurrent children don't drift.
    - Inherits priced identity + usage_metadata (INV-C2 → non-zero cost).
    Fail-fast on unscripted / 0-or-multi selector match / exhaustion.
    """

    responses: list[str] = [""]

    def model_post_init(self, _ctx) -> None:
        object.__setattr__(self, "_planner_response", None)
        object.__setattr__(self, "_child_decks", {})
        object.__setattr__(self, "_locks", {})
        object.__setattr__(self, "_selectors", [])
        object.__setattr__(self, "parent_plan_calls", 0)
        object.__setattr__(self, "child_plan_calls", 0)
        object.__setattr__(self, "build_llm_call_count", 0)
        object.__setattr__(self, "_parent_executor_response", None)

    def setup_responses(
        self,
        *,
        planner_response,
        child_responses=None,
        parent_executor_response=None,
        **legacy,
    ):
        object.__setattr__(self, "_parent_executor_response", parent_executor_response)
        from collections import deque
        object.__setattr__(self, "_planner_response", planner_response)
        step0 = planner_response["steps"][0]
        units = step0.get("parallel_work_units", {}).get("work_units", []) if step0.get("parallel_work_units") else []
        if child_responses is None:
            child_responses = {}
            for i, unit in enumerate(units, start=1):
                turns = legacy.get(f"child_{i}_write_calls")
                if turns is not None:
                    child_responses[unit["objective"]] = turns
        decks, locks, selectors = {}, {}, []
        for selector, turns in child_responses.items():
            if "\n" in selector or "**Objective**:" in selector:
                raise ValueError(
                    f"routing fake: selector {selector!r} must be a single line "
                    "without the '**Objective**:' marker (avoids ambiguous routing)"
                )
            decks[selector] = deque(self._build_child_turns(turns))
            locks[selector] = asyncio.Lock()
            selectors.append(selector)
        object.__setattr__(self, "_child_decks", decks)
        object.__setattr__(self, "_locks", locks)
        object.__setattr__(self, "_selectors", selectors)

    def _build_child_turns(self, turns):
        from langchain_core.messages import AIMessage
        msgs = []
        for t in turns:
            if "tool" in t:
                msgs.append(AIMessage(
                    content="",
                    tool_calls=[{"name": t["tool"], "args": t["args"], "id": f"tc-{len(msgs)}"}],
                    usage_metadata={"input_tokens": self.input_tokens,
                                    "output_tokens": self.output_tokens,
                                    "total_tokens": self.input_tokens + self.output_tokens},
                ))
            else:
                msgs.append(("control", t))
        msgs.append(AIMessage(
            content="done",
            usage_metadata={"input_tokens": self.input_tokens,
                            "output_tokens": self.output_tokens,
                            "total_tokens": self.input_tokens + self.output_tokens},
        ))
        return msgs

    def _match_selector(self, messages):
        text = "\n".join(getattr(m, "content", "") or "" for m in messages)
        targets = {f"**Objective**: {s}": s for s in self._selectors}
        hits: list[str] = []
        for line in text.splitlines():
            s = targets.get(line.strip())
            if s is not None and s not in hits:
                hits.append(s)
        if len(hits) == 1:
            return hits[0]
        if len(hits) == 0:
            return None
        raise AssertionError(f"routing fake: multi-selector match {hits} in messages")

    def with_structured_output(self, schema=None, **kwargs):  # type: ignore[override]
        from langchain_core.runnables import RunnableLambda

        async def _route(messages):
            name = getattr(schema, "__name__", "")
            if name == "PlanResponse":
                selector = self._match_selector(messages)
                if selector is None:
                    object.__setattr__(self, "parent_plan_calls", self.parent_plan_calls + 1)
                    return self._build_parent_plan()
                object.__setattr__(self, "child_plan_calls", self.child_plan_calls + 1)
                return self._build_child_single_step_plan(selector)
            if name == "PlanUpdateResponse":
                return self._build_keep_complete_update()
            if name == "ConversationSummaryResponse":
                return self._build_empty_summary()
            raise AssertionError(f"routing fake: unscripted structured schema {name}")

        return RunnableLambda(_route)

    def bind_tools(self, tools, **kwargs):  # type: ignore[override]
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise NotImplementedError(
            "PricedRoutingFakeChatModel is async-only in the coordinator graph "
            "(the graph uses ainvoke → _agenerate)."
        )

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        from langchain_core.messages import AIMessage
        from langchain_core.outputs import ChatGeneration, ChatResult
        selector = self._match_selector(messages)
        if selector is None or selector not in self._child_decks:
            # [C2b rollout WS0] Parent (non-child) executor call. If a parent
            # executor response is configured, return it as a terminal AIMessage
            # so a SANITIZED flag-off step can run to completion (dark-launch).
            # Else preserve the original raise (existing coordinator tests rely on it).
            if selector is None and self._parent_executor_response is not None:
                msg = AIMessage(
                    content=self._parent_executor_response,
                    usage_metadata={"input_tokens": self.input_tokens,
                                    "output_tokens": self.output_tokens,
                                    "total_tokens": self.input_tokens + self.output_tokens},
                )
                return ChatResult(generations=[ChatGeneration(message=msg)])
            raise AssertionError("routing fake: no child deck for selector in messages")
        async with self._locks[selector]:
            deck = self._child_decks[selector]
            if not deck:
                raise AssertionError(f"routing fake: deck exhausted for {selector!r}")
            turn = deck.popleft()
        if isinstance(turn, tuple) and turn[0] == "control":
            turn = await self._apply_control(turn[1])
        return ChatResult(generations=[ChatGeneration(message=turn)])

    async def _apply_control(self, spec):
        from langchain_core.messages import AIMessage
        if "sleep_seconds" in spec:
            await asyncio.sleep(spec["sleep_seconds"])
            return AIMessage(content="done", usage_metadata={
                "input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
                "total_tokens": self.input_tokens + self.output_tokens})
        exc_type = {"RuntimeError": RuntimeError}.get(spec.get("exc_type"), RuntimeError)
        raise exc_type(spec.get("exc_msg", "scripted child failure"))

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        # [finish-core R1-P2] The ONLY `astream` caller on the shared snap.llm is
        # the post-graph background summary (agent_task_runner._do_postprocess
        # Phase 3 → run_background_summary → summary_llm.astream). The child/ReAct
        # path uses ainvoke→_agenerate (selector-routed, fail-fast); the in-graph
        # summary uses with_structured_output. So `astream` is unambiguously the
        # background-summary surface — yield ONE priced benign summary chunk so
        # that surface (user-visible summary + its background_summary cost entry)
        # is COVERED instead of silently degrading on a no-selector AssertionError.
        from langchain_core.outputs import ChatGenerationChunk
        from langchain_core.messages import AIMessageChunk
        chunk = ChatGenerationChunk(message=AIMessageChunk(
            content="summary",
            usage_metadata={"input_tokens": self.input_tokens,
                            "output_tokens": self.output_tokens,
                            "total_tokens": self.input_tokens + self.output_tokens},
        ))
        if run_manager is not None:
            await run_manager.on_llm_new_token("summary", chunk=chunk)
        yield chunk

    def _build_parent_plan(self):
        from app.domain.models.llm_responses import PlanResponse
        return PlanResponse.model_validate(self._planner_response | {
            "title": "coordinator", "goal": "parallel", "language": "zh", "message": "ok",
        })

    def _build_child_single_step_plan(self, selector):
        from app.domain.models.llm_responses import PlanResponse, StepDef
        return PlanResponse(
            title="child", goal=selector, language="zh", message="ok",
            steps=[StepDef(description=f"do: {selector}")],
        )

    def _build_keep_complete_update(self):
        from app.domain.models.llm_responses import PlanUpdateResponse
        return PlanUpdateResponse(steps=[])

    def _build_empty_summary(self):
        from app.domain.models.llm_responses import ConversationSummaryResponse
        return ConversationSummaryResponse()


# ── C2 async_session (INV-C1) ───────────────────────────────────────────────


@pytest.fixture
async def async_session(async_session_factory):
    """[PR-9b-C C2 / INV-C1] Yield the production session **factory**.

    NOT a single ``AsyncSession`` — tests open scopes themselves::

        async with async_session() as s:
            ...

    Reuses the suite's ``async_session_factory`` (conftest.py:82, an
    ``async_sessionmaker`` bound to the shared test ``async_engine``); same
    factory production uses, no parallel engine. The ``async``/``await``-free
    body still needs ``async def`` so anyio's fixture machinery treats it as an
    async fixture consistent with the rest of this conftest.
    """
    yield async_session_factory


# ── C3 redis_real ───────────────────────────────────────────────────────────


@pytest.fixture
async def redis_real(redis_client):
    """[PR-9b-C C3] Thin alias over the existing ``redis_client`` fixture.

    ``redis_client`` (conftest.py:597) already connects to DB 15 and
    ``flushdb()`` before/after each test for isolation; we re-expose it under
    the name the coordinator E2E tests request.
    """
    yield redis_client


# ── C4 minio_real (INV-C3) ──────────────────────────────────────────────────


class _MinioRealHandle:
    """Handle exposing the raw ``minio.Minio`` SDK client + per-test bucket."""

    def __init__(self, client: Any, bucket_name: str) -> None:
        self.client = client
        self.bucket_name = bucket_name


@pytest.fixture
async def minio_real():
    """[PR-9b-C C4 / INV-C3] Per-test MinIO bucket via the raw SDK.

    Production storage adapters (``MinioStore`` /
    ``minio_file_storage.py``) don't expose bucket create/drop, so we drive the
    underlying ``minio.Minio`` SDK directly — reading the SAME env vars the
    production adapter reads (``settings.minio_endpoint`` /
    ``minio_access_key`` / ``minio_secret_key`` / ``minio_secure`` /
    ``minio_region``; core/config.py:74-79).

    Bucket name is ``actus-test-<uuid4hex8>`` for cross-test isolation.
    Teardown best-effort removes all objects + the bucket.  Sync SDK calls are
    wrapped in ``asyncio.to_thread`` to stay off the event loop.
    """
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

    bucket_name = f"actus-test-{_uuid.uuid4().hex[:8]}"
    await asyncio.to_thread(client.make_bucket, bucket_name)

    handle = _MinioRealHandle(client=client, bucket_name=bucket_name)
    try:
        yield handle
    finally:
        # Best-effort recursive teardown: list -> bulk delete, then remove the
        # bucket in its OWN try so an object-delete failure still attempts
        # bucket removal.  Teardown must never fail a test, but a leak MUST be
        # logged so CI doesn't silently accumulate ``actus-test-*`` buckets.
        try:
            objects = await asyncio.to_thread(
                lambda: list(client.list_objects(bucket_name, recursive=True))
            )
            delete_list = [DeleteObject(o.object_name) for o in objects]
            if delete_list:
                # remove_objects returns a generator of errors; drain it.
                errors = await asyncio.to_thread(
                    lambda: list(client.remove_objects(bucket_name, delete_list))
                )
                if errors:
                    logger.warning(
                        "minio_real teardown: %d object(s) failed to delete in "
                        "bucket %s: %r",
                        len(errors),
                        bucket_name,
                        errors,
                    )
        except Exception as exc:  # noqa: BLE001 — best-effort teardown
            logger.warning(
                "minio_real teardown: object cleanup failed for bucket %s: %s",
                bucket_name,
                exc,
            )
        try:
            await asyncio.to_thread(client.remove_bucket, bucket_name)
        except Exception as exc:  # noqa: BLE001 — best-effort teardown
            logger.warning(
                "minio_real teardown: bucket %s leaked (remove_bucket failed): %s",
                bucket_name,
                exc,
            )


# ── C5 sandbox_real (INV-C3) ────────────────────────────────────────────────


@pytest.fixture
async def sandbox_real():
    """[PR-9b-C C5 / INV-C3] Per-test ``DockerSandbox`` via the real factory.

    Construction is ``await DockerSandbox.create(...)`` (the classmethod
    factory; api/app/infrastructure/external/sandbox/docker_sandbox.py:238),
    NOT ctor + start().  The real signature is
    ``create(cls, user_id: Optional[str] = None)`` — there is **no**
    ``session_id`` / ``image`` / ``workspace`` parameter (plan snippet was a
    hint; adapted to the real API).  We pass a unique ``user_id`` so any
    per-user memory mount path is isolated.

    Teardown is ``await sandbox.destroy()``.
    """
    from app.infrastructure.external.sandbox.docker_sandbox import DockerSandbox

    user_id = f"test-{_uuid.uuid4().hex[:8]}"
    sandbox = await DockerSandbox.create(user_id=user_id)
    try:
        yield sandbox
    finally:
        # ``DockerSandbox.destroy()`` (docker_sandbox.py:334) catches its OWN
        # exceptions and returns ``bool`` — ``True`` on success (incl. the
        # already-gone ``NotFound`` idempotent case), ``False`` on a genuine
        # Docker remove/close failure (the ``except`` at :384-386 swallows and
        # returns ``False`` rather than raising). A bare ``except`` therefore
        # NEVER fires on the leak-causing return-False path, so a leaked
        # container would go unlogged. Handle BOTH: the return-False path AND
        # the (defensive) raise path. Best-effort — neither must mask the test
        # result; both MUST log so CI doesn't silently accumulate containers
        # (parallel to ``minio_real`` teardown).
        try:
            ok = await sandbox.destroy()
            if not ok:
                logger.warning(
                    "sandbox_real teardown: destroy() returned False — "
                    "container for user %s may have leaked",
                    user_id,
                )
        except Exception as exc:  # noqa: BLE001 — best-effort teardown
            logger.warning(
                "sandbox_real teardown: destroy() raised for user %s: %s",
                user_id,
                exc,
            )


# ── C6 priced fake-LLM fixtures (INV-C2) ────────────────────────────────────
#
# ⚠️  C↔D CONTRACT GAPS (3) — pre-existing E2E tests assume a richer harness ⚠️
# ----------------------------------------------------------------------------
# ROOT CAUSE (all 3): the pre-existing PR-9b-D skipped E2E tests
# (test_coordinator_e2e_{3_work_units,apply_rollback,sibling_cancel}.py) were
# written against a fixture-harness API that the PR-9b-C plan does NOT build.
# None of these break any C-phase (non-skipped) test; ALL are D1–D3 reconciliation.
#
# GAP 1 — fixture_mock_llm_3_workers shape (detailed below).
# GAP 2 — async_client auth shim: the E2E tests POST /api/sessions with
#   headers={"X-Test-User-Id": ...} (e.g. test_coordinator_e2e_sibling_cancel.py:73),
#   but async_client mounts the real app.main:app with real JWT auth and injects
#   NO such shim → 401 on unskip. (C9 smoke sidesteps this by minting a real JWT
#   via create_access_token.) D1–D3: add the test-only shim OR mint real JWTs.
# GAP 3 — sandbox_real file API: sandbox_real yields a raw DockerSandbox whose API
#   is write_file(...) + read_file(...) -> ToolResult. The E2E tests call
#   sandbox_real.atomic_write_file(path, bytes) and compare
#   `await sandbox_real.read_file(path) == b"..."` as bytes (e.g.
#   test_coordinator_e2e_3_work_units.py:89,164) — a method/return-shape the raw
#   DockerSandbox does NOT provide → AttributeError / wrong-type compare on unskip.
#   D1–D3: wrap sandbox_real to expose that API OR rewrite the E2E seed/read to
#   use write_file + read_file(->ToolResult).
# ----------------------------------------------------------------------------
# GAP 1 detail — fixture_mock_llm_3_workers shape:
# The PLAN specs ``fixture_mock_llm_3_workers`` as a ``list[FakeListChatModel]``
# and Task C9's smoke (``test_coordinator_fixture_harness_smoke.py``) asserts
# ``len(fixture_mock_llm_3_workers) == 3`` and indexes ``[0]``.  A LIST is
# therefore plan-correct and MUST NOT be converted into a harness object here.
#
# HOWEVER, the pre-existing PR-9b-D skipped E2E tests
# (``test_coordinator_e2e_{3_work_units,apply_rollback,sibling_cancel}.py``)
# instead call ``fixture_mock_llm_3_workers.setup_responses(planner_response,
# child_1_write_calls, …)`` — a harness API the plan does NOT define.  When
# D1–D3 unskip those tests, the list will ``AttributeError`` on
# ``.setup_responses``.
#
# D1–D3 MUST reconcile the mismatch, choosing ONE of:
#   (a) script the list directly (rewrite the E2E tests to assign each
#       worker's ``responses`` instead of calling ``.setup_responses``), or
#   (b) introduce a harness object that ALSO satisfies ``len()``/``[i]`` AND
#       update C9's list-shape assertions accordingly.
# Until then this is a tracked C↔D contract gap; do not paper over it by
# changing the shape on the C side (that silently breaks C9).
# ----------------------------------------------------------------------------


@pytest.fixture
def fixture_mock_llm_3_workers() -> List[PricedFakeListChatModel]:
    """[PR-9b-C C6 / INV-C2] Three priced fake-LLM workers.

    Each is a ``PricedFakeListChatModel`` emitting ONE AIMessage with a real
    ``usage_metadata`` block and a priced identity
    (``model='gpt-4o-mini'`` / ``provider_id='openai_official'``) so the
    coordinator cost rollup records non-zero, correctly-attributed cost.

    Returned as a list of three independent models (one per child work unit) —
    plan-correct shape (C9 asserts ``len == 3``).  See the ⚠️ C↔D CONTRACT GAP
    note above re: the D-phase ``setup_responses(...)`` mismatch.
    """
    return [
        PricedFakeListChatModel(responses=[f"worker {i} done"])
        for i in range(3)
    ]


@pytest.fixture
def fixture_mock_llm_parallel_planner() -> PricedPlannerFakeListChatModel:
    """[PR-9b-C C6 / INV-C2] Priced fake-LLM planner that drives parallel dispatch.

    The real planner calls ``self._llm.with_structured_output(PlanResponse)
    .ainvoke(messages)`` (``graphs/main_graph.py`` planner_node ~616,
    ``flows/planner_react.py:1157``) inside a ``try/except Exception`` that
    SILENTLY falls back to a single-step plan with no parallel_work_units on
    failure.  Stock ``FakeListChatModel.with_structured_output`` raises
    ``NotImplementedError`` (verified), which would trigger that silent
    fallback and make the 3-worker dispatch never fire — so this fixture uses
    ``PricedPlannerFakeListChatModel``, whose ``with_structured_output``
    override returns a ``RunnableLambda`` emitting a validated ``PlanResponse``
    with three WRITE ``parallel_work_units`` (``_build_parallel_plan``).

    The base ``responses`` is set to that plan's JSON so the raw
    ``ainvoke``/``_generate`` path still yields valid ``PlanResponse`` JSON
    content with the priced identity + usage_metadata intact (INV-C2).
    """
    plan = _build_parallel_plan()
    return PricedPlannerFakeListChatModel(responses=[plan.model_dump_json()])


# ── C7 env_with_coordinator_flag_on ─────────────────────────────────────────


@pytest.fixture
def env_with_coordinator_flag_on(monkeypatch):
    """[PR-9b-C C7] Turn the coordinator feature flag ON for the test.

    Sets ``ACTUS_C2_COORDINATOR_ENABLED=true`` — the exact env var read (each
    call, no cache) by
    ``app/domain/services/coordinator_feature_flag.py:17``
    (``is_coordinator_enabled`` / ``assert_coordinator_enabled``).
    ``monkeypatch`` auto-reverts after the test.
    """
    monkeypatch.setenv("ACTUS_C2_COORDINATOR_ENABLED", "true")
    yield


# ── C8 coordinator_truncation ───────────────────────────────────────────────
#
# The coordinator E2E tests (PR-9b-D) write rows into the session cost tree +
# the coordinator audit/store tables.  The default integration isolation
# pattern (``db_session``'s ``begin()...rollback()`` block — conftest.py:64-71)
# does NOT cover them because the E2E flows commit through their OWN
# independent sessions (``async_session_factory`` per call), so those rows
# survive ``db_session`` rollback and bleed across tests.  ``coordinator_truncation``
# gives the E2E tests an explicit, deterministic reset.
#
# FK topology (verified against app/infrastructure/models/*.py):
#   - cost_records.session_id   → sessions.id   (ON DELETE CASCADE)
#   - cost_records.user_id      → users.id      (ON DELETE CASCADE)
#   - sessions.parent_session_id → sessions.id  (ON DELETE RESTRICT, SELF-REF)
#   - sessions.user_id          → users.id      (ON DELETE SET NULL)
#   - coordinator_apply_audit / coordinator_result_envelope_store /
#     mailbox_envelope_audit    → NO foreign keys (standalone audit tables)
#
# The self-referential RESTRICT FK on ``sessions.parent_session_id`` means a
# plain ordered ``DELETE FROM sessions`` would block on child rows.  We
# therefore use ``TRUNCATE ... CASCADE`` over the full coordinator footprint in
# a single statement — TRUNCATE resolves self-refs and cascades atomically.
#
# ``users`` is deliberately NOT truncated: the FK-parent rows seeded by
# ``sample_user`` / ``fresh_test_user`` live there, and CASCADE-truncating
# users would nuke unrelated committed test data.  Truncating the
# session-scoped tables (which CASCADE-clears cost_records via the
# session_id FK) plus the standalone coordinator tables is the correct scope.
_COORDINATOR_TRUNCATE_TABLES = (
    "cost_records",
    "coordinator_apply_audit",
    "coordinator_result_envelope_store",
    "mailbox_envelope_audit",
    "sessions",
)


async def _truncate_coordinator_tables(session_factory) -> None:
    """Single source of truth for the coordinator-footprint TRUNCATE.

    Opens a short-lived session from the production ``async_session_factory``
    and runs ONE ``TRUNCATE ... RESTART IDENTITY CASCADE`` over the full
    ``_COORDINATOR_TRUNCATE_TABLES`` list (cost_records + the three
    coordinator/mailbox audit tables + sessions). TRUNCATE resolves the
    self-referential RESTRICT FK on ``sessions.parent_session_id`` and cascades
    atomically (a plain ordered ``DELETE FROM sessions`` would block on child
    rows).

    Both the ``coordinator_truncation`` fixture AND the C9 truncation smoke
    test call this helper so the table list is asserted from a single
    definition — if a future edit drops a table from
    ``_COORDINATOR_TRUNCATE_TABLES``, the smoke test's per-table
    assert-empty loop fails (INV-C4 regression protection).

    ⚠️ Destructive: clears the listed tables in the test database. Safe under
    the integration contract (the DB is the throwaway ``manus_test``, never the
    dev ``manus`` library). Do NOT run against a non-test database.
    """
    from sqlalchemy import text

    statement = text(
        "TRUNCATE TABLE "
        + ", ".join(_COORDINATOR_TRUNCATE_TABLES)
        + " RESTART IDENTITY CASCADE"
    )
    async with session_factory() as session:
        await session.execute(statement)
        await session.commit()


@pytest.fixture
async def coordinator_truncation(async_session_factory):
    """[PR-9b-C C8] Explicit (NOT autouse) coordinator-table reset, before + after.

    Delegates to the module-level ``_truncate_coordinator_tables`` helper (the
    single source of truth for the TRUNCATE SQL + table list).  Runs once on
    setup (so a test starts from a clean slate even if a prior test leaked) and
    once on teardown (so this test's committed rows don't bleed forward).

    Explicit-request only — the E2E tests that need a deterministic cost-tree /
    audit-table baseline ``async def test_x(coordinator_truncation, ...)`` it;
    the broader integration suite (which relies on ``db_session`` rollback
    isolation) is unaffected.

    ⚠️ Destructive: TRUNCATE CASCADE clears the listed tables in the test
    database.  Safe under the integration contract (conftest.py header — the
    DB is the throwaway ``manus_test``, never the dev ``manus`` library).  Do
    NOT request this fixture against a non-test database.
    """
    await _truncate_coordinator_tables(async_session_factory)
    try:
        yield
    finally:
        await _truncate_coordinator_tables(async_session_factory)


# ── C8 async_client (CI-infra only — lifespan needs pg/redis) ────────────────


@pytest.fixture
async def async_client():
    """[PR-9b-C C8] Real ASGI ``httpx.AsyncClient`` over the production app.

    Wraps the REAL ``app.main:app`` FastAPI singleton in
    ``asgi_lifespan.LifespanManager`` so the app's ``lifespan`` startup
    (DB / Redis / MinIO / checkpointer pool init — app/main.py:613 ``lifespan=``)
    actually runs, then drives it via ``httpx.ASGITransport``.  This is the
    full-stack E2E entry point the coordinator HTTP flows need.

    ⚠️ Runs ONLY under CI infra: the lifespan connects to Postgres + Redis +
    MinIO on startup, so entering ``LifespanManager(app)`` raises locally
    without those services (per CLAUDE.md — host has no published pg/redis
    ports).  The module still imports cleanly; the fixture body only touches
    infra when actually requested by a test.

    Distinct from the existing ``asgi_client`` (conftest.py:661), which
    deliberately SKIPS lifespan + installs auth/DI overrides for endpoint-unit
    anchors.  ``async_client`` runs the genuine lifespan for E2E.
    """
    import httpx
    from asgi_lifespan import LifespanManager

    from app.main import app as fastapi_app

    async with LifespanManager(fastapi_app):
        transport = httpx.ASGITransport(app=fastapi_app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            yield client


# ── C8 fresh_test_user (INV-C1 — FK parent for coordinator rows) ─────────────


@pytest.fixture
async def fresh_test_user(async_session_factory):
    """[PR-9b-C C8] Commit a minimal valid ``UserModel`` row; yield it.

    Inserts via the production ``async_session_factory`` and **commits** (unlike
    ``sample_user`` at conftest.py:275, which only ``flush()``-es inside the
    rolled-back ``db_session`` transaction).  A committed row is required because
    the coordinator E2E flows open their OWN independent sessions to write
    ``cost_records`` / ``sessions`` rows whose FKs reference ``users.id`` — an
    uncommitted user row is invisible to those sessions (read-committed isolation).

    Minimal valid shape (app/infrastructure/models/user.py): ``id`` is the only
    required column without a default; ``role`` / ``status`` / ``created_at`` /
    ``updated_at`` have server defaults, and ``username`` / ``password_hash`` are
    nullable.  We set ``username`` (unique) + ``password_hash`` to mirror the
    existing user fixtures.

    Yields the persisted ``UserModel`` (exposes ``.id``).  Explicit cleanup is
    unnecessary — ``coordinator_truncation`` does not touch ``users``, but the
    throwaway ``manus_test`` DB tolerates the residual row; tests that need a
    pristine users table can truncate it themselves.
    """
    from app.infrastructure.models.user import UserModel

    user_id = str(_uuid.uuid4())
    user = UserModel(
        id=user_id,
        username=f"coord-c8-{user_id[:8]}",
        password_hash="x",
    )
    async with async_session_factory() as session:
        session.add(user)
        await session.commit()
    yield user


# ── F4.2 routing-fake injection + JWT auth + dependency-ordered client ────────


@pytest.fixture
def inject_routing_fake_llm(monkeypatch):
    """[finish-core §5.6/§5.9] Patch _build_llm so the lifespan-built
    _ConfigSnapshot.llm IS the routing fake. MUST be requested BEFORE the app
    fixture (async_client) so the patch lands before lifespan builds AgentService.
    Yields the fake so the test scripts it via setup_responses(...). The patched
    builder increments fake.build_llm_call_count so coord_async_client can PROVE
    the patch was consumed by the lifespan (R1-P1 ordering self-check)."""
    fake = PricedRoutingFakeChatModel()
    import app.interfaces.service_dependencies as deps

    def _counting_build_llm(*a, **k):
        object.__setattr__(fake, "build_llm_call_count", fake.build_llm_call_count + 1)
        return fake

    monkeypatch.setattr(deps, "_build_llm", _counting_build_llm)
    # Ensure summary_llm stays None so ConversationSummaryResponse routes to snap.llm.
    # (default config leaves summary_model unset; assert if a test config sets it.)
    yield fake


@pytest.fixture
def coord_jwt_headers(fresh_test_user):
    """Authorization header with a real JWT for a COMMITTED user (G6)."""
    from core.security import create_access_token

    token = create_access_token({"sub": str(fresh_test_user.id)})
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
async def coord_async_client(inject_routing_fake_llm, async_client):
    """[R7 P1] Dependency-ordered wrapper. Requesting `inject_routing_fake_llm`
    BEFORE `async_client` guarantees the `_build_llm` monkeypatch is applied
    BEFORE `async_client`'s lifespan builds the AgentService snapshot — pytest
    instantiates a fixture's own dependencies in signature order, so this is a
    real ordering contract (param-order on the TEST is not). The coordinator
    E2E / smoke / dark-launch use THIS client (not `async_client`)."""
    # [finish-core R1-P1] Self-verifying ordering invariant: by the time this
    # fixture body runs, async_client's lifespan has already entered and built
    # the AgentService snapshot. If the _build_llm patch landed FIRST (correct
    # order), the patched builder ran >=1 time during lifespan. A zero count
    # means the snapshot was built with the REAL builder (ordering regressed) —
    # fail LOUDLY here instead of silently running the harness against a real LLM.
    assert inject_routing_fake_llm.build_llm_call_count >= 1, (
        "routing-fake injection did not land before the lifespan built the "
        "AgentService snapshot (build_llm_call_count == 0); coord_async_client "
        "fixture ordering is broken — the harness would run against a real LLM"
    )
    yield async_client


# ── F4.3 session-sandbox bind helper (module-level, NOT a fixture) ────────────


async def bind_session_sandbox_adapter(app, session_id, user_id):
    """[R6 P0] Bind the SESSION's sandbox (the coordinator parent path) and wrap
    it in ParentSandboxAdapter so the E2E seeds/asserts the SAME sandbox the
    apply writes. Call AFTER POST /api/sessions, BEFORE the chat that dispatches.
    Paths are subdir-relative per G2b (e.g. 'workspace/a.py')."""
    from app.infrastructure.external.sandbox.parent_sandbox_adapter import (
        ParentSandboxAdapter,
    )
    handle = await app.state.sandbox_lifecycle_service.bind_new(session_id, user_id=user_id)
    return ParentSandboxAdapter(handle)
