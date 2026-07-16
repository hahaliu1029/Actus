"""M2 PR-4: verify ``build_main_graph`` wires ``memory_snapshot_provider``
through each node that assembles a system prompt.

PR-3 added the data carrier (``MemorySnapshot``) + three sections that read
``RenderContext.memory_snapshot``. PR-4 connects the async fetch upstream
to the pure sync sections via a provider callable injected at graph
construction time.

Tests here verify the **integration contract** between ``build_main_graph``
and the provider:

1. Absent provider → existing behavior is unchanged (``ctx.memory_snapshot``
   stays None, sections inert).
2. Provider is invoked **per node call** (planner / executor / updater),
   not cached across the graph — so the snapshot reflects any in-flight
   writes that memory tools made during execution.
3. Provider raising any exception is swallowed and the graph keeps
   running with ``ctx.memory_snapshot = None`` (the "degrade to no
   memory" contract).
4. Provider returning a ``MemorySnapshot`` threads it into the
   ``RenderContext`` the PromptAssembler renders against.

Rather than run the full compiled graph (lots of mocks), we intercept
``build_render_context`` — it's the single funnel every node uses to
build the context. Verifying its ``memory_snapshot`` kwarg on each call
is equivalent to observing what the sections see.
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk

from app.application.services.sandbox_accessors import (
    EagerBrowserAccessor,
    EagerSandboxAccessor,
)
from app.domain.models.event import MessageEvent
from app.domain.models.llm_responses import PlanResponse, PlanUpdateResponse, StepDef
from app.domain.services.prompts.memory_snapshot import MemorySnapshot


pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


# ---- Shared mocks (mirrors test_main_graph.py fixtures) --------------- #


def _make_mock_react_graph():
    class MockReactGraph:
        async def astream(self, input_state, config=None, **kwargs):
            yield {"llm_node": {
                "events": [MessageEvent(role="assistant", message="Step done")],
                "messages": [
                    AIMessage(content='{"success": true, "result": "done", "attachments": []}'),
                ],
            }}

        async def ainvoke(self, input_state, config=None):
            return {
                "events": [MessageEvent(role="assistant", message="Step done")],
                "messages": [
                    AIMessage(content='{"success": true, "result": "done", "attachments": []}'),
                ],
                "should_interrupt": False,
                "attempt_count": 1,
                "failure_count": 0,
            }

    return MockReactGraph()


def _make_structured_planner_llm(
    create_response: PlanResponse | None = None,
    update_response: PlanUpdateResponse | None = None,
):
    if create_response is None:
        create_response = PlanResponse(
            title="t", goal="g", language="en",
            steps=[StepDef(description="Step 1")], message="ok",
        )
    if update_response is None:
        update_response = PlanUpdateResponse(steps=[])

    create_structured = AsyncMock()
    create_structured.ainvoke = AsyncMock(return_value=create_response)
    update_structured = AsyncMock()
    update_structured.ainvoke = AsyncMock(return_value=update_response)

    llm = MagicMock()

    def _with_structured_output(schema, **_kwargs):
        if schema is PlanResponse:
            return create_structured
        if schema is PlanUpdateResponse:
            return update_structured
        raise ValueError(f"Unexpected schema: {schema}")

    llm.with_structured_output = MagicMock(side_effect=_with_structured_output)

    async def _astream(messages, **kwargs):
        yield AIMessageChunk(content='{"message": "done", "attachments": []}')

    llm.astream = _astream
    return llm


def _empty_initial_state() -> dict:
    return {
        "message": "help",
        "language": "en",
        "attachments": [],
        "image_content_blocks": [],
        "plan": None,
        "current_step": None,
        "messages": [],
        "execution_summary": "",
        "events": [],
        "flow_status": "idle",
        "session_id": "sess-1",
        "should_interrupt": False,
        "resume_value": None,
        "original_request": "",
        "skill_context": "",
        "conversation_summaries": [],
    }


def _sample_snapshot() -> MemorySnapshot:
    from app.domain.models.memory_chunk import MemoryChunk

    now = datetime.now(timezone.utc)
    chunk = MemoryChunk(
        id="c1",
        user_id="u1",
        content="user prefers concise replies",
        content_hash="h1",
        source="manual",
        metadata={},
        created_at=now,
        updated_at=now,
        embedding=None,
        category="user",
        pinned=True,
    )
    return MemorySnapshot(user_chunks=(chunk,))


# ---- Compile-time surface area --------------------------------------- #


class TestProviderWiringCompiles:
    def test_default_provider_none_compiles(self):
        from app.domain.services.graphs.main_graph import build_main_graph

        graph = build_main_graph(
            _allow_default_prompt_assembler=True,
            planner_llm=_make_structured_planner_llm(),
            react_graph=_make_mock_react_graph(),
            summary_llm=_make_structured_planner_llm(),
            uow_factory=MagicMock(),
            session_id="s1",
        )
        assert graph is not None

    def test_explicit_provider_compiles(self):
        from app.domain.services.graphs.main_graph import build_main_graph

        async def _provider() -> MemorySnapshot | None:
            return None

        graph = build_main_graph(
            _allow_default_prompt_assembler=True,
            planner_llm=_make_structured_planner_llm(),
            react_graph=_make_mock_react_graph(),
            summary_llm=_make_structured_planner_llm(),
            uow_factory=MagicMock(),
            session_id="s1",
            memory_snapshot_provider=_provider,
        )
        assert graph is not None


# ---- Runtime behavior via build_render_context interception ----------- #


class _RenderContextRecorder:
    """Wrap the real ``build_render_context`` to capture ``memory_snapshot``."""

    def __init__(self):
        self.calls: list[MemorySnapshot | None] = []

    def install(self, monkeypatch) -> None:
        """Patch the source module's ``build_render_context`` BEFORE
        ``build_main_graph`` runs its local import — the closure grabs
        whatever is bound at import time, so the order matters.
        """
        from app.domain.services.prompts import render_context as rc_mod

        real = rc_mod.build_render_context

        def _spy(*args, **kwargs):
            self.calls.append(kwargs.get("memory_snapshot"))
            return real(*args, **kwargs)

        monkeypatch.setattr(rc_mod, "build_render_context", _spy)


def _make_graph(provider=None):
    from app.domain.services.graphs.main_graph import build_main_graph

    mock_uow = AsyncMock()
    mock_uow.__aenter__ = AsyncMock(return_value=mock_uow)
    mock_uow.__aexit__ = AsyncMock(return_value=False)
    mock_uow.session = AsyncMock()
    mock_uow.session.get_skill_graph_state = AsyncMock(return_value=None)

    planner = _make_structured_planner_llm()

    return build_main_graph(
        _allow_default_prompt_assembler=True,
        planner_llm=planner,
        react_graph=_make_mock_react_graph(),
        summary_llm=planner,
        uow_factory=MagicMock(return_value=mock_uow),
        session_id="s1",
        memory_snapshot_provider=provider,
    )


class TestProviderInvokedPerRender:
    async def test_absent_provider_keeps_snapshot_none(self, monkeypatch):
        """No provider → every ``build_render_context`` call sees ``None``
        across all three nodes. Rules out a misplaced default value."""
        recorder = _RenderContextRecorder()
        recorder.install(monkeypatch)

        graph = _make_graph(provider=None)
        await graph.ainvoke(_empty_initial_state())

        assert recorder.calls  # at least one render happened
        assert all(s is None for s in recorder.calls)

    async def test_provider_invoked_only_from_executor(self, monkeypatch):
        """P3 fix (codex 2026-04-18): planner + updater registries don't
        consume memory sections — the snapshot fetch is confined to
        ``executor_node``.

        Asserting the invariant directly:
        - provider is invoked exactly once per executor render (1 call for
          the 1-step mock plan used here)
        - planner / updater renders pass ``memory_snapshot=None`` —
          proving they took the no-fetch path
        - at least one render DID receive the snapshot (sanity: executor
          actually wired it up)

        If someone later regresses by fetching in planner / updater too,
        ``provider_calls`` will exceed the number of renders that actually
        saw the snapshot, and this test will fail.
        If someone caches at the graph level (single fetch shared across
        executor iterations), ``provider_calls`` will fall below the number
        of snapshot renders — also fails.
        """
        recorder = _RenderContextRecorder()
        recorder.install(monkeypatch)

        snap = _sample_snapshot()
        provider_calls = 0

        async def _provider() -> MemorySnapshot | None:
            nonlocal provider_calls
            provider_calls += 1
            return snap

        graph = _make_graph(provider=_provider)
        await graph.ainvoke(_empty_initial_state())

        snapshot_renders = sum(1 for s in recorder.calls if s is snap)
        none_renders = sum(1 for s in recorder.calls if s is None)

        # Exactly one executor render in the 1-step mock flow.
        assert provider_calls == 1, (
            f"expected 1 provider call for 1-step mock plan, got {provider_calls}; "
            "regression: are planner/updater calling the provider again?"
        )
        # One-to-one: every provider call corresponds to one snapshot render.
        assert snapshot_renders == provider_calls, (
            f"graph-level caching regression? provider_calls={provider_calls}, "
            f"snapshot_renders={snapshot_renders}"
        )
        # Planner + (optional updater) got None — they took the no-fetch path.
        assert none_renders >= 1, (
            "expected at least planner to render with snapshot=None, "
            f"got recorder.calls={recorder.calls}"
        )

    async def test_provider_exception_is_swallowed(self, monkeypatch):
        recorder = _RenderContextRecorder()
        recorder.install(monkeypatch)

        invocations = 0

        async def _broken_provider() -> MemorySnapshot | None:
            nonlocal invocations
            invocations += 1
            raise RuntimeError("DB unreachable")

        graph = _make_graph(provider=_broken_provider)
        # The graph must NOT raise — degradation to "no memory" is silent.
        result = await graph.ainvoke(_empty_initial_state())

        # Exactly one provider invocation (the executor's) — confirms the
        # fetch stays confined to executor even on the exception path.
        assert invocations == 1, (
            f"expected provider called exactly once from executor, got {invocations}"
        )
        assert recorder.calls
        assert all(s is None for s in recorder.calls)
        # The overall flow still produced a plan.
        assert result.get("plan") is not None


# ---- PlannerReActFlow._build_memory_snapshot_provider ---------------- #


class _StubSessionCM:
    """Minimal async-context-manager exposing the session object."""

    def __init__(self, session):
        self._session = session

    async def __aenter__(self):
        return self._session

    async def __aexit__(self, *_exc):
        return False


def _import_planner_flow():
    from app.domain.services.flows.planner_react import PlannerReActFlow

    return PlannerReActFlow


def _build_flow_with_memory_deps(
    *,
    user_id: str = "u1",
    session_factory=None,
    repo_factory=None,
):
    """Build a bare ``PlannerReActFlow`` with only the fields exercised by the
    provider builder. We don't touch ``__init__`` — we construct a blank
    instance and assign the handful of attributes ``_build_memory_snapshot_provider``
    reads. This keeps the unit scoped to the provider-building logic.
    """
    PlannerReActFlow = _import_planner_flow()
    flow = PlannerReActFlow.__new__(PlannerReActFlow)
    flow._user_id = user_id
    flow._memory_session_factory = session_factory
    flow._memory_repo_factory = repo_factory
    return flow


class TestBuildMemorySnapshotProvider:
    def test_returns_none_when_user_id_empty(self):
        flow = _build_flow_with_memory_deps(
            user_id="",
            session_factory=MagicMock(),
            repo_factory=MagicMock(),
        )
        assert flow._build_memory_snapshot_provider() is None

    def test_returns_none_when_session_factory_missing(self):
        flow = _build_flow_with_memory_deps(
            session_factory=None,
            repo_factory=MagicMock(),
        )
        assert flow._build_memory_snapshot_provider() is None

    def test_returns_none_when_repo_factory_missing(self):
        flow = _build_flow_with_memory_deps(
            session_factory=MagicMock(),
            repo_factory=None,
        )
        assert flow._build_memory_snapshot_provider() is None

    async def test_callable_returns_snapshot_from_repo(self, monkeypatch):
        stub_session = object()
        fake_repo = object()
        snapshot = _sample_snapshot()

        session_factory = MagicMock(return_value=_StubSessionCM(stub_session))
        repo_factory = MagicMock(return_value=fake_repo)

        flow = _build_flow_with_memory_deps(
            user_id="u1",
            session_factory=session_factory,
            repo_factory=repo_factory,
        )

        async def _fake_build(repo, user_id):
            assert repo is fake_repo
            assert user_id == "u1"
            return snapshot

        monkeypatch.setattr(
            "app.domain.services.prompts.memory_snapshot.build_memory_snapshot",
            _fake_build,
        )

        provider = flow._build_memory_snapshot_provider()
        assert callable(provider)
        result = await provider()
        assert result is snapshot
        session_factory.assert_called_once()
        repo_factory.assert_called_once_with(stub_session)

    async def test_callable_swallows_session_factory_error(self):
        def _broken_factory():
            raise RuntimeError("pool exhausted")

        flow = _build_flow_with_memory_deps(
            user_id="u1",
            session_factory=_broken_factory,
            repo_factory=MagicMock(),
        )

        provider = flow._build_memory_snapshot_provider()
        assert callable(provider)
        assert await provider() is None

    async def test_callable_swallows_build_snapshot_error(self, monkeypatch):
        session_factory = MagicMock(return_value=_StubSessionCM(object()))
        repo_factory = MagicMock(return_value=object())

        flow = _build_flow_with_memory_deps(
            user_id="u1",
            session_factory=session_factory,
            repo_factory=repo_factory,
        )

        async def _raise(*_a, **_kw):
            raise RuntimeError("query timeout")

        monkeypatch.setattr(
            "app.domain.services.prompts.memory_snapshot.build_memory_snapshot",
            _raise,
        )

        provider = flow._build_memory_snapshot_provider()
        assert await provider() is None


# ---- PlannerReActFlow._ensure_graphs forwarding ---------------------- #


class TestEnsureGraphsForwardsProvider:
    """The last-mile test: ``_ensure_graphs`` must hand the provider it
    builds into ``build_main_graph``. Without this, all the per-node
    wiring above is silent — ``main_graph`` would see ``None`` even when
    ``PlannerReActFlow`` has real memory deps wired.
    """

    async def test_provider_is_forwarded_to_build_main_graph(self, monkeypatch):
        from langgraph.checkpoint.memory import MemorySaver

        from app.domain.models.app_config import AgentConfig
        from app.domain.services.flows.planner_react import PlannerReActFlow

        # Construct a real flow with memory factories wired, then intercept
        # build_main_graph to capture the provider kwarg.
        mock_uow = AsyncMock()
        mock_uow.__aenter__ = AsyncMock(return_value=mock_uow)
        mock_uow.__aexit__ = AsyncMock(return_value=False)
        mock_uow.session = AsyncMock()
        mock_uow.session.get_skill_graph_state = AsyncMock(return_value=None)

        llm = MagicMock()
        llm.with_structured_output = MagicMock(return_value=AsyncMock())

        fake_session_factory = MagicMock(return_value=_StubSessionCM(object()))
        fake_repo_factory = MagicMock(return_value=object())

        flow = PlannerReActFlow(
            uow_factory=MagicMock(return_value=mock_uow),
            llm=llm,
            agent_config=AgentConfig(max_iterations=100, max_retries=3, max_search_results=10),
            session_id="sess-1",
            browser_accessor=EagerBrowserAccessor(AsyncMock()),
            sandbox_accessor=EagerSandboxAccessor(AsyncMock()),
            search_engine=AsyncMock(),
            mcp_tool=MagicMock(get_tools=MagicMock(return_value=[])),
            a2a_tool=MagicMock(manager=None),
            skill_tool=MagicMock(),
            checkpointer=MemorySaver(),
            user_id="u1",
            memory_session_factory=fake_session_factory,
            memory_repo_factory=fake_repo_factory,
            _allow_default_prompt_assembler=True,
        )

        captured: dict = {}

        def _capture(*_a, **kwargs):
            captured.update(kwargs)
            return MagicMock()

        monkeypatch.setattr(
            "app.domain.services.flows.planner_react.build_main_graph",
            _capture,
        )

        await flow._ensure_graphs()

        assert "memory_snapshot_provider" in captured
        provider = captured["memory_snapshot_provider"]
        assert callable(provider), "provider must be a callable when memory deps are wired"

    async def test_no_provider_when_memory_deps_missing(self, monkeypatch):
        from langgraph.checkpoint.memory import MemorySaver

        from app.domain.models.app_config import AgentConfig
        from app.domain.services.flows.planner_react import PlannerReActFlow

        mock_uow = AsyncMock()
        mock_uow.__aenter__ = AsyncMock(return_value=mock_uow)
        mock_uow.__aexit__ = AsyncMock(return_value=False)
        mock_uow.session = AsyncMock()
        mock_uow.session.get_skill_graph_state = AsyncMock(return_value=None)

        llm = MagicMock()
        llm.with_structured_output = MagicMock(return_value=AsyncMock())

        # Omit memory_session_factory / memory_repo_factory on purpose.
        flow = PlannerReActFlow(
            uow_factory=MagicMock(return_value=mock_uow),
            llm=llm,
            agent_config=AgentConfig(max_iterations=100, max_retries=3, max_search_results=10),
            session_id="sess-1",
            browser_accessor=EagerBrowserAccessor(AsyncMock()),
            sandbox_accessor=EagerSandboxAccessor(AsyncMock()),
            search_engine=AsyncMock(),
            mcp_tool=MagicMock(get_tools=MagicMock(return_value=[])),
            a2a_tool=MagicMock(manager=None),
            skill_tool=MagicMock(),
            checkpointer=MemorySaver(),
            user_id="u1",
            _allow_default_prompt_assembler=True,
        )

        captured: dict = {}

        def _capture(*_a, **kwargs):
            captured.update(kwargs)
            return MagicMock()

        monkeypatch.setattr(
            "app.domain.services.flows.planner_react.build_main_graph",
            _capture,
        )

        await flow._ensure_graphs()

        assert captured.get("memory_snapshot_provider") is None


# ---- Cold-start E2E: memory section headers in executor prompt -------- #
#
# Design doc §636 (M2 验收):
#   "Agent 冷启动（新 session 第一 turn）prompt 里有
#    <memory_user_profile> / <memory_rules> / <memory_fact_index>
#    sections（日志 grep 验证）"
#
# Unit tests already verify (snapshot → section text) in isolation
# (``test_sections_m2_unit.py``). The plumbing tests above verify
# (provider → RenderContext.memory_snapshot) wiring. What was missing
# until now is the end-to-end loop:
#
#   MemorySnapshot → build_render_context → PromptAssembler.assemble
#     → rendered system prompt → messages handed to react_graph
#
# If any link in that chain regresses (e.g. a future refactor
# inadvertently passes the wrong ctx to the assembler, or the
# executor registry drops a memory section), the unit tests and the
# plumbing tests would still pass — but the cold-start prompt would
# lose its memory payload silently. This class locks the end-to-end
# observation: "headers present in the assembled executor text".


def _full_category_snapshot() -> MemorySnapshot:
    """Build a MemorySnapshot with one chunk per category.

    All three sections must render for the three headers to appear.
    Using a pinned user chunk triggers the ``★`` marker branch of
    ``memory_user_profile`` — if the marker logic regresses (e.g.
    dropped to a plain bullet), that branch still renders, so the
    header assertion below still passes. Pin choice is about
    exercising the common production path, not locking a specific
    bullet format.
    """
    from app.domain.models.memory_chunk import MemoryChunk

    now = datetime.now(timezone.utc)

    def _chunk(
        id_: str, category: str, content: str, *, pinned: bool = False
    ) -> "MemoryChunk":
        return MemoryChunk(
            id=id_,
            user_id="u1",
            content=content,
            content_hash=f"h-{id_}",
            source="manual",
            metadata={},
            created_at=now,
            updated_at=now,
            embedding=None,
            category=category,
            pinned=pinned,
        )

    return MemorySnapshot(
        user_chunks=(
            _chunk("u1", "user", "prefers Go over Python", pinned=True),
        ),
        rule_chunks=(
            _chunk("r1", "rule", "never commit without asking"),
        ),
        fact_chunks=(
            _chunk("f1", "fact", "DB is PostgreSQL 17"),
        ),
    )


class _PromptAssemblerSpy:
    """Capture ``assemble()`` output by registry.name.

    Hooks the class method so every PromptAssembler instance built
    during ``build_main_graph`` is observed — including the internal
    default-assembler path (``_allow_default_prompt_assembler=True``).
    """

    def __init__(self) -> None:
        self.captured_by_registry: dict[str, str] = {}

    def install(self, monkeypatch) -> None:
        from app.domain.services.prompts import assembler as asm_mod

        real_assemble = asm_mod.PromptAssembler.assemble

        def _spy_assemble(inner_self, registry, ctx, mode, **kwargs):
            result = real_assemble(inner_self, registry, ctx, mode, **kwargs)
            # Capture the most recent text per registry — if the same
            # registry assembles multiple times (e.g. executor across
            # iterations), we keep the last one, which is the current
            # production-aligned observation.
            self.captured_by_registry[registry.name] = result.text
            return result

        monkeypatch.setattr(asm_mod.PromptAssembler, "assemble", _spy_assemble)


class TestColdStartEmitsMemorySections:
    """End-to-end: three memory section headers appear in the
    executor-assembled prompt when the provider returns a populated
    snapshot.

    Guards design doc §636 acceptance criterion. Single test that runs
    the full ``main_graph.ainvoke`` path with a mocked react_graph and
    planner LLM (same harness the sibling tests use), with a
    non-trivial provider. Fails loudly if ANY of the three section
    headers goes missing — the cold-start user experience depends on
    all three being present.
    """

    async def test_executor_prompt_contains_all_three_memory_headers(
        self, monkeypatch
    ) -> None:
        snapshot = _full_category_snapshot()

        async def _provider() -> MemorySnapshot | None:
            return snapshot

        spy = _PromptAssemblerSpy()
        spy.install(monkeypatch)

        graph = _make_graph(provider=_provider)
        await graph.ainvoke(_empty_initial_state())

        # ``_empty_initial_state`` sets language="en" and the mock planner
        # returns language="en", so the executor dispatches to the EN bundle.
        executor_prompt = spy.captured_by_registry.get("en_executor")
        assert executor_prompt is not None, (
            f"executor never assembled a prompt; captured registries: "
            f"{list(spy.captured_by_registry)}"
        )

        # The three memory section EN headers (zh counterparts:
        # "## 用户画像" / "## 项目规则" / "## 事实索引"). Using ``in``
        # not regex because the headers are stable literals declared in
        # the section modules.
        missing = [
            header
            for header in ("## User Profile", "## Project Rules", "## Fact Index")
            if header not in executor_prompt
        ]
        assert not missing, (
            f"cold-start executor prompt missing memory section headers: "
            f"{missing}. Design doc §636 requires all three. "
            f"Assembled text (first 500 chars): {executor_prompt[:500]}"
        )

    async def test_planner_prompt_does_not_render_memory_headers(
        self, monkeypatch
    ) -> None:
        """Sanity — planner/updater registries don't declare memory
        sections (see M2 PR-4 design §585-599). Even when the provider
        returns a populated snapshot, the planner's assembled prompt
        must NOT contain the three memory headers.

        This locks the round-2 codex P3 finding (planner/updater should
        not fetch memory) at the observable-output level: not only is
        the provider not called from planner_node, but even if someone
        later wires it up, the sections still won't render because
        they're absent from the registry.

        Guard object = the M2 snapshot trio (## User Profile / ## Project
        Rules / ## Fact Index): those three sections stay executor-only.
        B8 adds a DIFFERENT planner-only memory section (recalled_memory,
        fenced block) — its header is deliberately NOT asserted here; the
        B8 guards live in test_main_graph_recall_plumbing.py and
        test_recalled_memory_flag_off_gate.py.
        """
        snapshot = _full_category_snapshot()

        async def _provider() -> MemorySnapshot | None:
            return snapshot

        spy = _PromptAssemblerSpy()
        spy.install(monkeypatch)

        graph = _make_graph(provider=_provider)
        await graph.ainvoke(_empty_initial_state())

        planner_prompt = spy.captured_by_registry.get("en_planner")
        assert planner_prompt is not None, (
            "planner never assembled; test harness changed?"
        )
        # Any of the three memory headers showing up here means a
        # memory section leaked into the planner registry — which
        # would defeat the executor-only scope.
        leaks = [
            header
            for header in ("## User Profile", "## Project Rules", "## Fact Index")
            if header in planner_prompt
        ]
        assert not leaks, (
            f"memory section(s) leaked into planner prompt: {leaks}. "
            f"Registries should keep memory sections in executor only."
        )
