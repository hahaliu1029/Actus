"""SubagentResearchService orchestration: canonical try/finally + as_completed."""
import asyncio
from unittest.mock import AsyncMock, MagicMock
import pytest

from app.application.services.subagent_research_service import (
    SubagentResearchService,
    ChildResult,
)
from app.domain.services.subagent_research_classifier import ClassifierResult
from app.interfaces.schemas.subagent import ChildOutcome

pytestmark = pytest.mark.anyio


@pytest.fixture
def mock_deps():
    """Mock all SubagentResearchService dependencies."""
    return {
        "session_service": MagicMock(),
        "agent_service": MagicMock(),
        "execution_supervisor": MagicMock(),
        "token_estimator": MagicMock(),
        "summary_llm": MagicMock(),
        "classifier": MagicMock(),
        "sandbox_lifecycle_service": MagicMock(),
        "quota_service": MagicMock(),
    }


@pytest.fixture
def service(mock_deps):
    return SubagentResearchService(**mock_deps)


def test_child_result_dataclass_fields():
    cr = ChildResult(
        child_id="c-1",
        prompt="p",
        outcome=ChildOutcome.COMPLETED,
        final_answer="answer",
        transcript_tokens=1200,
        error_summary=None,
    )
    assert cr.child_id == "c-1"
    assert cr.outcome == ChildOutcome.COMPLETED


async def test_service_constructor_accepts_required_deps(mock_deps):
    """Constructor wires all deps into instance fields."""
    svc = SubagentResearchService(**mock_deps)
    assert svc._session_service is mock_deps["session_service"]
    assert svc._quota_service is mock_deps["quota_service"]


# ---------- Task 18b: _consume_child ----------

async def test_consume_child_normal_complete(service, mock_deps):
    """child stream emits MessageEvent then DoneEvent → outcome=COMPLETED."""
    from app.domain.models.event import MessageEvent, DoneEvent

    async def fake_chat(**kwargs):
        yield MessageEvent(id="ev-1", role="assistant", message="final answer text")
        yield DoneEvent(id="ev-2")

    mock_deps["agent_service"].chat = fake_chat
    mock_deps["token_estimator"].count = MagicMock(return_value=500)

    mock_session = MagicMock()
    mock_session.status = MagicMock(value="completed")
    mock_deps["session_service"].get_session = AsyncMock(return_value=mock_session)

    result = await service._consume_child(
        child_session_id="c-1", user_id="u-1", prompt="research X"
    )

    assert result.child_id == "c-1"
    assert result.outcome == ChildOutcome.COMPLETED
    assert result.final_answer == "final answer text"
    assert result.transcript_tokens >= 0


async def test_consume_child_error_event(service, mock_deps):
    """ErrorEvent → outcome=FAILED with error_summary."""
    from app.domain.models.event import ErrorEvent

    async def fake_chat(**kwargs):
        yield ErrorEvent(id="ev-1", error="LLM rate limit hit")

    mock_deps["agent_service"].chat = fake_chat
    mock_deps["token_estimator"].count = MagicMock(return_value=0)

    result = await service._consume_child(
        child_session_id="c-2", user_id="u-1", prompt="x"
    )

    assert result.outcome == ChildOutcome.FAILED
    assert "rate limit" in (result.error_summary or "")


async def test_consume_child_cancelled(service, mock_deps):
    """asyncio.CancelledError → outcome=CANCELLED."""
    async def fake_chat(**kwargs):
        raise asyncio.CancelledError()
        yield  # unreachable; for type signature

    mock_deps["agent_service"].chat = fake_chat
    mock_deps["token_estimator"].count = MagicMock(return_value=0)

    result = await service._consume_child(
        child_session_id="c-3", user_id="u-1", prompt="x"
    )

    assert result.outcome == ChildOutcome.CANCELLED


# ---------- Task 18c: _do_summary_join_with_retry ----------

async def test_summary_join_validator_pass_first_try(service, mock_deps):
    """Validator passes → no retry."""
    completed = [
        ChildResult(child_id="c-abcdef01", prompt="p1", outcome=ChildOutcome.COMPLETED,
                    final_answer="A", transcript_tokens=100, error_summary=None),
    ]
    mock_deps["summary_llm"].ainvoke = AsyncMock(
        return_value=MagicMock(content="background [[C1:c-abcdef]] conclusion.")
    )

    summary, warnings = await service._do_summary_join_with_retry(
        prompts=["p1"], completed=completed, dropped=[],
    )

    assert warnings == []
    assert "[[C1:c-abcdef]]" in summary
    mock_deps["summary_llm"].ainvoke.assert_called_once()


async def test_summary_join_validator_fail_then_retry_pass(service, mock_deps):
    """First attempt fails validation (missing citation) → retry → pass."""
    completed = [
        ChildResult(child_id="c-abcdef01", prompt="p1", outcome=ChildOutcome.COMPLETED,
                    final_answer="A", transcript_tokens=100, error_summary=None),
    ]
    responses = [
        MagicMock(content="no citation here"),
        MagicMock(content="[[C1:c-abcdef]] cited"),
    ]
    mock_deps["summary_llm"].ainvoke = AsyncMock(side_effect=responses)

    summary, warnings = await service._do_summary_join_with_retry(
        prompts=["p1"], completed=completed, dropped=[],
    )

    assert warnings == []
    assert "[[C1:c-abcdef]]" in summary
    assert mock_deps["summary_llm"].ainvoke.call_count == 2


async def test_summary_join_both_attempts_fail_yield_warnings(service, mock_deps):
    """Both attempts fail → warnings populated, summary still returned."""
    completed = [
        ChildResult(child_id="c-abcdef01", prompt="p1", outcome=ChildOutcome.COMPLETED,
                    final_answer="A", transcript_tokens=100, error_summary=None),
    ]
    mock_deps["summary_llm"].ainvoke = AsyncMock(
        return_value=MagicMock(content="no citation in either attempt")
    )

    summary, warnings = await service._do_summary_join_with_retry(
        prompts=["p1"], completed=completed, dropped=[],
    )

    assert len(warnings) > 0
    assert mock_deps["summary_llm"].ainvoke.call_count == 2


# ---------- Task 18d: run_research full path ----------

async def test_run_research_happy_path(service, mock_deps):
    """Full happy path: 3 prompts → 3 children → join → metric write."""
    from app.domain.models.event import DoneEvent, MessageEvent

    mock_deps["classifier"].classify_batch = AsyncMock(
        return_value=[
            ClassifierResult(approved=True, reason="ok"),
            ClassifierResult(approved=True, reason="ok"),
            ClassifierResult(approved=True, reason="ok"),
        ]
    )
    mock_deps["quota_service"].acquire = AsyncMock(return_value=True)
    mock_deps["quota_service"].release = AsyncMock()

    children_created = []

    async def fake_create(user_id, sample_session_id):
        child = MagicMock()
        child.id = f"child-{len(children_created)+1}"
        children_created.append(child)
        return child

    mock_deps["session_service"].create_session_with_parent = AsyncMock(
        side_effect=fake_create
    )
    parent_session = MagicMock()
    parent_session.id = "parent-1"
    parent_session.user_id = "u-1"
    parent_session.status = MagicMock(value="completed")
    mock_deps["session_service"].get_session = AsyncMock(return_value=parent_session)

    async def fake_chat(**kwargs):
        yield MessageEvent(id="ev", role="assistant", message="answer")
        yield DoneEvent(id="done")

    mock_deps["agent_service"].chat = fake_chat
    mock_deps["token_estimator"].count = MagicMock(return_value=100)
    mock_deps["summary_llm"].ainvoke = AsyncMock(return_value=MagicMock(
        content="[[C1:child-1]] [[C2:child-2]] [[C3:child-3]] integrated"
    ))
    mock_deps["sandbox_lifecycle_service"].suspend = AsyncMock()

    events = []
    async for ev in service.run_research(
        sample_session_id="parent-1",
        user_id="u-1",
        prompts=["q1", "q2", "q3"],
        max_children=3,
    ):
        events.append(ev)

    started = [e for e in events if e.type == "child_started"]
    done = [e for e in events if e.type == "child_done"]
    summary = [e for e in events if e.type == "joined_summary"]

    assert len(started) == 3
    assert len(done) == 3
    assert len(summary) == 1
    assert mock_deps["sandbox_lifecycle_service"].suspend.call_count == 3
    mock_deps["quota_service"].release.assert_called_once()


async def test_run_research_quota_exceeded_raises(service, mock_deps):
    """Quota acquire fails → ConflictError, no children created."""
    from app.application.errors.exceptions import ConflictError

    mock_deps["classifier"].classify_batch = AsyncMock(return_value=[
        ClassifierResult(approved=True, reason="ok")
    ])
    mock_deps["quota_service"].acquire = AsyncMock(return_value=False)
    mock_deps["session_service"].get_session = AsyncMock(return_value=MagicMock())

    with pytest.raises(ConflictError):
        async for _ in service.run_research(
            sample_session_id="parent-1", user_id="u-1",
            prompts=["q1"], max_children=1,
        ):
            pass


async def test_run_research_finally_runs_release_on_consumer_aclose(
    service, mock_deps
):
    """Locks in codex R1 P1: client disconnect / aclose() must still release
    the quota slot and suspend sandboxes. Simulates parent SSE disconnect by
    breaking out of `async for` mid-stream — the async generator's finally
    block must run."""
    from app.domain.models.event import DoneEvent, MessageEvent

    mock_deps["classifier"].classify_batch = AsyncMock(
        return_value=[ClassifierResult(approved=True, reason="ok")]
    )
    mock_deps["quota_service"].acquire = AsyncMock(return_value=True)
    mock_deps["quota_service"].release = AsyncMock()

    async def fake_create(user_id, sample_session_id):
        child = MagicMock()
        child.id = "child-disconnect"
        return child

    mock_deps["session_service"].create_session_with_parent = AsyncMock(
        side_effect=fake_create
    )
    mock_deps["session_service"].get_session = AsyncMock(return_value=MagicMock())

    async def fake_chat(**kwargs):
        yield MessageEvent(id="ev", role="assistant", message="x")
        yield DoneEvent(id="done")

    mock_deps["agent_service"].chat = fake_chat
    mock_deps["token_estimator"].count = MagicMock(return_value=10)
    mock_deps["sandbox_lifecycle_service"].suspend = AsyncMock()

    gen = service.run_research(
        sample_session_id="parent-1", user_id="u-1",
        prompts=["q1"], max_children=1,
    )
    # Consume one event then close the generator (simulates SSE client
    # disconnect before completion).
    await gen.__anext__()
    await gen.aclose()

    # finally must have run: quota release + (per-child) sandbox suspend.
    mock_deps["quota_service"].release.assert_called_once()
    mock_deps["sandbox_lifecycle_service"].suspend.assert_called()


async def test_classifier_word_boundary_no_false_positive_on_prefix(
    service, mock_deps
):
    """Locks in codex R1 P2#3: ASCII keyword `fix` must NOT match `prefix`."""
    # `prefix` should pass static; classifier LLM call returns yes.
    mock_deps["classifier"].classify_batch = AsyncMock(
        return_value=[ClassifierResult(approved=True, reason="ok")]
    )
    mock_deps["quota_service"].acquire = AsyncMock(return_value=True)
    mock_deps["quota_service"].release = AsyncMock()
    mock_deps["session_service"].get_session = AsyncMock(return_value=MagicMock())
    mock_deps["session_service"].create_session_with_parent = AsyncMock(
        return_value=MagicMock(id="c-1")
    )
    mock_deps["sandbox_lifecycle_service"].suspend = AsyncMock()
    mock_deps["token_estimator"].count = MagicMock(return_value=10)
    mock_deps["summary_llm"].ainvoke = AsyncMock(
        return_value=MagicMock(content="[[C1:c-1xxxxx]] ok")
    )
    from app.domain.models.event import DoneEvent, MessageEvent

    async def fake_chat(**kwargs):
        yield MessageEvent(id="ev", role="assistant", message="x")
        yield DoneEvent(id="done")

    mock_deps["agent_service"].chat = fake_chat

    # If `fix` falsely matched `prefix`, classifier would never be called
    # (static block short-circuits). The fact we configure classifier here
    # means service should accept the prompt. We just confirm no raise.
    async for _ in service.run_research(
        sample_session_id="parent-1", user_id="u-1",
        prompts=["research prefix sum patterns"], max_children=1,
    ):
        pass


async def test_run_research_aclose_mid_fanout_cancels_pending_children(
    service, mock_deps
):
    """Locks in codex R3 P1: client aclose() AFTER tasks were created MUST
    cancel still-running children via supervisor.request_cancel. aclose raises
    GeneratorExit, not CancelledError, so the previous code path missed this.
    """
    from app.domain.models.event import DoneEvent, MessageEvent

    mock_deps["classifier"].classify_batch = AsyncMock(
        return_value=[
            ClassifierResult(approved=True, reason="ok"),
            ClassifierResult(approved=True, reason="ok"),
        ]
    )
    mock_deps["quota_service"].acquire = AsyncMock(return_value=True)
    mock_deps["quota_service"].release = AsyncMock()

    created = []

    async def fake_create(user_id, sample_session_id):
        child = MagicMock()
        child.id = f"child-{len(created) + 1}"
        created.append(child)
        return child

    mock_deps["session_service"].create_session_with_parent = AsyncMock(
        side_effect=fake_create
    )
    mock_deps["session_service"].get_session = AsyncMock(return_value=MagicMock())

    # First child completes immediately; second blocks until cancel.
    block_forever = asyncio.Event()
    call_state = {"n": 0}

    async def fake_chat(**kwargs):
        call_state["n"] += 1
        n = call_state["n"]
        if n == 1:
            yield MessageEvent(id=f"ev-{n}", role="assistant", message="quick")
            yield DoneEvent(id=f"done-{n}")
        else:
            # Block until cancelled (simulates a still-running child).
            try:
                await block_forever.wait()
            except asyncio.CancelledError:
                raise
            yield DoneEvent(id="never")

    mock_deps["agent_service"].chat = fake_chat
    mock_deps["token_estimator"].count = MagicMock(return_value=10)
    mock_deps["sandbox_lifecycle_service"].suspend = AsyncMock()
    mock_deps["execution_supervisor"].request_cancel = AsyncMock()
    mock_deps["agent_service"].stop_session = AsyncMock()

    gen = service.run_research(
        sample_session_id="parent-1", user_id="u-1",
        prompts=["q1", "q2"], max_children=2,
    )

    # Drive the generator until we've seen ChildStarted for BOTH children
    # AND one ChildDone (proving fanout is past the create_task line).
    seen_started = 0
    seen_done = 0
    async for ev in gen:
        if ev.type == "child_started":
            seen_started += 1
        if ev.type == "child_done":
            seen_done += 1
            break
    assert seen_started == 2
    assert seen_done == 1

    # Now disconnect — close the generator while child 2 is still running.
    await gen.aclose()

    # request_cancel must have been called for the pending child (child-2).
    cancel_calls = mock_deps["execution_supervisor"].request_cancel.call_args_list
    cancelled_ids = [c.kwargs.get("session_id") for c in cancel_calls]
    assert "child-2" in cancelled_ids, (
        f"expected child-2 to receive request_cancel, got calls: {cancel_calls}"
    )
    # Quota also released.
    mock_deps["quota_service"].release.assert_called_once()
    # Sandbox suspended for both children.
    assert mock_deps["sandbox_lifecycle_service"].suspend.call_count == 2


def test_classifier_word_boundary_directly():
    """Direct unit assertion: SubagentResearchClassifier._static_block must
    distinguish word-class `fix` from `prefix`."""
    from app.domain.services.subagent_research_classifier import (
        SubagentResearchClassifier,
    )

    clf = SubagentResearchClassifier(llm=MagicMock())
    # Should NOT block on `prefix` (contains substring `fix`).
    assert clf._static_block("research prefix sum patterns") is None
    # Should block on `fix` as whole word.
    assert clf._static_block("fix the bug in main.py") is not None
    # Should NOT block on `arm` (rm is in word block; `arm` contains `rm`).
    assert clf._static_block("research arm architecture") is None
    # Should block on `rm` as whole word.
    assert clf._static_block("rm -rf /tmp/foo") is not None
