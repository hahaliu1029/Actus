import asyncio
import pytest
from tests.integration.coordinator_fixtures import PricedRoutingFakeChatModel

pytestmark = pytest.mark.anyio


def _planner_response(objectives):
    return {"steps": [{
        "id": "step1", "description": "parallel",
        "parallel_work_units": {"work_units": [
            {"objective": o, "phase": "write",
             "allowed_tools": ["file_read", "file_write"],
             "proposed_paths": [{"path": f"workspace/{i}.py", "op": "modify"}]}
            for i, o in enumerate(objectives)
        ]},
    }]}


def _child_msgs(objective):
    """Mimic a child planner/ReAct message list carrying the objective line."""
    from langchain_core.messages import SystemMessage, HumanMessage
    return [SystemMessage(content=f"**Objective**: {objective}\nDo the work."), HumanMessage(content="go")]


def _parent_msgs():
    from langchain_core.messages import SystemMessage, HumanMessage
    return [SystemMessage(content="You are the planner."), HumanMessage(content="patch files")]


async def test_parent_plan_is_parallel_child_plan_is_single_step():
    from app.domain.models.llm_responses import PlanResponse
    fake = PricedRoutingFakeChatModel()
    fake.setup_responses(
        planner_response=_planner_response(["rewrite a", "rewrite b"]),
        child_responses={
            "rewrite a": [{"tool": "file_write", "args": {"filepath": "workspace/0.py", "content": "A"}}],
            "rewrite b": [{"tool": "file_write", "args": {"filepath": "workspace/1.py", "content": "B"}}],
        },
    )
    parent_plan = await fake.with_structured_output(PlanResponse).ainvoke(_parent_msgs())
    assert parent_plan.steps[0].parallel_work_units is not None  # parent → parallel
    child_plan = await fake.with_structured_output(PlanResponse).ainvoke(_child_msgs("rewrite a"))
    assert child_plan.steps[0].parallel_work_units is None  # child → single-step, no nesting


async def test_child_bind_tools_routes_by_selector_then_finalizes():
    fake = PricedRoutingFakeChatModel()
    fake.setup_responses(
        planner_response=_planner_response(["rewrite a"]),
        child_responses={"rewrite a": [
            {"tool": "file_write", "args": {"filepath": "workspace/0.py", "content": "A"}},
        ]},
    )
    bound = fake.bind_tools([])
    r1 = await bound.ainvoke(_child_msgs("rewrite a"))
    assert r1.tool_calls and r1.tool_calls[0]["name"] == "file_write"
    r2 = await bound.ainvoke(_child_msgs("rewrite a"))
    assert not r2.tool_calls


async def test_fail_fast_on_unknown_selector():
    fake = PricedRoutingFakeChatModel()
    fake.setup_responses(planner_response=_planner_response(["a"]), child_responses={"a": []})
    with pytest.raises(Exception):
        await fake.bind_tools([]).ainvoke(_child_msgs("not-registered"))


async def test_concurrent_children_no_index_drift():
    """[finish-core R3] Verifies that INDEPENDENT per-selector decks don't
    cross-contaminate under concurrent children: three children with three
    DISTINCT selectors/decks run concurrently and each gets its own scripted
    content. NOTE this does NOT prove the per-selector asyncio.Lock is
    load-bearing — the pops are synchronous (no `await` between the empty-check
    and `popleft`), so contention on a SHARED deck is not exercised here. The
    guarantee under test is deck isolation, not lock contention."""
    fake = PricedRoutingFakeChatModel()
    fake.setup_responses(
        planner_response=_planner_response(["a", "b", "c"]),
        child_responses={
            "a": [{"tool": "file_write", "args": {"filepath": "workspace/0.py", "content": "A"}}],
            "b": [{"tool": "file_write", "args": {"filepath": "workspace/1.py", "content": "B"}}],
            "c": [{"tool": "file_write", "args": {"filepath": "workspace/2.py", "content": "C"}}],
        },
    )
    bound = fake.bind_tools([])
    results = await asyncio.gather(
        bound.ainvoke(_child_msgs("a")),
        bound.ainvoke(_child_msgs("b")),
        bound.ainvoke(_child_msgs("c")),
    )
    contents = {r.tool_calls[0]["args"]["content"] for r in results}
    assert contents == {"A", "B", "C"}  # no cross-child drift


async def test_selector_prefix_collision_routes_exactly():
    """[R5 P2] selector 'a' must NOT match the line '**Objective**: abc'."""
    fake = PricedRoutingFakeChatModel()
    fake.setup_responses(
        planner_response=_planner_response(["a", "abc"]),
        child_responses={
            "a": [{"tool": "file_write", "args": {"filepath": "workspace/0.py", "content": "A"}}],
            "abc": [{"tool": "file_write", "args": {"filepath": "workspace/1.py", "content": "ABC"}}],
        },
    )
    r = await fake.bind_tools([]).ainvoke(_child_msgs("a"))
    assert r.tool_calls[0]["args"]["content"] == "A"  # exact-line match, no prefix bleed
    r_abc = await fake.bind_tools([]).ainvoke(_child_msgs("abc"))
    assert r_abc.tool_calls[0]["args"]["content"] == "ABC"  # 'abc' line must NOT bleed to selector 'a'


async def test_background_summary_astream_yields_priced_chunk_without_selector():
    """[finish-core R1-P2] astream is the background-summary surface (no objective
    selector in the messages). It must yield a benign priced chunk, NOT raise the
    _agenerate 'no child deck' fail-fast. Mutation guard: deleting _astream makes
    astream fall back to _agenerate → AssertionError → this test fails."""
    from langchain_core.messages import SystemMessage, HumanMessage
    fake = PricedRoutingFakeChatModel()
    fake.setup_responses(planner_response=_planner_response(["a"]),
                         child_responses={"a": []})
    msgs = [SystemMessage(content="Summarize the conversation."),
            HumanMessage(content="please summarize")]
    chunks = [c async for c in fake.astream(msgs)]
    assert chunks, "astream yielded no chunks"
    text = "".join((getattr(c, "content", "") or "") for c in chunks)
    assert text == "summary"  # our override won, not the canned responses=[""]
    md = chunks[0].usage_metadata
    assert md is not None and md["total_tokens"] > 0  # priced background_summary cost surface


def test_build_llm_call_count_exposed_and_starts_zero():
    """[finish-core R1-P1] The fixture's ordering self-check reads
    fake.build_llm_call_count; it must exist and start at 0 (the patched builder
    increments it during the lifespan)."""
    fake = PricedRoutingFakeChatModel()
    assert fake.build_llm_call_count == 0


async def test_multi_selector_match_raises():
    """[finish-core R3] messages carrying two distinct registered objective lines
    must fail-fast (ambiguous routing), not silently pick one."""
    from langchain_core.messages import SystemMessage, HumanMessage
    fake = PricedRoutingFakeChatModel()
    fake.setup_responses(planner_response=_planner_response(["a", "b"]),
                         child_responses={"a": [], "b": []})
    msgs = [SystemMessage(content="**Objective**: a\n**Objective**: b"), HumanMessage(content="go")]
    with pytest.raises(AssertionError):
        await fake.bind_tools([]).ainvoke(msgs)


async def test_deck_exhaustion_raises_after_scripted_turns():
    """[finish-core R3] a child that calls the LLM more times than scripted must
    fail-fast on an exhausted deck (catches an extra-LLM-call regression)."""
    fake = PricedRoutingFakeChatModel()
    fake.setup_responses(planner_response=_planner_response(["a"]),
                         child_responses={"a": [
                             {"tool": "file_write", "args": {"filepath": "workspace/0.py", "content": "A"}},
                         ]})
    bound = fake.bind_tools([])
    await bound.ainvoke(_child_msgs("a"))   # turn 1: tool call
    await bound.ainvoke(_child_msgs("a"))   # turn 2: trailing final
    with pytest.raises(AssertionError):
        await bound.ainvoke(_child_msgs("a"))  # turn 3: deck exhausted


async def test_plan_update_routes_to_keep_complete():
    """[finish-core R3] PlanUpdateResponse branch → keep-complete (steps==[]).
    Guards the updater_node surface (main_graph)."""
    from app.domain.models.llm_responses import PlanUpdateResponse
    fake = PricedRoutingFakeChatModel()
    fake.setup_responses(planner_response=_planner_response(["a"]), child_responses={"a": []})
    out = await fake.with_structured_output(PlanUpdateResponse).ainvoke(_parent_msgs())
    assert isinstance(out, PlanUpdateResponse) and out.steps == []


async def test_conversation_summary_routes_to_empty():
    """[finish-core R3] ConversationSummaryResponse branch → default empty object.
    Guards the planner_react conversation-summary surface."""
    from app.domain.models.llm_responses import ConversationSummaryResponse
    fake = PricedRoutingFakeChatModel()
    fake.setup_responses(planner_response=_planner_response(["a"]), child_responses={"a": []})
    out = await fake.with_structured_output(ConversationSummaryResponse).ainvoke(_parent_msgs())
    assert isinstance(out, ConversationSummaryResponse)


def test_setup_responses_rejects_multiline_or_marker_selector():
    """[finish-core R3] reject selectors with a newline or the Objective marker —
    they would break the exact-line routing contract."""
    fake = PricedRoutingFakeChatModel()
    with pytest.raises(ValueError):
        fake.setup_responses(planner_response=_planner_response(["a"]),
                             child_responses={"bad\nname": []})
    fake2 = PricedRoutingFakeChatModel()
    with pytest.raises(ValueError):
        fake2.setup_responses(planner_response=_planner_response(["a"]),
                             child_responses={"**Objective**: x": []})
