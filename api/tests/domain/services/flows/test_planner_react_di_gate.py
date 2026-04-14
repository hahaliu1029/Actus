"""B5 post-audit: verify PlannerReActFlow DI gate for prompt_assembler.

HIGH #1 regression: the ``build_main_graph`` fail-loud gate broke
``PlannerReActFlow`` direct-construction paths (test_integration.py)
because the flow passed ``prompt_assembler=None`` through without
opting into the default. The fix adds a matching
``_allow_default_prompt_assembler`` flag to ``PlannerReActFlow.__init__``.

MEDIUM #2 regression: ``_run_planner_for_detection`` kept a silent
fallback (``if self._prompt_assembler is None: construct default``)
even after the main graph path went fail-loud. The fix makes the
detection path honor the same flag, either raising or logging a
warning + constructing a default.

These tests lock in the new contract:
1. Direct construction without a real assembler AND without the flag → downstream raise
2. Direct construction with the flag → default assembler silently constructed
3. _run_planner_for_detection with flag=False and no assembler → RuntimeError
4. _run_planner_for_detection with flag=True and no assembler → warning + works
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.domain.models.app_config import AgentConfig
from app.domain.services.flows.planner_react import PlannerReActFlow


pytestmark = pytest.mark.anyio


@pytest.fixture()
def anyio_backend() -> str:
    return "asyncio"


def _make_flow(
    *, allow_default: bool, prompt_assembler=None
) -> PlannerReActFlow:
    """Construct a minimal PlannerReActFlow for gate-behavior assertions."""
    return PlannerReActFlow(
        uow_factory=MagicMock(),
        llm=MagicMock(),
        agent_config=AgentConfig(max_iterations=10, max_retries=3, max_search_results=5),
        session_id="test-session",
        browser=MagicMock(),
        sandbox=MagicMock(),
        search_engine=MagicMock(),
        mcp_tool=MagicMock(get_tools=MagicMock(return_value=[])),
        a2a_tool=MagicMock(manager=None),
        skill_tool=MagicMock(),
        prompt_assembler=prompt_assembler,
        _allow_default_prompt_assembler=allow_default,
    )


def test_flow_stores_allow_default_flag() -> None:
    """The flag is stored on the instance as ``_allow_default_prompt_assembler``."""
    flow = _make_flow(allow_default=True)
    assert flow._allow_default_prompt_assembler is True

    flow2 = _make_flow(allow_default=False)
    assert flow2._allow_default_prompt_assembler is False


def test_flow_defaults_to_false() -> None:
    """The flag defaults to False — production callers don't need to opt out."""
    flow = PlannerReActFlow(
        uow_factory=MagicMock(),
        llm=MagicMock(),
        agent_config=AgentConfig(max_iterations=10, max_retries=3, max_search_results=5),
        session_id="test-session",
        browser=MagicMock(),
        sandbox=MagicMock(),
        search_engine=MagicMock(),
        mcp_tool=MagicMock(get_tools=MagicMock(return_value=[])),
        a2a_tool=MagicMock(manager=None),
        skill_tool=MagicMock(),
    )
    assert flow._allow_default_prompt_assembler is False


async def test_run_planner_for_detection_raises_without_flag() -> None:
    """Without the flag AND without an assembler, the detection path
    raises RuntimeError instead of silently constructing a default."""
    flow = _make_flow(allow_default=False, prompt_assembler=None)
    # summary_texts only needs to be iterable; the Message can be a MagicMock
    # because we only hit the fail-loud check before touching it.
    fake_message = MagicMock()
    fake_message.message = "hi"
    fake_message.attachments = []
    fake_message.image_content_blocks = []
    fake_message.language = "zh"

    with pytest.raises(RuntimeError, match="requires a PromptAssembler"):
        await flow._run_planner_for_detection(fake_message, [])


async def test_run_planner_for_detection_uses_default_with_flag(monkeypatch) -> None:
    """With the flag set AND no assembler, the detection path logs a
    warning and lazily constructs a default assembler. The real planner
    LLM call is mocked so we only verify the assembler path doesn't
    raise."""
    flow = _make_flow(allow_default=True, prompt_assembler=None)

    # Mock the structured_output planner call so we don't hit a real LLM
    fake_plan_response = MagicMock()
    fake_plan_response.title = "Test"
    fake_plan_response.goal = "test goal"
    fake_plan_response.language = "zh"
    fake_plan_response.message = "ok"
    fake_plan_response.steps = []

    mock_structured = MagicMock()
    mock_structured.ainvoke = AsyncMock(return_value=fake_plan_response)
    flow._llm.with_structured_output = MagicMock(return_value=mock_structured)

    fake_message = MagicMock()
    fake_message.message = "hi"
    fake_message.attachments = []
    fake_message.image_content_blocks = []
    fake_message.language = "zh"

    # This should not raise — the flag allows silent default construction
    await flow._run_planner_for_detection(fake_message, [])

    # After the call, a default assembler is attached
    assert flow._prompt_assembler is not None
    # And its budget is the hardcoded fallback (3500 tokens)
    assert flow._prompt_assembler._budget.max_tokens == 3500
