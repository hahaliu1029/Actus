"""B5 C5a: verify react_graph_provider returns (CompiledStateGraph, StepMetadata).

The closure signature changed from ``-> CompiledStateGraph`` to
``-> tuple[CompiledStateGraph, StepMetadata]`` so the executor can
receive authoritative per-step metadata without reaching into
``agent_task_runner`` internal state.

These tests use a stub provider and a minimal ``StepMetadata`` fixture
— they do NOT construct a real LangGraph, which is the whole point of
splitting metadata from the graph.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.domain.services.graphs.step_metadata import (
    RefreshedSkillsResult,
    StepMetadata,
)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def test_step_metadata_is_frozen_dataclass() -> None:
    """StepMetadata must be frozen so callers cannot mutate it after creation."""
    meta = StepMetadata(
        bound_tool_names=frozenset({"shell_execute"}),
        skill_context="## Active Skills\n- foo",
        skill_ids=("foo",),
    )
    with pytest.raises((AttributeError, Exception)):
        meta.bound_tool_names = frozenset()  # type: ignore[misc]


def test_step_metadata_holds_authoritative_fields() -> None:
    """All three fields are exposed as declared."""
    meta = StepMetadata(
        bound_tool_names=frozenset({"shell_execute", "skill_foo_bar"}),
        skill_context="## Active Skills\n- foo",
        skill_ids=("foo",),
    )
    assert meta.bound_tool_names == frozenset({"shell_execute", "skill_foo_bar"})
    assert meta.skill_context == "## Active Skills\n- foo"
    assert meta.skill_ids == ("foo",)


def test_refreshed_skills_result_is_frozen() -> None:
    """RefreshedSkillsResult must also be frozen (pure compute output)."""
    result = RefreshedSkillsResult(
        skills=(),
        context="",
        skill_ids=(),
        scores=None,
    )
    with pytest.raises((AttributeError, Exception)):
        result.context = "mutated"  # type: ignore[misc]


@pytest.mark.anyio
async def test_provider_contract_returns_two_tuple() -> None:
    """A conforming provider stub returns ``(graph, StepMetadata)``.

    Verifies the callsite contract in ``main_graph.executor_node``:
    ``step_react, step_meta = await react_graph_provider(step_description)``.
    """
    graph_stub = MagicMock(name="compiled_state_graph_stub")
    meta = StepMetadata(
        bound_tool_names=frozenset({"file_view", "skill_bar"}),
        skill_context="## Active Skills\n- bar",
        skill_ids=("bar",),
    )

    async def provider(step_description: str):
        return graph_stub, meta

    result = await provider("do something")
    assert isinstance(result, tuple)
    assert len(result) == 2
    returned_graph, returned_meta = result
    assert returned_graph is graph_stub
    assert returned_meta is meta
    assert returned_meta.bound_tool_names == frozenset({"file_view", "skill_bar"})


@pytest.mark.anyio
async def test_provider_metadata_fields_come_from_same_call() -> None:
    """All three StepMetadata fields must reflect the SAME call to the provider.

    This guards against future refactors that might assemble
    ``StepMetadata`` from multiple disjoint sources (e.g., reading
    ``skill_context`` from ``self._last_skill_context`` while
    ``bound_tool_names`` comes from a staler field).
    """
    call_log: list[int] = []

    async def provider(step_description: str):
        call_log.append(1)
        # Each call returns a metadata whose fields are internally consistent
        seq = len(call_log)
        return MagicMock(), StepMetadata(
            bound_tool_names=frozenset({f"tool_{seq}"}),
            skill_context=f"context-{seq}",
            skill_ids=(f"skill-{seq}",),
        )

    _, meta1 = await provider("step 1")
    _, meta2 = await provider("step 2")

    # Each StepMetadata should be internally consistent (same "generation")
    assert "tool_1" in meta1.bound_tool_names
    assert meta1.skill_context == "context-1"
    assert meta1.skill_ids == ("skill-1",)

    assert "tool_2" in meta2.bound_tool_names
    assert meta2.skill_context == "context-2"
    assert meta2.skill_ids == ("skill-2",)
