"""[PR-9b-C C6] Priced fake-LLM fixtures carry usage_metadata + priced identity.

INV-C2: each coordinator worker LLM must surface ``usage_metadata`` AND a
priced identity (``model='gpt-4o-mini'`` / ``provider_id='openai_official'``)
so the cost rollup records non-zero, correctly-attributed cost — NOT the
heuristic ``'openai'`` that misses the pricing table.

The assertion BODIES touch no infra (they only construct the fake LLMs and
``ainvoke`` them).  BUT this file lives under ``tests/integration/`` per the
plan, and that package's ``conftest.py`` has a module-scoped autouse
``_migrate`` fixture that connects to Postgres — so running it via pytest
locally requires a reachable DB (CI-validated; or run against a live test DB).
The pure logic can be exercised without pytest/DB by importing
``tests.integration.coordinator_fixtures`` directly (see the C6 manual check
in the PR-9b-C task report).
"""
import pytest

from langchain_core.messages import HumanMessage

pytestmark = [pytest.mark.integration, pytest.mark.anyio, pytest.mark.coordinator_recovery]


async def test_workers_emit_usage_metadata_and_priced_identity(
    fixture_mock_llm_3_workers,
):
    """All 3 workers: priced identity + usage_metadata on the emitted message."""
    assert len(fixture_mock_llm_3_workers) == 3
    for worker in fixture_mock_llm_3_workers:
        ident = worker._identifying_params
        assert ident["model"] == "gpt-4o-mini", ident
        assert ident["provider_id"] == "openai_official", ident

        out = await worker.ainvoke([HumanMessage(content="go")])
        um = out.usage_metadata
        assert um is not None, "worker AIMessage must carry usage_metadata"
        assert um["input_tokens"] > 0
        assert um["output_tokens"] > 0
        assert um["total_tokens"] == um["input_tokens"] + um["output_tokens"]


async def test_planner_flavor_priced_and_parallel(
    fixture_mock_llm_parallel_planner,
):
    """Planner flavor: same priced identity + usage_metadata, emits parallel plan JSON."""
    import json

    ident = fixture_mock_llm_parallel_planner._identifying_params
    assert ident["model"] == "gpt-4o-mini"
    assert ident["provider_id"] == "openai_official"

    out = await fixture_mock_llm_parallel_planner.ainvoke([HumanMessage(content="plan")])
    assert out.usage_metadata is not None
    assert out.usage_metadata["total_tokens"] > 0

    plan = json.loads(out.content)
    units = plan["steps"][0]["parallel_work_units"]["work_units"]
    assert len(units) == 3
    assert all(u["phase"] == "write" for u in units)


async def test_planner_with_structured_output_returns_plan(
    fixture_mock_llm_parallel_planner,
):
    """FIX 1: ``with_structured_output(PlanResponse)`` must NOT raise / fall back.

    The real planner calls ``.with_structured_output(PlanResponse).ainvoke(...)``
    in a ``try/except`` that silently degrades to a single-step plan with no
    parallel_work_units.  Stock ``FakeListChatModel`` raises there, so the
    fixture must return a runnable emitting a real ``PlanResponse`` with 3 write
    units — exercised via the async ``ainvoke`` path both call sites use.
    """
    from app.domain.models.llm_responses import PlanResponse

    structured = fixture_mock_llm_parallel_planner.with_structured_output(PlanResponse)
    parsed = await structured.ainvoke([HumanMessage(content="plan")])

    assert isinstance(parsed, PlanResponse)
    assert len(parsed.steps) == 1
    assert parsed.steps[0].parallel_work_units is not None
    units = parsed.steps[0].parallel_work_units.work_units
    assert len(units) == 3
    assert all(u.phase == "write" for u in units)
