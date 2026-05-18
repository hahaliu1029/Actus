# api/tests/invariants/test_inv5_behavior_pe_call_order.py
"""INV-5 behavior: with a fake PE, every native tool call MUST trigger
pe.evaluate (path A) or consume pe_resume_outcomes (path B) before
wrapper execution.

This is the runtime complement to the static AST gate above.

NOTE: The plan referenced a non-existent `graph_runner_with_fake_pe` fixture.
This test instead directly invokes tool_node (via _build_tool_node_fn) using
the pattern established in test_react_graph_pe_dispatch.py (Phase 9 tests).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.domain.models.tool_result import AllowSuccess, DecisionReason

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class _RecordingPE:
    """PE that records evaluate calls and returns AllowSuccess."""

    def __init__(self):
        self._evaluate_count = 0

    @property
    def evaluate_count(self) -> int:
        return self._evaluate_count

    async def evaluate(self, call, ctx):
        self._evaluate_count += 1
        return AllowSuccess(content="ok", data={})

    async def preflight_resume(self, *a, **kw):
        return None

    async def commit_resume(self, *a, **kw):
        return AllowSuccess(content="ok", data={})


def _make_fake_ssm():
    from app.domain.models.session import SessionStatus
    ssm = AsyncMock()
    ssm.get_mode_with_revision = AsyncMock(return_value=(SessionStatus.RUNNING, 1))
    return ssm


def _make_state(tool_name: str, tool_args: dict, call_id: str = "tc1") -> dict:
    from langchain_core.messages import AIMessage
    return {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[{"id": call_id, "name": tool_name, "args": tool_args, "type": "tool_call"}],
            )
        ],
        "llm_input_messages": [],
        "step_description": "test",
        "original_request": "test",
        "language": "en",
        "attachments": [],
        "image_content_blocks": [],
        "events": [],
        "should_interrupt": False,
        "soft_hint_sent": False,
        "attempt_count": 0,
        "failure_count": 0,
        "completed_tool_call_prefix": [],
        "approved_tool_call_ids": [],
        "pending_ask_outcome": None,
        "pending_ask_tool_call_id": None,
        "pending_ask_artifact": None,
        "pending_ask_tool_args": None,
        "pe_resume_outcomes": None,
    }


def _make_config(fake_pe, fake_ssm, *, user_id: str = "u", session_id: str = "s") -> dict:
    """PE-1 §2.5: include ``tool_confirmation_config`` so the per-call
    ``is_pe_enabled_for_source`` gate inside ``_pe_dispatch`` passes for
    source=native."""
    from types import SimpleNamespace

    tc_cfg = SimpleNamespace(
        enabled=True,
        permission_engine_native_enabled=True,
        permission_engine_skill_enabled=True,
    )
    return {
        "configurable": {
            "permission_engine": fake_pe,
            "session_state_machine": fake_ssm,
            "permission_engine_native_enabled": True,
            "tool_confirmation_config": tc_cfg,
            "user_id": user_id,
            "session_id": session_id,
            "thread_id": session_id,
        }
    }


def _build_tool_node_fn():
    """Build a minimal react_graph and return the tool_node closure."""
    from langchain_core.tools import tool as lc_tool
    from app.domain.services.graphs.react_graph import build_react_graph

    @lc_tool
    async def file_write(path: str, content: str = "") -> str:
        """Write to a file."""
        return f"wrote {path}"

    stub_llm = AsyncMock()
    from langchain_core.messages import AIMessage as _AIMessage
    stub_llm.ainvoke = AsyncMock(
        return_value=_AIMessage(content='{"success":true,"result":"done","attachments":[]}')
    )
    stub_llm.bind_tools = MagicMock(return_value=stub_llm)

    graph = build_react_graph(stub_llm, [file_write])
    tool_node_fn = graph.nodes["tool_node"].bound.afunc
    return tool_node_fn


async def test_native_tool_invocation_calls_pe_evaluate_first():
    """PE.evaluate must be called before the tool wrapper executes.

    Verifies INV-5 path A: pe.evaluate is called at least once when a
    native tool (file_write) is dispatched through _pe_dispatch.
    """
    tool_node_fn = _build_tool_node_fn()
    fake_pe = _RecordingPE()
    fake_ssm = _make_fake_ssm()

    state = _make_state("file_write", {"path": "/x", "content": "hello"})
    config = _make_config(fake_pe, fake_ssm)

    await tool_node_fn(state, config)

    assert fake_pe.evaluate_count >= 1, (
        "INV-5 behavior: pe.evaluate was not called before tool wrapper execution. "
        f"evaluate_count={fake_pe.evaluate_count}"
    )


async def test_pe_evaluate_count_matches_tool_calls():
    """PE.evaluate is called exactly once per native tool call in the batch."""
    tool_node_fn = _build_tool_node_fn()
    fake_pe = _RecordingPE()
    fake_ssm = _make_fake_ssm()

    # Single tool call → exactly one evaluate
    state = _make_state("file_write", {"path": "/y"}, call_id="tc42")
    config = _make_config(fake_pe, fake_ssm, session_id="sess-inv5")

    await tool_node_fn(state, config)

    assert fake_pe.evaluate_count == 1, (
        f"Expected exactly 1 pe.evaluate call, got {fake_pe.evaluate_count}"
    )


# ---------- PE-1 INV-5 extension: SkillSource route ----------


class _FakeRecordingSource:
    """Records assess_risk invocations for a single source.

    Mirrors the ``PermissionSource`` ABC contract just enough for the
    fake-PE harness: it owns a ``tool_source`` ClassVar and an async
    ``assess_risk(call)`` that returns a stub ``RiskAssessment``.
    """

    def __init__(self, tool_source: str):
        type(self).tool_source = tool_source  # set ClassVar dynamically for fake
        self.tool_source = tool_source
        self.assess_risk_calls = 0

    async def assess_risk(self, call):
        self.assess_risk_calls += 1
        from app.domain.services.risk_assessor import RiskAssessment, RiskLevel
        return RiskAssessment(
            tool_name=call.tool_name,
            tool_args=dict(call.tool_args),
            static_level=RiskLevel.LOW,
            dynamic_level=RiskLevel.NONE,
            final_level=RiskLevel.LOW,
            risk_reason="fake",
            matched_patterns=[],
            suggested_alternative=None,
            primary_arg="",
            dir_arg=None,
            arg_digest="d",
        )


class _RecordingPEWithSources:
    """PE that owns a ``sources`` dict (spec §2.4) and records every
    evaluate call's source-routing decision.

    On each ``evaluate`` it looks up ``call.tool_source`` in ``_sources``
    and invokes the matching ``PermissionSource.assess_risk`` so the
    test can assert the correct source was hit exactly once.
    """

    def __init__(self):
        self._sources = {
            "native": _FakeRecordingSource("native"),
            "skill": _FakeRecordingSource("skill"),
        }
        self.evaluate_calls: list = []

    async def evaluate(self, call, ctx):
        self.evaluate_calls.append(call)
        source = self._sources.get(call.tool_source)
        if source is not None:
            await source.assess_risk(call)
        return AllowSuccess(content="ok", data={"via": "test"})

    async def preflight_resume(self, *a, **kw):
        return None

    async def commit_resume(self, *a, **kw):
        return AllowSuccess(content="ok", data={})


class _FakeSkillTool:
    """Minimal SkillTool fake exposing only what ``_pe_dispatch`` and
    ``build_skill_call_metadata`` actually touch:
      - ``has_tool(name)``
      - ``_tool_bindings[name]`` with ``skill / runtime_type / final_risk /
        trust_origin / scan_verdict``
    """

    def __init__(self, tool_name: str):
        from app.domain.models.skill import (
            Skill,
            SkillRuntimeType,
            SkillSourceType,
        )

        self._skill = Skill(
            id="skill-inv5-test",
            slug="inv5-test",
            name="INV5 test skill",
            description="fake skill for INV-5 SkillSource routing test",
            source_type=SkillSourceType.LOCAL,
            source_ref="local:/tmp/inv5-test",
            runtime_type=SkillRuntimeType.NATIVE,
            manifest={"runtime_type": "native", "tools": []},
            enabled=True,
            trust_origin="user_installed",
            scan_report={"verdict": "safe", "content_hash": "deadbeef"},
        )
        self._tool_bindings: dict[str, dict] = {
            tool_name: {
                "skill": self._skill,
                "runtime_type": SkillRuntimeType.NATIVE,
                "manifest_tool": {"name": tool_name},
                "final_risk": "low",
                "trust_origin": "user_installed",
                "scan_verdict": "safe",
            }
        }

    def has_tool(self, tool_name: str) -> bool:
        return tool_name in self._tool_bindings


def _build_skill_tool_node_fn(skill_tool_name: str):
    """Build a react_graph that includes a fake skill-flavored tool
    registered with the ``skill`` ToolSource (via the ``skill_`` prefix
    heuristic + ``annotate_and_register_tool_source`` to write
    ``tool.metadata['_actus_source']``).
    """
    from langchain_core.messages import AIMessage as _AIMessage
    from langchain_core.tools import StructuredTool
    from pydantic import Field, create_model

    from app.domain.services.graphs.react_graph import build_react_graph
    from app.domain.services.tools.tool_source_resolver import (
        annotate_and_register_tool_source,
    )

    args_model = create_model(
        f"{skill_tool_name}_args",
        path=(str, Field(description="path")),
    )

    async def _invoke(**kwargs):
        from app.domain.models.tool_result import AllowSuccess as _AS
        outcome = _AS(content=f"skill ran on {kwargs.get('path')}", data={})
        return outcome.content, outcome

    lc_tool = StructuredTool.from_function(
        coroutine=_invoke,
        name=skill_tool_name,
        description="fake skill tool for INV-5 routing test",
        args_schema=args_model,
        response_format="content_and_artifact",
    )
    annotate_and_register_tool_source(lc_tool, source="skill", category="skill")
    # Match what langchain_dynamic_skill_tools writes onto tool.metadata
    # so any callers that introspect it (e.g. risk_level snapshot) see
    # the canonical shape — build_skill_call_metadata still reads from
    # _tool_bindings as the source of truth.
    lc_tool.metadata = {
        **(lc_tool.metadata or {}),
        "risk_level": "low",
        "runtime_type": "native",
        "trust_origin": "user_installed",
        "scan_verdict": "safe",
    }

    stub_llm = AsyncMock()
    stub_llm.ainvoke = AsyncMock(
        return_value=_AIMessage(
            content='{"success":true,"result":"done","attachments":[]}'
        )
    )
    stub_llm.bind_tools = MagicMock(return_value=stub_llm)

    graph = build_react_graph(stub_llm, [lc_tool])
    tool_node_fn = graph.nodes["tool_node"].bound.afunc
    return tool_node_fn


def _make_skill_config(
    fake_pe,
    fake_ssm,
    fake_skill_tool,
    *,
    user_id: str = "u",
    session_id: str = "s",
) -> dict:
    """Same as ``_make_config`` but with the ``skill`` source flag on
    and ``skill_tool`` wired into configurable so ``_pe_dispatch`` can
    build SkillCallMetadata via ``build_skill_call_metadata``.
    """
    from types import SimpleNamespace

    tc_cfg = SimpleNamespace(
        enabled=True,
        permission_engine_native_enabled=True,
        permission_engine_skill_enabled=True,
    )
    return {
        "configurable": {
            "permission_engine": fake_pe,
            "session_state_machine": fake_ssm,
            "permission_engine_native_enabled": True,
            "permission_engine_skill_enabled": True,
            "tool_confirmation_config": tc_cfg,
            "skill_tool": fake_skill_tool,
            "user_id": user_id,
            "session_id": session_id,
            "thread_id": session_id,
        }
    }


async def test_skill_tool_call_routes_through_skill_source():
    """PE-1 INV-5 extension: when a skill tool is invoked through the
    react_graph PE path, ``SkillSource.assess_risk`` MUST be called
    exactly once per tool call AND the native source must NOT be touched.

    Wires the ``_pe_dispatch`` harness with:
      - a ``StructuredTool`` whose name starts with ``skill_`` so the
        tool-source resolver routes it to ``ToolSource(source="skill")``;
      - a ``_FakeSkillTool`` exposing ``has_tool`` + ``_tool_bindings``
        so ``build_skill_call_metadata`` succeeds;
      - a ``_RecordingPEWithSources`` whose ``evaluate`` dispatches to
        the matching ``_sources[tool_source].assess_risk``.

    The assertion proves PE-1 §3.2 wiring: skill calls land on SkillSource,
    not NativeSource. (The native counterpart is covered by the two
    pre-existing tests above.)
    """
    skill_tool_name = "skill_test_inv5_foo"
    tool_node_fn = _build_skill_tool_node_fn(skill_tool_name)
    fake_pe = _RecordingPEWithSources()
    fake_ssm = _make_fake_ssm()
    fake_skill_tool = _FakeSkillTool(skill_tool_name)

    state = _make_state(skill_tool_name, {"path": "/skill_inv5"}, call_id="tc-skill-1")
    config = _make_skill_config(
        fake_pe, fake_ssm, fake_skill_tool, session_id="sess-inv5-skill"
    )

    await tool_node_fn(state, config)

    assert len(fake_pe.evaluate_calls) == 1, (
        f"Expected exactly 1 pe.evaluate call for skill tool, "
        f"got {len(fake_pe.evaluate_calls)}"
    )
    assert fake_pe.evaluate_calls[0].tool_source == "skill", (
        "ToolCallSpec.tool_source should be 'skill' for a skill_*-prefixed tool"
    )
    skill_source = fake_pe._sources["skill"]
    native_source = fake_pe._sources["native"]
    assert skill_source.assess_risk_calls == 1, (
        f"INV-5 SkillSource route: skill source should be evaluated exactly "
        f"once, got {skill_source.assess_risk_calls}"
    )
    assert native_source.assess_risk_calls == 0, (
        f"Native source must not be touched for a skill tool call, "
        f"got {native_source.assess_risk_calls}"
    )
