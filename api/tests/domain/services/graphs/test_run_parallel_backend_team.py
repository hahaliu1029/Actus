"""C2-full S4 Task 3.3 — team expander wired into ``_run_parallel_backend``.

Spec ref: S4 §10 + R8-3. The team resolve runs INSIDE the existing ``try``
(before ``subgraph.ainvoke``); the ``except`` is widened from
``CoordinatorPathContractError`` alone to also catch ``TeamCapabilityError``
(resolve-time OR merge-time inside ainvoke) and ``TeamArtifactError`` (a
malformed ``TEAM.md`` raised from ``get_by_slug``). All become a graceful
FAILED ``ParallelBackendOutcome`` — never an uncaught abort.

INV-0: when ``team_slug`` is absent OR either flag is OFF, ``team_member_map``
MUST be None and the only payload delta vs. pre-S4 is the new key carrying None.

Project convention: pytest-anyio (NOT pytest-asyncio); see tests/conftest.py.
"""
from __future__ import annotations

from typing import Any

import pytest

from app.domain.services.graphs.main_graph import _run_parallel_backend
from app.domain.services.team_expander import TeamCapabilityError  # noqa: F401

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class _Step:
    id = "step-1"

    class parallel_work_units:
        work_units: list[Any] = []


class _FakeSubgraph:
    def __init__(self) -> None:
        self.invoked_with: dict[str, Any] | None = None

    async def ainvoke(self, payload: dict[str, Any], config: Any = None) -> dict[str, Any]:
        self.invoked_with = payload
        return {"step_result_candidate": "ok", "group_outcome": None}


class _BadTeamRepo:
    async def get_by_slug(self, slug: str) -> None:  # team requested but missing
        return None


class _MalformedTeamRepo:
    async def get_by_slug(self, slug: str) -> Any:  # present-but-malformed TEAM.md
        from app.domain.repositories.agent_team_repository import TeamArtifactError

        raise TeamArtifactError(f"malformed team {slug}")


async def test_malformed_team_artifact_returns_graceful_failed(monkeypatch) -> None:
    # [codex-R3-F1] TeamArtifactError from the repo must be caught → graceful FAILED,
    # NOT an uncaught abort.
    monkeypatch.setenv("ACTUS_C2_COORDINATOR_ENABLED", "true")
    monkeypatch.setenv("ACTUS_C2_AGENT_TEAMS_ENABLED", "true")
    sub = _FakeSubgraph()
    cfg = {
        "configurable": {
            "parallel_execution_subgraph": sub,
            "team_repository": _MalformedTeamRepo(),
            "skill_repository": object(),
            "user_id": "u",
        }
    }
    state = {"session_id": "s", "user_id": "u", "team_slug": "broken"}
    outcome = await _run_parallel_backend(state, cfg, _Step())
    assert outcome.success is False
    assert sub.invoked_with is None


async def test_unresolvable_team_returns_graceful_failed(monkeypatch) -> None:
    monkeypatch.setenv("ACTUS_C2_COORDINATOR_ENABLED", "true")
    monkeypatch.setenv("ACTUS_C2_AGENT_TEAMS_ENABLED", "true")
    sub = _FakeSubgraph()
    cfg = {
        "configurable": {
            "parallel_execution_subgraph": sub,
            "team_repository": _BadTeamRepo(),
            "skill_repository": object(),
            "user_id": "u",
        }
    }
    state = {"session_id": "s", "user_id": "u", "team_slug": "ghost"}
    outcome = await _run_parallel_backend(state, cfg, _Step())
    assert outcome.success is False  # graceful FAILED, NOT an uncaught raise
    assert sub.invoked_with is None  # never reached ainvoke


async def test_no_team_slug_passes_none_map(monkeypatch) -> None:
    monkeypatch.setenv("ACTUS_C2_COORDINATOR_ENABLED", "true")
    monkeypatch.setenv("ACTUS_C2_AGENT_TEAMS_ENABLED", "true")
    sub = _FakeSubgraph()
    cfg = {"configurable": {"parallel_execution_subgraph": sub, "user_id": "u"}}
    state = {"session_id": "s", "user_id": "u"}  # no team_slug → INV-0
    await _run_parallel_backend(state, cfg, _Step())
    assert sub.invoked_with is not None
    assert sub.invoked_with["team_member_map"] is None
