"""C2 PR-3 §6.2 — coordinator envelope extension contract tests.

Covers:
- ResultReadyOutcome.TIMED_OUT + NEEDS_AUTHORIZATION enum values
- SpawnRequestPayload.coordinator_context iff agent_kind == "coordinator_step"
- CoordinatorChildContext + CoordinatorBudgetSnapshot wire schema
- Backward-compat: existing agent_kind="research" still rejects coordinator_context
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.domain.models.mailbox_envelope import (
    CoordinatorBudgetSnapshot,
    CoordinatorChildContext,
    ResultReadyOutcome,
    SpawnRequestPayload,
)


class TestResultReadyOutcomeExtended:
    def test_new_values_present(self) -> None:
        assert ResultReadyOutcome.TIMED_OUT.value == "timed_out"
        assert ResultReadyOutcome.NEEDS_AUTHORIZATION.value == "needs_authorization"

    def test_old_values_intact(self) -> None:
        assert ResultReadyOutcome.SUCCESS.value == "success"
        assert ResultReadyOutcome.FAILED.value == "failed"
        assert ResultReadyOutcome.CANCELLED.value == "cancelled"

    def test_value_set_is_exactly_five(self) -> None:
        assert {m.value for m in ResultReadyOutcome} == {
            "success", "failed", "cancelled", "timed_out", "needs_authorization",
        }


def _mk_budget() -> CoordinatorBudgetSnapshot:
    return CoordinatorBudgetSnapshot(
        max_tool_calls=10, max_token_cost_usd=0.5, max_wallclock_seconds=300,
    )


def _mk_ctx() -> CoordinatorChildContext:
    return CoordinatorChildContext(
        coordinator_run_id="p1:abcd1234abcd1234:a1",
        work_unit_id="abcd1234abcd1234.a1.0",
        parent_session_id="p1",
        spawn_manifest_ref="minio://m",
        spawn_manifest_sha256="abc",
        session_mode_revision=1,
        budget=_mk_budget(),
    )


class TestSpawnRequestPayloadCoordinator:
    def test_research_default_no_coordinator_context(self) -> None:
        p = SpawnRequestPayload(agent_kind="research", task_prompt="x")
        assert p.coordinator_context is None

    def test_default_agent_kind_remains_research(self) -> None:
        p = SpawnRequestPayload(task_prompt="x")
        assert p.agent_kind == "research"

    def test_general_no_coordinator_context(self) -> None:
        p = SpawnRequestPayload(agent_kind="general", task_prompt="x")
        assert p.coordinator_context is None

    def test_coordinator_step_requires_coordinator_context(self) -> None:
        with pytest.raises(ValidationError) as ei:
            SpawnRequestPayload(agent_kind="coordinator_step", task_prompt="x")
        assert "coordinator_context required" in str(ei.value)

    def test_research_with_coordinator_context_rejected(self) -> None:
        with pytest.raises(ValidationError) as ei:
            SpawnRequestPayload(
                agent_kind="research", task_prompt="x",
                coordinator_context=_mk_ctx(),
            )
        assert "coordinator_context only valid" in str(ei.value)

    def test_general_with_coordinator_context_rejected(self) -> None:
        with pytest.raises(ValidationError) as ei:
            SpawnRequestPayload(
                agent_kind="general", task_prompt="x",
                coordinator_context=_mk_ctx(),
            )
        assert "coordinator_context only valid" in str(ei.value)

    def test_default_agent_kind_with_coordinator_context_rejected(self) -> None:
        """[r7 P2 fix] When agent_kind defaults (omitted, falls to 'research')
        AND coordinator_context is supplied, the iff validator must still reject.
        Otherwise a caller omitting agent_kind could smuggle a coordinator_context
        into a research-mode envelope."""
        with pytest.raises(ValidationError) as ei:
            SpawnRequestPayload(
                task_prompt="x",
                coordinator_context=_mk_ctx(),
            )
        assert "coordinator_context only valid" in str(ei.value)

    def test_coordinator_step_complete(self) -> None:
        p = SpawnRequestPayload(
            agent_kind="coordinator_step", task_prompt="do x",
            coordinator_context=_mk_ctx(),
        )
        assert p.coordinator_context is not None
        assert p.coordinator_context.coordinator_run_id == "p1:abcd1234abcd1234:a1"
        assert p.coordinator_context.budget.max_tool_calls == 10
        assert p.coordinator_context.session_mode_revision == 1

    def test_unknown_agent_kind_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SpawnRequestPayload(agent_kind="planner", task_prompt="x")


class TestCoordinatorBudgetSnapshotFrozen:
    def test_required_fields(self) -> None:
        with pytest.raises(ValidationError):
            CoordinatorBudgetSnapshot()

    def test_extra_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            CoordinatorBudgetSnapshot(
                max_tool_calls=1, max_token_cost_usd=0.1,
                max_wallclock_seconds=10, extra_field="nope",
            )

    def test_frozen(self) -> None:
        b = _mk_budget()
        with pytest.raises(ValidationError):
            b.max_tool_calls = 999


class TestCoordinatorChildContextFrozen:
    def test_extra_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            CoordinatorChildContext(
                coordinator_run_id="r", work_unit_id="w",
                parent_session_id="p", spawn_manifest_ref="m",
                spawn_manifest_sha256="s", session_mode_revision=0,
                budget=_mk_budget(),
                unexpected="nope",
            )

    def test_frozen(self) -> None:
        c = _mk_ctx()
        with pytest.raises(ValidationError):
            c.coordinator_run_id = "other"


class TestSpawnRequestPayloadWireRoundtrip:
    """coordinator_context survives model_dump + model_validate cycle.

    MailboxEnvelope._validate_payload_matches_type re-validates via model_dump → model_validate.
    """
    def test_dump_load_preserves_coordinator_context(self) -> None:
        p = SpawnRequestPayload(
            agent_kind="coordinator_step", task_prompt="do x",
            coordinator_context=_mk_ctx(),
        )
        dumped = p.model_dump(mode="python")
        assert dumped["agent_kind"] == "coordinator_step"
        assert dumped["coordinator_context"]["coordinator_run_id"] == "p1:abcd1234abcd1234:a1"
        p2 = SpawnRequestPayload.model_validate(dumped)
        assert p2.coordinator_context is not None
        assert p2.coordinator_context.budget.max_tool_calls == 10
