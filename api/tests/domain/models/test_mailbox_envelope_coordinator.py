"""C2 PR-3 §6.2 — coordinator envelope extension contract tests.

Covers:
- ResultReadyOutcome.TIMED_OUT + NEEDS_AUTHORIZATION enum values
- SpawnRequestPayload.coordinator_context iff agent_kind == "coordinator_step"
- CoordinatorChildContext + CoordinatorBudgetSnapshot wire schema
- Backward-compat: existing agent_kind="research" still rejects coordinator_context

PR-4 §6.2 additions (TestResultReadyPayloadExtended below):
- ResultReadyPayload.patch_manifest optional (default None)
- ResultReadyPayload.needs_authorization_details optional (default None)
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.domain.models.mailbox_envelope import (
    CoordinatorBudgetSnapshot,
    CoordinatorChildContext,
    ResultReadyOutcome,
    ResultReadyPayload,
    SpawnRequestPayload,
)
from app.domain.models.needs_authorization_details import (
    NeedsAuthorizationDetails,
    ProposedWritePlan,
)
from app.domain.models.patch_manifest import FilePatchEntry, PatchManifest


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


# ---------------------------------------------------------------------------
# PR-4 Task 4.3 — ResultReadyPayload optional fields
# ---------------------------------------------------------------------------

def _mk_patch_manifest() -> PatchManifest:
    return PatchManifest(
        patch_id="r1:wu1:p",
        coordinator_run_id="r1",
        work_unit_id="wu1",
        files=(
            FilePatchEntry(
                path="x", op="add",
                new_digest="b" * 64, content_ref="minio://ref", content_size=10,
            ),
        ),
    )


def _mk_needs_auth(reason: str = "exploration_proposal") -> NeedsAuthorizationDetails:
    return NeedsAuthorizationDetails(
        reason=reason,  # type: ignore[arg-type]
        proposed_write_plan=ProposedWritePlan(
            proposed_paths=(), proposed_tools=frozenset({"file_write"}),
        ),
    )


class TestResultReadyPayloadExtended:
    def test_default_optional_fields_none(self) -> None:
        """Backward-compat: existing producers omitting the new fields still
        succeed and both new optionals default to None."""
        p = ResultReadyPayload(summary="x", outcome=ResultReadyOutcome.SUCCESS)
        assert p.patch_manifest is None
        assert p.needs_authorization_details is None

    def test_with_patch_manifest(self) -> None:
        m = _mk_patch_manifest()
        p = ResultReadyPayload(
            summary="x", outcome=ResultReadyOutcome.SUCCESS, patch_manifest=m,
        )
        assert p.patch_manifest is not None
        assert p.patch_manifest.patch_id == "r1:wu1:p"
        assert p.needs_authorization_details is None

    def test_with_needs_auth_details(self) -> None:
        d = _mk_needs_auth(reason="exploration_proposal")
        p = ResultReadyPayload(
            summary="x",
            outcome=ResultReadyOutcome.NEEDS_AUTHORIZATION,
            needs_authorization_details=d,
        )
        assert p.needs_authorization_details is not None
        assert p.needs_authorization_details.reason == "exploration_proposal"

    def test_outcome_field_matrix_success_with_needs_auth_rejected(self) -> None:
        """[r3 P2#3] SUCCESS + needs_authorization_details → ValidationError
        (prevents authorization-confusion via completion envelopes)."""
        with pytest.raises(ValidationError, match="forbids needs_authorization_details"):
            ResultReadyPayload(
                summary="x", outcome=ResultReadyOutcome.SUCCESS,
                needs_authorization_details=_mk_needs_auth(reason="hard_blocked"),
            )

    def test_outcome_field_matrix_needs_auth_missing_details_rejected(self) -> None:
        """[r3 P2#3] NEEDS_AUTHORIZATION without details → ValidationError
        (reducer routes on the details; missing them strands the parent)."""
        with pytest.raises(ValidationError, match="requires.*needs_authorization_details"):
            ResultReadyPayload(
                summary="x", outcome=ResultReadyOutcome.NEEDS_AUTHORIZATION,
            )

    def test_outcome_field_matrix_failed_with_patch_manifest_rejected(self) -> None:
        """[r3 P2#3] FAILED + patch_manifest → ValidationError (no meaningful
        write set when the runner crashed)."""
        with pytest.raises(ValidationError, match="forbids patch_manifest"):
            ResultReadyPayload(
                summary="x", outcome=ResultReadyOutcome.FAILED,
                patch_manifest=_mk_patch_manifest(),
            )

    def test_outcome_field_matrix_timed_out_with_patch_manifest_rejected(self) -> None:
        """[r3 P2#3] TIMED_OUT + patch_manifest → ValidationError."""
        with pytest.raises(ValidationError, match="forbids patch_manifest"):
            ResultReadyPayload(
                summary="x", outcome=ResultReadyOutcome.TIMED_OUT,
                patch_manifest=_mk_patch_manifest(),
            )

    def test_extra_field_still_forbidden(self) -> None:
        """The new optionals do NOT loosen extra=forbid."""
        with pytest.raises(ValidationError):
            ResultReadyPayload(  # type: ignore[call-arg]
                summary="x", outcome=ResultReadyOutcome.SUCCESS,
                untracked_field="nope",
            )

    def test_frozen_preserved(self) -> None:
        p = ResultReadyPayload(summary="x", outcome=ResultReadyOutcome.SUCCESS)
        with pytest.raises(ValidationError):
            p.summary = "y"  # type: ignore[misc]


class TestResultReadyPayloadWireRoundtrip:
    """MailboxEnvelope._validate_payload_matches_type re-validates via
    model_dump → model_validate; the new optional fields MUST survive."""

    def test_roundtrip_with_patch_manifest(self) -> None:
        original = ResultReadyPayload(
            summary="done", outcome=ResultReadyOutcome.SUCCESS,
            patch_manifest=_mk_patch_manifest(),
        )
        dumped = original.model_dump(mode="python")
        rehydrated = ResultReadyPayload.model_validate(dumped)
        assert rehydrated.patch_manifest is not None
        assert rehydrated.patch_manifest.files[0].op == "add"

    def test_roundtrip_with_needs_auth_and_plan(self) -> None:
        original = ResultReadyPayload(
            summary="ask",
            outcome=ResultReadyOutcome.NEEDS_AUTHORIZATION,
            needs_authorization_details=_mk_needs_auth(reason="exploration_proposal"),
        )
        dumped = original.model_dump(mode="python")
        rehydrated = ResultReadyPayload.model_validate(dumped)
        assert rehydrated.needs_authorization_details is not None
        assert rehydrated.needs_authorization_details.reason == "exploration_proposal"
        assert rehydrated.needs_authorization_details.proposed_write_plan is not None
