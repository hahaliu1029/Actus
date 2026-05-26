"""C2 PR-4 Task 4.2 — NeedsAuthorizationDetails + ProposedWritePlan tests.

Spec ref: §6.2 新 schema 2 + r14 P1-2 ProposedWritePlan for exploration phase.

NeedsAuthorizationDetails is a structured grievance from the coordinator
child to the parent reducer / orchestrator explaining why the child stopped
before producing a PatchManifest. The reducer uses ``reason`` to decide
whether to escalate to the human (via PE / SSE), retry with widened lease,
or fail the group.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.domain.models.needs_authorization_details import (
    NeedsAuthorizationDetails,
    ProposedWritePlan,
)
from app.domain.models.work_unit import ProposedPath


class TestProposedWritePlanMinimal:
    def test_minimal_default_confidence(self) -> None:
        p = ProposedWritePlan(
            proposed_paths=(ProposedPath(path="/x", op="add"),),
            proposed_tools=frozenset({"file_write"}),
        )
        assert p.confidence == "medium"
        assert p.artifact_count == 0
        assert p.rationale_ref is None

    def test_empty_paths_allowed(self) -> None:
        """Exploration phase may emit a proposal with zero paths (e.g. abort
        recommendation). Schema must accept empty tuple/frozenset."""
        p = ProposedWritePlan(proposed_paths=(), proposed_tools=frozenset())
        assert p.proposed_paths == ()


class TestProposedWritePlanFullFields:
    def test_with_all_fields(self) -> None:
        p = ProposedWritePlan(
            proposed_paths=(
                ProposedPath(path="/a", op="add"),
                ProposedPath(path="/b", op="modify"),
            ),
            proposed_tools=frozenset({"file_write", "file_str_replace"}),
            rationale_ref="minio://r/rationale.md",
            confidence="high",
            artifact_count=5,
        )
        assert p.rationale_ref == "minio://r/rationale.md"
        assert p.confidence == "high"
        assert p.artifact_count == 5
        assert "file_str_replace" in p.proposed_tools

    def test_low_confidence_allowed(self) -> None:
        p = ProposedWritePlan(
            proposed_paths=(), proposed_tools=frozenset(),
            confidence="low",
        )
        assert p.confidence == "low"


class TestProposedWritePlanInvariants:
    def test_invalid_confidence(self) -> None:
        with pytest.raises(ValidationError):
            ProposedWritePlan(
                proposed_paths=(), proposed_tools=frozenset(),
                confidence="absolute",  # type: ignore[arg-type]
            )

    def test_extra_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            ProposedWritePlan(  # type: ignore[call-arg]
                proposed_paths=(), proposed_tools=frozenset(),
                surprise="nope",
            )

    def test_frozen(self) -> None:
        p = ProposedWritePlan(proposed_paths=(), proposed_tools=frozenset())
        with pytest.raises(ValidationError):
            p.confidence = "high"  # type: ignore[misc]


class TestNeedsAuthorizationDetailsOutOfLease:
    def test_out_of_path_lease_no_proposal(self) -> None:
        d = NeedsAuthorizationDetails(
            reason="out_of_path_lease",
            requested_tool="file_write",
            requested_paths=("/forbidden",),
        )
        assert d.proposed_write_plan is None
        assert d.requested_paths == ("/forbidden",)


class TestNeedsAuthorizationDetailsExplorationProposal:
    def test_with_plan(self) -> None:
        d = NeedsAuthorizationDetails(
            reason="exploration_proposal",
            proposed_write_plan=ProposedWritePlan(
                proposed_paths=(),
                proposed_tools=frozenset({"file_write"}),
            ),
        )
        assert d.proposed_write_plan is not None
        assert "file_write" in d.proposed_write_plan.proposed_tools

    def test_without_plan_is_legal_but_unusual(self) -> None:
        """exploration_proposal WITHOUT a write plan is legal at the schema
        level (no validator enforces it) — the producer is responsible for
        building one. Pin the behavior so a future tightening is explicit."""
        d = NeedsAuthorizationDetails(reason="exploration_proposal")
        assert d.proposed_write_plan is None


class TestNeedsAuthorizationDetailsKnownDigests:
    def test_known_digests_default_empty(self) -> None:
        d = NeedsAuthorizationDetails(reason="hard_blocked")
        assert d.known_digests == {}

    def test_known_digests_carried(self) -> None:
        import hashlib
        sha = hashlib.sha256(b"known").hexdigest()
        d = NeedsAuthorizationDetails(
            reason="op_mismatch",
            requested_paths=("/x",),
            known_digests={"/x": sha},
        )
        assert d.known_digests == {"/x": sha}

    def test_known_digests_rejects_non_sha256(self) -> None:
        """[r7 P2#1] free-form strings are not SHA-256 — wire schema rejects."""
        with pytest.raises(ValidationError, match="SHA-256"):
            NeedsAuthorizationDetails(
                reason="op_mismatch", known_digests={"/x": "not-a-digest"},
            )


class TestNeedsAuthorizationDetailsInvariants:
    def test_invalid_reason_rejected(self) -> None:
        with pytest.raises(ValidationError):
            NeedsAuthorizationDetails(reason="not_a_real_reason")  # type: ignore[arg-type]

    def test_extra_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            NeedsAuthorizationDetails(  # type: ignore[call-arg]
                reason="hard_blocked",
                undeclared="x",
            )

    def test_frozen(self) -> None:
        d = NeedsAuthorizationDetails(reason="hard_blocked")
        with pytest.raises(ValidationError):
            d.reason = "budget_exhausted"  # type: ignore[misc]

    def test_all_reason_values_accepted(self) -> None:
        """Pin the closed reason set. Any add/remove must update this test."""
        for r in (
            "out_of_tool_allowlist", "out_of_path_lease", "op_mismatch",
            "hard_blocked", "budget_exhausted", "lease_expired", "revision_drift",
            "exploration_proposal",
        ):
            NeedsAuthorizationDetails(reason=r)  # type: ignore[arg-type]


class TestWireRoundtrip:
    """model_dump → model_validate preserves all fields including the nested
    ProposedWritePlan + frozenset coercion."""
    def test_roundtrip_with_plan(self) -> None:
        import hashlib
        sha_a = hashlib.sha256(b"a").hexdigest()
        original = NeedsAuthorizationDetails(
            reason="exploration_proposal",
            proposed_write_plan=ProposedWritePlan(
                proposed_paths=(ProposedPath(path="/a", op="add"),),
                proposed_tools=frozenset({"file_write"}),
                rationale_ref="minio://r",
                confidence="high",
                artifact_count=3,
            ),
            known_digests={"/a": sha_a},
        )
        dumped = original.model_dump(mode="python")
        rehydrated = NeedsAuthorizationDetails.model_validate(dumped)
        assert rehydrated.reason == "exploration_proposal"
        assert rehydrated.proposed_write_plan is not None
        assert rehydrated.proposed_write_plan.confidence == "high"
        assert rehydrated.proposed_write_plan.artifact_count == 3
        assert rehydrated.known_digests == {"/a": sha_a}
