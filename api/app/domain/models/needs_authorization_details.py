"""C2 v1 NeedsAuthorizationDetails + ProposedWritePlan (spec §6.2 + r14 P1-2).

The coordinator child publishes a ``NeedsAuthorizationDetails`` (wrapped in a
RESULT_READY(NEEDS_AUTHORIZATION) envelope) whenever it stops short of producing
a PatchManifest and needs an out-of-band decision before progress can resume:

- ChildScopeGate violation (tool not allowlisted, path out of lease, op mismatch,
  hard-blocked, lease_expired, revision_drift)
- Budget watchdog tripped (token / wallclock — PR-6 wires evidence)
- Exploration-phase ``proposed_write_plan`` for a write phase that hasn't been
  approved (r13 phase-aware finalizer)

The reducer + orchestrator (PR-5 + PR-6) inspect ``reason`` to decide:
escalate to human via PE, retry with widened lease, or fail the group.

Free-text fields (rationale) MUST NOT be inlined into the
``coordinator_result_envelope_store`` (which keeps only the minimal structured
fields per spec §11) — they are stored as MinIO refs (``rationale_ref``) and
fetched lazily by the UI.
"""
from __future__ import annotations

import re
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.domain.models.work_unit import ProposedPath


_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")


class ProposedWritePlan(BaseModel):
    """[r14 P1-2] Exploration-phase child output — planner-facing write plan.

    Emitted by the coordinator child when it finishes the EXPLORATION phase
    naturally (i.e. without scope/budget violation) and the orchestrator needs
    a structured proposal to seed the WRITE phase.

    ``rationale_ref`` MUST be a MinIO ref (not inlined text) — free-text
    rationale stays out of the canonical envelope store per spec §11.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    proposed_paths: tuple[ProposedPath, ...]
    proposed_tools: frozenset[str]
    rationale_ref: Optional[str] = None
    confidence: Literal["high", "medium", "low"] = "medium"
    artifact_count: int = 0


class NeedsAuthorizationDetails(BaseModel):
    """Structured grievance from a coordinator child that stopped before
    producing a PatchManifest.

    ``reason`` is a closed Literal set (now including the §3.4 snapshot-diff
    group zero-apply reject codes); any expansion is an explicit wire-schema
    change tracked by ``TestNeedsAuthorizationDetailsInvariants
    .test_all_reason_values_accepted``.

    ``known_digests`` carries SHA-256 hashes the child observed for paths
    relevant to its grievance — the reducer compares against the live parent
    digests to detect revision drift vs. true conflict (PR-5).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    reason: Literal[
        "out_of_tool_allowlist",
        "out_of_path_lease",
        "op_mismatch",
        "hard_blocked",
        "budget_exhausted",
        "lease_expired",
        "revision_drift",
        "exploration_proposal",
        # [C2-full S2 §3.4] snapshot-diff group zero-apply reject codes — an
        # EXPLICIT wire-schema expansion of the closed Literal (the docstring
        # contract). Mirrored in the runner's _SNAPSHOT_REJECT_REASONS (Task 4.3)
        # and pinned by test_all_reason_values_accepted.
        "out_of_tree_lease",
        "special_file",
        "symlink",
        "mode_only_change",
        "indeterminate_kind",
        "scan_truncated",
        "tree_add_target_exists",
        "parent_not_regular",
    ]
    requested_tool: Optional[str] = None
    requested_paths: tuple[str, ...] = ()
    observed_evidence: Optional[str] = None
    # [C2-full S2 §3.4] bounded first-N offending paths for a group zero-apply,
    # so a snapshot reject is diagnosable on the envelope without an apply-audit
    # row. Empty for non-snapshot grievances. Bounded by the populating site
    # (differ/finalizer pass at most N — see Task 4.4/4.6 _SNAPSHOT_SUMMARY_CAP).
    rejection_summary: tuple[str, ...] = ()
    proposed_write_plan: Optional[ProposedWritePlan] = None
    known_digests: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _known_digests_are_sha256(self) -> "NeedsAuthorizationDetails":
        """[r7 P2#1] Each value in ``known_digests`` must be a SHA-256 hex
        digest (64 lowercase hex chars). The field is documented as carrying
        "SHA-256 hashes the child observed" — accepting free-form strings
        would let a malicious child smuggle phantom digests through the
        reducer's lease-vs-known compare.

        ``frozen=True`` on the outer model does NOT prevent post-construction
        mutation of the inner dict — Python dict is mutable. The validator
        guards the construction-time contract; deep-frozen semantics would
        require switching the field type to ``tuple[tuple[str, str], ...]``
        which sacrifices ergonomics. Pragmatic: validate, document, trust
        same-codebase consumers."""
        for path, digest in self.known_digests.items():
            if not _SHA256_HEX_RE.match(digest):
                raise ValueError(
                    f"known_digests[{path!r}] must be SHA-256 hex, got {digest!r}"
                )
        return self
