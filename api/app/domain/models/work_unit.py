"""C2 v1 work unit schema (spec §4.2 r7 P0-1).

Two-model split:
- ParallelWorkUnitGroupRequest: planner output, no coordinator_run_id.
- ParallelRunSpec: dispatch-time runtime, has coordinator_run_id +
  fully-resolved WorkUnits with base_digest + seed_content_ref filled.
"""
from __future__ import annotations
from typing import Annotated, Literal, Optional
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

from app.domain.models.path_validation import (
    validate_coordinator_tree_prefix,
    validate_relative_path,
)


class ProposedPath(BaseModel):
    """[C2 PR-4 r1 P1#4 deep-freeze] frozen + extra=forbid so that nested
    ``ProposedWritePlan.proposed_paths`` cannot have its elements mutated
    out from under a "frozen" wire contract."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: Annotated[
        str, Field(min_length=1, description="non-empty target path"),
        AfterValidator(validate_relative_path),
    ]
    op: Literal["add", "modify", "delete"]


class PathLease(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: Annotated[
        str, Field(min_length=1, description="non-empty target path"),
        AfterValidator(validate_relative_path),
    ]
    op: Literal["add", "modify", "delete"]
    base_digest: Optional[str] = None
    seed_content_ref: Optional[str] = None

    @model_validator(mode="after")
    def _add_op_invariants(self) -> "PathLease":
        if self.op == "add":
            if self.base_digest is not None:
                raise ValueError("op=add must have base_digest=None")
            if self.seed_content_ref is not None:
                raise ValueError("op=add must have seed_content_ref=None")
        return self


class TreeLease(BaseModel):
    """[S2 §3.3] ADD-ONLY directory-tree write authorization for a coordinator
    child. The prefix names a workspace-relative directory under which the child
    may CREATE new files (raw-shell writes captured by snapshot diff in PR-2).
    v1 is ADD-ONLY: ``ops`` is a frozenset of the single literal ``"add"`` —
    modify/delete inside a leased tree are NOT authorized (the planner must lease
    those paths explicitly via ``PathLease``)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    prefix: Annotated[
        str,
        Field(min_length=1, description="workspace-relative directory prefix"),
        AfterValidator(validate_coordinator_tree_prefix),
    ]
    ops: Annotated[frozenset[Literal["add"]], Field(min_length=1)]


class ProposedTree(BaseModel):
    """[S2 §3.3] Planner-proposed tree prefix (the request-side sibling of
    ``TreeLease``, mirroring ``ProposedPath`` vs ``PathLease``)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    prefix: Annotated[
        str,
        Field(min_length=1, description="workspace-relative directory prefix"),
        AfterValidator(validate_coordinator_tree_prefix),
    ]
    ops: Annotated[frozenset[Literal["add"]], Field(min_length=1)]


class WorkUnitRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    objective: str
    phase: Literal["exploration", "write"]
    allowed_tools: list[str] = Field(default_factory=list)
    proposed_paths: list[ProposedPath] = Field(default_factory=list)
    proposed_trees: list[ProposedTree] = Field(default_factory=list)
    # [S2 §3.5] positive, INDEPENDENTLY-REQUESTABLE shell-capable signal on the
    # REQUEST. Default False = typed-only. A request may set shell_mode=True with
    # ONLY exact file leases (proposed_paths, NO proposed_trees) — tree leases
    # are SUFFICIENT to imply shell_mode (§3.3) but NOT NECESSARY for it. Without
    # this field, PR-5/PR-6 planner payloads `WorkUnitRequest(..., shell_mode=
    # True, ...)` are rejected by extra="forbid" before dispatch, and a
    # shell-mode unit with only exact leases can never activate shell mode.
    shell_mode: bool = False
    expected_result_schema: Optional[str] = None
    # [S4 §9] planner-authored member role tag. None ⇒ no specialization
    # (INV-0 passthrough). Optional default keeps extra="forbid" accepting
    # pre-S4 payloads.
    role: Optional[str] = None

    @model_validator(mode="after")
    def _write_must_have_paths(self) -> "WorkUnitRequest":
        # [S2 §3.3] write phase is satisfied by proposed_paths OR proposed_trees.
        if (
            self.phase == "write"
            and not self.proposed_paths
            and not self.proposed_trees
        ):
            raise ValueError(
                "phase=write requires at least one proposed_path or proposed_tree"
            )
        return self

    @model_validator(mode="after")
    def _request_shell_mode_consistency(self) -> "WorkUnitRequest":
        """[S2 §3.3/§3.5] Reconcile shell_mode with the request's leases:

        - phase=exploration ⇒ shell_mode=False (read-only contradicts raw-shell
          write). The rejection is evaluated against the EFFECTIVE shell intent
          — the explicit ``shell_mode`` flag OR the tree-implied one (non-empty
          ``proposed_trees`` ⇒ shell mode, §3.3) — and MUST run BEFORE the
          coercion below. Otherwise an exploration request carrying non-empty
          ``proposed_trees`` with default ``shell_mode=False`` would slip past a
          flag-only check, get coerced to True, and return a contradictory
          object (exploration WITH a tree lease) that only the downstream
          ``WorkUnit._phase_lease_consistency`` would catch — defeating the
          whole point of rejecting the contradiction at the REQUEST boundary.
        - proposed_trees non-empty ⇒ shell_mode COERCED True (a tree lease is a
          shell-mode-only concept; §3.3 "write_tree_lease implies shell_mode" —
          tree leases are SUFFICIENT for shell mode). shell_mode stays
          INDEPENDENTLY requestable: a request with ONLY proposed_paths may set
          shell_mode=True (tree leases are NOT necessary).
        """
        effective_shell_mode = self.shell_mode or bool(self.proposed_trees)
        if self.phase == "exploration" and effective_shell_mode:
            raise ValueError(
                "phase=exploration must have shell_mode=False and empty "
                "proposed_trees (read-only requests cannot do raw-shell writes)"
            )
        # WorkUnitRequest is NOT frozen, so a mode="after" validator may assign.
        # Safe to coerce now: any exploration contradiction was already rejected
        # above against the effective intent (explicit flag OR tree-implied).
        if self.proposed_trees and not self.shell_mode:
            self.shell_mode = True
        return self


class WorkUnit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    work_unit_id: str
    objective: str
    phase: Literal["exploration", "write"]
    allowed_tools: list[str] = Field(default_factory=list)
    write_lease: list[PathLease] = Field(default_factory=list)
    write_tree_lease: list[TreeLease] = Field(default_factory=list)
    shell_mode: bool = False
    expected_result_schema: Optional[str] = None
    # [S4 §9] carried verbatim from WorkUnitRequest in
    # _build_work_units_from_requests; re-threaded through the :658 enrichment.
    role: Optional[str] = None
    # [S4 §9] member persona, expander-set, rides the IN-MEMORY WorkUnit only.
    # NEVER serialized to the content-addressed manifest (avoids perturbing
    # spawn_manifest_sha256 + keeps large prompts out of MinIO).
    system_prompt: Optional[str] = None
    # [S4 §9/§11] the BIND floor carrier (preset ∪ member_skill_tools) — the
    # GENERATED skill_{slug}_{tool} names. Serialized omit-when-empty + sorted.
    member_skill_tools: frozenset[str] = Field(default_factory=frozenset)
    # [S4 §9/§12 hop-3] the child-side carve-out carrier — source Skill slugs,
    # so the child can force-include them past _filter_skills_by_user_preferences.
    member_skill_slugs: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _phase_lease_consistency(self) -> "WorkUnit":
        """[C2 PR-4 r3 P1#5 + S2 §3.3/§3.5] Phase ↔ lease ↔ shell_mode invariants:
        - phase=write       ⇒ non-empty write_lease OR non-empty write_tree_lease
        - phase=exploration ⇒ empty write_lease AND empty write_tree_lease
          (read-only by design — no write authorization smuggled in)
        - phase=exploration ⇒ shell_mode=False (read-only contradicts raw-shell write)
        - non-empty write_tree_lease ⇒ shell_mode=True (a tree lease is only
          meaningful for the raw-shell-write capture path; a tree lease with
          shell_mode off can never be satisfied)
        """
        if self.phase == "write" and not self.write_lease and not self.write_tree_lease:
            raise ValueError(
                "phase=write requires non-empty write_lease or write_tree_lease"
            )
        if self.phase == "exploration" and (self.write_lease or self.write_tree_lease):
            raise ValueError(
                "phase=exploration must have empty write_lease and "
                "write_tree_lease (exploration children are read-only by design)"
            )
        if self.phase == "exploration" and self.shell_mode:
            raise ValueError(
                "phase=exploration must have shell_mode=False "
                "(read-only children cannot do raw-shell writes)"
            )
        if self.write_tree_lease and not self.shell_mode:
            raise ValueError(
                "non-empty write_tree_lease requires shell_mode=True "
                "(tree leases only authorize the raw-shell-write capture path)"
            )
        return self


class ParallelWorkUnitGroupRequest(BaseModel):
    """Planner output type. NO coordinator_run_id."""
    work_units: list[WorkUnitRequest]


class ParallelRunSpec(BaseModel):
    """Runtime spec built by dispatch_node."""
    coordinator_run_id: str
    work_units: list[WorkUnit]
