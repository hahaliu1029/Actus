"""C2 v1 work unit schema (spec §4.2 r7 P0-1).

Two-model split:
- ParallelWorkUnitGroupRequest: planner output, no coordinator_run_id.
- ParallelRunSpec: dispatch-time runtime, has coordinator_run_id +
  fully-resolved WorkUnits with base_digest + seed_content_ref filled.
"""
from __future__ import annotations
from typing import Annotated, Literal, Optional
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

from app.domain.models.path_validation import validate_relative_path


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


class WorkUnitRequest(BaseModel):
    objective: str
    phase: Literal["exploration", "write"]
    allowed_tools: list[str] = Field(default_factory=list)
    proposed_paths: list[ProposedPath] = Field(default_factory=list)
    expected_result_schema: Optional[str] = None

    @model_validator(mode="after")
    def _write_must_have_paths(self) -> "WorkUnitRequest":
        if self.phase == "write" and not self.proposed_paths:
            raise ValueError("phase=write requires at least one proposed_path")
        return self


class WorkUnit(BaseModel):
    work_unit_id: str
    objective: str
    phase: Literal["exploration", "write"]
    allowed_tools: list[str] = Field(default_factory=list)
    write_lease: list[PathLease] = Field(default_factory=list)
    expected_result_schema: Optional[str] = None

    @model_validator(mode="after")
    def _phase_lease_consistency(self) -> "WorkUnit":
        """[C2 PR-4 r3 P1#5] Phase ↔ lease invariants:
        - phase=write   ⇒ non-empty write_lease (existing rule)
        - phase=exploration ⇒ empty write_lease (NEW: prevents a planner from
          smuggling write authorization into a read-only phase, which the
          coordinator child prompt would silently render as authorized paths
          and ChildScopeGate would honor at runtime).
        """
        if self.phase == "write" and not self.write_lease:
            raise ValueError("phase=write requires non-empty write_lease")
        if self.phase == "exploration" and self.write_lease:
            raise ValueError(
                "phase=exploration must have empty write_lease "
                "(exploration children are read-only by design)"
            )
        return self


class ParallelWorkUnitGroupRequest(BaseModel):
    """Planner output type. NO coordinator_run_id."""
    work_units: list[WorkUnitRequest]


class ParallelRunSpec(BaseModel):
    """Runtime spec built by dispatch_node."""
    coordinator_run_id: str
    work_units: list[WorkUnit]
