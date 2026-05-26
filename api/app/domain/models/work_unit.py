"""C2 v1 work unit schema (spec §4.2 r7 P0-1).

Two-model split:
- ParallelWorkUnitGroupRequest: planner output, no coordinator_run_id.
- ParallelRunSpec: dispatch-time runtime, has coordinator_run_id +
  fully-resolved WorkUnits with base_digest + seed_content_ref filled.
"""
from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field, model_validator


class ProposedPath(BaseModel):
    path: str
    op: Literal["add", "modify", "delete"]


class PathLease(BaseModel):
    path: str
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
    def _write_must_have_lease(self) -> "WorkUnit":
        if self.phase == "write" and not self.write_lease:
            raise ValueError("phase=write requires non-empty write_lease")
        return self


class ParallelWorkUnitGroupRequest(BaseModel):
    """Planner output type. NO coordinator_run_id."""
    work_units: list[WorkUnitRequest]


class ParallelRunSpec(BaseModel):
    """Runtime spec built by dispatch_node."""
    coordinator_run_id: str
    work_units: list[WorkUnit]
