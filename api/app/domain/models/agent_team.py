"""C2-full S4 agent-team named bundles — domain model (spec §4).

Domain-pure: pydantic only. A TeamBundle is a named, reusable roster of
role-specialized members. Capability validation (skill-slug resolution,
generated tool names) is RESOLVE-time and lives in the expander (§13), NOT
here — these are the pure STRUCTURAL validators only.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

# The child prompt is minimal by security design (assembler.py:223); cap member
# prompt size so an over-long persona block can't bloat / destabilize it (§4/R10-4).
MEMBER_SYSTEM_PROMPT_MAX_LENGTH = 4000


class TeamMember(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    role: str                                    # member-local id; unique within the team
    description: str                             # routing hint shown to the planner
    system_prompt: str = Field(max_length=MEMBER_SYSTEM_PROMPT_MAX_LENGTH)
    skills: tuple[str, ...] = ()                 # existing Skill slugs
    default_phase: Optional[Literal["exploration", "write"]] = None
    shell_mode: bool = False

    @model_validator(mode="after")
    def _exploration_is_read_only(self) -> "TeamMember":
        # Mirror WorkUnit._phase_lease_consistency: exploration ⇒ no raw-shell write.
        if self.default_phase == "exploration" and self.shell_mode:
            raise ValueError(
                f"member {self.role!r}: default_phase=exploration must have shell_mode=False"
            )
        return self


class TeamBundle(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    slug: str
    name: str
    description: str = ""
    version: str = "0.1.0"
    members: tuple[TeamMember, ...]

    @model_validator(mode="after")
    def _members_valid(self) -> "TeamBundle":
        if not self.members:
            raise ValueError(f"team {self.slug!r}: members must be non-empty")
        roles = [m.role for m in self.members]
        if len(roles) != len(set(roles)):
            raise ValueError(f"team {self.slug!r}: member roles must be unique, got {roles}")
        return self
