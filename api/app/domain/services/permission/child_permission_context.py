"""C2 v1 ChildPermissionContext carrier (spec §5.3).

Carried inside EvaluationContext.child_permission_context. None = root.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from app.domain.models.work_unit import PathLease


class ChildRuntimeCap(StrEnum):
    NO_RESPAWN = "no_respawn"
    NO_SKILL_INSTALL = "no_skill_install"
    NO_PERMISSION_MUTATION = "no_permission_mutation"
    NO_RAW_SHELL_WRITE = "no_raw_shell_write"
    NO_USER_QUESTION = "no_user_question"
    NO_MAILBOX_PUBLISH_CONTROL = "no_mailbox_publish_control"


@dataclass(frozen=True)
class ChildBudget:
    max_tool_calls: int
    max_token_cost_usd: float
    max_wallclock_seconds: int


@dataclass(frozen=True)
class SpawnManifest:
    allowed_tools: frozenset[str]
    path_leases: "tuple[PathLease, ...]"
    runtime_caps: frozenset[ChildRuntimeCap]


@dataclass(frozen=True)
class ChildPermissionContext:
    parent_session_id: str
    child_session_id: str
    coordinator_run_id: str
    work_unit_id: str
    spawn_manifest: SpawnManifest
    session_mode_revision: int
    budget: ChildBudget
    lease_expiry: Optional[datetime] = None
