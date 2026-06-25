"""C2 v1 ChildPermissionContext carrier (spec §5.3).

Carried inside EvaluationContext.child_permission_context. None = root.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from app.domain.models.work_unit import PathLease, TreeLease


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
    # [S2 §3.3] ADD-only directory-tree leases. Default () keeps every existing
    # construction (3-field) valid + inert.
    tree_leases: "tuple[TreeLease, ...]" = ()
    # [S2 §3.5] positive shell-capable signal. Default False = typed-only (the
    # pre-S2 behavior). NO reuse of ChildRuntimeCap.NO_RAW_SHELL_WRITE.
    shell_mode: bool = False
    # [S4 §9/§11] member-skill GENERATED tool names = the bind floor carrier.
    # Default empty ⇒ omitted from the manifest JSON ⇒ byte-identical manifest.
    member_skill_tools: frozenset[str] = frozenset()
    # [S4 §9/§12] member-skill source slugs = the child-side preference carve-out
    # carrier. Default empty ⇒ omitted ⇒ byte-identical.
    member_skill_slugs: tuple[str, ...] = ()


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
    # [S2 §3.5] mirror of spawn_manifest.shell_mode hoisted to the top-level
    # carrier so the two gate callers can read cpc.shell_mode without reaching
    # into the manifest. Dormant in PR-3 (no caller reads it yet).
    shell_mode: bool = False
    # [S4 §11/§12] mirrors of the manifest fields hoisted onto the carrier so the
    # factory bind (member_skill_tools) and the child startup carve-out
    # (member_skill_slugs) read them without reaching into the manifest. Dormant
    # until PR-3 wires the readers.
    member_skill_tools: frozenset[str] = frozenset()
    member_skill_slugs: tuple[str, ...] = ()
