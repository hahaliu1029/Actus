"""[PR-9b-B] Group-scope lineage fields for CoordinatorApplyEvent.

The 3rd field `coordinator_run_id` lives on `PatchApplyPlan` already; this
value object carries only the 2 that aren't on the plan.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class GroupLineageFields:
    root_session_id: Optional[str] = None
    parent_session_id: Optional[str] = None
