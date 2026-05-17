"""IUnitOfWork Protocol contract — slot annotations for PE-0.

The Protocol exposes repo slots used by domain services. PE-0 adds
`user_tool_approval_policy` so PermissionEngine can do per-call
policy reads via uow.user_tool_approval_policy.get(...) without
importing infrastructure from domain.
"""

from typing import get_type_hints

from app.domain.repositories.uow import IUnitOfWork
from app.domain.repositories.user_tool_approval_policy_repository import (
    UserToolApprovalPolicyRepository,
)


def test_iunitofwork_exposes_user_tool_approval_policy_slot():
    hints = get_type_hints(IUnitOfWork)
    assert "user_tool_approval_policy" in hints, (
        "PE-0 contract: IUnitOfWork must expose a "
        "user_tool_approval_policy: UserToolApprovalPolicyRepository slot"
    )
    assert hints["user_tool_approval_policy"] is UserToolApprovalPolicyRepository


def test_existing_slots_preserved():
    hints = get_type_hints(IUnitOfWork)
    for required in (
        "session", "file", "tool_approval_log",
        "sandbox_lifecycle_log", "approval_grants", "compaction",
    ):
        assert required in hints, (
            f"existing UoW slot {required!r} must be preserved"
        )
