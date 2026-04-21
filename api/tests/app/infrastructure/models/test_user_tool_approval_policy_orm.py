"""Unit tests for UserToolApprovalPolicyModel ORM round-trip."""

from datetime import datetime

from app.domain.models.user_tool_approval_policy import (
    ApprovalPolicy,
    UserToolApprovalPolicy,
)
from app.infrastructure.models.user_tool_approval_policy import (
    UserToolApprovalPolicyModel,
)


class TestUserToolApprovalPolicyOrmRoundTrip:
    def test_from_domain_preserves_fields(self):
        domain = UserToolApprovalPolicy(
            user_id="user-1",
            tool_name="shell_execute",
            policy=ApprovalPolicy.ASK,
        )
        orm = UserToolApprovalPolicyModel.from_domain(domain)
        assert orm.id == domain.id
        assert orm.user_id == "user-1"
        assert orm.tool_name == "shell_execute"
        assert orm.policy == "ask"  # enum value stored as string
        assert orm.created_at == domain.created_at
        assert orm.updated_at == domain.updated_at

    def test_to_domain_restores_enum(self):
        orm = UserToolApprovalPolicyModel(
            id="id-1",
            user_id="user-1",
            tool_name="mcp_amap-maps_maps_weather",
            policy="auto",
            created_at=datetime(2026, 4, 21, 10, 0, 0),
            updated_at=datetime(2026, 4, 21, 10, 0, 0),
        )
        domain = orm.to_domain()
        assert isinstance(domain.policy, ApprovalPolicy)
        assert domain.policy == ApprovalPolicy.AUTO
        assert domain.tool_name == "mcp_amap-maps_maps_weather"

    def test_tablename(self):
        assert UserToolApprovalPolicyModel.__tablename__ == "user_tool_approval_policies"

    def test_update_from_domain_bumps_updated_at(self):
        orm = UserToolApprovalPolicyModel(
            id="id-1",
            user_id="user-1",
            tool_name="shell_execute",
            policy="ask",
            created_at=datetime(2026, 4, 1, 0, 0, 0),
            updated_at=datetime(2026, 4, 1, 0, 0, 0),
        )
        before = orm.updated_at
        domain = UserToolApprovalPolicy(
            id="id-1",
            user_id="user-1",
            tool_name="shell_execute",
            policy=ApprovalPolicy.DENY,
        )
        orm.update_from_domain(domain)
        assert orm.policy == "deny"
        assert orm.updated_at > before
