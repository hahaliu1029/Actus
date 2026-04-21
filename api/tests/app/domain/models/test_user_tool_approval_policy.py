"""Unit tests for UserToolApprovalPolicy domain model."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from app.domain.models.user_tool_approval_policy import (
    ApprovalPolicy,
    UserToolApprovalPolicy,
)


class TestApprovalPolicyEnum:
    def test_enum_values(self):
        assert ApprovalPolicy.AUTO.value == "auto"
        assert ApprovalPolicy.ASK.value == "ask"
        assert ApprovalPolicy.DENY.value == "deny"

    def test_enum_is_str_enum(self):
        # 可以和 string 直接比较（SQLAlchemy 存值兼容）
        assert ApprovalPolicy.AUTO == "auto"


class TestUserToolApprovalPolicyModel:
    def test_construct_with_required_fields(self):
        policy = UserToolApprovalPolicy(
            user_id="user-1",
            tool_name="shell_execute",
            policy=ApprovalPolicy.AUTO,
        )
        assert policy.user_id == "user-1"
        assert policy.tool_name == "shell_execute"
        assert policy.policy == ApprovalPolicy.AUTO
        # 自动生成 id 和时间戳
        assert policy.id  # uuid4 str
        assert isinstance(policy.created_at, datetime)
        assert isinstance(policy.updated_at, datetime)

    def test_accepts_tool_name_with_hyphen_mcp_style(self):
        # MCP canonical 名可含连字符（见 spec §6.2 / langchain_mcp_discovery.py:133）
        policy = UserToolApprovalPolicy(
            user_id="user-1",
            tool_name="mcp_amap-maps_maps_weather",
            policy=ApprovalPolicy.ASK,
        )
        assert policy.tool_name == "mcp_amap-maps_maps_weather"

    def test_rejects_invalid_policy_value(self):
        with pytest.raises(ValidationError):
            UserToolApprovalPolicy(
                user_id="user-1",
                tool_name="shell_execute",
                policy="allow",  # 不在 auto/ask/deny 内
            )

    def test_serializes_via_model_dump(self):
        policy = UserToolApprovalPolicy(
            user_id="user-1",
            tool_name="shell_execute",
            policy=ApprovalPolicy.DENY,
        )
        dumped = policy.model_dump()
        assert dumped["policy"] == "deny"
        assert dumped["user_id"] == "user-1"
