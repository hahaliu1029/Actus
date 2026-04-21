"""R6 concept guard — enablement 表和 approval_policy 表互相不污染。

TODO2 §06 R6 点名要求：锁死"两张表独立、不互相读写污染"。
"""

import pytest
from sqlalchemy import text

from app.application.services.user_tool_approval_policy_service import (
    UserToolApprovalPolicyService,
)
from app.application.services.user_tool_enablement_service import (
    UserToolEnablementService,
)
from app.domain.models.user_tool_approval_policy import ApprovalPolicy
from app.domain.models.user_tool_enablement import ToolType
from app.infrastructure.repositories.db_user_tool_approval_policy_repository import (
    DBUserToolApprovalPolicyRepository,
)
from app.infrastructure.repositories.db_user_tool_enablement_repository import (
    DBUserToolEnablementRepository,
)


pytestmark = [pytest.mark.anyio, pytest.mark.integration]


@pytest.fixture
async def _user(db_session):
    user_id = "r6-concept-guard-user"
    await db_session.execute(text(
        "INSERT INTO users (id, username, email, password_hash, status, created_at, updated_at) "
        "VALUES (:id, :u, :e, 'x', 'active', NOW(), NOW())"
    ), {"id": user_id, "u": f"u_{user_id}", "e": f"{user_id}@t"})
    await db_session.flush()
    yield user_id


async def _make_services(db_session):
    enablement_svc = UserToolEnablementService(
        DBUserToolEnablementRepository(db_session),
    )
    policy_svc = UserToolApprovalPolicyService(
        DBUserToolApprovalPolicyRepository(db_session),
    )
    return enablement_svc, policy_svc


async def test_both_tables_persist_independently(db_session, _user):
    """同一 user 同一逻辑目标：两张表都能独立落行、独立读回。"""
    enablement_svc, policy_svc = await _make_services(db_session)

    await enablement_svc.set_tool_enabled(_user, ToolType.MCP, "github", enabled=False)
    await policy_svc.set_policy(_user, "mcp_github_create_issue", ApprovalPolicy.AUTO)

    assert (await enablement_svc.is_tool_enabled_for_user(
        _user, ToolType.MCP, "github"
    )) is False
    assert (await policy_svc.get_policy(
        _user, "mcp_github_create_issue"
    )) == ApprovalPolicy.AUTO


async def test_enablement_service_delete_by_tool_does_not_touch_policies(
    db_session, _user
):
    """删除 MCP enablement 行，不能影响任何 policy 行。"""
    enablement_svc, policy_svc = await _make_services(db_session)

    await enablement_svc.set_tool_enabled(_user, ToolType.MCP, "github", enabled=False)
    await policy_svc.set_policy(_user, "mcp_github_create_issue", ApprovalPolicy.AUTO)
    await policy_svc.set_policy(_user, "mcp_github_list_repos", ApprovalPolicy.ASK)

    await enablement_svc.delete_enablements_by_tool(ToolType.MCP, "github")

    assert (await policy_svc.get_policy(
        _user, "mcp_github_create_issue"
    )) == ApprovalPolicy.AUTO
    assert (await policy_svc.get_policy(
        _user, "mcp_github_list_repos"
    )) == ApprovalPolicy.ASK


async def test_policy_clear_does_not_touch_enablement(db_session, _user):
    """清掉 policy 行，不能影响 enablement 行。"""
    enablement_svc, policy_svc = await _make_services(db_session)

    await enablement_svc.set_tool_enabled(_user, ToolType.MCP, "github", enabled=False)
    await policy_svc.set_policy(_user, "mcp_github_create_issue", ApprovalPolicy.AUTO)

    await policy_svc.clear_policy(_user, "mcp_github_create_issue")

    assert (await enablement_svc.is_tool_enabled_for_user(
        _user, ToolType.MCP, "github"
    )) is False


async def test_partial_string_match_does_not_cross_tables(db_session, _user):
    """tool_id='github' 不等于 tool_name='mcp_github_create_issue' — 表隔离。"""
    enablement_svc, policy_svc = await _make_services(db_session)

    await enablement_svc.set_tool_enabled(_user, ToolType.MCP, "github", enabled=True)
    await policy_svc.set_policy(_user, "mcp_github_create_issue", ApprovalPolicy.ASK)

    await policy_svc.clear_policy(_user, "mcp_github_create_issue")

    row = await enablement_svc.get_enablement(_user, ToolType.MCP, "github")
    assert row is not None
    assert row.enabled is True


async def test_two_tables_have_separate_sqlalchemy_tables(db_session):
    """Sanity check: 两张表在 DB 里是独立实体，没有 JOIN/trigger 耦合。"""
    result = await db_session.execute(text(
        "SELECT tablename FROM pg_tables WHERE schemaname='public' "
        "AND tablename IN ('user_tool_enablements', 'user_tool_approval_policies')"
    ))
    names = {r[0] for r in result.fetchall()}
    assert names == {"user_tool_enablements", "user_tool_approval_policies"}
