"""Unit tests for UserToolApprovalPolicyService (with in-memory fake repo)."""

from typing import Optional

import pytest

from app.application.services.user_tool_approval_policy_service import (
    UserToolApprovalPolicyService,
)
from app.domain.models.user_tool_approval_policy import (
    ApprovalPolicy,
    UserToolApprovalPolicy,
)
from app.domain.repositories.user_tool_approval_policy_repository import (
    UserToolApprovalPolicyRepository,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class _FakeRepo(UserToolApprovalPolicyRepository):
    def __init__(self) -> None:
        self._data: dict[tuple[str, str], UserToolApprovalPolicy] = {}

    async def get(
        self, user_id: str, tool_name: str
    ) -> Optional[UserToolApprovalPolicy]:
        return self._data.get((user_id, tool_name))

    async def list_by_user(self, user_id: str) -> list[UserToolApprovalPolicy]:
        return [p for (uid, _), p in self._data.items() if uid == user_id]

    async def upsert(self, policy: UserToolApprovalPolicy) -> UserToolApprovalPolicy:
        self._data[(policy.user_id, policy.tool_name)] = policy
        return policy

    async def delete(self, user_id: str, tool_name: str) -> bool:
        return self._data.pop((user_id, tool_name), None) is not None


class TestGetPolicy:
    async def test_missing_returns_none_not_default(self) -> None:
        svc = UserToolApprovalPolicyService(_FakeRepo())
        assert await svc.get_policy("u1", "shell_execute") is None

    async def test_present_returns_enum(self) -> None:
        repo = _FakeRepo()
        await repo.upsert(
            UserToolApprovalPolicy(
                user_id="u1",
                tool_name="shell_execute",
                policy=ApprovalPolicy.AUTO,
            )
        )
        svc = UserToolApprovalPolicyService(repo)
        assert await svc.get_policy("u1", "shell_execute") == ApprovalPolicy.AUTO


class TestGetPolicyRecord:
    async def test_missing_returns_none(self) -> None:
        svc = UserToolApprovalPolicyService(_FakeRepo())
        assert await svc.get_policy_record("u1", "shell_execute") is None

    async def test_present_returns_full_record(self) -> None:
        repo = _FakeRepo()
        seeded = UserToolApprovalPolicy(
            user_id="u1",
            tool_name="shell_execute",
            policy=ApprovalPolicy.AUTO,
        )
        await repo.upsert(seeded)
        svc = UserToolApprovalPolicyService(repo)
        result = await svc.get_policy_record("u1", "shell_execute")
        assert result is not None
        # 完整 row 返回：不止 policy enum，还有 id/timestamps 供路由层用
        assert result.id == seeded.id
        assert result.tool_name == "shell_execute"
        assert result.policy == ApprovalPolicy.AUTO
        assert result.created_at == seeded.created_at
        assert result.updated_at == seeded.updated_at

    async def test_respects_user_isolation(self) -> None:
        svc = UserToolApprovalPolicyService(_FakeRepo())
        await svc.set_policy("u1", "shell_execute", ApprovalPolicy.ASK)
        # 另一个用户查同一 tool_name → None
        assert await svc.get_policy_record("u2", "shell_execute") is None


class TestSetAndClear:
    async def test_set_creates_row(self) -> None:
        svc = UserToolApprovalPolicyService(_FakeRepo())
        result = await svc.set_policy("u1", "shell_execute", ApprovalPolicy.ASK)
        assert result.policy == ApprovalPolicy.ASK
        assert await svc.get_policy("u1", "shell_execute") == ApprovalPolicy.ASK

    async def test_set_overwrites(self) -> None:
        svc = UserToolApprovalPolicyService(_FakeRepo())
        await svc.set_policy("u1", "shell_execute", ApprovalPolicy.ASK)
        await svc.set_policy("u1", "shell_execute", ApprovalPolicy.DENY)
        assert await svc.get_policy("u1", "shell_execute") == ApprovalPolicy.DENY

    async def test_clear_returns_true_when_existed(self) -> None:
        svc = UserToolApprovalPolicyService(_FakeRepo())
        await svc.set_policy("u1", "shell_execute", ApprovalPolicy.ASK)
        assert await svc.clear_policy("u1", "shell_execute") is True

    async def test_clear_returns_false_when_absent(self) -> None:
        svc = UserToolApprovalPolicyService(_FakeRepo())
        assert await svc.clear_policy("u1", "nonexistent") is False


class TestListUserPolicies:
    async def test_list_returns_all_for_user(self) -> None:
        svc = UserToolApprovalPolicyService(_FakeRepo())
        await svc.set_policy("u1", "shell_execute", ApprovalPolicy.ASK)
        await svc.set_policy("u1", "mcp_github_create_issue", ApprovalPolicy.AUTO)
        await svc.set_policy("u2", "shell_execute", ApprovalPolicy.DENY)
        u1_rows = await svc.list_user_policies("u1")
        assert {r.tool_name for r in u1_rows} == {
            "shell_execute",
            "mcp_github_create_issue",
        }
