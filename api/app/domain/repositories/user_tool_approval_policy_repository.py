"""用户工具审批偏好仓储抽象接口。"""

from abc import ABC, abstractmethod
from typing import Optional

from app.domain.models.user_tool_approval_policy import UserToolApprovalPolicy


class UserToolApprovalPolicyRepository(ABC):
    """R6 §4.3 — 仓储抽象。`get` 返回 Optional，不做隐式默认回落。"""

    @abstractmethod
    async def get(
        self, user_id: str, tool_name: str
    ) -> Optional[UserToolApprovalPolicy]:
        """显式查询。缺行返 None；不返回默认值（(ii′) 语义强制落点）。"""

    @abstractmethod
    async def list_by_user(
        self, user_id: str
    ) -> list[UserToolApprovalPolicy]:
        """列表当前 user 的所有 policy 行，按 tool_name 升序。"""

    @abstractmethod
    async def upsert(
        self, policy: UserToolApprovalPolicy
    ) -> UserToolApprovalPolicy:
        """幂等 upsert by UNIQUE(user_id, tool_name)；更新 updated_at。"""

    @abstractmethod
    async def delete(self, user_id: str, tool_name: str) -> bool:
        """删除指定 (user, tool) policy。返 True 当且仅当之前存在。"""
