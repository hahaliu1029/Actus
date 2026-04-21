"""用户工具审批偏好应用服务（CRUD only；不接入决策链）。"""

import logging
from typing import Optional

from app.domain.models.user_tool_approval_policy import (
    ApprovalPolicy,
    UserToolApprovalPolicy,
)
from app.domain.repositories.user_tool_approval_policy_repository import (
    UserToolApprovalPolicyRepository,
)

logger = logging.getLogger(__name__)


class UserToolApprovalPolicyService:
    """R6 §5.2 — Phase 1 CRUD；Phase 2 PermissionEngine 再接决策链。"""

    def __init__(self, repo: UserToolApprovalPolicyRepository) -> None:
        self._repo = repo

    async def get_policy(
        self, user_id: str, tool_name: str
    ) -> Optional[ApprovalPolicy]:
        """Phase 2 PermissionEngine 调用路径：只暴露 enum，
        缺行返 None，不做默认回落。裁剪掉 id/timestamps 以收窄下游
        决策链接口面。路由若需要完整 row（带 updated_at 等），
        请走 :meth:`get_policy_record`。
        """
        row = await self._repo.get(user_id, tool_name)
        return row.policy if row else None

    async def get_policy_record(
        self, user_id: str, tool_name: str
    ) -> Optional[UserToolApprovalPolicy]:
        """HTTP 路由单条查询路径：返回完整 domain 记录，缺行返 None。

        走仓储 :meth:`UserToolApprovalPolicyRepository.get` 单条
        WHERE，避免路由层 ``list_user_policies + 内存 filter`` 的
        O(n) 开销与 "单条查询合同未被路由使用" 的死分支。
        """
        return await self._repo.get(user_id, tool_name)

    async def list_user_policies(
        self, user_id: str
    ) -> list[UserToolApprovalPolicy]:
        return await self._repo.list_by_user(user_id)

    async def set_policy(
        self, user_id: str, tool_name: str, policy: ApprovalPolicy
    ) -> UserToolApprovalPolicy:
        record = UserToolApprovalPolicy(
            user_id=user_id,
            tool_name=tool_name,
            policy=policy,
        )
        result = await self._repo.upsert(record)
        logger.info(
            "User %s set tool %s approval policy=%s",
            user_id,
            tool_name,
            policy.value,
        )
        return result

    async def clear_policy(self, user_id: str, tool_name: str) -> bool:
        """删除 = 回到"缺行"fallback 语义。"""
        existed = await self._repo.delete(user_id, tool_name)
        if existed:
            logger.info(
                "User %s cleared tool %s approval policy", user_id, tool_name
            )
        return existed
