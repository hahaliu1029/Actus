"""用户工具扩展启用服务。"""

import logging
from typing import Optional

from app.domain.models.user_tool_enablement import ToolType, UserToolEnablement
from app.domain.repositories.user_tool_enablement_repository import (
    UserToolEnablementRepository,
)

logger = logging.getLogger(__name__)


class UserToolEnablementService:
    """用户工具扩展启用服务。

    管理用户对 MCP/A2A/Skill 工具的个人启用/禁用偏好
    （仅管扩展是否进工具池；审批策略走 UserToolApprovalPolicyService）。
    """

    def __init__(self, enablement_repository: UserToolEnablementRepository) -> None:
        self.enablement_repository = enablement_repository

    async def list_user_enablements(
        self,
        user_id: str,
        tool_type: Optional[ToolType] = None,
    ) -> list[UserToolEnablement]:
        """获取用户的工具扩展启用列表。

        Args:
            user_id: 用户 ID
            tool_type: 工具类型过滤，None 表示全部

        Returns:
            list: 用户工具扩展启用列表
        """
        return await self.enablement_repository.get_by_user_id(user_id, tool_type)

    async def get_enablement(
        self,
        user_id: str,
        tool_type: ToolType,
        tool_id: str,
    ) -> Optional[UserToolEnablement]:
        """获取用户对特定工具的扩展启用记录。

        Args:
            user_id: 用户 ID
            tool_type: 工具类型
            tool_id: 工具 ID

        Returns:
            Optional: 启用记录，不存在返回 None
        """
        return await self.enablement_repository.get_by_user_and_tool(
            user_id, tool_type, tool_id
        )

    async def is_tool_enabled_for_user(
        self,
        user_id: str,
        tool_type: ToolType,
        tool_id: str,
    ) -> bool:
        """检查用户是否启用了某个工具。

        如果用户没有设置偏好，默认返回 True（启用）。

        Args:
            user_id: 用户 ID
            tool_type: 工具类型
            tool_id: 工具 ID

        Returns:
            bool: 是否启用
        """
        enablement = await self.enablement_repository.get_by_user_and_tool(
            user_id, tool_type, tool_id
        )
        # 没有设置偏好时，默认启用
        return enablement.enabled if enablement else True

    async def set_tool_enabled(
        self,
        user_id: str,
        tool_type: ToolType,
        tool_id: str,
        enabled: bool,
    ) -> UserToolEnablement:
        """设置用户对某个工具的启用状态。

        Args:
            user_id: 用户 ID
            tool_type: 工具类型
            tool_id: 工具 ID
            enabled: 是否启用

        Returns:
            UserToolEnablement: 更新后的启用记录
        """
        enablement = UserToolEnablement(
            user_id=user_id,
            tool_type=tool_type,
            tool_id=tool_id,
            enabled=enabled,
        )
        result = await self.enablement_repository.upsert(enablement)
        logger.info(
            f"User {user_id} set {tool_type.value} tool {tool_id} enabled={enabled}"
        )
        return result

    async def delete_enablements_by_tool(
        self,
        tool_type: ToolType,
        tool_id: str,
    ) -> int:
        """删除某个工具的所有用户扩展启用记录。

        当工具被管理员删除时调用。

        Args:
            tool_type: 工具类型
            tool_id: 工具 ID

        Returns:
            int: 删除的记录数
        """
        count = await self.enablement_repository.delete_by_tool(tool_type, tool_id)
        logger.info(
            f"Deleted {count} enablement rows for {tool_type.value} tool {tool_id}"
        )
        return count
