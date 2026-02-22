"""用户工具偏好服务"""

import logging
from typing import Optional

from app.domain.models.user_tool_preference import ToolType, UserToolPreference
from app.domain.repositories.user_tool_preference_repository import (
    UserToolPreferenceRepository,
)

logger = logging.getLogger(__name__)


class UserToolPreferenceService:
    """用户工具偏好服务

    管理用户对 MCP/A2A 工具的个人启用/禁用偏好
    """

    def __init__(self, preference_repository: UserToolPreferenceRepository) -> None:
        self.preference_repository = preference_repository

    async def get_user_preferences(
        self,
        user_id: str,
        tool_type: Optional[ToolType] = None,
    ) -> list[UserToolPreference]:
        """获取用户的工具偏好列表

        Args:
            user_id: 用户 ID
            tool_type: 工具类型过滤，None 表示全部

        Returns:
            list: 用户工具偏好列表
        """
        return await self.preference_repository.get_by_user_id(user_id, tool_type)

    async def get_preference(
        self,
        user_id: str,
        tool_type: ToolType,
        tool_id: str,
    ) -> Optional[UserToolPreference]:
        """获取用户对特定工具的偏好

        Args:
            user_id: 用户 ID
            tool_type: 工具类型
            tool_id: 工具 ID

        Returns:
            Optional: 偏好设置，不存在返回 None
        """
        return await self.preference_repository.get_by_user_and_tool(
            user_id, tool_type, tool_id
        )

    async def is_tool_enabled_for_user(
        self,
        user_id: str,
        tool_type: ToolType,
        tool_id: str,
    ) -> bool:
        """检查用户是否启用了某个工具

        如果用户没有设置偏好，默认返回 True（启用）

        Args:
            user_id: 用户 ID
            tool_type: 工具类型
            tool_id: 工具 ID

        Returns:
            bool: 是否启用
        """
        pref = await self.preference_repository.get_by_user_and_tool(
            user_id, tool_type, tool_id
        )
        # 没有设置偏好时，默认启用
        return pref.enabled if pref else True

    async def set_tool_enabled(
        self,
        user_id: str,
        tool_type: ToolType,
        tool_id: str,
        enabled: bool,
    ) -> UserToolPreference:
        """设置用户对某个工具的启用状态

        Args:
            user_id: 用户 ID
            tool_type: 工具类型
            tool_id: 工具 ID
            enabled: 是否启用

        Returns:
            UserToolPreference: 更新后的偏好
        """
        preference = UserToolPreference(
            user_id=user_id,
            tool_type=tool_type,
            tool_id=tool_id,
            enabled=enabled,
        )
        result = await self.preference_repository.upsert(preference)
        logger.info(
            f"User {user_id} set {tool_type.value} tool {tool_id} enabled={enabled}"
        )
        return result

    async def delete_tool_preferences(
        self,
        tool_type: ToolType,
        tool_id: str,
    ) -> int:
        """删除某个工具的所有用户偏好

        当工具被管理员删除时调用

        Args:
            tool_type: 工具类型
            tool_id: 工具 ID

        Returns:
            int: 删除的记录数
        """
        count = await self.preference_repository.delete_by_tool(tool_type, tool_id)
        logger.info(f"Deleted {count} preferences for {tool_type.value} tool {tool_id}")
        return count
