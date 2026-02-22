"""用户工具偏好仓储接口"""

from abc import ABC, abstractmethod
from typing import Optional

from app.domain.models.user_tool_preference import ToolType, UserToolPreference


class UserToolPreferenceRepository(ABC):
    """用户工具偏好仓储抽象接口"""

    @abstractmethod
    async def create(self, preference: UserToolPreference) -> UserToolPreference:
        """创建用户工具偏好"""
        pass

    @abstractmethod
    async def get_by_id(self, preference_id: str) -> Optional[UserToolPreference]:
        """根据 ID 获取用户工具偏好"""
        pass

    @abstractmethod
    async def get_by_user_and_tool(
        self, user_id: str, tool_type: ToolType, tool_id: str
    ) -> Optional[UserToolPreference]:
        """根据用户 ID、工具类型和工具 ID 获取偏好"""
        pass

    @abstractmethod
    async def get_by_user_id(
        self, user_id: str, tool_type: Optional[ToolType] = None
    ) -> list[UserToolPreference]:
        """获取用户的所有工具偏好，可选按类型过滤"""
        pass

    @abstractmethod
    async def upsert(self, preference: UserToolPreference) -> UserToolPreference:
        """创建或更新用户工具偏好"""
        pass

    @abstractmethod
    async def delete(self, preference_id: str) -> bool:
        """删除用户工具偏好"""
        pass

    @abstractmethod
    async def delete_by_tool(self, tool_type: ToolType, tool_id: str) -> int:
        """删除指定工具的所有用户偏好（当工具被删除时）"""
        pass
