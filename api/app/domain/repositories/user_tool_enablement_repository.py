"""用户工具扩展启用仓储接口。"""

from abc import ABC, abstractmethod
from typing import Optional

from app.domain.models.user_tool_enablement import ToolType, UserToolEnablement


class UserToolEnablementRepository(ABC):
    """用户工具扩展启用仓储抽象接口。"""

    @abstractmethod
    async def create(self, enablement: UserToolEnablement) -> UserToolEnablement:
        """创建用户工具扩展启用记录。"""
        pass

    @abstractmethod
    async def get_by_id(self, enablement_id: str) -> Optional[UserToolEnablement]:
        """根据 ID 获取用户工具扩展启用记录。"""
        pass

    @abstractmethod
    async def get_by_user_and_tool(
        self, user_id: str, tool_type: ToolType, tool_id: str
    ) -> Optional[UserToolEnablement]:
        """根据用户 ID、工具类型和工具 ID 获取启用记录。"""
        pass

    @abstractmethod
    async def get_by_user_id(
        self, user_id: str, tool_type: Optional[ToolType] = None
    ) -> list[UserToolEnablement]:
        """获取用户的所有工具扩展启用记录，可选按类型过滤。"""
        pass

    @abstractmethod
    async def upsert(self, enablement: UserToolEnablement) -> UserToolEnablement:
        """创建或更新用户工具扩展启用记录。"""
        pass

    @abstractmethod
    async def delete(self, enablement_id: str) -> bool:
        """删除用户工具扩展启用记录。"""
        pass

    @abstractmethod
    async def delete_by_tool(self, tool_type: ToolType, tool_id: str) -> int:
        """删除指定工具的所有用户扩展启用记录（当工具被删除时）。"""
        pass
