"""OAuth 账户仓储接口"""

from abc import ABC, abstractmethod
from typing import Optional

from app.domain.models.oauth_account import OAuthAccount, OAuthProvider


class OAuthRepository(ABC):
    """OAuth 账户仓储抽象接口"""

    @abstractmethod
    async def create(self, oauth_account: OAuthAccount) -> OAuthAccount:
        """创建 OAuth 账户"""
        pass

    @abstractmethod
    async def get_by_id(self, oauth_id: str) -> Optional[OAuthAccount]:
        """根据 ID 获取 OAuth 账户"""
        pass

    @abstractmethod
    async def get_by_provider_user_id(
        self, provider: OAuthProvider, provider_user_id: str
    ) -> Optional[OAuthAccount]:
        """根据提供商和提供商用户 ID 获取 OAuth 账户"""
        pass

    @abstractmethod
    async def get_by_user_id(self, user_id: str) -> list[OAuthAccount]:
        """根据用户 ID 获取所有 OAuth 账户"""
        pass

    @abstractmethod
    async def update(self, oauth_account: OAuthAccount) -> OAuthAccount:
        """更新 OAuth 账户"""
        pass

    @abstractmethod
    async def delete(self, oauth_id: str) -> bool:
        """删除 OAuth 账户"""
        pass
