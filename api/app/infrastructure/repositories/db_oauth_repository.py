"""OAuth 账户仓储实现"""

from typing import Optional

from app.domain.models.oauth_account import OAuthAccount, OAuthProvider
from app.domain.repositories.oauth_repository import OAuthRepository
from app.infrastructure.models.oauth_account import OAuthAccountModel
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession


class DBOAuthRepository(OAuthRepository):
    """基于数据库的 OAuth 账户仓储实现"""

    def __init__(self, db_session: AsyncSession) -> None:
        """构造函数，完成数据仓储初始化"""
        self.db_session = db_session

    async def create(self, oauth_account: OAuthAccount) -> OAuthAccount:
        """创建 OAuth 账户"""
        record = OAuthAccountModel.from_domain(oauth_account)
        self.db_session.add(record)
        await self.db_session.flush()
        return record.to_domain()

    async def get_by_id(self, oauth_id: str) -> Optional[OAuthAccount]:
        """根据 ID 获取 OAuth 账户"""
        stmt = select(OAuthAccountModel).where(OAuthAccountModel.id == oauth_id)
        result = await self.db_session.execute(stmt)
        record = result.scalar_one_or_none()
        return record.to_domain() if record else None

    async def get_by_provider_user_id(
        self, provider: OAuthProvider, provider_user_id: str
    ) -> Optional[OAuthAccount]:
        """根据提供商和提供商用户 ID 获取 OAuth 账户"""
        stmt = select(OAuthAccountModel).where(
            and_(
                OAuthAccountModel.provider == provider.value,
                OAuthAccountModel.provider_user_id == provider_user_id,
            )
        )
        result = await self.db_session.execute(stmt)
        record = result.scalar_one_or_none()
        return record.to_domain() if record else None

    async def get_by_user_id(self, user_id: str) -> list[OAuthAccount]:
        """根据用户 ID 获取所有 OAuth 账户"""
        stmt = select(OAuthAccountModel).where(OAuthAccountModel.user_id == user_id)
        result = await self.db_session.execute(stmt)
        records = result.scalars().all()
        return [record.to_domain() for record in records]

    async def update(self, oauth_account: OAuthAccount) -> OAuthAccount:
        """更新 OAuth 账户"""
        stmt = select(OAuthAccountModel).where(OAuthAccountModel.id == oauth_account.id)
        result = await self.db_session.execute(stmt)
        record = result.scalar_one_or_none()

        if not record:
            raise ValueError(f"OAuthAccount with id {oauth_account.id} not found")

        record.update_from_domain(oauth_account)
        await self.db_session.flush()
        return record.to_domain()

    async def delete(self, oauth_id: str) -> bool:
        """删除 OAuth 账户"""
        stmt = select(OAuthAccountModel).where(OAuthAccountModel.id == oauth_id)
        result = await self.db_session.execute(stmt)
        record = result.scalar_one_or_none()

        if record:
            await self.db_session.delete(record)
            return True
        return False
