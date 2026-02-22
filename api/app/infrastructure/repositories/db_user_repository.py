"""用户仓储实现"""

from typing import Optional

from app.domain.models.user import User
from app.domain.repositories.user_repository import UserRepository
from app.infrastructure.models.user import UserModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession


class DBUserRepository(UserRepository):
    """基于数据库的用户仓储实现"""

    def __init__(self, db_session: AsyncSession) -> None:
        """构造函数，完成数据仓储初始化"""
        self.db_session = db_session

    async def create(self, user: User) -> User:
        """创建用户"""
        record = UserModel.from_domain(user)
        self.db_session.add(record)
        await self.db_session.flush()
        return record.to_domain()

    async def get_by_id(self, user_id: str) -> Optional[User]:
        """根据 ID 获取用户"""
        stmt = select(UserModel).where(UserModel.id == user_id)
        result = await self.db_session.execute(stmt)
        record = result.scalar_one_or_none()
        return record.to_domain() if record else None

    async def get_by_username(self, username: str) -> Optional[User]:
        """根据用户名获取用户"""
        stmt = select(UserModel).where(UserModel.username == username)
        result = await self.db_session.execute(stmt)
        record = result.scalar_one_or_none()
        return record.to_domain() if record else None

    async def get_by_email(self, email: str) -> Optional[User]:
        """根据邮箱获取用户"""
        stmt = select(UserModel).where(UserModel.email == email)
        result = await self.db_session.execute(stmt)
        record = result.scalar_one_or_none()
        return record.to_domain() if record else None

    async def get_by_phone(self, phone: str) -> Optional[User]:
        """根据手机号获取用户"""
        stmt = select(UserModel).where(UserModel.phone == phone)
        result = await self.db_session.execute(stmt)
        record = result.scalar_one_or_none()
        return record.to_domain() if record else None

    async def update(self, user: User) -> User:
        """更新用户"""
        stmt = select(UserModel).where(UserModel.id == user.id)
        result = await self.db_session.execute(stmt)
        record = result.scalar_one_or_none()

        if not record:
            raise ValueError(f"User with id {user.id} not found")

        record.update_from_domain(user)
        await self.db_session.flush()
        return record.to_domain()

    async def delete(self, user_id: str) -> bool:
        """删除用户"""
        stmt = select(UserModel).where(UserModel.id == user_id)
        result = await self.db_session.execute(stmt)
        record = result.scalar_one_or_none()

        if record:
            await self.db_session.delete(record)
            return True
        return False

    async def list_all(self, skip: int = 0, limit: int = 100) -> list[User]:
        """获取用户列表"""
        stmt = (
            select(UserModel)
            .offset(skip)
            .limit(limit)
            .order_by(UserModel.created_at.desc())
        )
        result = await self.db_session.execute(stmt)
        records = result.scalars().all()
        return [record.to_domain() for record in records]

    async def count(self) -> int:
        """获取用户总数"""
        stmt = select(func.count()).select_from(UserModel)
        result = await self.db_session.execute(stmt)
        return result.scalar() or 0

    async def exists_by_role(self, role: str) -> bool:
        """检查是否存在指定角色的用户"""
        stmt = select(func.count()).select_from(UserModel).where(UserModel.role == role)
        result = await self.db_session.execute(stmt)
        count = result.scalar() or 0
        return count > 0
