#!/usr/bin/env python3
"""
创建超级管理员账户的CLI脚本

使用方法:
    python scripts/create_super_admin.py

脚本会交互式提示输入用户名、邮箱和密码，然后创建超级管理员账户。
"""

import asyncio
import getpass
import re
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.domain.models.user import User, UserRole, UserStatus
from app.infrastructure.models.user import UserModel
from core.config import settings
from core.security import get_password_hash
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker


def validate_email(email: str) -> bool:
    """验证邮箱格式"""
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return bool(re.match(pattern, email))


def validate_username(username: str) -> bool:
    """验证用户名格式 (至少3个字符，只允许字母数字和下划线)"""
    pattern = r"^[a-zA-Z0-9_]{3,50}$"
    return bool(re.match(pattern, username))


def validate_password(password: str) -> bool:
    """验证密码强度 (至少8个字符)"""
    return len(password) >= 8


async def check_super_admin_exists(session: AsyncSession) -> bool:
    """检查是否已存在超级管理员"""
    stmt = select(UserModel).where(UserModel.role == UserRole.SUPER_ADMIN.value)
    result = await session.execute(stmt)
    return result.scalar_one_or_none() is not None


async def check_user_exists(
    session: AsyncSession, username: str = None, email: str = None
) -> bool:
    """检查用户名或邮箱是否已存在"""
    if username:
        stmt = select(UserModel).where(UserModel.username == username)
        result = await session.execute(stmt)
        if result.scalar_one_or_none():
            return True
    if email:
        stmt = select(UserModel).where(UserModel.email == email)
        result = await session.execute(stmt)
        if result.scalar_one_or_none():
            return True
    return False


async def create_super_admin(
    session: AsyncSession, username: str, email: str, password: str
) -> User:
    """创建超级管理员账户"""
    user = User(
        username=username,
        email=email,
        password_hash=get_password_hash(password),
        role=UserRole.SUPER_ADMIN,
        status=UserStatus.ACTIVE,
    )
    user_model = UserModel.from_domain(user)
    session.add(user_model)
    await session.commit()
    await session.refresh(user_model)
    return user_model.to_domain()


async def main():
    print("=" * 50)
    print("  创建超级管理员账户")
    print("=" * 50)
    print()

    # 创建数据库连接
    engine = create_async_engine(settings.sqlalchemy_database_url, echo=False)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as session:
        # 检查是否已存在超级管理员
        if await check_super_admin_exists(session):
            print("❌ 错误: 系统中已存在超级管理员账户")
            print("   如需重新创建，请先删除现有的超级管理员账户")
            return 1

        # 输入用户名
        while True:
            username = input("请输入用户名 (3-50个字符，仅限字母数字下划线): ").strip()
            if not validate_username(username):
                print("❌ 用户名格式不正确，请重新输入")
                continue
            if await check_user_exists(session, username=username):
                print("❌ 该用户名已被使用，请选择其他用户名")
                continue
            break

        # 输入邮箱 (可选)
        while True:
            email = input("请输入邮箱 (可选，直接回车跳过): ").strip()
            if not email:
                email = None
                break
            if not validate_email(email):
                print("❌ 邮箱格式不正确，请重新输入")
                continue
            if await check_user_exists(session, email=email):
                print("❌ 该邮箱已被使用，请使用其他邮箱")
                continue
            break

        # 输入密码
        while True:
            password = getpass.getpass("请输入密码 (至少8个字符): ")
            if not validate_password(password):
                print("❌ 密码长度至少8个字符，请重新输入")
                continue
            password_confirm = getpass.getpass("请再次输入密码确认: ")
            if password != password_confirm:
                print("❌ 两次输入的密码不一致，请重新输入")
                continue
            break

        # 确认创建
        print()
        print("-" * 50)
        print(f"用户名: {username}")
        print(f"邮箱: {email or '(未设置)'}")
        print(f"角色: 超级管理员 (super_admin)")
        print("-" * 50)

        confirm = input("确认创建? (y/N): ").strip().lower()
        if confirm != "y":
            print("已取消创建")
            return 0

        # 创建超级管理员
        try:
            user = await create_super_admin(session, username, email, password)
            print()
            print("✅ 超级管理员账户创建成功!")
            print(f"   用户ID: {user.id}")
            print(f"   用户名: {user.username}")
            return 0
        except Exception as e:
            print(f"❌ 创建失败: {e}")
            return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
