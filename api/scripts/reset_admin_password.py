#!/usr/bin/env python3
"""
重置超级管理员密码的 CLI 脚本

用法示例:
    python scripts/reset_admin_password.py --username admin
    python scripts/reset_admin_password.py --email admin@example.com

说明:
1. 必须指定 --username 或 --email 二选一。
2. 仅允许重置 role=super_admin 的账户密码。
3. 密码最少 8 位，会进行二次确认。
"""

import argparse
import asyncio
import getpass
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.domain.models.user import UserRole
from app.infrastructure.models.user import UserModel
from core.config import settings
from core.security import get_password_hash
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="重置超级管理员密码")
    parser.add_argument("--username", type=str, help="管理员用户名")
    parser.add_argument("--email", type=str, help="管理员邮箱")
    parser.add_argument(
        "--password",
        type=str,
        help="新密码（不建议明文传参，留空会交互输入）",
    )
    return parser.parse_args()


def validate_password(password: str) -> bool:
    return len(password) >= 8


def read_new_password(cli_password: str | None) -> str:
    if cli_password:
        if not validate_password(cli_password):
            raise ValueError("密码长度至少 8 位")
        return cli_password

    password = getpass.getpass("请输入新密码 (至少8位): ")
    if not validate_password(password):
        raise ValueError("密码长度至少 8 位")

    password_confirm = getpass.getpass("请再次输入新密码确认: ")
    if password != password_confirm:
        raise ValueError("两次输入的密码不一致")

    return password


async def find_admin_user(
    session: AsyncSession, username: str | None, email: str | None
) -> UserModel | None:
    stmt = select(UserModel).where(UserModel.role == UserRole.SUPER_ADMIN.value)

    if username:
        stmt = stmt.where(UserModel.username == username)
    elif email:
        stmt = stmt.where(UserModel.email == email)
    else:
        return None

    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def main() -> int:
    args = parse_args()
    if not args.username and not args.email:
        print("❌ 请至少提供 --username 或 --email 之一")
        return 1

    if args.username and args.email:
        print("❌ 请只提供一个标识参数：--username 或 --email")
        return 1

    try:
        new_password = read_new_password(args.password)
    except ValueError as exc:
        print(f"❌ {exc}")
        return 1

    engine = create_async_engine(settings.sqlalchemy_database_url, echo=False)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as session:
        user = await find_admin_user(session, args.username, args.email)
        if not user:
            print("❌ 未找到匹配的超级管理员账户")
            return 1

        user.password_hash = get_password_hash(new_password)
        await session.commit()

        print("✅ 超级管理员密码已重置")
        print(f"   用户ID: {user.id}")
        print(f"   用户名: {user.username or '(未设置)'}")
        print(f"   邮箱: {user.email or '(未设置)'}")
        return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
