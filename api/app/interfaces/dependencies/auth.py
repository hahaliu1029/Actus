"""认证依赖模块"""

import asyncio
from collections.abc import AsyncGenerator
from typing import Annotated

from app.application.services.auth_service import AuthService
from app.domain.models.user import User, UserRole
from app.domain.repositories.user_repository import UserRepository
from app.infrastructure.repositories.db_user_repository import DBUserRepository
from app.infrastructure.storage.postgres import get_postgres
from core.security import decode_token
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

# HTTP Bearer 认证方案
security = HTTPBearer(auto_error=False)


async def get_user_repository() -> AsyncGenerator[UserRepository, None]:
    """获取用户仓储实例"""
    postgres = get_postgres()
    session = postgres.session_factory()
    try:
        yield DBUserRepository(session)
    except asyncio.CancelledError:
        try:
            await asyncio.shield(session.rollback())
        except Exception:
            pass
        raise
    except Exception:
        await session.rollback()
        raise
    finally:
        try:
            await asyncio.shield(session.close())
        except Exception:
            pass


async def get_auth_service(
    user_repository: UserRepository = Depends(get_user_repository),
) -> AuthService:
    """获取认证服务实例"""
    return AuthService(user_repository)


async def resolve_user_from_access_token(token: str) -> User:
    """解析 access token 并返回登录用户"""
    payload = decode_token(token)

    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的访问令牌",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if payload.get("type") != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的令牌类型",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的令牌",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # 从数据库获取用户
    postgres = get_postgres()
    async with postgres.session_factory() as session:
        from app.infrastructure.repositories.db_user_repository import DBUserRepository

        user_repo = DBUserRepository(session)
        user = await user_repo.get_by_id(user_id)

        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="用户不存在",
                headers={"WWW-Authenticate": "Bearer"},
            )

        if not user.is_active():
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="账户已被禁用",
            )

        return user


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(security)],
) -> User:
    """获取当前登录用户

    从 Authorization 头中提取 Bearer token，解析并验证

    Raises:
        HTTPException: 401 未授权
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="未提供认证信息",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials
    return await resolve_user_from_access_token(token)


async def get_current_user_ws_query(token: str | None) -> User:
    """WebSocket Query 参数鉴权"""
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="未提供认证信息",
        )
    return await resolve_user_from_access_token(token)


async def get_current_user_optional(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(security)],
) -> User | None:
    """获取当前登录用户（可选）

    如果未提供认证信息，返回 None
    """
    if not credentials:
        return None

    try:
        return await get_current_user(credentials)
    except HTTPException:
        return None


async def require_admin(
    current_user: Annotated[User, Depends(get_current_user)],
) -> User:
    """要求超级管理员权限

    Raises:
        HTTPException: 403 权限不足
    """
    if not current_user.is_admin():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="需要管理员权限",
        )
    return current_user


# 类型别名，方便使用
CurrentUser = Annotated[User, Depends(get_current_user)]
OptionalUser = Annotated[User | None, Depends(get_current_user_optional)]
AdminUser = Annotated[User, Depends(require_admin)]
