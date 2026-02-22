"""认证服务"""

import logging
from typing import Optional

from app.domain.models.user import User, UserRole, UserStatus
from app.domain.repositories.user_repository import UserRepository
from core.security import (
    create_tokens,
    decode_token,
    get_password_hash,
    verify_password,
)

logger = logging.getLogger(__name__)


class AuthService:
    """认证服务，处理用户注册、登录、token 刷新等"""

    def __init__(self, user_repository: UserRepository) -> None:
        self.user_repository = user_repository

    async def register(
        self,
        username: Optional[str] = None,
        email: Optional[str] = None,
        password: str = "",
        nickname: Optional[str] = None,
    ) -> tuple[User, dict[str, str]]:
        """用户注册

        Args:
            username: 用户名（与 email 至少提供一个）
            email: 邮箱（与 username 至少提供一个）
            password: 密码
            nickname: 昵称

        Returns:
            tuple: (用户对象, tokens 字典)

        Raises:
            ValueError: 参数校验失败
        """
        # 参数校验
        if not username and not email:
            raise ValueError("用户名或邮箱至少提供一个")
        if not password:
            raise ValueError("密码不能为空")
        if len(password) < 6:
            raise ValueError("密码长度至少6位")

        # 检查用户名是否已存在
        if username:
            existing = await self.user_repository.get_by_username(username)
            if existing:
                raise ValueError("用户名已被使用")

        # 检查邮箱是否已存在
        if email:
            existing = await self.user_repository.get_by_email(email)
            if existing:
                raise ValueError("邮箱已被注册")

        # 创建用户
        user = User(
            username=username,
            email=email,
            password_hash=get_password_hash(password),
            nickname=nickname or username or email.split("@")[0] if email else None,
            role=UserRole.USER,
            status=UserStatus.ACTIVE,
        )

        created_user = await self.user_repository.create(user)
        tokens = create_tokens(
            created_user.id, created_user.username or "", created_user.role.value
        )

        logger.info(f"User registered: {created_user.id}")
        return created_user, tokens

    async def login(
        self,
        username: Optional[str] = None,
        email: Optional[str] = None,
        password: str = "",
    ) -> tuple[User, dict[str, str]]:
        """用户登录

        Args:
            username: 用户名
            email: 邮箱
            password: 密码

        Returns:
            tuple: (用户对象, tokens 字典)

        Raises:
            ValueError: 登录失败
        """
        # 根据用户名或邮箱查找用户
        user = None
        if username:
            user = await self.user_repository.get_by_username(username)
        elif email:
            user = await self.user_repository.get_by_email(email)
        else:
            raise ValueError("请提供用户名或邮箱")

        if not user:
            raise ValueError("用户不存在")

        if not user.password_hash:
            raise ValueError("该账户未设置密码，请使用第三方登录")

        if not verify_password(password, user.password_hash):
            raise ValueError("密码错误")

        if not user.is_active():
            raise ValueError("账户已被禁用")

        tokens = create_tokens(user.id, user.username or "", user.role.value)
        logger.info(f"User logged in: {user.id}")
        return user, tokens

    async def refresh_token(self, refresh_token: str) -> dict[str, str]:
        """刷新 access token

        Args:
            refresh_token: 刷新令牌

        Returns:
            dict: 新的 tokens

        Raises:
            ValueError: token 无效
        """
        payload = decode_token(refresh_token)
        if not payload:
            raise ValueError("无效的刷新令牌")

        if payload.get("type") != "refresh":
            raise ValueError("无效的令牌类型")

        user_id = payload.get("sub")
        if not user_id:
            raise ValueError("无效的令牌")

        user = await self.user_repository.get_by_id(user_id)
        if not user:
            raise ValueError("用户不存在")

        if not user.is_active():
            raise ValueError("账户已被禁用")

        tokens = create_tokens(user.id, user.username or "", user.role.value)
        logger.info(f"Token refreshed for user: {user.id}")
        return tokens

    async def get_current_user(self, access_token: str) -> User:
        """根据 access token 获取当前用户

        Args:
            access_token: 访问令牌

        Returns:
            User: 用户对象

        Raises:
            ValueError: token 无效或用户不存在
        """
        payload = decode_token(access_token)
        if not payload:
            raise ValueError("无效的访问令牌")

        if payload.get("type") != "access":
            raise ValueError("无效的令牌类型")

        user_id = payload.get("sub")
        if not user_id:
            raise ValueError("无效的令牌")

        user = await self.user_repository.get_by_id(user_id)
        if not user:
            raise ValueError("用户不存在")

        if not user.is_active():
            raise ValueError("账户已被禁用")

        return user
