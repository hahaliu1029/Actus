"""安全工具模块：JWT 签发验证、密码哈希"""

from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import bcrypt
from core.config import get_settings
from jose import JWTError, jwt


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """验证密码是否正确

    Args:
        plain_password: 明文密码
        hashed_password: 哈希后的密码

    Returns:
        bool: 密码是否匹配
    """
    # bcrypt 限制密码最大长度为 72 字节，需要与哈希时保持一致
    password_bytes = plain_password.encode("utf-8")[:72]
    try:
        return bcrypt.checkpw(password_bytes, hashed_password.encode("utf-8"))
    except (ValueError, TypeError):
        # 兼容历史异常哈希数据，统一按密码不匹配处理，避免抛 500
        return False


def get_password_hash(password: str) -> str:
    """生成密码哈希

    Args:
        password: 明文密码

    Returns:
        str: 哈希后的密码
    """
    # bcrypt 限制密码最大长度为 72 字节，超过需要截断
    password_bytes = password.encode("utf-8")[:72]
    return bcrypt.hashpw(password_bytes, bcrypt.gensalt()).decode("utf-8")


def create_access_token(
    data: dict[str, Any],
    expires_delta: Optional[timedelta] = None,
) -> str:
    """创建访问令牌

    Args:
        data: 要编码到 token 中的数据
        expires_delta: 过期时间间隔，默认使用配置中的值

    Returns:
        str: JWT access token
    """
    settings = get_settings()
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(
            minutes=settings.jwt_access_token_expire_minutes
        )

    to_encode.update({"exp": expire, "type": "access"})
    encoded_jwt = jwt.encode(
        to_encode,
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm,
    )
    return encoded_jwt


def create_refresh_token(
    data: dict[str, Any],
    expires_delta: Optional[timedelta] = None,
) -> str:
    """创建刷新令牌

    Args:
        data: 要编码到 token 中的数据
        expires_delta: 过期时间间隔，默认使用配置中的值

    Returns:
        str: JWT refresh token
    """
    settings = get_settings()
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(
            days=settings.jwt_refresh_token_expire_days
        )

    to_encode.update({"exp": expire, "type": "refresh"})
    encoded_jwt = jwt.encode(
        to_encode,
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm,
    )
    return encoded_jwt


def decode_token(token: str) -> Optional[dict[str, Any]]:
    """解码并验证 JWT token

    Args:
        token: JWT token 字符串

    Returns:
        Optional[dict]: 解码后的 payload，验证失败返回 None
    """
    settings = get_settings()
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
        )
        return payload
    except JWTError:
        return None


def create_tokens(user_id: str, username: str, role: str) -> dict[str, str]:
    """创建 access_token 和 refresh_token

    Args:
        user_id: 用户 ID
        username: 用户名
        role: 用户角色

    Returns:
        dict: 包含 access_token 和 refresh_token
    """
    token_data = {
        "sub": user_id,
        "username": username,
        "role": role,
    }
    return {
        "access_token": create_access_token(token_data),
        "refresh_token": create_refresh_token(token_data),
        "token_type": "bearer",
    }
