"""用户领域模型"""

from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, EmailStr, Field


class UserRole(str, Enum):
    """用户角色枚举"""

    SUPER_ADMIN = "super_admin"
    USER = "user"


class UserStatus(str, Enum):
    """用户状态枚举"""

    ACTIVE = "active"
    INACTIVE = "inactive"
    BANNED = "banned"


class User(BaseModel):
    """用户领域模型"""

    id: str = Field(default_factory=lambda: str(uuid4()))
    username: Optional[str] = None
    email: Optional[EmailStr] = None
    phone: Optional[str] = None  # 预留手机号字段
    password_hash: Optional[str] = None
    nickname: Optional[str] = None
    avatar: Optional[str] = None
    role: UserRole = UserRole.USER
    status: UserStatus = UserStatus.ACTIVE
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    class Config:
        from_attributes = True

    def is_admin(self) -> bool:
        """检查用户是否为超级管理员"""
        return self.role == UserRole.SUPER_ADMIN

    def is_active(self) -> bool:
        """检查用户是否处于活跃状态"""
        return self.status == UserStatus.ACTIVE
