"""认证相关 Schema"""

from typing import Optional

from pydantic import BaseModel, EmailStr, Field

# ============ 请求 Schema ============


class RegisterRequest(BaseModel):
    """用户注册请求"""

    username: Optional[str] = Field(
        None, min_length=3, max_length=64, description="用户名"
    )
    email: Optional[EmailStr] = Field(None, description="邮箱")
    password: str = Field(..., min_length=6, max_length=128, description="密码")
    nickname: Optional[str] = Field(None, max_length=64, description="昵称")

    class Config:
        json_schema_extra = {
            "example": {
                "username": "testuser",
                "email": "test@example.com",
                "password": "password123",
                "nickname": "Test User",
            }
        }


class LoginRequest(BaseModel):
    """用户登录请求"""

    username: Optional[str] = Field(None, description="用户名")
    email: Optional[EmailStr] = Field(None, description="邮箱")
    password: str = Field(..., description="密码")

    class Config:
        json_schema_extra = {
            "example": {"username": "testuser", "password": "password123"}
        }


class RefreshTokenRequest(BaseModel):
    """刷新令牌请求"""

    refresh_token: str = Field(..., description="刷新令牌")


class UpdateUserRequest(BaseModel):
    """更新用户信息请求"""

    nickname: Optional[str] = Field(None, max_length=64, description="昵称")
    avatar: Optional[str] = Field(None, max_length=512, description="头像 URL")


# ============ 响应 Schema ============


class TokenResponse(BaseModel):
    """令牌响应"""

    access_token: str = Field(..., description="访问令牌")
    refresh_token: str = Field(..., description="刷新令牌")
    token_type: str = Field(default="bearer", description="令牌类型")


class UserResponse(BaseModel):
    """用户信息响应"""

    id: str = Field(..., description="用户 ID")
    username: Optional[str] = Field(None, description="用户名")
    email: Optional[str] = Field(None, description="邮箱")
    nickname: Optional[str] = Field(None, description="昵称")
    avatar: Optional[str] = Field(None, description="头像 URL")
    role: str = Field(..., description="角色")
    status: str = Field(..., description="状态")
    created_at: str = Field(..., description="创建时间")

    class Config:
        from_attributes = True


class LoginResponse(BaseModel):
    """登录响应"""

    user: UserResponse
    tokens: TokenResponse
