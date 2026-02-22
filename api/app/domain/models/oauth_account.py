"""OAuth 账户领域模型"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class OAuthProvider(str, Enum):
    """OAuth 提供商枚举"""

    WECHAT = "wechat"
    # 可扩展其他提供商
    # GITHUB = "github"
    # GOOGLE = "google"


class OAuthAccount(BaseModel):
    """OAuth 账户领域模型"""

    id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    provider: OAuthProvider
    provider_user_id: str  # openid
    unionid: Optional[str] = None  # 微信 unionid
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    expires_at: Optional[datetime] = None
    raw_data: Optional[dict[str, Any]] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    class Config:
        from_attributes = True
