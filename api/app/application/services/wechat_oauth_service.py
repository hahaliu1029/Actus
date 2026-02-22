"""微信 OAuth 服务"""

import logging
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from urllib.parse import urlencode

import httpx
from app.domain.models.oauth_account import OAuthAccount, OAuthProvider
from app.domain.models.user import User, UserRole, UserStatus
from app.domain.repositories.oauth_repository import OAuthRepository
from app.domain.repositories.user_repository import UserRepository
from core.config import get_settings
from core.security import create_tokens

logger = logging.getLogger(__name__)


class WeChatOAuthService:
    """微信公众号 OAuth 服务

    处理微信公众号网页授权流程：
    1. 生成授权 URL
    2. 处理回调，用 code 换取 access_token
    3. 获取用户信息
    4. 自动注册或登录
    """

    # 微信 OAuth 相关 URL
    AUTHORIZE_URL = "https://open.weixin.qq.com/connect/oauth2/authorize"
    ACCESS_TOKEN_URL = "https://api.weixin.qq.com/sns/oauth2/access_token"
    USERINFO_URL = "https://api.weixin.qq.com/sns/userinfo"
    REFRESH_TOKEN_URL = "https://api.weixin.qq.com/sns/oauth2/refresh_token"

    def __init__(
        self,
        user_repository: UserRepository,
        oauth_repository: OAuthRepository,
    ) -> None:
        self.user_repository = user_repository
        self.oauth_repository = oauth_repository
        self.settings = get_settings()

    def get_authorize_url(
        self, state: Optional[str] = None, scope: str = "snsapi_userinfo"
    ) -> str:
        """生成微信授权 URL

        Args:
            state: 防 CSRF 状态参数，不传则自动生成
            scope: 授权作用域
                - snsapi_base: 静默授权，只能获取 openid
                - snsapi_userinfo: 需用户确认，可获取用户信息

        Returns:
            str: 微信授权 URL
        """
        if not state:
            state = secrets.token_urlsafe(16)

        params = {
            "appid": self.settings.wechat_app_id,
            "redirect_uri": self.settings.wechat_redirect_uri,
            "response_type": "code",
            "scope": scope,
            "state": state,
        }

        url = f"{self.AUTHORIZE_URL}?{urlencode(params)}#wechat_redirect"
        return url

    async def get_access_token(self, code: str) -> dict[str, Any]:
        """用授权码换取 access_token

        Args:
            code: 微信回调带的授权码

        Returns:
            dict: 包含 access_token, openid, unionid 等信息

        Raises:
            ValueError: 获取失败
        """
        params = {
            "appid": self.settings.wechat_app_id,
            "secret": self.settings.wechat_app_secret,
            "code": code,
            "grant_type": "authorization_code",
        }

        async with httpx.AsyncClient() as client:
            response = await client.get(self.ACCESS_TOKEN_URL, params=params)
            data = response.json()

        if "errcode" in data:
            logger.error(f"WeChat get access_token failed: {data}")
            raise ValueError(f"微信授权失败: {data.get('errmsg', '未知错误')}")

        return data

    async def get_user_info(self, access_token: str, openid: str) -> dict[str, Any]:
        """获取微信用户信息

        Args:
            access_token: 访问令牌
            openid: 用户 openid

        Returns:
            dict: 用户信息，包含 nickname, headimgurl 等

        Raises:
            ValueError: 获取失败
        """
        params = {
            "access_token": access_token,
            "openid": openid,
            "lang": "zh_CN",
        }

        async with httpx.AsyncClient() as client:
            response = await client.get(self.USERINFO_URL, params=params)
            data = response.json()

        if "errcode" in data:
            logger.error(f"WeChat get user info failed: {data}")
            raise ValueError(f"获取用户信息失败: {data.get('errmsg', '未知错误')}")

        return data

    async def handle_callback(
        self, code: str, state: str
    ) -> tuple[User, dict[str, str]]:
        """处理微信授权回调

        1. 用 code 换取 access_token
        2. 获取用户信息
        3. 查找或创建用户
        4. 返回用户信息和 JWT tokens

        Args:
            code: 授权码
            state: 状态参数（可用于验证）

        Returns:
            tuple: (用户对象, tokens 字典)

        Raises:
            ValueError: 授权失败
        """
        # 1. 获取 access_token
        token_data = await self.get_access_token(code)
        access_token = token_data["access_token"]
        openid = token_data["openid"]
        unionid = token_data.get("unionid")
        refresh_token = token_data.get("refresh_token")
        expires_in = token_data.get("expires_in", 7200)

        # 2. 查找已绑定的 OAuth 账户
        oauth_account = await self.oauth_repository.get_by_provider_user_id(
            OAuthProvider.WECHAT, openid
        )

        if oauth_account:
            # 已存在，更新 token 并返回用户
            oauth_account.access_token = access_token
            oauth_account.refresh_token = refresh_token
            oauth_account.expires_at = datetime.now(timezone.utc) + timedelta(
                seconds=expires_in
            )
            await self.oauth_repository.update(oauth_account)

            user = await self.user_repository.get_by_id(oauth_account.user_id)
            if not user:
                raise ValueError("关联用户不存在")

            if not user.is_active():
                raise ValueError("账户已被禁用")

            tokens = create_tokens(user.id, user.username or "", user.role.value)
            logger.info(f"WeChat user logged in: {user.id}")
            return user, tokens

        # 3. 新用户，获取用户信息
        user_info = await self.get_user_info(access_token, openid)

        # 4. 创建新用户
        nickname = user_info.get("nickname", f"微信用户_{openid[-6:]}")
        avatar = user_info.get("headimgurl", "")

        user = User(
            nickname=nickname,
            avatar=avatar,
            role=UserRole.USER,
            status=UserStatus.ACTIVE,
        )
        created_user = await self.user_repository.create(user)

        # 5. 创建 OAuth 账户绑定
        oauth_account = OAuthAccount(
            user_id=created_user.id,
            provider=OAuthProvider.WECHAT,
            provider_user_id=openid,
            unionid=unionid,
            access_token=access_token,
            refresh_token=refresh_token,
            expires_at=datetime.now(timezone.utc) + timedelta(seconds=expires_in),
            raw_data=user_info,
        )
        await self.oauth_repository.create(oauth_account)

        tokens = create_tokens(
            created_user.id, created_user.username or "", created_user.role.value
        )
        logger.info(f"WeChat user registered: {created_user.id}")
        return created_user, tokens

    def get_frontend_redirect_url(self, tokens: dict[str, str]) -> str:
        """生成前端重定向 URL

        Args:
            tokens: JWT tokens

        Returns:
            str: 带 token 参数的前端 URL
        """
        base_url = self.settings.wechat_frontend_redirect_uri
        params = {
            "access_token": tokens["access_token"],
            "refresh_token": tokens["refresh_token"],
        }
        return f"{base_url}?{urlencode(params)}"
