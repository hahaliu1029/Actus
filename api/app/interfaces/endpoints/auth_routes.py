"""认证路由模块"""

import logging
from collections.abc import AsyncGenerator

from app.application.services.auth_service import AuthService
from app.domain.models.user import User
from app.domain.repositories.user_repository import UserRepository
from app.infrastructure.repositories.db_user_repository import DBUserRepository
from app.infrastructure.storage.postgres import get_postgres
from app.interfaces.dependencies import CurrentUser
from app.interfaces.schemas import Response
from app.interfaces.schemas.auth import (
    LoginRequest,
    LoginResponse,
    RefreshTokenRequest,
    RegisterRequest,
    TokenResponse,
    UpdateUserRequest,
    UserResponse,
)
from fastapi import APIRouter, Depends, HTTPException, status

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/auth", tags=["认证模块"])


async def get_auth_service_dep() -> AsyncGenerator[AuthService, None]:
    """获取认证服务依赖"""
    postgres = get_postgres()
    async with postgres.session_factory() as session:
        user_repo = DBUserRepository(session)
        yield AuthService(user_repo)


def user_to_response(user: User) -> UserResponse:
    """将用户领域模型转换为响应模型"""
    return UserResponse(
        id=user.id,
        username=user.username,
        email=user.email,
        nickname=user.nickname,
        avatar=user.avatar,
        role=user.role.value,
        status=user.status.value,
        created_at=user.created_at.isoformat(),
    )


@router.post(
    "/register",
    response_model=Response[LoginResponse],
    summary="用户注册",
    description="通过用户名或邮箱注册新账户",
)
async def register(
    request: RegisterRequest,
) -> Response:
    """用户注册"""
    postgres = get_postgres()
    async with postgres.session_factory() as session:
        user_repo = DBUserRepository(session)
        auth_service = AuthService(user_repo)

        try:
            user, tokens = await auth_service.register(
                username=request.username,
                email=request.email,
                password=request.password,
                nickname=request.nickname,
            )
            await session.commit()

            return Response.success(
                data=LoginResponse(
                    user=user_to_response(user),
                    tokens=TokenResponse(**tokens),
                ),
                msg="注册成功",
            )
        except ValueError as e:
            return Response.fail(code=400, msg=str(e))
        except Exception as e:
            logger.exception("Register failed")
            return Response.fail(code=500, msg="注册失败")


@router.post(
    "/login",
    response_model=Response[LoginResponse],
    summary="用户登录",
    description="通过用户名或邮箱登录",
)
async def login(
    request: LoginRequest,
) -> Response:
    """用户登录"""
    postgres = get_postgres()
    async with postgres.session_factory() as session:
        user_repo = DBUserRepository(session)
        auth_service = AuthService(user_repo)

        try:
            user, tokens = await auth_service.login(
                username=request.username,
                email=request.email,
                password=request.password,
            )

            return Response.success(
                data=LoginResponse(
                    user=user_to_response(user),
                    tokens=TokenResponse(**tokens),
                ),
                msg="登录成功",
            )
        except ValueError as e:
            return Response.fail(code=401, msg=str(e))
        except Exception as e:
            logger.exception("Login failed")
            return Response.fail(code=500, msg="登录失败")


@router.post(
    "/refresh",
    response_model=Response[TokenResponse],
    summary="刷新令牌",
    description="使用 refresh_token 获取新的 access_token",
)
async def refresh_token(
    request: RefreshTokenRequest,
) -> Response:
    """刷新令牌"""
    postgres = get_postgres()
    async with postgres.session_factory() as session:
        user_repo = DBUserRepository(session)
        auth_service = AuthService(user_repo)

        try:
            tokens = await auth_service.refresh_token(request.refresh_token)
            return Response.success(
                data=TokenResponse(**tokens),
                msg="刷新成功",
            )
        except ValueError as e:
            return Response.fail(code=401, msg=str(e))
        except Exception as e:
            logger.exception("Refresh token failed")
            return Response.fail(code=500, msg="刷新失败")


@router.get(
    "/me",
    response_model=Response[UserResponse],
    summary="获取当前用户信息",
    description="获取当前登录用户的详细信息",
)
async def get_me(
    current_user: CurrentUser,
) -> Response:
    """获取当前用户信息"""
    return Response.success(
        data=user_to_response(current_user),
        msg="获取成功",
    )


@router.put(
    "/me",
    response_model=Response[UserResponse],
    summary="更新当前用户信息",
    description="更新当前登录用户的昵称、头像等信息",
)
async def update_me(
    request: UpdateUserRequest,
    current_user: CurrentUser,
) -> Response:
    """更新当前用户信息"""
    postgres = get_postgres()
    async with postgres.session_factory() as session:
        user_repo = DBUserRepository(session)

        # 更新用户信息
        if request.nickname is not None:
            current_user.nickname = request.nickname
        if request.avatar is not None:
            current_user.avatar = request.avatar

        try:
            updated_user = await user_repo.update(current_user)
            await session.commit()

            return Response.success(
                data=user_to_response(updated_user),
                msg="更新成功",
            )
        except Exception as e:
            logger.exception("Update user failed")
            return Response.fail(code=500, msg="更新失败")


# ============ 微信登录路由 ============

from app.application.services.wechat_oauth_service import WeChatOAuthService
from app.infrastructure.repositories.db_oauth_repository import DBOAuthRepository
from core.config import get_settings
from fastapi.responses import RedirectResponse


@router.get(
    "/wechat/authorize",
    response_model=Response,
    summary="获取微信授权 URL",
    description="获取微信公众号网页授权 URL，前端跳转此 URL 让用户扫码授权",
)
async def wechat_authorize(
    state: str | None = None,
    scope: str = "snsapi_userinfo",
) -> Response:
    """获取微信授权 URL"""
    settings = get_settings()

    if not settings.wechat_app_id or not settings.wechat_redirect_uri:
        return Response.fail(code=500, msg="微信登录未配置")

    postgres = get_postgres()
    async with postgres.session_factory() as session:
        user_repo = DBUserRepository(session)
        oauth_repo = DBOAuthRepository(session)
        wechat_service = WeChatOAuthService(user_repo, oauth_repo)

        authorize_url = wechat_service.get_authorize_url(state=state, scope=scope)
        return Response.success(
            data={"authorize_url": authorize_url},
            msg="获取成功",
        )


@router.get(
    "/wechat/callback",
    summary="微信授权回调",
    description="微信授权后的回调接口，自动注册/登录用户并重定向到前端",
)
async def wechat_callback(
    code: str,
    state: str = "",
) -> RedirectResponse:
    """微信授权回调

    微信授权成功后会重定向到此接口，带上 code 和 state 参数。
    处理完成后重定向到前端页面，URL 参数带上 token。
    """
    settings = get_settings()
    postgres = get_postgres()

    async with postgres.session_factory() as session:
        user_repo = DBUserRepository(session)
        oauth_repo = DBOAuthRepository(session)
        wechat_service = WeChatOAuthService(user_repo, oauth_repo)

        try:
            user, tokens = await wechat_service.handle_callback(code, state)
            await session.commit()

            # 重定向到前端页面
            redirect_url = wechat_service.get_frontend_redirect_url(tokens)
            return RedirectResponse(url=redirect_url)

        except ValueError as e:
            logger.error(f"WeChat callback failed: {e}")
            # 重定向到前端错误页面
            error_url = f"{settings.wechat_frontend_redirect_uri}?error={str(e)}"
            return RedirectResponse(url=error_url)
        except Exception as e:
            logger.exception("WeChat callback failed")
            error_url = f"{settings.wechat_frontend_redirect_uri}?error=微信登录失败"
            return RedirectResponse(url=error_url)
