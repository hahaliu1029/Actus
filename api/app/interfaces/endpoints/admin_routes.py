"""管理员路由模块"""

import logging

from app.domain.models.user import UserStatus
from app.infrastructure.repositories.db_user_repository import DBUserRepository
from app.infrastructure.storage.postgres import get_postgres
from app.interfaces.dependencies import AdminUser
from app.interfaces.schemas import Response
from app.interfaces.schemas.auth import UserResponse
from app.interfaces.schemas.user import UserListResponse, UserStatusUpdateRequest
from fastapi import APIRouter

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/admin", tags=["管理员模块"])


def user_to_response(user) -> UserResponse:
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


@router.get(
    "/users",
    response_model=Response[UserListResponse],
    summary="获取用户列表",
    description="获取所有用户列表（仅限超级管理员）",
)
async def list_users(
    admin_user: AdminUser,
    skip: int = 0,
    limit: int = 100,
) -> Response:
    """获取用户列表"""
    postgres = get_postgres()

    async with postgres.session_factory() as session:
        user_repo = DBUserRepository(session)
        users = await user_repo.list_all(skip=skip, limit=limit)
        total = await user_repo.count()

        return Response.success(
            data=UserListResponse(
                users=[user_to_response(u) for u in users],
                total=total,
            )
        )


@router.get(
    "/users/{user_id}",
    response_model=Response[UserResponse],
    summary="获取用户详情",
    description="根据用户 ID 获取用户详情（仅限超级管理员）",
)
async def get_user(
    user_id: str,
    admin_user: AdminUser,
) -> Response:
    """获取用户详情"""
    postgres = get_postgres()

    async with postgres.session_factory() as session:
        user_repo = DBUserRepository(session)
        user = await user_repo.get_by_id(user_id)

        if not user:
            return Response.fail(code=404, msg="用户不存在")

        return Response.success(data=user_to_response(user))


@router.put(
    "/users/{user_id}/status",
    response_model=Response,
    summary="更新用户状态",
    description="更新用户状态（仅限超级管理员）",
)
async def update_user_status(
    user_id: str,
    request: UserStatusUpdateRequest,
    admin_user: AdminUser,
) -> Response:
    """更新用户状态"""
    # 验证状态值
    try:
        new_status = UserStatus(request.status)
    except ValueError:
        return Response.fail(code=400, msg="无效的状态值")

    # 不能修改自己的状态
    if user_id == admin_user.id:
        return Response.fail(code=400, msg="不能修改自己的状态")

    postgres = get_postgres()

    async with postgres.session_factory() as session:
        user_repo = DBUserRepository(session)
        user = await user_repo.get_by_id(user_id)

        if not user:
            return Response.fail(code=404, msg="用户不存在")

        # 不能修改其他管理员
        if user.is_admin():
            return Response.fail(code=403, msg="不能修改管理员状态")

        user.status = new_status
        await user_repo.update(user)
        await session.commit()

        return Response.success(msg="更新成功")


@router.delete(
    "/users/{user_id}",
    response_model=Response,
    summary="删除用户",
    description="删除用户（仅限超级管理员）",
)
async def delete_user(
    user_id: str,
    admin_user: AdminUser,
) -> Response:
    """删除用户"""
    # 不能删除自己
    if user_id == admin_user.id:
        return Response.fail(code=400, msg="不能删除自己")

    postgres = get_postgres()

    async with postgres.session_factory() as session:
        user_repo = DBUserRepository(session)
        user = await user_repo.get_by_id(user_id)

        if not user:
            return Response.fail(code=404, msg="用户不存在")

        # 不能删除其他管理员
        if user.is_admin():
            return Response.fail(code=403, msg="不能删除管理员")

        success = await user_repo.delete(user_id)
        if success:
            await session.commit()
            return Response.success(msg="删除成功")
        else:
            return Response.fail(code=500, msg="删除失败")
