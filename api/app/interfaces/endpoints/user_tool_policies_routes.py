"""用户工具审批偏好 v2 路由（self-only CRUD；R6 §6.2）。"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.application.services.user_tool_approval_policy_service import (
    UserToolApprovalPolicyService,
)
from app.infrastructure.storage.postgres import get_db_session
from app.interfaces.dependencies import CurrentUser
from app.interfaces.schemas import Response
from app.interfaces.schemas.user import ToolPolicyRequest, ToolPolicyResponse
from app.interfaces.service_dependencies import (
    get_user_tool_approval_policy_service,
)

router = APIRouter(prefix="/v2/user/tool-policies", tags=["用户工具审批偏好v2"])


@router.get(
    "",
    response_model=Response,
    summary="列出当前用户的所有工具审批 policy",
)
async def list_policies(
    current_user: CurrentUser,
    svc: UserToolApprovalPolicyService = Depends(
        get_user_tool_approval_policy_service
    ),
) -> Response:
    rows = await svc.list_user_policies(current_user.id)
    policies = [
        ToolPolicyResponse(
            tool_name=r.tool_name,
            policy=r.policy,
            updated_at=r.updated_at,
        )
        for r in rows
    ]
    return Response.success(data={"policies": policies})


@router.get(
    "/{tool_name}",
    response_model=Response,
    summary="获取当前用户对单个工具的 policy（缺行返 404）",
)
async def get_policy(
    tool_name: str,
    current_user: CurrentUser,
    svc: UserToolApprovalPolicyService = Depends(
        get_user_tool_approval_policy_service
    ),
) -> Response:
    _validate_tool_name(tool_name)
    match = await svc.get_policy_record(current_user.id, tool_name)
    if match is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="该工具无 policy 偏好",
        )
    return Response.success(data=ToolPolicyResponse(
        tool_name=match.tool_name,
        policy=match.policy,
        updated_at=match.updated_at,
    ))


@router.put(
    "/{tool_name}",
    response_model=Response,
    summary="设置/更新当前用户对单个工具的 policy",
)
async def set_policy(
    tool_name: str,
    request: ToolPolicyRequest,
    current_user: CurrentUser,
    svc: UserToolApprovalPolicyService = Depends(
        get_user_tool_approval_policy_service
    ),
    db_session: AsyncSession = Depends(get_db_session),
) -> Response:
    _validate_tool_name(tool_name)
    record = await svc.set_policy(current_user.id, tool_name, request.policy)
    await db_session.commit()
    return Response.success(data=ToolPolicyResponse(
        tool_name=record.tool_name,
        policy=record.policy,
        updated_at=record.updated_at,
    ))


@router.delete(
    "/{tool_name}",
    response_model=Response,
    summary="清除当前用户对单个工具的 policy（幂等）",
)
async def delete_policy(
    tool_name: str,
    current_user: CurrentUser,
    svc: UserToolApprovalPolicyService = Depends(
        get_user_tool_approval_policy_service
    ),
    db_session: AsyncSession = Depends(get_db_session),
) -> Response:
    _validate_tool_name(tool_name)
    await svc.clear_policy(current_user.id, tool_name)
    await db_session.commit()
    return Response.success(msg="清除成功")


def _validate_tool_name(tool_name: str) -> None:
    """Path 参数校验：非空 + 长度 ≤ 255；**不限制字符集**（MCP 名可含连字符）。"""
    if not tool_name or len(tool_name) > 255:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="tool_name 非空且长度不得超过 255",
        )
