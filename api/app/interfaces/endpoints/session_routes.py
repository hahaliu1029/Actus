import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import AsyncGenerator, Dict, Optional
from urllib.parse import quote

import anyio
import websockets
from app.application.errors.exceptions import (
    BadRequestError,
    ConflictError,
    ForbiddenError,
    NotFoundError,
    ServiceUnavailableError,
    TooManyRequestsError,
)
from app.application.services.agent_service import AgentService
from app.application.services.session_service import SessionService
from app.domain.errors.supervisor import SupervisorContractError
from app.domain.models.event import (
    ExecutionStateChangedEvent,
    ExecutionStatePayload,
    OwnerConflictEvent,
    OwnerConflictPayload,
)
from app.domain.models.session import Session, SessionStatus
from app.domain.services.execution_supervisor import ExecutionSupervisor
from app.interfaces.dependencies import (
    CurrentUser,
    RateLimitBucket,
    RateLimitChannel,
    acquire_connection_limit,
    enforce_request_limit,
    get_current_user_ws_query,
    rate_limit_chat,
    rate_limit_read,
    rate_limit_write,
)
from app.interfaces.schemas import Response
from app.interfaces.schemas.event import EventMapper
from app.interfaces.schemas.session import (
    BackgroundQuotaResponse,
    CancelSessionRequest,
    ChatRequest,
    CreateSessionResponse,
    EndTakeoverRequest,
    EndTakeoverResponse,
    EventsSinceResponse,
    FileReadRequest,
    FileReadResponse,
    GetTakeoverResponse,
    GetSessionFilesResponse,
    GetSessionResponse,
    ListSessionItem,
    ListSessionResponse,
    RejectTakeoverRequest,
    RejectTakeoverResponse,
    RenewTakeoverRequest,
    RenewTakeoverResponse,
    ReopenTakeoverResponse,
    RetryFromSuspendResponse,
    ShellReadRequest,
    ShellReadResponse,
    StartTakeoverRequest,
    StartTakeoverResponse,
)
from app.interfaces.service_dependencies import (
    get_agent_service,
    get_session_service,
    get_subagent_research_service,
    get_supervisor,
)
from app.application.services.subagent_research_service import (
    SubagentResearchService,
)
from app.interfaces.schemas.subagent import ResearchSubagentRequest
from app.infrastructure.storage.redis import RedisClient, get_redis
from core.config import get_settings
from fastapi import APIRouter, Body, Depends, Request, Response as FastAPIResponse
from fastapi.responses import StreamingResponse
from sse_starlette import EventSourceResponse, ServerSentEvent
from starlette.websockets import WebSocket, WebSocketDisconnect
from websockets import ConnectionClosed

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/sessions", tags=["会话模块"])

# 流式获取会话详情睡眠间隔
SESSION_SLEEP_INTERVAL = 5
SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}
_PENDING_AUTO_DEGRADE_TASKS: set[asyncio.Task[None]] = set()


def _track_auto_degrade_task(task: asyncio.Task[None]) -> None:
    _PENDING_AUTO_DEGRADE_TASKS.add(task)
    task.add_done_callback(_PENDING_AUTO_DEGRADE_TASKS.discard)


async def _do_auto_degrade(
    session_id: str,
    user_id: str,
    agent_service: AgentService,
    supervisor: ExecutionSupervisor,
) -> None:
    try:
        sess = await agent_service.get_session(session_id)
        if (
            sess is None
            or sess.execution_mode != "foreground"
            or sess.status in (SessionStatus.COMPLETED, SessionStatus.TIMED_OUT)
            or sess.execution_phase in ("terminating", "terminated")
        ):
            return

        expires_at = datetime.now(timezone.utc) + timedelta(hours=2)
        supervisor_user_id = str(sess.user_id or user_id)
        try:
            retry_budget_remaining = await supervisor.promote(
                session_id=session_id,
                user_id=supervisor_user_id,
                expires_at=expires_at,
            )
        except SupervisorContractError:
            logger.info("auto-degrade rejected for session %s", session_id)
            return
        if retry_budget_remaining is None:
            logger.info("auto-degrade skipped stale session %s", session_id)
            return

        await agent_service._emit_event(
            session_id,
            ExecutionStateChangedEvent(
                payload=ExecutionStatePayload(
                    execution_mode="background",
                    execution_phase="running",
                    transition_reason="auto_degrade_sse_disconnect",
                    background_reason="auto_degrade",
                    expires_at=expires_at,
                    retry_budget_remaining=retry_budget_remaining,
                )
            ),
        )
    except Exception:
        logger.exception("auto-degrade failed for session %s", session_id)


async def _build_list_session_item(
    session: Session,
    agent_service: AgentService,
) -> ListSessionItem:
    supervisor_snapshot = None
    if session.execution_mode == "background":
        supervisor_snapshot = await agent_service.build_supervisor_snapshot(session)

    return ListSessionItem(
        session_id=session.id,
        title=session.title,
        sample_session_id=session.sample_session_id,
        parent_session_id=session.parent_session_id or session.sample_session_id,
        worker_type=session.worker_type,
        latest_message=session.latest_message,
        latest_message_at=session.latest_message_at,
        status=session.status,
        unread_message_count=session.unread_message_count,
        supervisor_snapshot=supervisor_snapshot,
    )


@router.post(
    path="",
    response_model=Response[CreateSessionResponse],
    summary="创建新任务会话",
    description="为当前用户创建一个空白的新任务会话",
    dependencies=[Depends(rate_limit_write)],
)
async def create_session(
    current_user: CurrentUser,
    session_service: SessionService = Depends(get_session_service),
) -> Response[CreateSessionResponse]:
    """创建一个空白的新任务会话"""
    session = await session_service.create_session(current_user.id)
    return Response.success(
        msg="创建任务会话成功", data=CreateSessionResponse(session_id=session.id)
    )


@router.post(
    path="/stream",
    summary="流式获取所有会话基础信息列表",
    description="间隔指定时间流式获取所有会话基础信息列表",
    dependencies=[Depends(rate_limit_read)],
)
async def stream_sessions(
    current_user: CurrentUser,
    session_service: SessionService = Depends(get_session_service),
    agent_service: AgentService = Depends(get_agent_service),
    redis_client: RedisClient = Depends(get_redis),
) -> EventSourceResponse:
    """间隔指定时间流式获取所有会话基础信息列表"""
    lease = await acquire_connection_limit(
        channel=RateLimitChannel.SSE,
        user_id=current_user.id,
        redis_client=redis_client,
    )
    lease.start_heartbeat()

    async def event_generator() -> AsyncGenerator[ServerSentEvent, None]:
        """定义一个异步迭代器，用于获取所有会话列表"""
        try:
            while True:
                # 1.获取所有会话列表
                sessions = await session_service.get_all_sessions(
                    current_user.id, current_user.is_admin()
                )

                # 2.循环遍历并组装数据
                session_items = [
                    await _build_list_session_item(session, agent_service)
                    for session in sessions
                ]

                # 3.将会话列表转换为流式事件数据并返回
                yield ServerSentEvent(
                    event="sessions",
                    data=ListSessionResponse(sessions=session_items).model_dump_json(),
                )

                # 4.睡眠指定时间避免高频响应
                await asyncio.sleep(SESSION_SLEEP_INTERVAL)
        finally:
            await lease.release()

    return EventSourceResponse(event_generator(), headers=SSE_HEADERS)


@router.get(
    path="",
    response_model=Response[ListSessionResponse],
    summary="获取会话列表基础信息",
    description="获取当前用户的任务会话基础信息列表",
    dependencies=[Depends(rate_limit_read)],
)
async def get_all_sessions(
    current_user: CurrentUser,
    session_service: SessionService = Depends(get_session_service),
    agent_service: AgentService = Depends(get_agent_service),
) -> Response[ListSessionResponse]:
    """获取当前用户的任务会话基础信息列表"""
    sessions = await session_service.get_all_sessions(
        current_user.id, current_user.is_admin()
    )
    session_items = [
        await _build_list_session_item(session, agent_service) for session in sessions
    ]
    return Response.success(
        msg="获取任务会话列表成功", data=ListSessionResponse(sessions=session_items)
    )


@router.get(
    path="/background-quota",
    response_model=Response[BackgroundQuotaResponse],
    summary="获取后台任务额度",
    description="获取当前用户和系统后台任务额度使用情况",
    dependencies=[Depends(rate_limit_read)],
)
async def get_background_quota(
    current_user: CurrentUser,
    supervisor: ExecutionSupervisor = Depends(get_supervisor),
) -> Response[BackgroundQuotaResponse]:
    """获取后台任务额度读模型"""
    quota = await supervisor.get_background_quota(str(current_user.id))
    return Response.success(
        msg="获取后台任务额度成功",
        data=BackgroundQuotaResponse.model_validate(quota),
    )


@router.post(
    path="/{session_id}/clear-unread-message-count",
    response_model=Response[Optional[Dict]],
    summary="清除指定任务会话未读消息数",
    description="清除当前用户指定任务会话未读消息数",
    dependencies=[Depends(rate_limit_write)],
)
async def clear_unread_message_count(
    session_id: str,
    current_user: CurrentUser,
    session_service: SessionService = Depends(get_session_service),
) -> Response[Optional[Dict]]:
    """根据传递的会话id清空当前用户未读消息数"""
    await session_service.clear_unread_message_count(
        session_id=session_id,
        user_id=current_user.id,
        is_admin=current_user.is_admin(),
    )
    return Response.success(msg="清除未读消息数成功")


@router.post(
    path="/{session_id}/delete",
    response_model=Response[Optional[Dict]],
    summary="删除指定任务会话",
    description="根据传递的会话id删除当前用户的指定任务会话",
    dependencies=[Depends(rate_limit_write)],
)
async def delete_session(
    session_id: str,
    current_user: CurrentUser,
    session_service: SessionService = Depends(get_session_service),
) -> Response[Optional[Dict]]:
    """根据传递的会话id删除当前用户指定任务会话"""
    await session_service.delete_session(
        session_id=session_id,
        user_id=current_user.id,
        is_admin=current_user.is_admin(),
    )
    return Response.success(msg="删除任务会话成功")


@router.post(
    path="/{session_id}/chat",
    summary="向指定任务会话发起聊天请求",
    description="向指定任务会话发起聊天请求",
    dependencies=[Depends(rate_limit_chat)],
)
async def chat(
    session_id: str,
    request: ChatRequest,
    fastapi_request: Request,
    current_user: CurrentUser,
    agent_service: AgentService = Depends(get_agent_service),
    session_service: SessionService = Depends(get_session_service),
    supervisor: ExecutionSupervisor = Depends(get_supervisor),
    redis_client: RedisClient = Depends(get_redis),
) -> EventSourceResponse:
    """根据传递的会话id+chat请求数据向指定会话发起聊天请求"""
    session = await session_service.get_session(
        session_id=session_id,
        user_id=current_user.id,
        is_admin=current_user.is_admin(),
    )
    has_user_input = bool(request.message) or bool(request.attachments)
    if (
        has_user_input
        and getattr(session, "status", None) == SessionStatus.RUNNING
        and getattr(session, "execution_mode", None) == "background"
        and getattr(session, "execution_phase", None) == "suspended"
    ):
        raise ConflictError("后台任务已挂起，请先重试后台任务")

    # R5b-3 + B3-core PR-3c: 访问校验通过后、response 前必须先拿连接
    # lease，再做 owner conflict gate，最后才允许 tool-confirmation preflight
    # 产生 claim/mark processing/audit 等副作用。非冲突路径仍在
    # EventSourceResponse 之前完成 preflight 的 404/409/400 判定，避免退化成
    # 200 后的 SSE 异常。
    lease = await acquire_connection_limit(
        channel=RateLimitChannel.SSE,
        user_id=current_user.id,
        redis_client=redis_client,
    )
    lease.start_heartbeat()

    raw_connection_id = fastapi_request.headers.get("X-Connection-Id")
    connection_id = f"{current_user.id}:{raw_connection_id or uuid.uuid4()}"
    subscriber_scope = supervisor.subscriber_scope(
        session_id=session_id,
        connection_id=connection_id,
    )
    try:
        scope = await subscriber_scope.__aenter__()
    except BaseException:
        await lease.release()
        raise

    if scope.is_conflict:
        async def event_generator() -> AsyncGenerator[ServerSentEvent, None]:
            """定义事件生成器，用于配合EventSourceResponse生成流式响应数据"""
            try:
                event = OwnerConflictEvent(
                    payload=OwnerConflictPayload(
                        current_owner_connection_id=scope.current_owner or "",
                        conflicting_connection_id=connection_id,
                        session_id=session_id,
                        suggested_action="request_takeover",
                    )
                )
                sse_event = EventMapper.event_to_sse_event(event)
                yield ServerSentEvent(
                    id=event.id,
                    event=sse_event.event,
                    data=sse_event.to_sse_data_json(),
                )
            finally:
                try:
                    await subscriber_scope.__aexit__(None, None, None)
                finally:
                    await lease.release()

        return EventSourceResponse(event_generator(), headers=SSE_HEADERS)

    resume_state = None
    if request.tool_confirmation is not None:
        try:
            resume_state = await agent_service.preflight_resume_tool_confirmation(
                session_id=session_id,
                user_id=current_user.id,
                is_admin=current_user.is_admin(),
                tool_confirmation=request.tool_confirmation,
            )
        except BaseException:
            try:
                await subscriber_scope.__aexit__(None, None, None)
            finally:
                await lease.release()
            raise

    auto_degrade_scheduled = False

    def schedule_auto_degrade() -> None:
        nonlocal auto_degrade_scheduled
        if auto_degrade_scheduled:
            return
        auto_degrade_scheduled = True
        task = asyncio.create_task(
            _do_auto_degrade(
                session_id,
                current_user.id,
                agent_service,
                supervisor,
            )
        )
        _track_auto_degrade_task(task)

    async def handle_client_close(_message: dict[str, object]) -> None:
        schedule_auto_degrade()

    async def event_generator() -> AsyncGenerator[ServerSentEvent, None]:
        """定义事件生成器，用于配合EventSourceResponse生成流式响应数据"""
        try:
            # 1.分派：tool_confirmation 走 drive（preflight 已拿 claim）；否则正常 chat
            if resume_state is not None:
                event_stream = agent_service.drive_resume_tool_confirmation(
                    resume_state
                )
            else:
                event_stream = agent_service.chat(
                    session_id=session_id,
                    user_id=current_user.id,
                    is_admin=current_user.is_admin(),
                    message=request.message,
                    attachments=request.attachments,
                    skill_confirmation_action=request.skill_confirmation_action,
                    tool_confirmation=request.tool_confirmation,
                    latest_event_id=request.event_id,
                    timestamp=(
                        datetime.fromtimestamp(request.timestamp)
                        if request.timestamp
                        else None
                    ),
                )
            async for event in event_stream:
                # 2.将Agent事件转换为sse数据(因为普通的event没法通过流式事件传输)
                sse_event = EventMapper.event_to_sse_event(event)
                if sse_event:
                    yield ServerSentEvent(
                        id=event.id,
                        event=sse_event.event,
                        data=sse_event.to_sse_data_json(),
                    )
        except (
            asyncio.CancelledError,
            ConnectionResetError,
            GeneratorExit,
            anyio.EndOfStream,
        ):
            schedule_auto_degrade()
            raise
        finally:
            try:
                await subscriber_scope.__aexit__(None, None, None)
            finally:
                await lease.release()

    return EventSourceResponse(
        event_generator(),
        headers=SSE_HEADERS,
        client_close_handler_callable=handle_client_close,
    )


@router.post(
    path="/{session_id}/cancel",
    response_model=Response[Dict[str, str]],
    summary="取消指定任务会话",
    description="请求取消当前用户指定任务会话",
    dependencies=[Depends(rate_limit_write)],
)
async def cancel_session(
    session_id: str,
    current_user: CurrentUser,
    body: CancelSessionRequest = Body(default_factory=CancelSessionRequest),
    agent_service: AgentService = Depends(get_agent_service),
    session_service: SessionService = Depends(get_session_service),
    supervisor: ExecutionSupervisor = Depends(get_supervisor),
) -> Response[Dict[str, str]]:
    """请求取消指定任务会话，并由 supervisor 统一标记热状态。"""
    await _request_user_cancel(
        session_id=session_id,
        current_user=current_user,
        reason=body.reason,
        agent_service=agent_service,
        session_service=session_service,
        supervisor=supervisor,
    )

    return Response.success(
        msg="取消任务会话请求已提交",
        data={
            "status": "cancel_requested",
            "session_id": session_id,
            "reason": body.reason,
        },
    )


async def _request_user_cancel(
    *,
    session_id: str,
    current_user: CurrentUser,
    reason: str,
    agent_service: AgentService,
    session_service: SessionService,
    supervisor: ExecutionSupervisor,
) -> None:
    await session_service.get_session(
        session_id=session_id,
        user_id=current_user.id,
        is_admin=False,
    )

    async def _stop_session(*, session_id: str, user_id: str) -> None:
        await agent_service.stop_session(
            session_id=session_id,
            user_id=user_id,
            is_admin=False,
        )

    await supervisor.request_cancel(
        session_id=session_id,
        user_id=str(current_user.id),
        reason=reason,
        stop_session=_stop_session,
    )


@router.get(
    path="/{session_id}",
    response_model=Response[GetSessionResponse],
    summary="获取指定会话详情信息",
    description="根据传递的会话id获取该会话的对话详情",
    dependencies=[Depends(rate_limit_read)],
)
async def get_session(
    session_id: str,
    current_user: CurrentUser,
    session_service: SessionService = Depends(get_session_service),
    agent_service: AgentService = Depends(get_agent_service),
) -> Response[GetSessionResponse]:
    """传递指定会话id获取该会话的对话详情"""
    session = await session_service.get_session(
        session_id=session_id,
        user_id=current_user.id,
        is_admin=current_user.is_admin(),
    )
    if not session:
        raise NotFoundError("该会话不存在，请核实后重试")
    supervisor_snapshot = None
    if session.execution_mode == "background":
        supervisor_snapshot = await agent_service.build_supervisor_snapshot(session)
    return Response.success(
        msg="获取会话详情成功",
        data=GetSessionResponse(
            session_id=session.id,
            title=session.title,
            status=session.status,
            events=EventMapper.events_to_sse_events(session.events),
            supervisor_snapshot=supervisor_snapshot,
        ),
    )


@router.get(
    path="/{session_id}/events",
    response_model=Response[EventsSinceResponse],
    summary="获取会话增量事件",
    description="获取指定 event_id (or seq) 之后的增量事件，用于断线重连后的状态恢复",
    dependencies=[Depends(rate_limit_read)],
)
async def get_events_since(
    session_id: str,
    current_user: CurrentUser,
    since: Optional[str] = None,
    since_seq: Optional[int] = None,  # B3-core PR-1 §3.3 — preferred cursor
    agent_service: AgentService = Depends(get_agent_service),
) -> Response[EventsSinceResponse]:
    """获取 session 在 since (or since_seq) 之后的增量事件 + 当前状态。

    B3-core PR-1 §3.3: ``since_seq`` is the preferred monotonic cursor for
    sequenced events. When both are provided, agent_service keeps ``since`` as
    the legacy-event fallback.
    """
    result = await agent_service.get_events_since(
        session_id=session_id,
        since_event_id=since,
        user_id=current_user.id,
        is_admin=current_user.is_admin(),
        since_seq=since_seq,
    )
    return Response.success(
        msg="获取增量事件成功",
        data=EventsSinceResponse(
            events=EventMapper.events_to_sse_events(result["events"]),
            session_status=result["session_status"],
            has_more=result["has_more"],
            last_seq=result["last_seq"],
            supervisor_snapshot=result["supervisor_snapshot"],
        ),
    )


@router.get(
    path="/{session_id}/takeover",
    response_model=Response[GetTakeoverResponse],
    summary="获取指定会话接管状态",
    description="根据会话ID获取当前会话接管状态",
    dependencies=[Depends(rate_limit_read)],
)
async def get_takeover(
    session_id: str,
    current_user: CurrentUser,
    agent_service: AgentService = Depends(get_agent_service),
) -> Response[GetTakeoverResponse]:
    """获取指定会话接管状态"""
    result = await agent_service.get_takeover(
        session_id=session_id,
        user_id=current_user.id,
        is_admin=current_user.is_admin(),
        user_role=current_user.role.value,
    )
    return Response.success(
        msg="获取会话接管状态成功",
        data=GetTakeoverResponse.model_validate(result),
    )


@router.post(
    path="/{session_id}/takeover/start",
    response_model=Response[StartTakeoverResponse],
    summary="启动指定会话接管",
    description="用户主动接管指定会话控制权",
    dependencies=[Depends(rate_limit_write)],
)
async def start_takeover(
    session_id: str,
    request: StartTakeoverRequest,
    current_user: CurrentUser,
    http_response: FastAPIResponse,
    agent_service: AgentService = Depends(get_agent_service),
) -> Response[StartTakeoverResponse]:
    """启动指定会话接管"""
    result = await agent_service.start_takeover(
        session_id=session_id,
        user_id=current_user.id,
        scope=request.scope,
        is_admin=current_user.is_admin(),
        user_role=current_user.role.value,
    )
    if result.get("request_status") == "starting":
        http_response.status_code = 202
    return Response.success(
        msg="启动会话接管成功",
        data=StartTakeoverResponse.model_validate(result),
    )


@router.post(
    path="/{session_id}/takeover/renew",
    response_model=Response[RenewTakeoverResponse],
    summary="续期指定会话接管",
    description="续期当前会话的接管租约",
    dependencies=[Depends(rate_limit_write)],
)
async def renew_takeover(
    session_id: str,
    request: RenewTakeoverRequest,
    current_user: CurrentUser,
    agent_service: AgentService = Depends(get_agent_service),
) -> Response[RenewTakeoverResponse]:
    """续期指定会话接管"""
    result = await agent_service.renew_takeover(
        session_id=session_id,
        user_id=current_user.id,
        takeover_id=request.takeover_id,
        is_admin=current_user.is_admin(),
        user_role=current_user.role.value,
    )
    return Response.success(
        msg="续期会话接管成功",
        data=RenewTakeoverResponse.model_validate(result),
    )


@router.post(
    path="/{session_id}/takeover/reject",
    response_model=Response[RejectTakeoverResponse],
    summary="处理指定会话接管请求",
    description="处理AI发起的接管请求，可继续或终止",
    dependencies=[Depends(rate_limit_write)],
)
async def reject_takeover(
    session_id: str,
    request: RejectTakeoverRequest,
    current_user: CurrentUser,
    agent_service: AgentService = Depends(get_agent_service),
) -> Response[RejectTakeoverResponse]:
    """处理指定会话接管请求"""
    result = await agent_service.reject_takeover(
        session_id=session_id,
        user_id=current_user.id,
        decision=request.decision,
        is_admin=current_user.is_admin(),
        user_role=current_user.role.value,
    )
    return Response.success(
        msg="处理接管请求成功",
        data=RejectTakeoverResponse.model_validate(result),
    )


@router.post(
    path="/{session_id}/takeover/end",
    response_model=Response[EndTakeoverResponse],
    summary="结束指定会话接管",
    description="结束用户接管并选择继续执行或直接完成",
    dependencies=[Depends(rate_limit_write)],
)
async def end_takeover(
    session_id: str,
    request: EndTakeoverRequest,
    current_user: CurrentUser,
    agent_service: AgentService = Depends(get_agent_service),
) -> Response[EndTakeoverResponse]:
    """结束指定会话接管"""
    result = await agent_service.end_takeover(
        session_id=session_id,
        user_id=current_user.id,
        handoff_mode=request.handoff_mode,
        is_admin=current_user.is_admin(),
        user_role=current_user.role.value,
    )
    return Response.success(
        msg="结束会话接管成功",
        data=EndTakeoverResponse.model_validate(result),
    )


@router.post(
    path="/{session_id}/takeover/reopen",
    response_model=Response[ReopenTakeoverResponse],
    summary="补救接管已完成的会话",
    description="在完成窗口期内恢复已完成会话到接管待决状态",
    dependencies=[Depends(rate_limit_write)],
)
async def reopen_takeover(
    session_id: str,
    current_user: CurrentUser,
    agent_service: AgentService = Depends(get_agent_service),
) -> Response[ReopenTakeoverResponse]:
    """补救接管已完成的会话"""
    result = await agent_service.reopen_takeover(
        session_id=session_id,
        user_id=current_user.id,
        is_admin=current_user.is_admin(),
        user_role=current_user.role.value,
    )
    return Response.success(
        msg="补救接管成功",
        data=ReopenTakeoverResponse.model_validate(result),
    )


@router.post(
    path="/{session_id}/retry-from-suspend",
    response_model=Response[RetryFromSuspendResponse],
    summary="重试挂起的后台任务",
    description="将可恢复的后台挂起任务重新加入后台执行",
    dependencies=[Depends(rate_limit_write)],
)
async def retry_from_suspend(
    session_id: str,
    current_user: CurrentUser,
    agent_service: AgentService = Depends(get_agent_service),
) -> Response[RetryFromSuspendResponse]:
    """重试指定的后台挂起任务"""
    result = await agent_service.retry_from_suspend(
        session_id=session_id,
        user_id=current_user.id,
        is_admin=current_user.is_admin(),
        user_role=current_user.role.value,
    )
    return Response.success(
        msg="重试后台任务成功",
        data=RetryFromSuspendResponse.model_validate(result),
    )


@router.post(
    path="/{session_id}/stop",
    response_model=Response[Optional[Dict]],
    summary="停止指定任务会话",
    description="根据传递的指定会话id停止对应任务会话",
    dependencies=[Depends(rate_limit_write)],
)
async def stop_session(
    session_id: str,
    current_user: CurrentUser,
    agent_service: AgentService = Depends(get_agent_service),
    session_service: SessionService = Depends(get_session_service),
    supervisor: ExecutionSupervisor = Depends(get_supervisor),
) -> Response[Optional[Dict]]:
    """根据传递的指定会话id停止对应任务会话"""
    await _request_user_cancel(
        session_id=session_id,
        current_user=current_user,
        reason="user_cancel",
        agent_service=agent_service,
        session_service=session_service,
        supervisor=supervisor,
    )
    return Response.success(msg="停止任务会话成功")


@router.get(
    path="/{session_id}/files",
    response_model=Response[GetSessionFilesResponse],
    summary="获取指定任务会话文件列表信息",
    description="获取指定任务会话文件列表信息",
    dependencies=[Depends(rate_limit_read)],
)
async def get_session_files(
    session_id: str,
    current_user: CurrentUser,
    session_service: SessionService = Depends(get_session_service),
) -> Response[GetSessionFilesResponse]:
    """获取指定任务会话文件列表信息"""
    files = await session_service.get_session_files(
        session_id=session_id,
        user_id=current_user.id,
        is_admin=current_user.is_admin(),
    )
    return Response.success(
        msg="获取会话文件列表成功", data=GetSessionFilesResponse(files=files)
    )


@router.post(
    path="/{session_id}/file",
    response_model=Response[FileReadResponse],
    summary="查看会话沙箱中指定文件的内容",
    description="根据传递的会话id+文件路径查看沙箱中文件的内容信息",
    dependencies=[Depends(rate_limit_read)],
)
async def read_file(
    session_id: str,
    request: FileReadRequest,
    current_user: CurrentUser,
    session_service: SessionService = Depends(get_session_service),
) -> Response[FileReadResponse]:
    """根据传递的会话id+文件路径查看沙箱中文件的内容信息"""
    result = await session_service.read_file(
        session_id=session_id,
        filepath=request.filepath,
        user_id=current_user.id,
        is_admin=current_user.is_admin(),
    )
    return Response.success(msg="获取会话文件内容成功", data=result)


@router.get(
    path="/{session_id}/file/download",
    summary="从沙箱中下载文件",
    description="通过会话ID和文件路径直接从沙箱中下载文件（支持二进制文件）",
    dependencies=[Depends(rate_limit_read)],
)
async def download_sandbox_file(
    session_id: str,
    filepath: str,
    current_user: CurrentUser,
    session_service: SessionService = Depends(get_session_service),
) -> StreamingResponse:
    """通过 session + filepath 直接从沙箱下载文件"""
    import mimetypes
    import os

    file_data = await session_service.download_file(
        session_id, filepath, current_user.id, current_user.is_admin()
    )
    filename = os.path.basename(filepath)
    content_type, _ = mimetypes.guess_type(filename)

    return StreamingResponse(
        file_data,
        media_type=content_type or "application/octet-stream",
        headers={
            "Content-Disposition": f'attachment; filename="{quote(filename)}"',
        },
    )


@router.post(
    path="/{session_id}/shell",
    response_model=Response[ShellReadResponse],
    summary="查看会话的shell内容输出",
    description="传递指定会话id与shell会话标识，查看shell内容输出",
    dependencies=[Depends(rate_limit_read)],
)
async def read_shell_output(
    session_id: str,
    request: ShellReadRequest,
    current_user: CurrentUser,
    session_service: SessionService = Depends(get_session_service),
) -> Response[ShellReadResponse]:
    """查看会话的shell内容输出"""
    result = await session_service.read_shell_output(
        session_id=session_id,
        shell_session_id=request.session_id,
        user_id=current_user.id,
        is_admin=current_user.is_admin(),
    )
    return Response.success(
        msg="获取Shell内容输出结果成功",
        data=result,
    )


@router.websocket(
    path="/{session_id}/takeover/shell/ws",
)
async def takeover_shell_websocket(
    websocket: WebSocket,
    session_id: str,
    takeover_id: str | None = None,
    token: str | None = None,
    session_service: SessionService = Depends(get_session_service),
    agent_service: AgentService = Depends(get_agent_service),
    redis_client: RedisClient = Depends(get_redis),
) -> None:
    """终端接管 WebSocket 端点，提供接管态下的双向交互。"""
    lease = None
    current_user = None

    # 先 accept 连接，再进行认证/鉴权，失败时通过 status 消息告知后关闭。
    # Starlette 不允许对未 accept 的 WebSocket 调用 close()。
    await websocket.accept()

    if not takeover_id:
        await websocket.send_text(
            json.dumps({"type": "status", "state": "error", "message": "缺少takeover_id"}, ensure_ascii=False)
        )
        await websocket.close(code=4400, reason="缺少takeover_id")
        return

    try:
        current_user = await get_current_user_ws_query(token)
        await enforce_request_limit(
            bucket=RateLimitBucket.READ,
            current_user=current_user,
            redis_client=redis_client,
        )
        lease = await acquire_connection_limit(
            channel=RateLimitChannel.WS,
            user_id=current_user.id,
            redis_client=redis_client,
        )
        lease.start_heartbeat()
        await agent_service.assert_takeover_shell_access(
            session_id=session_id,
            user_id=current_user.id,
            takeover_id=takeover_id,
            is_admin=current_user.is_admin(),
            user_role=current_user.role.value,
        )
        sandbox, shell_session_id = await session_service.ensure_takeover_shell_session(
            session_id=session_id,
            takeover_id=takeover_id,
            user_id=current_user.id,
            is_admin=current_user.is_admin(),
        )
    except TooManyRequestsError as exc:
        retry_after = (exc.data or {}).get("retry_after", 1)
        await websocket.send_text(
            json.dumps({"type": "status", "state": "error", "message": f"请求过多，请{retry_after}秒后重试"}, ensure_ascii=False)
        )
        await websocket.close(code=1013, reason=f"请求过多，请{retry_after}秒后重试")
        return
    except ServiceUnavailableError:
        await websocket.send_text(
            json.dumps({"type": "status", "state": "error", "message": "限流服务不可用"}, ensure_ascii=False)
        )
        await websocket.close(code=1011, reason="限流服务不可用")
        return
    except ForbiddenError as exc:
        await websocket.send_text(
            json.dumps({"type": "status", "state": "forbidden", "message": str(exc)}, ensure_ascii=False)
        )
        await websocket.close(code=4403, reason=str(exc))
        return
    except (BadRequestError, ConflictError) as exc:
        await websocket.send_text(
            json.dumps({"type": "status", "state": "error", "message": str(exc)}, ensure_ascii=False)
        )
        await websocket.close(code=4409, reason=str(exc))
        return
    except Exception as exc:
        await websocket.send_text(
            json.dumps({"type": "status", "state": "error", "message": str(exc)}, ensure_ascii=False)
        )
        await websocket.close(code=4401, reason=str(exc))
        return
    await websocket.send_text(
        json.dumps({"type": "status", "state": "connected"}, ensure_ascii=False)
    )

    try:
        closed = asyncio.Event()
        last_output = ""
        last_output_hash = 0

        async def check_takeover_lease() -> bool:
            try:
                await agent_service.assert_takeover_shell_access(
                    session_id=session_id,
                    user_id=current_user.id,
                    takeover_id=takeover_id,
                    is_admin=current_user.is_admin(),
                    user_role=current_user.role.value,
                )
                return True
            except ConflictError:
                await websocket.send_text(
                    json.dumps(
                        {"type": "status", "state": "lease_expired"},
                        ensure_ascii=False,
                    )
                )
                return False
            except (ForbiddenError, BadRequestError):
                await websocket.send_text(
                    json.dumps(
                        {"type": "status", "state": "forbidden"},
                        ensure_ascii=False,
                    )
                )
                return False

        async def lease_guard() -> None:
            guard_interval = get_settings().feature_takeover_lease_guard_interval_seconds
            while not closed.is_set():
                lease_ok = await check_takeover_lease()
                if not lease_ok:
                    break
                await asyncio.sleep(guard_interval)

        sandbox_shell_ws_url = str(getattr(sandbox, "shell_ws_url", "") or "").strip()

        # Get lifecycle registry for WS holder registration (P2 quiesce barrier)
        _lifecycle_svc = getattr(
            getattr(session_service, "_lifecycle", None), "registry", None
        ) if hasattr(session_service, "_lifecycle") and session_service._lifecycle else None

        if sandbox_shell_ws_url:
            target_url = (
                f"{sandbox_shell_ws_url}?session_id={quote(shell_session_id, safe='')}"
            )
            logger.info("接管终端走沙箱WS透传: %s", target_url)

            async with websockets.connect(target_url) as sandbox_ws:
                async def forward_to_sandbox() -> None:
                    while not closed.is_set():
                        try:
                            message = await websocket.receive()
                        except WebSocketDisconnect:
                            break

                        if message.get("type") == "websocket.disconnect":
                            break

                        payload_bytes = message.get("bytes")
                        if payload_bytes is not None:
                            await sandbox_ws.send(payload_bytes)
                            continue

                        payload_text = message.get("text")
                        if payload_text is not None:
                            await sandbox_ws.send(payload_text)

                async def forward_from_sandbox() -> None:
                    while not closed.is_set():
                        try:
                            data = await sandbox_ws.recv()
                        except ConnectionClosed:
                            break
                        if isinstance(data, bytes):
                            await websocket.send_bytes(data)
                        else:
                            await websocket.send_text(str(data))

                tasks = [
                    asyncio.create_task(forward_to_sandbox()),
                    asyncio.create_task(forward_from_sandbox()),
                    asyncio.create_task(lease_guard()),
                ]

                # Register WS holder for quiesce barrier (I6/P2)
                ws_holder = None
                if _lifecycle_svc is not None:
                    from app.interfaces.endpoints._ws_holders import TakeoverShellWebSocketHolder
                    ws_holder = TakeoverShellWebSocketHolder(
                        session_id=session_id,
                        client_ws=websocket,
                        sandbox_ws=sandbox_ws,
                        tasks=tasks,
                        closed_event=closed,
                    )
                    _lifecycle_svc.register_ws_holder(session_id, ws_holder)

                try:
                    done, pending = await asyncio.wait(
                        tasks,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                finally:
                    if ws_holder is not None and _lifecycle_svc is not None:
                        _lifecycle_svc.release_ws_holder(session_id, ws_holder)
        else:
            logger.warning("沙箱不支持shell_ws_url，降级为HTTP轮询转发")

            async def forward_to_sandbox_via_http() -> None:
                while not closed.is_set():
                    try:
                        message = await websocket.receive()
                    except WebSocketDisconnect:
                        break

                    if message.get("type") == "websocket.disconnect":
                        break

                    payload_bytes = message.get("bytes")
                    if payload_bytes is not None:
                        input_text = payload_bytes.decode("utf-8", errors="replace")
                        if not input_text:
                            continue
                        result = await sandbox.write_shell_input(
                            session_id=shell_session_id,
                            input_text=input_text,
                            press_enter=False,
                        )
                        if not result.success:
                            await websocket.send_text(
                                json.dumps(
                                    {
                                        "type": "error",
                                        "code": "write_failed",
                                        "message": result.message or "写入终端失败",
                                    },
                                    ensure_ascii=False,
                                )
                            )
                        continue

                    payload_text = message.get("text")
                    if payload_text is None:
                        continue
                    try:
                        payload = json.loads(payload_text)
                    except Exception:
                        continue
                    if payload.get("type") == "resize":
                        try:
                            cols = max(1, min(int(payload.get("cols", 0)), 500))
                            rows = max(1, min(int(payload.get("rows", 0)), 200))
                        except (TypeError, ValueError):
                            await websocket.send_text(
                                json.dumps(
                                    {
                                        "type": "error",
                                        "code": "invalid_resize",
                                        "message": "无效的终端尺寸参数",
                                    },
                                    ensure_ascii=False,
                                )
                            )
                            continue
                        resize_result = await sandbox.resize_shell_session(
                            session_id=shell_session_id,
                            cols=cols,
                            rows=rows,
                        )
                        if not resize_result.success:
                            await websocket.send_text(
                                json.dumps(
                                    {
                                        "type": "error",
                                        "code": "resize_failed",
                                        "message": resize_result.message or "调整终端尺寸失败",
                                    },
                                    ensure_ascii=False,
                                )
                            )
                        continue

            async def forward_from_sandbox_via_http() -> None:
                nonlocal last_output, last_output_hash
                while not closed.is_set():
                    read_result = await sandbox.read_shell_output(
                        session_id=shell_session_id, console=False
                    )
                    if not read_result.success:
                        await websocket.send_text(
                            json.dumps(
                                {
                                    "type": "error",
                                    "code": "read_failed",
                                    "message": read_result.message or "读取终端输出失败",
                                },
                                ensure_ascii=False,
                            )
                        )
                        await asyncio.sleep(0.3)
                        continue

                    latest_output = str((read_result.data or {}).get("output") or "")
                    latest_hash = hash(latest_output)

                    # 内容完全相同（含空），跳过
                    if latest_hash == last_output_hash and latest_output == last_output:
                        await asyncio.sleep(0.2)
                        continue

                    # 判断新内容是否是旧内容的追加延续
                    if (
                        len(latest_output) >= len(last_output)
                        and latest_output[: len(last_output)] == last_output
                    ):
                        delta = latest_output[len(last_output) :]
                    else:
                        # 缓冲区被截断/重置/内容不连续 → 全量重传
                        delta = latest_output

                    if delta:
                        await websocket.send_bytes(delta.encode("utf-8"))
                    last_output = latest_output
                    last_output_hash = latest_hash
                    await asyncio.sleep(0.2)

            tasks = [
                asyncio.create_task(forward_to_sandbox_via_http()),
                asyncio.create_task(forward_from_sandbox_via_http()),
                asyncio.create_task(lease_guard()),
            ]

            # Register HTTP-fallback WS holder for quiesce barrier (I6/P2)
            http_ws_holder = None
            if _lifecycle_svc is not None:
                from app.interfaces.endpoints._ws_holders import TakeoverShellWebSocketHolder
                http_ws_holder = TakeoverShellWebSocketHolder(
                    session_id=session_id,
                    client_ws=websocket,
                    sandbox_ws=None,  # No upstream WS in HTTP fallback mode
                    tasks=tasks,
                    closed_event=closed,
                )
                _lifecycle_svc.register_ws_holder(session_id, http_ws_holder)

            try:
                done, pending = await asyncio.wait(
                    tasks,
                    return_when=asyncio.FIRST_COMPLETED,
                )
            finally:
                if http_ws_holder is not None and _lifecycle_svc is not None:
                    _lifecycle_svc.release_ws_holder(session_id, http_ws_holder)

        closed.set()
        for task in pending:
            task.cancel()
        for task in pending:
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        for task in done:
            if task.cancelled():
                continue
            try:
                exc = task.exception()
            except Exception as task_exc:  # noqa: BLE001 - 保护收尾阶段不被次生异常打断
                logger.warning(
                    "接管终端任务收尾异常: session_id=%s error=%s",
                    session_id,
                    str(task_exc),
                )
                continue
            if exc:
                logger.warning(
                    "接管终端任务异常退出: session_id=%s error=%s",
                    session_id,
                    str(exc),
                )
    except WebSocketDisconnect:
        logger.info("接管终端WebSocket连接已断开, session_id=%s", session_id)
    except Exception as exc:
        logger.error("接管终端WebSocket异常: %s", str(exc))
        await websocket.close(code=1011, reason=f"WebSocket异常: {str(exc)}")
    finally:
        if lease:
            await lease.release()


@router.websocket(
    path="/{session_id}/vnc",
)
async def vnc_websocket(
    websocket: WebSocket,
    session_id: str,
    token: str | None = None,
    session_service: SessionService = Depends(get_session_service),
    redis_client: RedisClient = Depends(get_redis),
) -> None:
    """VNC Websocket端点，用于建立与沙箱环境的vnc连接，并双向转发数据"""
    lease = None
    try:
        current_user = await get_current_user_ws_query(token)
        await enforce_request_limit(
            bucket=RateLimitBucket.READ,
            current_user=current_user,
            redis_client=redis_client,
        )
        lease = await acquire_connection_limit(
            channel=RateLimitChannel.WS,
            user_id=current_user.id,
            redis_client=redis_client,
        )
        lease.start_heartbeat()
    except TooManyRequestsError as exc:
        retry_after = (exc.data or {}).get("retry_after", 1)
        await websocket.close(code=1013, reason=f"请求过多，请{retry_after}秒后重试")
        return
    except ServiceUnavailableError:
        await websocket.close(code=1011, reason="限流服务不可用")
        return
    except Exception as exc:
        await websocket.close(code=4401, reason=str(exc))
        return

    # 1.从客户端noVNC接收子协议
    protocols_str = websocket.headers.get("sec-websocket-protocol", "")
    protocols = [p.strip() for p in protocols_str.split(",")]

    # 2.判断使用不同协议(noVNC首选binary)
    selected_protocol = None
    if "binary" in protocols:
        selected_protocol = "binary"
    elif "base64" in protocols:
        selected_protocol = "base64"

    # 3.使用对应协议接收websocket连接
    logger.info(f"为会话[{session_id}]开启WebSocket连接")
    await websocket.accept(subprotocol=selected_protocol)

    try:
        # 4.获取对应会话的vnc链接
        sandbox_vnc_url = await session_service.get_vnc_url(
            session_id=session_id,
            user_id=current_user.id,
            is_admin=current_user.is_admin(),
        )
        logger.info(f"连接WebSocket VNC： {sandbox_vnc_url}")

        # 5.创建上下文并连接到vnc
        async with websockets.connect(sandbox_vnc_url) as sandbox_ws:
            # 6.创建两个异步协程来完成数据的双向转发
            async def forward_to_sandbox():
                try:
                    while True:
                        # 接收来自客户端的数据
                        data = await websocket.receive_bytes()
                        await sandbox_ws.send(data)
                except WebSocketDisconnect:
                    logger.info(f"Web->VNC连接终端")
                except Exception as forward_e:
                    logger.error(f"forward_to_sandbox出错: {str(forward_e)}")

            async def forward_from_sandbox():
                try:
                    while True:
                        # 接收来自沙箱的数据并转发
                        data = await sandbox_ws.recv()
                        await websocket.send_bytes(data)
                except ConnectionClosed:
                    logger.info("VNC->Web连接关闭")
                except Exception as forward_e:
                    logger.error(f"forward_from_sandbox出错: {str(forward_e)}")

            # 7.并行运行两个任务
            forward_task1 = asyncio.create_task(forward_to_sandbox())
            forward_task2 = asyncio.create_task(forward_from_sandbox())
            vnc_tasks = [forward_task1, forward_task2]

            # Register VNC WS holder for quiesce barrier (I6/P2)
            vnc_holder = None
            _vnc_lifecycle_svc = getattr(
                getattr(session_service, "_lifecycle", None), "registry", None
            ) if hasattr(session_service, "_lifecycle") and session_service._lifecycle else None
            if _vnc_lifecycle_svc is not None:
                from app.interfaces.endpoints._ws_holders import VncWebSocketHolder
                vnc_holder = VncWebSocketHolder(
                    session_id=session_id,
                    client_ws=websocket,
                    sandbox_ws=sandbox_ws,
                    tasks=vnc_tasks,
                )
                _vnc_lifecycle_svc.register_ws_holder(session_id, vnc_holder)

            try:
                # 8.等待任意任务结束意味WebSocket连接终端
                done, pending = await asyncio.wait(
                    vnc_tasks,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                logger.info("WebSocket连接已关闭")
            finally:
                if vnc_holder is not None and _vnc_lifecycle_svc is not None:
                    _vnc_lifecycle_svc.release_ws_holder(session_id, vnc_holder)

            # 9.如果任一任务完成则取消其他任务(关闭全部链接)
            for task in pending:
                task.cancel()
            for task in pending:
                try:
                    await task
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass
    except ConnectionError as connection_e:
        # 连接沙箱环境失败，关闭websocket
        logger.error(f"连接沙箱环境失败: {str(connection_e)}")
        await websocket.close(
            code=1011, reason=f"连接沙箱环境失败: {str(connection_e)}"
        )
    except Exception as e:
        # 其他错误记录日志并关闭websocket
        logger.error(f"WebSocket异常: {str(e)}")
        await websocket.close(code=1011, reason=f"WebSocket异常: {str(e)}")
    finally:
        if lease:
            await lease.release()


async def _drain_subagent_cleanup(agen, lease) -> None:
    """Run agen.aclose() + lease.release() to completion under cancel storm.

    Codex R4 P2#1: a plain ``await asyncio.shield(coro)`` becomes a
    "fire-and-forget task" the moment outer cancellation fires; the await
    returns CancelledError and execution continues, but the inner coro is
    now a detached task that the event loop is free to cancel during
    shutdown (e.g. worker termination). For the SSE-slot/quota cleanup
    we actually need both cleanups to RUN TO COMPLETION on the current
    loop tick. Pattern: spawn each cleanup as a task, then loop
    ``await asyncio.shield(task)`` until ``task.done()``. Any
    CancelledError on the outer is captured and re-raised after BOTH
    cleanups finish, so cooperative cancellation is honored without
    leaking the lease slot via the ``_released=True`` / Redis-zrem race
    at ``rate_limit.py:113``.
    """
    pending_cancel: BaseException | None = None
    agen_task = asyncio.create_task(agen.aclose())
    release_task = asyncio.create_task(lease.release())
    for cleanup_task, label in (
        (agen_task, "agen.aclose"),
        (release_task, "lease.release"),
    ):
        while not cleanup_task.done():
            try:
                await asyncio.shield(cleanup_task)
            except asyncio.CancelledError as exc:
                pending_cancel = exc
            except Exception as exc:
                logger.warning(
                    "subagent_research cleanup %s raised: %s", label, exc,
                )
                break
    if pending_cancel is not None:
        raise pending_cancel


@router.post(
    path="/{parent_session_id}/subagents/research",
    summary="Phase 1 minimal: spawn read-only research subagents",
    description="Spawn 1-3 read-only research subagents under the parent session, fan-out execution with deterministic summary join, multi-metric jsonl probe record.",
    dependencies=[Depends(rate_limit_chat)],
)
async def subagent_research(
    parent_session_id: str,
    request: ResearchSubagentRequest,
    current_user: CurrentUser,
    redis_client: RedisClient = Depends(get_redis),
    session_service: SessionService = Depends(get_session_service),
    service: SubagentResearchService = Depends(get_subagent_research_service),
) -> EventSourceResponse:
    """Canonical C1a route — delegates to ``_run_subagent_research``."""
    return await _run_subagent_research(
        resolved_parent_id=parent_session_id,
        request=request,
        current_user=current_user,
        redis_client=redis_client,
        session_service=session_service,
        service=service,
    )


@router.post(
    path="/{sample_session_id}/subagents/research",
    summary="Phase 1 minimal: spawn read-only research subagents (legacy path)",
    description="Deprecated legacy path; PR-4 removes. Prefer /{parent_session_id}/subagents/research.",
    dependencies=[Depends(rate_limit_chat)],
    include_in_schema=False,
)
async def subagent_research_legacy(
    sample_session_id: str,
    request: ResearchSubagentRequest,
    current_user: CurrentUser,
    redis_client: RedisClient = Depends(get_redis),
    session_service: SessionService = Depends(get_session_service),
    service: SubagentResearchService = Depends(get_subagent_research_service),
) -> EventSourceResponse:
    """Legacy alias — preserves the pre-C1a URL while PR-1..PR-3 migrate
    callers. ``include_in_schema=False`` keeps it out of OpenAPI so docs
    only advertise the canonical path. PR-4 removes both this handler and
    the ``_run_subagent_research`` indirection."""
    return await _run_subagent_research(
        resolved_parent_id=sample_session_id,
        request=request,
        current_user=current_user,
        redis_client=redis_client,
        session_service=session_service,
        service=service,
    )


async def _run_subagent_research(
    resolved_parent_id: str,
    request: ResearchSubagentRequest,
    current_user: CurrentUser,
    redis_client: RedisClient,
    session_service: SessionService,
    service: SubagentResearchService,
) -> EventSourceResponse:
    """Shared implementation for canonical + legacy subagent-research routes.

    C1a PR-1: both ``/{parent_session_id}/subagents/research`` (canonical,
    OpenAPI-visible) and ``/{sample_session_id}/subagents/research``
    (legacy, hidden via ``include_in_schema=False``) delegate here. Each
    route declares its own path-param kwarg so the OpenAPI schema for
    the canonical route advertises ``parent_session_id`` as a path
    parameter and NOT ``sample_session_id`` as a query parameter. PR-4
    drops the legacy route + this helper indirection.

    Order of side effects MUST match the chat endpoint preflight contract:
    1. Parent ownership check (404 on miss/cross-tenant → no existence leak)
    2. Acquire SSE connection lease (rate_limit_chat already enforced via
       dependency declaration; lease is a separate per-connection guard)
    3. Prime the inner ``run_research`` generator so preflight errors
       (ConflictError/BadRequestError) surface as proper HTTP 409/400 via
       FastAPI exception handlers, NOT as 200-then-broken-SSE-stream
    4. Stream remaining events through EventMapper → ServerSentEvent

    CS3 invariant: ``ServerSentEvent.id == event.id`` (== payload event_id).
    All probe event domain models extend BaseEvent which carries ``id``;
    EventMapper falls through to CommonSSEEvent for unknown event types so
    ``probe_run_id`` / ``child_session_id`` / ``metrics`` etc. survive the
    wire envelope via ``CommonEventData(extra="allow")``.

    Cancellation contract: when the SSE client disconnects, the outer
    generator's ``aclose()`` triggers ``GeneratorExit`` at its yield
    point. The outer ``finally`` MUST explicitly ``aclose()`` the inner
    ``run_research`` generator (an ``async for`` does NOT propagate
    close to the producer), otherwise ``run_research``'s PR-4 R3
    GeneratorExit safety (child cancel + sandbox suspend + quota
    release) is delayed until garbage collection — i.e. effectively
    leaks until the event loop is shut down.
    """
    parent = await session_service.get_session(
        session_id=resolved_parent_id,
        user_id=current_user.id,
        is_admin=current_user.is_admin(),
    )
    if parent is None:
        raise NotFoundError(
            f"Session {resolved_parent_id} not found or not accessible"
        )

    lease = await acquire_connection_limit(
        channel=RateLimitChannel.SSE,
        user_id=current_user.id,
        redis_client=redis_client,
    )
    lease.start_heartbeat()

    # Prime the inner generator BEFORE returning EventSourceResponse so
    # preflight errors (ConflictError / BadRequestError / NotFoundError
    # raised in `run_research` before its first yield) become real HTTP
    # 409 / 400 / 404 responses. Once we return EventSourceResponse the
    # status line is already 200 and exceptions can only manifest as a
    # truncated body. On any exception OR an empty stream we own the
    # lease release here; on success the streaming generator owns it.
    # PR-1: service still uses legacy ``sample_session_id`` kwarg. PR-2
    # flips the signature; until then we pass the resolved id under the
    # legacy keyword to avoid coupling two PRs together.
    agen = service.run_research(
        sample_session_id=resolved_parent_id,
        user_id=current_user.id,
        prompts=request.prompts,
        max_children=request.max_children,
    )
    first_event = None
    try:
        first_event = await agen.__anext__()
    except StopAsyncIteration:
        # Legitimate empty stream — fall through to the streaming response
        # with first_event=None; the generator below will just not yield.
        pass
    except BaseException as preflight_exc:
        # Includes ConflictError, BadRequestError, NotFoundError, asyncio
        # CancelledError, and any other startup failure. Drain cleanup to
        # completion (Codex R4 P2 helper handles cancel storm), then
        # re-raise the ORIGINAL preflight exception so FastAPI's exception
        # chain produces the right HTTP status (409 / 400 / 404 / 500).
        try:
            await _drain_subagent_cleanup(agen, lease)
        except asyncio.CancelledError:
            # If the route task is being cancelled, prefer the
            # cancellation over the preflight error (cooperative
            # cancel semantics — outer is going away regardless).
            raise
        except Exception as cleanup_exc:
            logger.warning(
                "subagent_research priming cleanup raised: %s", cleanup_exc,
            )
        raise preflight_exc

    async def event_generator() -> AsyncGenerator[ServerSentEvent, None]:
        try:
            if first_event is not None:
                sse_event = EventMapper.event_to_sse_event(first_event)
                yield ServerSentEvent(
                    id=first_event.id,
                    event=sse_event.event,
                    data=sse_event.to_sse_data_json(),
                )
            async for event in agen:
                sse_event = EventMapper.event_to_sse_event(event)
                yield ServerSentEvent(
                    id=event.id,
                    event=sse_event.event,
                    data=sse_event.to_sse_data_json(),
                )
        finally:
            # Codex R3 P1 + R4 P2: drain via _drain_subagent_cleanup so
            # `ConnectionLease.release()` reaches its Redis zrem
            # (rate_limit.py:113-126) even under cancel storm (task_group
            # cancel triggered by sibling _ping / handler finishing while
            # we are inside this finally) AND even on the second iteration
            # of the drain loop if the first cleanup task itself absorbs
            # a cancel. The helper re-raises CancelledError after both
            # cleanups complete, preserving cooperative cancellation.
            try:
                await _drain_subagent_cleanup(agen, lease)
            except asyncio.CancelledError:
                # Generator is being torn down — re-raising would just
                # propagate into sse-starlette which is already cancelling
                # its task group; let GeneratorExit/StopAsyncIteration
                # take its natural course.
                pass

    # Codex R2 P1: sse-starlette's _stream_response only aclose()s the
    # body_iterator on send_timeout (sse_starlette/sse.py:186-187), NOT on
    # client http.disconnect — that path just calls the close handler then
    # cancels the task group, leaving the body iterator in a suspended
    # state until GC. Without an explicit aclose() in the close handler,
    # the outer `event_generator` finally above (and therefore the inner
    # `agen.aclose()` + `lease.release()` chain) runs non-deterministically
    # on GC pressure rather than synchronously on disconnect. The chat
    # endpoint at session_routes.py:489-493 uses the same
    # `client_close_handler_callable` pattern.
    stream = event_generator()

    async def handle_client_close(_message: dict[str, object]) -> None:
        try:
            await stream.aclose()
        except RuntimeError as exc:
            # Race when the outer generator is mid-await on agen.__anext__()
            # at the moment disconnect fires. The subsequent task_group
            # cancel propagates CancelledError into that await, which still
            # triggers the generator's finally — so cleanup is preserved
            # even when we can't aclose synchronously here.
            if "already running" not in str(exc):
                raise

    return EventSourceResponse(
        stream,
        headers=SSE_HEADERS,
        client_close_handler_callable=handle_client_close,
    )
