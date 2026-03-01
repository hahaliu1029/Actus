import asyncio
import json
import logging
from datetime import datetime
from typing import AsyncGenerator, Dict, Optional
from urllib.parse import quote

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
    ChatRequest,
    CreateSessionResponse,
    EndTakeoverRequest,
    EndTakeoverResponse,
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
    ShellReadRequest,
    ShellReadResponse,
    StartTakeoverRequest,
    StartTakeoverResponse,
)
from app.interfaces.service_dependencies import get_agent_service, get_session_service
from app.infrastructure.storage.redis import RedisClient, get_redis
from core.config import get_settings
from fastapi import APIRouter, Depends, Response as FastAPIResponse
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
                    ListSessionItem(
                        session_id=session.id,
                        title=session.title,
                        latest_message=session.latest_message,
                        latest_message_at=session.latest_message_at,
                        status=session.status,
                        unread_message_count=session.unread_message_count,
                    )
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
) -> Response[ListSessionResponse]:
    """获取当前用户的任务会话基础信息列表"""
    sessions = await session_service.get_all_sessions(
        current_user.id, current_user.is_admin()
    )
    session_items = [
        ListSessionItem(
            session_id=session.id,
            title=session.title,
            latest_message=session.latest_message,
            latest_message_at=session.latest_message_at,
            status=session.status,
            unread_message_count=session.unread_message_count,
        )
        for session in sessions
    ]
    return Response.success(
        msg="获取任务会话列表成功", data=ListSessionResponse(sessions=session_items)
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
    current_user: CurrentUser,
    agent_service: AgentService = Depends(get_agent_service),
    redis_client: RedisClient = Depends(get_redis),
) -> EventSourceResponse:
    """根据传递的会话id+chat请求数据向指定会话发起聊天请求"""
    lease = await acquire_connection_limit(
        channel=RateLimitChannel.SSE,
        user_id=current_user.id,
        redis_client=redis_client,
    )
    lease.start_heartbeat()

    async def event_generator() -> AsyncGenerator[ServerSentEvent, None]:
        """定义事件生成器，用于配合EventSourceResponse生成流式响应数据"""
        try:
            # 1.调用Agent服务发起聊天
            async for event in agent_service.chat(
                session_id=session_id,
                user_id=current_user.id,
                is_admin=current_user.is_admin(),
                message=request.message,
                attachments=request.attachments,
                latest_event_id=request.event_id,
                timestamp=(
                    datetime.fromtimestamp(request.timestamp)
                    if request.timestamp
                    else None
                ),
            ):
                # 2.将Agent事件转换为sse数据(因为普通的event没法通过流式事件传输)
                sse_event = EventMapper.event_to_sse_event(event)
                if sse_event:
                    yield ServerSentEvent(
                        event=sse_event.event,
                        data=sse_event.data.model_dump_json(),
                    )
        finally:
            await lease.release()

    return EventSourceResponse(event_generator(), headers=SSE_HEADERS)


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
) -> Response[GetSessionResponse]:
    """传递指定会话id获取该会话的对话详情"""
    session = await session_service.get_session(
        session_id=session_id,
        user_id=current_user.id,
        is_admin=current_user.is_admin(),
    )
    if not session:
        raise NotFoundError("该会话不存在，请核实后重试")
    return Response.success(
        msg="获取会话详情成功",
        data=GetSessionResponse(
            session_id=session.id,
            title=session.title,
            status=session.status,
            events=EventMapper.events_to_sse_events(session.events),
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
) -> Response[Optional[Dict]]:
    """根据传递的指定会话id停止对应任务会话"""
    await agent_service.stop_session(
        session_id=session_id,
        user_id=current_user.id,
        is_admin=current_user.is_admin(),
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
                done, pending = await asyncio.wait(
                    tasks,
                    return_when=asyncio.FIRST_COMPLETED,
                )
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
            done, pending = await asyncio.wait(
                tasks,
                return_when=asyncio.FIRST_COMPLETED,
            )

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

            # 8.等待任意任务结束意味WebSocket连接终端
            done, pending = await asyncio.wait(
                [forward_task1, forward_task2],
                return_when=asyncio.FIRST_COMPLETED,
            )
            logger.info("WebSocket连接已关闭")

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
