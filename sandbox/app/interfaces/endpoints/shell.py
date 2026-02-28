import asyncio
import json
import logging
import os
from contextlib import suppress

from fastapi import APIRouter, Depends
from starlette.websockets import WebSocket, WebSocketDisconnect

from app.interfaces.errors.exceptions import BadRequestException
from app.interfaces.schemas.base import Response
from app.interfaces.schemas.shell import (
    ShellExecuteRequest,
    ShellKillRequest,
    ShellReadRequest,
    ShellResizeRequest,
    ShellWaitRequest,
    ShellWriteRequest,
)
from app.interfaces.service_dependencies import get_shell_service
from app.models.shell import (
    ShellExecuteResult,
    ShellKillResult,
    ShellReadResult,
    ShellWaitResult,
    ShellWriteResult,
)
from app.services.shell import ShellService

router = APIRouter(prefix="/shell", tags=["Shell模块"])
logger = logging.getLogger(__name__)


@router.post(
    path="/exec-command",
    response_model=Response[ShellExecuteResult],
)
async def exec_command(
    request: ShellExecuteRequest,
    shell_service: ShellService = Depends(get_shell_service),
) -> Response[ShellExecuteResult]:
    """在指定的Shell会话中运行命令"""
    # 1.判断下是否传递了session_id，如果不存在则新建一个session_id
    if not request.session_id or request.session_id == "":
        request.session_id = shell_service.create_session_id()

    # 2.判断下是否传递了执行目录，如果未传递则使用根目录作为执行路径
    if not request.exec_dir or request.exec_dir == "":
        request.exec_dir = os.path.expanduser("~")

    # 3.调用服务执行命令获取结果
    result = await shell_service.exec_command(
        session_id=request.session_id,
        exec_dir=request.exec_dir,
        command=request.command,
    )

    return Response.success(data=result)


@router.post(path="/read-shell-output", response_model=Response[ShellReadResult])
async def read_shell_output(
    request: ShellReadRequest,
    shell_service: ShellService = Depends(get_shell_service),
) -> Response[ShellReadResult]:
    """根据传递的会话id+是否返回控制台标识获取Shell命令执行结果"""
    # 1.判断下Shell会话id是否存在
    if not request.session_id or request.session_id == "":
        raise BadRequestException("Shell会话ID为空, 请核实后重试")

    # 2.调用服务获取命令执行结果
    result = await shell_service.read_shell_output(request.session_id, request.console)

    return Response.success(data=result)


@router.post(
    path="/wait-process",
    response_model=Response[ShellWaitResult],
)
async def wait_process(
    request: ShellWaitRequest,
    shell_service: ShellService = Depends(get_shell_service),
) -> Response[ShellWaitResult]:
    """传递会话id+描述执行等待并获取等待结果"""
    # 1.判断下Shell会话id是否存在
    if not request.session_id or request.session_id == "":
        raise BadRequestException("Shell会话ID为空, 请核实后重试")

    # 2.调用服务等待子进程
    result = await shell_service.wait_process(request.session_id, request.seconds)

    return Response.success(
        msg=f"进程结束, 返回状态码(returncode): {result.returncode}",
        data=result,
    )


@router.post(
    path="/write-shell-input",
    response_model=Response[ShellWriteResult],
)
async def write_shell_input(
    request: ShellWriteRequest,
    shell_service: ShellService = Depends(get_shell_service),
) -> Response[ShellWriteResult]:
    """根据传递的会话+写入内容+按下回车标识向指定子进程写入数据"""
    # 1.判断下Shell会话id是否存在
    if not request.session_id or request.session_id == "":
        raise BadRequestException("Shell会话ID为空, 请核实后重试")

    # 2.调用服务向子进程写入数据
    result = await shell_service.write_shell_input(
        session_id=request.session_id,
        input_text=request.input_text,
        press_enter=request.press_enter,
    )

    return Response.success(
        msg="向进程写入数据成功",
        data=result,
    )


@router.post(
    path="/kill-process",
    response_model=Response[ShellKillResult],
)
async def kill_process(
    request: ShellKillRequest,
    shell_service: ShellService = Depends(get_shell_service),
) -> Response[ShellKillResult]:
    """传递Shell会话id关闭指定会话"""
    # 1.判断下Shell会话id是否存在
    if not request.session_id or request.session_id == "":
        raise BadRequestException("Shell会话ID为空, 请核实后重试")

    # 2.调用服务关闭Shell会话
    result = await shell_service.kill_process(request.session_id)

    return Response.success(
        msg="进程终止" if result.status == "terminated" else "进程已结束",
        data=result,
    )


@router.post(
    path="/resize-shell",
    response_model=Response[ShellWriteResult],
)
async def resize_shell(
    request: ShellResizeRequest,
    shell_service: ShellService = Depends(get_shell_service),
) -> Response[ShellWriteResult]:
    """根据传递的会话ID调整PTY终端窗口大小"""
    shell_service.resize_pty_session(
        request.session_id,
        cols=request.cols,
        rows=request.rows,
    )
    return Response.success(
        msg="调整终端窗口大小成功",
        data=ShellWriteResult(status="success"),
    )


@router.websocket(path="/ws")
async def shell_websocket(
    websocket: WebSocket,
    session_id: str | None = None,
    shell_service: ShellService = Depends(get_shell_service),
) -> None:
    """Shell交互WebSocket，支持二进制输入与输出流。"""
    if not session_id:
        await websocket.close(code=4400, reason="缺少session_id")
        return

    await websocket.accept()
    try:
        await shell_service.ensure_pty_session(
            session_id=session_id,
            exec_dir=os.path.expanduser("~"),
        )
    except Exception as exc:
        await websocket.send_text(
            json.dumps(
                {
                    "type": "error",
                    "code": "session_init_failed",
                    "message": str(exc),
                },
                ensure_ascii=False,
            )
        )
        await websocket.close(code=1011, reason="session_init_failed")
        return

    await websocket.send_text(json.dumps({"type": "status", "state": "connected"}))

    closed = asyncio.Event()

    async def forward_to_shell() -> None:
        while not closed.is_set():
            try:
                message = await websocket.receive()
            except WebSocketDisconnect:
                break

            if message.get("type") == "websocket.disconnect":
                break

            payload_bytes = message.get("bytes")
            if payload_bytes is not None:
                try:
                    await shell_service.write_pty_input(session_id, payload_bytes)
                except Exception as exc:
                    await websocket.send_text(
                        json.dumps(
                            {
                                "type": "error",
                                "code": "write_failed",
                                "message": str(exc),
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
            except json.JSONDecodeError:
                continue

            if payload.get("type") != "resize":
                continue

            try:
                cols = max(1, min(int(payload.get("cols", 0)), 500))
                rows = max(1, min(int(payload.get("rows", 0)), 200))
                shell_service.resize_pty_session(
                    session_id,
                    cols=cols,
                    rows=rows,
                )
            except Exception as exc:
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "error",
                            "code": "resize_failed",
                            "message": str(exc),
                        },
                        ensure_ascii=False,
                    )
                )

    async def forward_from_shell() -> None:
        while not closed.is_set():
            try:
                chunk = await shell_service.read_pty_output(
                    session_id,
                    timeout_seconds=0.2,
                )
            except Exception as exc:
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "error",
                            "code": "read_failed",
                            "message": str(exc),
                        },
                        ensure_ascii=False,
                    )
                )
                return

            if chunk:
                await websocket.send_bytes(chunk)
                continue

            if not shell_service.is_pty_session_alive(session_id):
                await websocket.send_text(
                    json.dumps({"type": "status", "state": "closed"})
                )
                return

    forward_task1 = asyncio.create_task(forward_to_shell())
    forward_task2 = asyncio.create_task(forward_from_shell())
    try:
        done, pending = await asyncio.wait(
            [forward_task1, forward_task2],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in done:
            if task.cancelled():
                continue
            exc = task.exception()
            if exc and not isinstance(exc, WebSocketDisconnect):
                logger.debug("shell ws worker exited with error: %s", str(exc))
    finally:
        closed.set()
        for task in (forward_task1, forward_task2):
            if not task.done():
                task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        logger.debug("shell ws closed: session_id=%s", session_id)
