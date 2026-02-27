import asyncio
import logging
import os
import time
import uuid
from datetime import datetime
from typing import AsyncGenerator, Callable, Dict, List, Optional, Type

from app.application.errors.exceptions import (
    BadRequestError,
    ConflictError,
    ForbiddenError,
    NotFoundError,
)
from app.domain.external.file_storage import FileStorage
from app.domain.external.json_parser import JSONParser
from app.domain.external.llm import LLM
from app.domain.external.sandbox import Sandbox
from app.domain.external.search import SearchEngine
from app.domain.external.task import Task
from app.domain.models.app_config import (
    A2AConfig,
    AgentConfig,
    MCPConfig,
    SkillRiskPolicy,
)
from app.domain.models.context_overflow_config import ContextOverflowConfig
from app.domain.models.event import (
    BaseEvent,
    ControlAction,
    ControlEvent,
    ControlScope,
    ControlSource,
    DoneEvent,
    ErrorEvent,
    Event,
    MessageEvent,
    WaitEvent,
)
from app.domain.models.file import File
from app.domain.models.session import Session, SessionStatus

# from app.domain.repositories.file_repository import FileRepository
# from app.domain.repositories.session_repository import SessionRepository
from app.domain.repositories.uow import IUnitOfWork
from app.domain.services.agent_task_runner import AgentTaskRunner
from core.config import get_settings
from pydantic import TypeAdapter

logger = logging.getLogger(__name__)
OUTPUT_STREAM_POLL_BLOCK_MS = 1000
TAKEOVER_CANCEL_TIMEOUT_SECONDS = 15
TAKEOVER_LEASE_TTL_SECONDS = 15 * 60


class AgentService:
    """Manus智能体服务"""

    def __init__(
        self,
        uow_factory: Callable[[], IUnitOfWork],
        llm: LLM,
        agent_config: AgentConfig,
        mcp_config: MCPConfig,
        a2a_config: A2AConfig,
        sandbox_cls: Type[Sandbox],
        task_cls: Type[Task],
        json_parser: JSONParser,
        search_engine: SearchEngine,
        file_storage: FileStorage,
        skill_risk_policy: SkillRiskPolicy | None = None,
        overflow_config: ContextOverflowConfig | None = None,
        redis_client: object | None = None,
        # file_repository: FileRepository,
    ) -> None:
        """构造函数，完成Agent服务初始化"""
        self._uow_factory = uow_factory
        self._uow = uow_factory()
        self._llm = llm
        self._agent_config = agent_config
        self._mcp_config = mcp_config
        self._a2a_config = a2a_config
        self._skill_risk_policy = skill_risk_policy or SkillRiskPolicy()
        self._overflow_config = overflow_config or ContextOverflowConfig()
        self._sandbox_cls = sandbox_cls
        self._task_cls = task_cls
        self._json_parser = json_parser
        self._search_engine = search_engine
        self._file_storage = file_storage
        self._redis_client = redis_client
        self._background_tasks: set[asyncio.Task] = set()
        self._pending_timeout_tasks: dict[str, asyncio.Task] = {}
        self._settings = get_settings()
        # self._file_repository = file_repository
        logger.info(f"AgentService初始化成功")

    async def _get_task(self, session: Session) -> Optional[Task]:
        """根据传递的任务会话获取任务实例"""
        # 1.从会话中取出任务id
        task_id = session.task_id
        if not task_id:
            return None

        # 2.调用人物类的get方法获取对应的任务实例
        return self._task_cls.get(task_id)

    async def _create_task(self, session: Session) -> Task:
        """根据传递的会话创建一个新任务"""
        # 1.获取沙箱实例
        sandbox = None
        sandbox_id = session.sandbox_id
        if sandbox_id:
            sandbox = await self._sandbox_cls.get(sandbox_id)

        # 2.判断是否能获取到沙箱(如果没有则创建)
        if not sandbox:
            # 3.沙箱不存在则创建一个新的(有可能被释放了)
            sandbox = await self._sandbox_cls.create()
            session.sandbox_id = sandbox.id
            async with self._uow:
                await self._uow.session.save(session)

        # 4.从沙箱中获取浏览器实例
        browser = await sandbox.get_browser()
        if not browser:
            logger.error(f"获取沙箱[{sandbox.id}]中的浏览器实例失败")
            raise RuntimeError(f"获取沙箱[{sandbox.id}]中的浏览器实例失败")

        # 5.创建AgentTaskRunner
        task_runner = AgentTaskRunner(
            uow_factory=self._uow_factory,
            llm=self._llm,
            agent_config=self._agent_config,
            mcp_config=self._mcp_config,
            a2a_config=self._a2a_config,
            skill_risk_policy=self._skill_risk_policy,
            overflow_config=self._overflow_config,
            session_id=session.id,
            user_id=session.user_id,
            # session_repository=self._session_repository,
            file_storage=self._file_storage,
            # file_repository=self._file_repository,
            json_parser=self._json_parser,
            browser=browser,
            search_engine=self._search_engine,
            sandbox=sandbox,
        )

        # 6.创建任务Task并更新会话中的信息
        task = self._task_cls.create(task_runner=task_runner)
        session.task_id = task.id
        async with self._uow:
            await self._uow.session.save(session)

        return task

    async def _safe_update_unread_count(self, session_id: str) -> None:
        """在独立的后台任务中安全地更新未读消息计数

        该方法通过asyncio.create_task()调用，运行在一个全新的asyncio Task中，
        因此不受sse_starlette的anyio cancel scope影响，数据库操作可以正常完成。
        使用uow_factory创建全新的UoW实例，避免与被取消的上下文共享数据库连接。
        """
        try:
            uow = self._uow_factory()
            async with uow:
                await uow.session.update_unread_message_count(session_id, 0)
        except Exception as e:
            logger.warning(f"会话[{session_id}]后台更新未读消息计数失败: {e}")

    async def _get_accessible_session(
        self, session_id: str, user_id: str, is_admin: bool = False
    ) -> Session:
        """根据用户权限获取可访问会话"""
        async with self._uow:
            session = await self._uow.session.get_by_id(session_id)
        if not session:
            logger.error(f"尝试访问不存在的会话[{session_id}]")
            raise NotFoundError("任务会话不存在, 请核实后重试")
        if not is_admin and session.user_id != user_id:
            logger.error(f"用户[{user_id}]无权访问会话[{session_id}]")
            raise ForbiddenError("无权访问此会话")
        return session

    async def _check_attachments_access(
        self, attachments: Optional[List[str]], user_id: str, is_admin: bool = False
    ) -> None:
        """检查用户是否有权限使用传递的附件"""
        if not attachments:
            return

        async with self._uow:
            for attachment_id in attachments:
                file = await self._uow.file.get_by_id(attachment_id)
                if not file:
                    raise NotFoundError(f"附件[{attachment_id}]不存在")
                if not is_admin and (not file.user_id or file.user_id != user_id):
                    raise ForbiddenError(f"无权使用附件[{attachment_id}]")

    async def chat(
        self,
        session_id: str,
        user_id: str,
        is_admin: bool = False,
        message: Optional[str] = None,
        attachments: Optional[List[str]] = None,
        latest_event_id: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ) -> AsyncGenerator[BaseEvent, None]:
        """根据传递的信息调用Agent服务发起对话请求"""
        try:
            # 1.检查会话是否存在
            session = await self._get_accessible_session(session_id, user_id, is_admin)
            await self._check_attachments_access(attachments, user_id, is_admin)

            # 2.获取对应会话任务
            task = await self._get_task(session)

            logger.info(
                "会话[%s] chat请求: message_present=%s task_exists=%s session_status=%s",
                session_id,
                bool(message),
                task is not None,
                session.status.value,
            )

            # 3.判断是否传递了message
            if message:
                if session.status in {
                    SessionStatus.TAKEOVER_PENDING,
                    SessionStatus.TAKEOVER,
                }:
                    raise BadRequestError("当前会话处于接管状态，暂不支持聊天输入")

                # 4.判断会话的状态是什么,如果不是运行中则表示已完成或者空闲中
                if session.status != SessionStatus.RUNNING or task is None:
                    # 5.不在运行中需要创建一个新的task并启动
                    task = await self._create_task(session)
                    if not task:
                        logger.error(f"会话[{session_id}]创建任务失败")
                        raise RuntimeError(f"会话[{session_id}]创建任务失败")

                # 6.传递了消息则更新会话中的最后一条消息
                async with self._uow:
                    await self._uow.session.update_latest_message(
                        session_id=session_id,
                        message=message,
                        timestamp=timestamp or datetime.now(),
                    )

                # 7.创建一个人类消息事件
                message_event = MessageEvent(
                    role="user",
                    message=message,
                    attachments=(
                        [File(id=attachment) for attachment in attachments]
                        if attachments
                        else []
                    ),
                )

                # 8.将事件添加到任务的输入流中，好让Agent获取到数据
                event_id = await task.input_stream.put(message_event.model_dump_json())
                message_event.id = event_id
                async with self._uow:
                    await self._uow.session.add_event(session_id, message_event)

                # 9.立刻把用户消息返回给前端，避免依赖后续拉取导致消息缺失
                yield message_event

                # 10.执行任务
                await task.invoke()
                logger.info(
                    f"往会话[{session_id}]输入消息队列写入消息: {message[:50]}..."
                )
            elif session.status == SessionStatus.RUNNING and task is None:
                logger.warning(
                    "会话[%s]状态自愈: status_reconciled=true from=running to=completed message_present=false task_exists=false",
                    session_id,
                )
                async with self._uow:
                    await self._uow.session.update_status(
                        session_id, SessionStatus.COMPLETED
                    )
                session = session.model_copy(update={"status": SessionStatus.COMPLETED})

            # 11.记录日志展示会话已启动
            logger.info(f"会话[{session_id}]已启动")
            logger.info(f"会话[{session_id}]任务实例: {task}")

            # 12.从任务的输出流中读取数据
            while task:
                # 13.从输出消息队列中获取数据
                event_id, event_str = await task.output_stream.get(
                    start_id=latest_event_id, block_ms=OUTPUT_STREAM_POLL_BLOCK_MS
                )
                if event_str is None:
                    logger.debug(f"在会话[{session_id}]输出队列中未发现事件内容")
                    if task.done:
                        break
                    continue
                latest_event_id = event_id

                # 14.使用Pydantic提供的类型适配器将event_str转换为指定类实例
                event = TypeAdapter(Event).validate_json(event_str)
                event.id = event_id
                logger.debug(f"从会话[{session_id}]中获取事件: {type(event).__name__}")

                if isinstance(event, ControlEvent):
                    if event.action == ControlAction.REQUESTED:
                        self._schedule_pending_timeout(session_id)
                    elif event.action in {
                        ControlAction.STARTED,
                        ControlAction.REJECTED,
                        ControlAction.ENDED,
                        ControlAction.EXPIRED,
                    }:
                        self._cancel_pending_timeout(session_id)

                # 15.将未读消息数重置为0
                async with self._uow:
                    await self._uow.session.update_unread_message_count(session_id, 0)

                # 16.将事件返回并判断事件类型是否为结束类型
                yield event
                if isinstance(event, (DoneEvent, ErrorEvent, WaitEvent, ControlEvent)):
                    break

            # 17.循环外面表示这次任务AI端的已结束
            logger.info(f"会话[{session_id}]本轮运行结束")
        except BadRequestError:
            raise
        except Exception as e:
            # 18.记录日志并返回错误事件
            logger.error(f"任务会话[{session_id}]对话出错: {str(e)}")
            event = ErrorEvent(error=str(e))
            try:
                async with self._uow:
                    await self._uow.session.add_event(session_id, event)
            except (asyncio.CancelledError, Exception) as add_err:
                logger.warning(
                    f"会话[{session_id}]添加错误事件失败(可能是客户端断开连接): {add_err}"
                )
            yield event
        finally:
            # 19.会话完整传递给前端后，表示至少用户肯定收到了这些消息，所以不应该有未读消息数
            # 注意：当SSE客户端断开连接时，sse_starlette使用anyio cancel scope取消当前Task中
            # 所有的await操作（asyncio.shield也无法对抗anyio的cancel scope）。
            # 如果在finally块中直接执行数据库操作，该操作会被立即取消，并且SQLAlchemy在尝试
            # 终止被中断的连接时也会被取消，从而产生ERROR日志并可能污染连接池。
            # 解决方案：将数据库更新操作放到独立的asyncio Task中执行，新Task不受当前
            # cancel scope的影响，可以正常完成数据库操作。
            try:
                asyncio.create_task(self._safe_update_unread_count(session_id))
            except RuntimeError:
                # 事件循环已关闭（如应用正在关闭），无法创建后台任务
                logger.warning(f"会话[{session_id}]无法创建后台任务更新未读消息计数")

    async def stop_session(
        self, session_id: str, user_id: str, is_admin: bool = False
    ) -> None:
        """根据传递的会话id停止指定会话"""
        # 1.查找会话是否存在
        session = await self._get_accessible_session(session_id, user_id, is_admin)

        # 2.根据会话获取任务信息
        task = await self._get_task(session)
        if task:
            task.cancel(reason="stop")

        # 3.更新会话任务状态
        async with self._uow:
            await self._uow.session.update_status(session_id, SessionStatus.COMPLETED)

    @staticmethod
    def _get_latest_control_event(session: Session) -> Optional[ControlEvent]:
        for event in reversed(session.events):
            if isinstance(event, ControlEvent):
                return event
        return None

    @staticmethod
    def _lease_key(session_id: str) -> str:
        return f"takeover:lease:{session_id}"

    @staticmethod
    def _lease_value(takeover_id: str, operator_user_id: str) -> str:
        return f"{takeover_id}:{operator_user_id}"

    def _get_redis_connection(self):
        if not self._redis_client:
            return None
        try:
            return self._redis_client.client
        except Exception as exc:
            raise BadRequestError("接管租约服务暂不可用，请稍后重试") from exc

    @staticmethod
    def _parse_csv(raw: str) -> set[str]:
        return {item.strip() for item in (raw or "").split(",") if item.strip()}

    def _resolve_operator_role(self, *, is_admin: bool, user_role: Optional[str]) -> str:
        if user_role:
            return user_role
        return "super_admin" if is_admin else "user"

    @staticmethod
    def _resolve_worker_count() -> int:
        for key in ("WEB_CONCURRENCY", "UVICORN_WORKERS"):
            raw_value = (os.getenv(key) or "").strip()
            if not raw_value:
                continue
            try:
                worker_count = int(raw_value)
            except ValueError:
                continue
            if worker_count > 0:
                return worker_count
        return 1

    def _assert_takeover_capability(
        self,
        *,
        user_id: str,
        is_admin: bool,
        user_role: Optional[str],
        scope: Optional[ControlScope] = None,
    ) -> None:
        if not self._settings.feature_takeover_enabled:
            raise ForbiddenError("接管功能未启用")

        if scope == ControlScope.BROWSER and not self._settings.feature_takeover_browser_enabled:
            raise ForbiddenError("浏览器接管功能未启用")

        role = self._resolve_operator_role(is_admin=is_admin, user_role=user_role)
        allowed_roles = self._parse_csv(self._settings.feature_takeover_allowed_roles)
        if allowed_roles and role not in allowed_roles:
            raise ForbiddenError("当前角色无接管权限")

        whitelist = self._parse_csv(self._settings.feature_takeover_user_whitelist)
        if whitelist and user_id not in whitelist:
            raise ForbiddenError("当前用户不在接管白名单中")

        if (
            self._settings.feature_takeover_single_worker_only
            and self._resolve_worker_count() > 1
        ):
            raise ForbiddenError("当前部署为多Worker模式，接管功能仅支持单Worker")

    async def _acquire_takeover_lease(
        self,
        session_id: str,
        *,
        takeover_id: str,
        operator_user_id: str,
        ttl_seconds: int = TAKEOVER_LEASE_TTL_SECONDS,
    ) -> bool:
        redis = self._get_redis_connection()
        if redis is None:
            return True

        lease_key = self._lease_key(session_id)
        lease_value = self._lease_value(takeover_id, operator_user_id)
        acquired = await redis.set(
            lease_key,
            lease_value,
            ex=max(ttl_seconds, 1),
            nx=True,
        )
        return bool(acquired)

    async def _renew_takeover_lease(
        self,
        session_id: str,
        *,
        takeover_id: str,
        operator_user_id: str,
        ttl_seconds: int = TAKEOVER_LEASE_TTL_SECONDS,
    ) -> bool:
        redis = self._get_redis_connection()
        if redis is None:
            return True

        lease_key = self._lease_key(session_id)
        lease_value = self._lease_value(takeover_id, operator_user_id)
        script = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
    redis.call('EXPIRE', KEYS[1], tonumber(ARGV[2]))
    return 1
else
    return 0
end
"""
        renewed = await redis.eval(
            script,
            1,
            lease_key,
            lease_value,
            max(ttl_seconds, 1),
        )
        return int(renewed) == 1

    async def _release_takeover_lease(
        self,
        session_id: str,
        *,
        takeover_id: str,
        operator_user_id: str,
    ) -> None:
        redis = self._get_redis_connection()
        if redis is None:
            return

        lease_key = self._lease_key(session_id)
        lease_value = self._lease_value(takeover_id, operator_user_id)
        script = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
    return redis.call('DEL', KEYS[1])
else
    return 0
end
"""
        await redis.eval(
            script,
            1,
            lease_key,
            lease_value,
        )

    async def _force_release_takeover_lease(self, session_id: str) -> None:
        redis = self._get_redis_connection()
        if redis is None:
            return
        await redis.delete(self._lease_key(session_id))

    def _track_background_task(self, task: asyncio.Task) -> None:
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    def _cancel_pending_timeout(self, session_id: str) -> None:
        timeout_task = self._pending_timeout_tasks.pop(session_id, None)
        if timeout_task and not timeout_task.done():
            timeout_task.cancel()

    def _schedule_pending_timeout(self, session_id: str) -> None:
        ttl = max(self._settings.feature_takeover_pending_ttl_seconds, 1)
        self._cancel_pending_timeout(session_id)
        timeout_task = asyncio.create_task(
            self._handle_takeover_pending_timeout(session_id=session_id, ttl_seconds=ttl)
        )
        self._pending_timeout_tasks[session_id] = timeout_task
        timeout_task.add_done_callback(lambda _: self._pending_timeout_tasks.pop(session_id, None))
        self._track_background_task(timeout_task)

    async def _handle_takeover_pending_timeout(
        self,
        *,
        session_id: str,
        ttl_seconds: int,
    ) -> None:
        try:
            await asyncio.sleep(max(ttl_seconds, 1))
            uow = self._uow_factory()
            async with uow:
                # 读取时加行锁，避免与 reject_takeover 等并发状态迁移发生 TOCTOU 竞态。
                session = await uow.session.get_by_id_for_update(session_id)
                if not session or session.status != SessionStatus.TAKEOVER_PENDING:
                    return

                latest_control = self._get_latest_control_event(session)
                takeover_id = latest_control.takeover_id if latest_control else None
                await uow.session.add_event(
                    session_id,
                    ControlEvent(
                        action=ControlAction.EXPIRED,
                        source=ControlSource.SYSTEM,
                        reason="pending_timeout",
                        request_status="expired",
                        takeover_id=takeover_id,
                    ),
                )
                await uow.session.update_status(session_id, SessionStatus.COMPLETED)
            await self._force_release_takeover_lease(session_id)
            # TODO(P2): 增加 takeover lease 到期监控，实现 takeover -> takeover_pending 自动迁移。
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("会话[%s]处理接管待决超时失败: %s", session_id, exc)

    def _schedule_takeover_completion(
        self,
        *,
        session_id: str,
        task: Task,
        scope: ControlScope,
        takeover_id: str,
        operator_user_id: str,
        cancel_timeout_seconds: int,
    ) -> None:
        background_task = asyncio.create_task(
            self._complete_takeover_after_cancel(
                session_id=session_id,
                task=task,
                scope=scope,
                takeover_id=takeover_id,
                operator_user_id=operator_user_id,
                cancel_timeout_seconds=cancel_timeout_seconds,
            )
        )
        self._track_background_task(background_task)

    async def _append_control_event(
        self,
        session_id: str,
        *,
        action: ControlAction,
        source: ControlSource,
        scope: Optional[ControlScope] = None,
        reason: Optional[str] = None,
        handoff_mode: Optional[str] = None,
        request_status: Optional[str] = None,
        takeover_id: Optional[str] = None,
        task: Optional[Task] = None,
    ) -> ControlEvent:
        control_event = ControlEvent(
            action=action,
            source=source,
            scope=scope,
            reason=reason,
            handoff_mode=handoff_mode,
            request_status=request_status,
            takeover_id=takeover_id,
        )
        if task:
            try:
                event_id = await task.output_stream.put(control_event.model_dump_json())
                control_event.id = event_id
            except Exception as exc:
                logger.warning(
                    "会话[%s]写入ControlEvent到输出流失败，降级为仅落库: %s",
                    session_id,
                    exc,
                )
        uow = self._uow_factory()
        async with uow:
            await uow.session.add_event(session_id, control_event)
        return control_event

    async def _append_error_event(
        self, session_id: str, *, error: str, task: Optional[Task] = None
    ) -> ErrorEvent:
        error_event = ErrorEvent(error=error)
        if task:
            try:
                event_id = await task.output_stream.put(error_event.model_dump_json())
                error_event.id = event_id
            except Exception as exc:
                logger.warning(
                    "会话[%s]写入ErrorEvent到输出流失败，降级为仅落库: %s",
                    session_id,
                    exc,
                )
        uow = self._uow_factory()
        async with uow:
            await uow.session.add_event(session_id, error_event)
        return error_event

    async def _inject_handoff_message(self, task: Task, text: str) -> str:
        """向任务输入流注入一条handoff消息，用于恢复执行上下文。"""
        handoff_event = MessageEvent(
            role="system",
            message=text,
        )
        return await task.input_stream.put(handoff_event.model_dump_json())

    async def _resume_task_with_handoff(self, session: Session, text: str) -> Task:
        """重建任务并注入handoff消息后启动任务。"""
        task = await self._create_task(session)
        await self._inject_handoff_message(task, text)
        await task.invoke()
        return task

    async def _rollback_resume_failed(
        self,
        session_id: str,
        *,
        error: Exception,
        task: Optional[Task],
        takeover_id: Optional[str] = None,
        operator_user_id: Optional[str] = None,
    ) -> None:
        logger.exception("会话[%s]恢复执行失败: %s", session_id, error)
        if task and not task.done:
            task.cancel(reason="takeover_timeout")
        await self._append_error_event(
            session_id,
            error=f"恢复执行失败: {str(error)}",
            task=task,
        )
        uow = self._uow_factory()
        async with uow:
            await uow.session.update_status(session_id, SessionStatus.COMPLETED)
        await self._append_control_event(
            session_id,
            action=ControlAction.ENDED,
            source=ControlSource.SYSTEM,
            handoff_mode="complete",
            reason="resume_failed",
            request_status="failed",
            takeover_id=takeover_id,
            task=task,
        )
        if takeover_id and operator_user_id:
            await self._release_takeover_lease(
                session_id,
                takeover_id=takeover_id,
                operator_user_id=operator_user_id,
            )

    async def _complete_takeover_after_cancel(
        self,
        *,
        session_id: str,
        task: Task,
        scope: ControlScope,
        takeover_id: str,
        operator_user_id: str,
        cancel_timeout_seconds: int,
    ) -> None:
        """等待running任务取消完成，异步推进接管状态。"""
        try:
            deadline = time.monotonic() + max(cancel_timeout_seconds, 1)
            while not task.done and time.monotonic() < deadline:
                await asyncio.sleep(0.05)

            if task.done:
                uow = self._uow_factory()
                async with uow:
                    await uow.session.update_status(
                        session_id, SessionStatus.TAKEOVER
                    )
                await self._append_control_event(
                    session_id,
                    action=ControlAction.STARTED,
                    source=ControlSource.SYSTEM,
                    scope=scope,
                    request_status="started",
                    takeover_id=takeover_id,
                    task=task,
                )
                return

            await self._append_control_event(
                session_id,
                action=ControlAction.REJECTED,
                source=ControlSource.SYSTEM,
                scope=scope,
                reason="cancel_timeout",
                request_status="rejected",
                takeover_id=takeover_id,
                task=task,
            )
            await self._release_takeover_lease(
                session_id,
                takeover_id=takeover_id,
                operator_user_id=operator_user_id,
            )
        except Exception as exc:
            logger.exception("会话[%s]异步确认接管失败: %s", session_id, exc)
            await self._append_error_event(
                session_id,
                error=f"接管确认失败: {str(exc)}",
                task=task,
            )
            await self._release_takeover_lease(
                session_id,
                takeover_id=takeover_id,
                operator_user_id=operator_user_id,
            )

    async def get_takeover(
        self,
        session_id: str,
        user_id: str,
        is_admin: bool = False,
        user_role: Optional[str] = None,
    ) -> Dict[str, object]:
        """获取会话接管状态"""
        self._assert_takeover_capability(
            user_id=user_id,
            is_admin=is_admin,
            user_role=user_role,
        )
        session = await self._get_accessible_session(session_id, user_id, is_admin)
        latest_control = self._get_latest_control_event(session)
        if not latest_control:
            return {"status": session.status}

        return {
            "status": session.status,
            "takeover_id": latest_control.takeover_id,
            "request_status": latest_control.request_status,
            "reason": latest_control.reason,
            "scope": latest_control.scope.value if latest_control.scope else None,
            "handoff_mode": latest_control.handoff_mode,
        }

    async def start_takeover(
        self,
        session_id: str,
        user_id: str,
        *,
        scope: str = "shell",
        is_admin: bool = False,
        user_role: Optional[str] = None,
        cancel_timeout_seconds: int = TAKEOVER_CANCEL_TIMEOUT_SECONDS,
    ) -> Dict[str, object]:
        """启动会话接管"""
        try:
            control_scope = ControlScope(scope)
        except ValueError as exc:
            raise BadRequestError("scope仅支持 shell 或 browser") from exc
        self._assert_takeover_capability(
            user_id=user_id,
            is_admin=is_admin,
            user_role=user_role,
            scope=control_scope,
        )
        session = await self._get_accessible_session(session_id, user_id, is_admin)

        if session.status == SessionStatus.TAKEOVER:
            latest_control = self._get_latest_control_event(session)
            return {
                "request_status": "started",
                "status": SessionStatus.TAKEOVER,
                "scope": (
                    latest_control.scope.value
                    if latest_control and latest_control.scope
                    else control_scope.value
                ),
                "takeover_id": latest_control.takeover_id if latest_control else None,
            }

        if session.status not in {
            SessionStatus.RUNNING,
            SessionStatus.WAITING,
            SessionStatus.TAKEOVER_PENDING,
        }:
            raise BadRequestError(
                f"当前状态[{session.status.value}]不支持启动接管"
            )

        takeover_id = f"tk_{uuid.uuid4().hex[:12]}"
        acquired = await self._acquire_takeover_lease(
            session_id,
            takeover_id=takeover_id,
            operator_user_id=user_id,
            ttl_seconds=max(self._settings.feature_takeover_lease_ttl_seconds, 1),
        )
        if not acquired:
            raise ConflictError("接管租约冲突，请稍后重试")

        if session.status == SessionStatus.RUNNING:
            task = await self._get_task(session)
            if task:
                task.cancel(reason="takeover_start")
                self._schedule_takeover_completion(
                    session_id=session_id,
                    task=task,
                    scope=control_scope,
                    takeover_id=takeover_id,
                    operator_user_id=user_id,
                    cancel_timeout_seconds=cancel_timeout_seconds,
                )
                return {
                    "takeover_id": takeover_id,
                    "request_status": "starting",
                    "status": SessionStatus.RUNNING,
                    "scope": control_scope.value,
                }
            logger.warning(
                "会话[%s]处于RUNNING但无活跃task，直接按无任务路径进入接管",
                session_id,
            )

        if session.status == SessionStatus.TAKEOVER_PENDING:
            self._cancel_pending_timeout(session_id)

        async with self._uow:
            await self._uow.session.update_status(session_id, SessionStatus.TAKEOVER)
        await self._append_control_event(
            session_id,
            action=ControlAction.STARTED,
            source=ControlSource.USER,
            scope=control_scope,
            request_status="started",
            takeover_id=takeover_id,
        )
        return {
            "takeover_id": takeover_id,
            "request_status": "started",
            "status": SessionStatus.TAKEOVER,
            "scope": control_scope.value,
        }

    async def renew_takeover(
        self,
        session_id: str,
        user_id: str,
        *,
        takeover_id: str,
        is_admin: bool = False,
        user_role: Optional[str] = None,
        lease_ttl_seconds: Optional[int] = None,
    ) -> Dict[str, object]:
        """续期会话接管租约"""
        self._assert_takeover_capability(
            user_id=user_id,
            is_admin=is_admin,
            user_role=user_role,
        )
        session = await self._get_accessible_session(session_id, user_id, is_admin)
        if session.status != SessionStatus.TAKEOVER:
            raise BadRequestError("当前会话不处于接管状态")

        effective_ttl_seconds = (
            lease_ttl_seconds
            if lease_ttl_seconds is not None and lease_ttl_seconds > 0
            else self._settings.feature_takeover_lease_ttl_seconds
        )
        renewed = await self._renew_takeover_lease(
            session_id,
            takeover_id=takeover_id,
            operator_user_id=user_id,
            ttl_seconds=max(effective_ttl_seconds, 1),
        )
        if not renewed:
            raise ConflictError("接管租约已失效或不匹配")

        await self._append_control_event(
            session_id,
            action=ControlAction.RENEWED,
            source=ControlSource.USER,
            request_status="renewed",
            takeover_id=takeover_id,
        )
        return {
            "status": SessionStatus.TAKEOVER,
            "request_status": "renewed",
            "takeover_id": takeover_id,
        }

    async def reject_takeover(
        self,
        session_id: str,
        user_id: str,
        *,
        decision: str,
        is_admin: bool = False,
        user_role: Optional[str] = None,
    ) -> Dict[str, object]:
        """处理接管请求拒绝"""
        self._assert_takeover_capability(
            user_id=user_id,
            is_admin=is_admin,
            user_role=user_role,
        )
        session = await self._get_accessible_session(session_id, user_id, is_admin)
        if session.status != SessionStatus.TAKEOVER_PENDING:
            raise BadRequestError("当前会话不处于待接管状态")
        self._cancel_pending_timeout(session_id)
        latest_control = self._get_latest_control_event(session)
        takeover_id = latest_control.takeover_id if latest_control else None

        decision_normalized = (decision or "").strip().lower()
        if decision_normalized == "continue":
            resumed_task: Optional[Task] = None
            try:
                resumed_task = await self._resume_task_with_handoff(
                    session,
                    "用户拒绝接管请求，请继续执行上次任务。",
                )
            except Exception as exc:
                await self._rollback_resume_failed(
                    session_id,
                    error=exc,
                    task=resumed_task,
                )
                return {"status": SessionStatus.COMPLETED, "reason": "resume_failed"}

            async with self._uow:
                await self._uow.session.update_status(session_id, SessionStatus.RUNNING)
            await self._append_control_event(
                session_id,
                action=ControlAction.REJECTED,
                source=ControlSource.USER,
                reason="continue",
                takeover_id=takeover_id,
                task=resumed_task,
            )
            return {"status": SessionStatus.RUNNING, "reason": "continue"}

        if decision_normalized == "terminate":
            async with self._uow:
                await self._uow.session.update_status(session_id, SessionStatus.COMPLETED)
            await self._append_control_event(
                session_id,
                action=ControlAction.REJECTED,
                source=ControlSource.USER,
                reason="terminate",
                takeover_id=takeover_id,
            )
            await self._force_release_takeover_lease(session_id)
            return {"status": SessionStatus.COMPLETED, "reason": "terminate"}

        raise BadRequestError("decision仅支持 continue 或 terminate")

    async def end_takeover(
        self,
        session_id: str,
        user_id: str,
        *,
        handoff_mode: str = "complete",
        is_admin: bool = False,
        user_role: Optional[str] = None,
    ) -> Dict[str, object]:
        """结束接管并交还控制"""
        self._assert_takeover_capability(
            user_id=user_id,
            is_admin=is_admin,
            user_role=user_role,
        )
        session = await self._get_accessible_session(session_id, user_id, is_admin)
        if session.status != SessionStatus.TAKEOVER:
            raise BadRequestError("当前会话不处于接管状态")
        latest_control = self._get_latest_control_event(session)
        takeover_id = latest_control.takeover_id if latest_control else None

        mode = (handoff_mode or "").strip().lower()
        if mode == "continue":
            resumed_task: Optional[Task] = None
            try:
                resumed_task = await self._resume_task_with_handoff(
                    session,
                    "用户已结束接管并交还控制，请继续执行未完成任务。",
                )
            except Exception as exc:
                await self._rollback_resume_failed(
                    session_id,
                    error=exc,
                    task=resumed_task,
                    takeover_id=takeover_id,
                    operator_user_id=user_id,
                )
                return {
                    "status": SessionStatus.COMPLETED,
                    "handoff_mode": "complete",
                }

            async with self._uow:
                await self._uow.session.update_status(session_id, SessionStatus.RUNNING)
            await self._append_control_event(
                session_id,
                action=ControlAction.ENDED,
                source=ControlSource.USER,
                handoff_mode="continue",
                takeover_id=takeover_id,
                task=resumed_task,
            )
            if takeover_id:
                await self._release_takeover_lease(
                    session_id,
                    takeover_id=takeover_id,
                    operator_user_id=user_id,
                )
            return {"status": SessionStatus.RUNNING, "handoff_mode": "continue"}

        if mode == "complete":
            async with self._uow:
                await self._uow.session.update_status(session_id, SessionStatus.COMPLETED)
            await self._append_control_event(
                session_id,
                action=ControlAction.ENDED,
                source=ControlSource.USER,
                handoff_mode="complete",
                takeover_id=takeover_id,
            )
            if takeover_id:
                await self._release_takeover_lease(
                    session_id,
                    takeover_id=takeover_id,
                    operator_user_id=user_id,
                )
            return {"status": SessionStatus.COMPLETED, "handoff_mode": "complete"}

        raise BadRequestError("handoff_mode仅支持 continue 或 complete")

    async def shutdown(self) -> None:
        """关闭Agent服务"""
        for task in list(self._pending_timeout_tasks.values()):
            if not task.done():
                task.cancel()
        self._pending_timeout_tasks.clear()
        for task in list(self._background_tasks):
            if not task.done():
                task.cancel()
        self._background_tasks.clear()
        logger.info("正在清除所有会话任务资源并释放")
        await self._task_cls.destroy()
        logger.info("所有会话任务资源清除成功")
