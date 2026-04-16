import logging
from typing import TYPE_CHECKING, Callable, List, Optional, Type

from app.application.errors.exceptions import (
    ForbiddenError,
    NotFoundError,
    ServerRequestsError,
)
from app.domain.errors.sandbox_lifecycle import (
    SandboxLifecycleError,
    SessionFinalizedError,
    SessionSuspendedError,
    SessionUnboundError,
)
from app.domain.external.sandbox import SandboxHandle
from app.domain.external.task import Task
from app.domain.models.file import File
from app.domain.models.session import DestroyReason, Session

# from app.domain.repositories.session_repository import SessionRepository
from app.domain.repositories.uow import IUnitOfWork
from app.interfaces.schemas.session import FileReadResponse, ShellReadResponse
from core.config import get_settings

if TYPE_CHECKING:
    from app.application.services.sandbox_lifecycle_service import SandboxLifecycleService

logger = logging.getLogger(__name__)


class SessionService:
    """会话服务"""

    def __init__(
        self,
        uow_factory: Callable[[], IUnitOfWork],
        task_cls: Optional[Type[Task]] = None,
        sandbox_lifecycle_service: Optional["SandboxLifecycleService"] = None,
    ) -> None:
        """构造函数，完成会话服务初始化"""
        self._uow_factory = uow_factory
        self._uow = uow_factory()
        self._task_cls = task_cls
        self._lifecycle = sandbox_lifecycle_service

    async def create_session(self, user_id: str) -> Session:
        """创建一个空白的新任务会话"""
        logger.info("创建一个空白新任务会话")
        session = Session(title="新对话", user_id=user_id)
        async with self._uow:
            await self._uow.session.save(session)
        logger.info(f"成功创建一个新任务会话: {session.id}")
        return session

    async def get_all_sessions(
        self, user_id: str, is_admin: bool = False
    ) -> List[Session]:
        """获取用户所有任务会话列表"""
        async with self._uow:
            if is_admin:
                return await self._uow.session.get_all()
            return await self._uow.session.get_all_by_user(user_id)

    async def clear_unread_message_count(
        self, session_id: str, user_id: str, is_admin: bool = False
    ) -> None:
        """清空指定会话未读消息数"""
        logger.info(f"清除会话[{session_id}]未读消息数")
        async with self._uow:
            await self._get_accessible_session(session_id, user_id, is_admin=is_admin)
            await self._uow.session.update_unread_message_count(session_id, 0)

    async def delete_session(
        self, session_id: str, user_id: str, is_admin: bool = False
    ) -> None:
        """根据传递的会话id删除任务会话

        Eng review #10: destroy first (quiesce), then hard delete row.
        """
        # 1.检查会话是否存在
        logger.info(f"正在删除会话, 会话id: {session_id}")
        async with self._uow:
            session = await self._get_accessible_session(
                session_id, user_id, is_admin=is_admin
            )

        # 2.清理会话关联的运行态资源（任务/容器）
        await self._cleanup_task(session.task_id)

        # 3. Lifecycle-managed sandbox destroy (I6: quiesce barrier)
        if self._lifecycle:
            try:
                await self._lifecycle.destroy(session_id, DestroyReason.SESSION_DELETE)
            except Exception:
                logger.warning(
                    "Sandbox lifecycle destroy failed for session %s, proceeding with delete",
                    session_id,
                    exc_info=True,
                )

        # 4.根据传递的会话id删除会话
        async with self._uow:
            await self._uow.session.delete_by_id(session_id)
        logger.info(f"删除会话[{session_id}]成功")

    async def _cleanup_task(self, task_id: Optional[str]) -> None:
        """清理会话关联任务，避免删除会话后后台任务继续运行。"""
        if not task_id or self._task_cls is None:
            return

        try:
            task = self._task_cls.get(task_id)
            if not task:
                logger.info(f"会话任务[{task_id}]不存在或已结束，无需清理")
                return
            task.cancel(reason="session_delete")
            logger.info(f"会话任务[{task_id}]已取消")
        except Exception as e:
            logger.warning(f"清理会话任务[{task_id}]失败: {e}")

    async def _acquire_sandbox(self, session_id: str) -> SandboxHandle:
        """Acquire a sandbox handle for a session via lifecycle service.

        Raises NotFoundError/ServerRequestsError with user-friendly message
        for all lifecycle error states.
        """
        if not self._lifecycle:
            raise ServerRequestsError("Sandbox lifecycle service not available")
        try:
            return await self._lifecycle.acquire(session_id)
        except SessionUnboundError:
            raise NotFoundError("当前会话无沙箱环境")
        except (SessionFinalizedError, SessionSuspendedError):
            raise NotFoundError("当前会话沙箱不存在或已销毁")
        except SandboxLifecycleError:
            # Catch-all for SessionCreatingError, SessionDestroyingError, etc.
            raise ServerRequestsError("沙箱正在初始化或销毁中，请稍后重试")

    async def download_file(
        self,
        session_id: str,
        filepath: str,
        user_id: str,
        is_admin: bool = False,
    ) -> "BinaryIO":
        """通过 session + filepath 直接从沙箱下载文件"""
        from typing import BinaryIO

        async with self._uow:
            await self._get_accessible_session(session_id, user_id, is_admin)

        handle = await self._acquire_sandbox(session_id)
        return await handle.download_file(filepath)

    async def get_session(
        self, session_id: str, user_id: str, is_admin: bool = False
    ) -> Session:
        """获取指定会话详情信息"""
        async with self._uow:
            return await self._get_accessible_session(session_id, user_id, is_admin)

    async def get_session_files(
        self, session_id: str, user_id: str, is_admin: bool = False
    ) -> List[File]:
        """根据传递的会话id获取指定会话的文件列表信息"""
        logger.info(f"获取指定会话[{session_id}]下的文件列表信息")
        async with self._uow:
            session = await self._get_accessible_session(session_id, user_id, is_admin)
        return session.files

    async def _get_accessible_session(
        self, session_id: str, user_id: str, is_admin: bool = False
    ) -> Session:
        """获取指定用户可访问的会话（需在 uow 上下文中调用）"""
        session = await self._uow.session.get_by_id(session_id)
        if not session:
            logger.error(f"会话[{session_id}]不存在")
            raise NotFoundError("会话不存在")
        if not is_admin and session.user_id != user_id:
            logger.error(f"会话[{session_id}]无权访问")
            raise ForbiddenError("无权访问此会话")
        return session

    async def read_file(
        self, session_id: str, filepath: str, user_id: str, is_admin: bool = False
    ) -> FileReadResponse:
        """根据传递的信息查看会话中指定文件的内容"""
        # 1.检查会话是否存在
        logger.info(f"获取会话[{session_id}]中的文件内容, 文件路径: {filepath}")
        async with self._uow:
            await self._get_accessible_session(session_id, user_id, is_admin)

        # 2.通过 lifecycle service 获取沙箱 handle
        handle = await self._acquire_sandbox(session_id)

        # 3.调用沙箱读取文件内容
        result = await handle.read_file(filepath)
        if result.success:
            return FileReadResponse(**result.data)

        raise ServerRequestsError(result.message)

    async def read_shell_output(
        self,
        session_id: str,
        shell_session_id: str,
        user_id: str,
        is_admin: bool = False,
    ) -> ShellReadResponse:
        """根据传递的任务会话id+Shell会话id获取Shell执行结果"""
        # 1.检查会话是否存在
        logger.info(
            f"获取会话[{session_id}]中的Shell内容输出, Shell标识符: {shell_session_id}"
        )
        async with self._uow:
            await self._get_accessible_session(session_id, user_id, is_admin)

        # 2.通过 lifecycle service 获取沙箱 handle
        handle = await self._acquire_sandbox(session_id)

        # 3.调用沙箱查看shell内容
        result = await handle.read_shell_output(
            session_id=shell_session_id, console=True
        )
        if result.success:
            return ShellReadResponse(**result.data)

        raise ServerRequestsError(result.message)

    async def get_vnc_url(
        self, session_id: str, user_id: str, is_admin: bool = False
    ) -> str:
        """获取指定会话的vnc链接"""
        # 1.检查会话是否存在
        logger.info(f"获取会话[{session_id}]的VNC链接")
        async with self._uow:
            session = await self._get_accessible_session(session_id, user_id, is_admin)

        if not self._lifecycle:
            raise ServerRequestsError("Sandbox lifecycle service not available")

        # 2. 获取或创建沙箱（I5: UNBOUND → bind_new, ACTIVE → acquire, SUSPENDED → resume）
        try:
            handle = await self._lifecycle.acquire(session_id)
        except SessionUnboundError:
            handle = await self._lifecycle.bind_new(session_id)
        except SessionSuspendedError:
            handle = await self._lifecycle.resume(session_id)

        return handle.vnc_url

    async def ensure_takeover_shell_session(
        self,
        session_id: str,
        takeover_id: str,
        user_id: str,
        is_admin: bool = False,
    ) -> tuple[SandboxHandle, str]:
        """确保接管终端会话存在并返回沙箱 handle 与 shell 会话ID。"""
        logger.info("确保会话[%s]接管终端可用，takeover_id=%s", session_id, takeover_id)
        async with self._uow:
            await self._get_accessible_session(session_id, user_id, is_admin)

        if not self._lifecycle:
            raise ServerRequestsError("Sandbox lifecycle service not available")

        # 获取或创建沙箱
        try:
            handle = await self._lifecycle.acquire(session_id)
        except SessionUnboundError:
            handle = await self._lifecycle.bind_new(session_id)
        except SessionSuspendedError:
            handle = await self._lifecycle.resume(session_id)

        shell_session_id = f"takeover_{session_id}_{takeover_id}"
        probe_result = await handle.read_shell_output(
            session_id=shell_session_id,
            console=False,
        )
        if not probe_result.success:
            sandbox_home = get_settings().sandbox_default_cwd or "/root"
            start_result = await handle.exec_command(
                session_id=shell_session_id,
                exec_dir=sandbox_home,
                command="bash -i",
            )
            if not start_result.success:
                raise ServerRequestsError(start_result.message)

        return handle, shell_session_id
