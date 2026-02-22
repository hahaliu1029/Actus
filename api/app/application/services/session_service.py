import logging
from typing import Callable, List, Optional, Type

from app.application.errors.exceptions import (
    ForbiddenError,
    NotFoundError,
    ServerRequestsError,
)
from app.domain.external.sandbox import Sandbox
from app.domain.external.task import Task
from app.domain.models.file import File
from app.domain.models.session import Session

# from app.domain.repositories.session_repository import SessionRepository
from app.domain.repositories.uow import IUnitOfWork
from app.interfaces.schemas.session import FileReadResponse, ShellReadResponse
from core.config import get_settings

logger = logging.getLogger(__name__)


class SessionService:
    """会话服务"""

    def __init__(
        self,
        uow_factory: Callable[[], IUnitOfWork],
        sandbox_cls: Type[Sandbox],
        task_cls: Optional[Type[Task]] = None,
    ) -> None:
        """构造函数，完成会话服务初始化"""
        self._uow_factory = uow_factory
        self._uow = uow_factory()
        self._sandbox_cls = sandbox_cls
        self._task_cls = task_cls

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
        """根据传递的会话id删除任务会话"""
        # 1.检查会话是否存在
        logger.info(f"正在删除会话, 会话id: {session_id}")
        async with self._uow:
            session = await self._get_accessible_session(
                session_id, user_id, is_admin=is_admin
            )

        # 2.清理会话关联的运行态资源（任务/容器）
        await self._cleanup_task(session.task_id)
        await self._cleanup_sandbox(session.sandbox_id)

        # 3.根据传递的会话id删除会话
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
            task.cancel()
            logger.info(f"会话任务[{task_id}]已取消")
        except Exception as e:
            logger.warning(f"清理会话任务[{task_id}]失败: {e}")

    async def _cleanup_sandbox(self, sandbox_id: Optional[str]) -> None:
        """清理会话关联沙箱容器。"""
        if not sandbox_id:
            return

        settings = get_settings()
        if settings.sandbox_address:
            logger.info(
                "当前启用共享沙箱地址(sandbox_address)，跳过按会话销毁沙箱[%s]",
                sandbox_id,
            )
            return

        try:
            sandbox = await self._sandbox_cls.get(sandbox_id)
            if not sandbox:
                logger.info(f"会话沙箱[{sandbox_id}]不存在或已销毁，无需清理")
                return

            destroyed = await sandbox.destroy()
            if destroyed:
                logger.info(f"会话沙箱[{sandbox_id}]已销毁")
            else:
                logger.warning(f"会话沙箱[{sandbox_id}]销毁失败")
        except Exception as e:
            logger.warning(f"清理会话沙箱[{sandbox_id}]失败: {e}")

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
            session = await self._get_accessible_session(session_id, user_id, is_admin)

        # 2.根据沙箱id获取沙箱并判断是否存在
        if not session.sandbox_id:
            raise NotFoundError("当前会话无沙箱环境")
        sandbox = await self._sandbox_cls.get(session.sandbox_id)
        if not sandbox:
            raise NotFoundError("当前会话沙箱不存在或已销毁")

        # 3.调用沙箱读取文件内容
        result = await sandbox.read_file(filepath)
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
            session = await self._get_accessible_session(session_id, user_id, is_admin)

        # 2.根据沙箱id获取沙箱并判断是否存在
        if not session.sandbox_id:
            raise NotFoundError("当前会话无沙箱环境")
        sandbox = await self._sandbox_cls.get(session.sandbox_id)
        if not sandbox:
            raise NotFoundError("当前会话沙箱不存在或已销毁")

        # 3.调用沙箱查看shell内容
        result = await sandbox.read_shell_output(
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

        # 2.懒创建沙箱，避免新会话访问VNC时直接失败
        sandbox = None
        if session.sandbox_id:
            sandbox = await self._sandbox_cls.get(session.sandbox_id)
        if not sandbox:
            sandbox = await self._sandbox_cls.create()
            session.sandbox_id = sandbox.id
            async with self._uow:
                await self._uow.session.save(session)

        # 3.确认沙箱服务就绪后再返回VNC链接
        await sandbox.ensure_sandbox()

        return sandbox.vnc_url
