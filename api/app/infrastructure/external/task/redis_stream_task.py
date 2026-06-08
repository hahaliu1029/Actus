import asyncio
import logging
import uuid
from typing import Any, Dict, Optional

from app.domain.external.message_queue import MessageQueue
from app.domain.external.task import Task, TaskRunner
from app.infrastructure.external.message_queue.redis_stream_message_queue import (
    RedisStreamMessageQueue,
)

logger = logging.getLogger(__name__)


class RedisStreamTask(Task):
    """基于Redis流的任务类"""

    # 定义一个全局变量用于存储所有已注册的任务
    _task_registry: Dict[str, "RedisStreamTask"] = {}

    def __init__(self, task_runner: TaskRunner) -> None:
        """构造函数，传递任务运行器完成Task初始化"""
        self._task_runner = task_runner
        self._id = str(uuid.uuid4())
        self._execution_task: Optional[asyncio.Task] = None  # 定义在后台执行的任务
        self._cancel_reason: str = "stop"

        # [C2b §4.4] Typed child-scope violation stash. Set by
        # AgentTaskRunner.invoke's `except ChildScopeViolation` BEFORE re-raising,
        # so it survives this task's _execute_task `except Exception` swallow and
        # the coordinator invoke-adapter can re-raise it (→ NEEDS_AUTHORIZATION).
        self._child_scope_violation: Optional[Any] = None

        input_stream_name = f"task:input:{self._id}"
        output_stream_name = f"task:output:{self._id}"

        self._input_stream = RedisStreamMessageQueue(input_stream_name)
        self._output_stream = RedisStreamMessageQueue(output_stream_name)

        # 将当前类实例注册到全局变量中
        RedisStreamTask._task_registry[self._id] = self

    def _cleanup_registry(self) -> None:
        """清除类全局变量中当前注册的任务"""
        if self._id in RedisStreamTask._task_registry:
            del RedisStreamTask._task_registry[self._id]
            logger.info(f"任务[{self._id}]从注册中心移除")

    def _on_task_done(self) -> None:
        """任务结束时的回调函数"""
        # 1.检测task_runner是否存在，如果存在则调用task_runner的回调函数
        if self._task_runner:
            asyncio.create_task(self._task_runner.on_done(self))

        # 2.清除当前任务对应的资源
        self._cleanup_registry()

    async def _execute_task(self) -> None:
        """使用TaskRunner执行任务"""
        try:
            await self._task_runner.invoke(self)
        except asyncio.CancelledError:
            logger.info(f"任务[{self._id}]执行被取消")
            raise
        except Exception as e:
            logger.error(f"任务[{self._id}]执行出现异常: {str(e)}")
        finally:
            self._on_task_done()

    async def invoke(self) -> None:
        """使用提供的task_runner来运行任务"""
        if self.done:
            self._execution_task = asyncio.create_task(self._execute_task())
            logger.info(f"任务[{self._id}]开始执行")

    async def _execute_resume(self, command: Any) -> None:
        """Execute resume in background task (mirrors _execute_task for invoke)."""
        try:
            await self._task_runner.resume(self, command)
        except asyncio.CancelledError:
            logger.info(f"任务[{self._id}] resume 被取消")
            raise
        except Exception as e:
            logger.error(f"任务[{self._id}] resume 出现异常: {str(e)}")
        finally:
            self._on_task_done()

    async def resume(self, command: Any) -> None:
        """Resume task in a background asyncio.Task (mirrors invoke pattern)."""
        self._execution_task = asyncio.create_task(self._execute_resume(command))
        logger.info(f"任务[{self._id}] resume 开始执行")

    def cancel(self, reason: str = "stop") -> bool:
        """取消当前执行的任务"""
        self._cancel_reason = reason or "stop"
        if not self.done:
            self._execution_task.cancel()
            logger.info(
                "任务[%s]已取消，reason=%s",
                self._id,
                self._cancel_reason,
            )
            return True

        # 任务已结束，无需重复取消
        return True

    @property
    def cancel_reason(self) -> str:
        return self._cancel_reason

    @property
    def child_scope_violation(self) -> Optional[Any]:
        return self._child_scope_violation

    def set_child_scope_violation(self, exc: Any) -> None:
        self._child_scope_violation = exc

    @property
    def input_stream(self) -> MessageQueue:
        return self._input_stream

    @property
    def output_stream(self) -> MessageQueue:
        return self._output_stream

    @property
    def id(self) -> str:
        return self._id

    @property
    def done(self) -> bool:
        if self._execution_task is None:
            return True
        return self._execution_task.done()

    @classmethod
    def get(cls, task_id: str) -> Optional["Task"]:
        return RedisStreamTask._task_registry.get(task_id)

    @classmethod
    def create(cls, task_runner: TaskRunner) -> "Task":
        return cls(task_runner)

    @classmethod
    async def destroy(cls) -> None:
        for task in list(RedisStreamTask._task_registry.values()):
            task.cancel()

            if task._task_runner:
                await task._task_runner.destroy()

        cls._task_registry.clear()
