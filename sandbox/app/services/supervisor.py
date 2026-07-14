import asyncio
import http.client
import logging
import math
import socket
import threading
import xmlrpc.client
from datetime import datetime, timedelta
from typing import Any, List, Optional

from app.core.config import get_settings
from app.interfaces.errors.exceptions import AppException, BadRequestException
from app.models.supervisor import ProcessInfo, SupervisorActionResult, SupervisorTimeout

"""
1.Supervisor启动后，通过一个Unix套接字文件来实现通信(rpc协议)
2.连接这个通信文件，/tmp/supervisor.sock (xml-rpc连接)
3.使用某种方式来完整转换，让xml-rpc实现连接supervisor.sock
4.连接之后我们就可以调用rpc对应的方法，getAllProcessInfo()
"""

logger = logging.getLogger(__name__)


class UnixStreamHTTPConnection(http.client.HTTPConnection):
    """基于Unix流的HTTP连接处理器"""

    def __init__(self, host: str, socket_path: str, timeout=None) -> None:
        """构造函数，完成连接处理器初始化"""
        http.client.HTTPConnection.__init__(self, host, timeout)
        self.socket_path = socket_path

    def connect(self) -> None:
        """重写连接方法，欺骗xml-rpc库让其觉得自己正在进行网络连接"""
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect(self.socket_path)


class UnixStreamTransport(xmlrpc.client.Transport):
    """基于Unix流传输层的适配器/转换器"""

    def __init__(self, socket_path: str) -> None:
        """构造函数，完成传输适配器的初始化"""
        xmlrpc.client.Transport.__init__(self)
        self.socket_path = socket_path

    def make_connection(self, host) -> http.client.HTTPConnection:
        return UnixStreamHTTPConnection(host, self.socket_path)


class SupervisorService:
    """Supervisor服务"""

    def __init__(self) -> None:
        """构造函数，完成supervisor服务链接"""
        # 1.连接supervisor配置
        self.rpc_url = "/tmp/supervisor.sock"
        self._connect_rpc()

        # 2.supervisor超时配置
        settings = get_settings()
        self.timeout_active = settings.server_timeout_minutes is not None
        self.shutdown_task = None
        self.shutdown_time = None
        self.shutdown_timer = None
        self._timeout_lock = threading.RLock()
        self._timeout_generation = 0
        self._timeout_fired_generation = None
        self._expand_enabled = True  # 是否自动保活(每调用一次接口就增加时间)

        # 3.检测是否配置了自动销毁
        if settings.server_timeout_minutes is not None:
            timeout_minutes = self._validate_timeout_minutes(
                settings.server_timeout_minutes
            )
            # 4.设置销毁时间+定时器
            generation = self._replace_timeout_deadline(
                datetime.now() + timedelta(minutes=timeout_minutes)
            )
            self._setup_timer(timeout_minutes, generation=generation)

    @property
    def expand_enabled(self) -> bool:
        """只读属性，返回是否自动保活"""
        return self._expand_enabled

    def enable_expand(self) -> None:
        """开启自动保活"""
        self._expand_enabled = True

    def disable_expand(self) -> None:
        """关闭自动保活"""
        self._expand_enabled = False

    @staticmethod
    def _validate_timeout_minutes(minutes: int | float) -> int | float:
        if (
            isinstance(minutes, bool)
            or not isinstance(minutes, (int, float))
            or not math.isfinite(minutes)
            or minutes <= 0
        ):
            raise BadRequestException("超时时间必须是大于0的分钟数")
        return minutes

    def _replace_timeout_deadline(self, deadline: datetime) -> int:
        """Install one authoritative deadline and return its generation."""
        with self._timeout_lock:
            self._timeout_generation += 1
            generation = self._timeout_generation
            self._timeout_fired_generation = None
            self.timeout_active = True
            self.shutdown_time = deadline
            return generation

    def _setup_timer(
        self,
        minutes: int | float,
        *,
        generation: Optional[int] = None,
        force_thread_timer: bool = False,
    ) -> None:
        """传递时间(分钟)并创建定时器，在时间结束之后关闭supervisord主进程"""
        generation = self._timeout_generation if generation is None else generation
        with self._timeout_lock:
            # A stale callback that woke early must never cancel the current
            # generation's timer while trying to reschedule itself.
            if generation != self._timeout_generation or not self.timeout_active:
                return
            previous_task = self.shutdown_task
            previous_timer = self.shutdown_timer
            delay_seconds = (
                max(0.0, (self.shutdown_time - datetime.now()).total_seconds())
                if self.shutdown_time is not None
                else minutes * 60
            )

        # 1.检测当前是否存在销毁任务，如果存在则先取消
        current_task = None
        try:
            current_task = asyncio.current_task()
        except RuntimeError:
            pass
        if previous_task and previous_task is not current_task:
            try:
                previous_task.cancel()
            except Exception as e:
                logger.warning(f"取消shutdown任务失败: {str(e)}")
        if previous_timer:
            previous_timer.cancel()

        # 2.创建一个异步定时器任务函数
        async def shutdown_after_timeout():
            try:
                await asyncio.sleep(delay_seconds)
                await self._handle_timeout_expiry(generation)
            except asyncio.CancelledError:
                return
            except Exception:
                logger.exception(
                    "sandbox timeout timer失败 generation=%s", generation
                )

        if not force_thread_timer:
            try:
                # 3.获取事件循环并添加任务
                loop = asyncio.get_running_loop()
                coroutine = shutdown_after_timeout()
                try:
                    task = loop.create_task(coroutine)
                except Exception:
                    coroutine.close()
                    raise
                with self._timeout_lock:
                    if (
                        generation == self._timeout_generation
                        and self.timeout_active
                    ):
                        self.shutdown_task = task
                    else:
                        task.cancel()
                return
            except Exception:
                pass

        # 4.没有持久running loop，或由thread timer提前唤醒：继续使用thread timer。
        # asyncio.run()内部的loop只活到callback返回，不能把重排任务挂到该loop。
        def run_thread_timeout() -> None:
            try:
                asyncio.run(
                    self._handle_timeout_expiry(
                        generation,
                        from_thread_timer=True,
                    )
                )
            except Exception:
                logger.exception(
                    "sandbox thread timeout timer失败 generation=%s",
                    generation,
                )

        timer = threading.Timer(
            delay_seconds,
            run_thread_timeout,
        )
        timer.daemon = True
        with self._timeout_lock:
            if (
                generation == self._timeout_generation
                and self.timeout_active
            ):
                self.shutdown_timer = timer
                timer.start()

    async def _handle_timeout_expiry(
        self,
        generation: int,
        *,
        from_thread_timer: bool = False,
    ) -> None:
        """Ignore stale timers, reschedule early wakes, and fire once."""
        remaining_seconds = 0.0
        with self._timeout_lock:
            if (
                generation != self._timeout_generation
                or not self.timeout_active
                or self.shutdown_time is None
                or self._timeout_fired_generation == generation
            ):
                return
            remaining_seconds = (self.shutdown_time - datetime.now()).total_seconds()
            if remaining_seconds <= 0:
                self._timeout_fired_generation = generation

        if remaining_seconds > 0:
            if from_thread_timer:
                self._setup_timer(
                    remaining_seconds / 60,
                    generation=generation,
                    force_thread_timer=True,
                )
            else:
                self._setup_timer(
                    remaining_seconds / 60,
                    generation=generation,
                )
            return

        await self.shutdown()

    def _connect_rpc(self) -> None:
        """使用python的xml-rpc客户端连接一个本地sock文件文件实现连接rpc服务"""
        try:
            self.server = xmlrpc.client.ServerProxy(
                "http://localhost",
                transport=UnixStreamTransport(self.rpc_url),
            )
        except Exception as e:
            logger.error(f"连接Supervisor服务失败: {str(e)}")
            raise BadRequestException(f"连接Supervisor服务失败: {str(e)}")

    @classmethod
    async def _call_rpc(cls, method, *args) -> Any:
        """根据传递的方法+参数调用rpc方法"""
        try:
            return await asyncio.to_thread(method, *args)
        except Exception as e:
            logger.error(f"RPC方法调用失败: {str(e)}")
            raise BadRequestException(f"RPC方法调用失败: {str(e)}")

    async def get_all_processes(self) -> List[ProcessInfo]:
        """获取当前supervisor管理的所有进程信息"""
        try:
            processes = await self._call_rpc(self.server.supervisor.getAllProcessInfo)
            return [ProcessInfo(**process) for process in processes]
        except Exception as e:
            logger.error(f"获取进程信息失败: {str(e)}")
            raise AppException(f"获取进程信息失败: {str(e)}")

    async def stop_all_processes(self) -> SupervisorActionResult:
        """停止supervisor管理的所有进程"""
        try:
            result = await self._call_rpc(self.server.supervisor.stopAllProcesses)
            return SupervisorActionResult(status="stopped", result=result)
        except Exception as e:
            logger.error(f"停止supervisor所有进程服务失败: {str(e)}")
            raise AppException(f"停止supervisor所有进程服务失败: {str(e)}")

    async def shutdown(self) -> SupervisorActionResult:
        """关闭supervisord服务"""
        try:
            shutdown_result = await self._call_rpc(self.server.supervisor.shutdown)
            return SupervisorActionResult(
                status="shutdown", shutdown_result=shutdown_result
            )
        except Exception as e:
            logger.error(f"关闭supervisord服务失败: {str(e)}")
            raise AppException(f"关闭supervisord服务失败: {str(e)}")

    async def restart(self) -> SupervisorActionResult:
        """重启Supervisor管理的进程"""
        try:
            stop_result = await self._call_rpc(self.server.supervisor.stopAllProcesses)
            start_result = await self._call_rpc(
                self.server.supervisor.startAllProcesses
            )
            return SupervisorActionResult(
                status="restarted",
                stop_result=stop_result,
                start_result=start_result,
            )
        except Exception as _:
            logger.error(f"重启Supervisor进程服务失败")
            raise AppException(f"重启Supervisor进程服务失败")

    async def activate_timeout(
        self, minutes: Optional[int] = None
    ) -> SupervisorTimeout:
        """传递指定分钟，并激活定时销毁任务同时关闭自动保活"""
        # 1.获取超时分钟数
        setting = get_settings()
        timeout_minutes = (
            setting.server_timeout_minutes if minutes is None else minutes
        )
        if timeout_minutes is None:
            raise BadRequestException("超时时间未配置, 并且未读取到系统默认超时时间")
        timeout_minutes = self._validate_timeout_minutes(timeout_minutes)

        # 2.更新超时配置
        deadline = datetime.now() + timedelta(minutes=timeout_minutes)
        generation = self._replace_timeout_deadline(deadline)

        # 3.创建一个新的定时器
        self._setup_timer(timeout_minutes, generation=generation)

        return SupervisorTimeout(
            status="timeout_activated",
            active=True,
            shutdown_time=self.shutdown_time.isoformat(),
            timeout_minutes=timeout_minutes,
            remaining_seconds=(self.shutdown_time - datetime.now()).total_seconds(),
        )

    async def reset_timeout(
        self, minutes: Optional[int] = None
    ) -> SupervisorTimeout:
        """把cleanup lease精确重置为 ``now + window``。"""
        settings = get_settings()
        timeout_minutes = (
            settings.server_timeout_minutes if minutes is None else minutes
        )
        if timeout_minutes is None:
            raise BadRequestException("超时时间未配置, 并且未读取到系统默认超时时间")
        timeout_minutes = self._validate_timeout_minutes(timeout_minutes)
        deadline = datetime.now() + timedelta(minutes=timeout_minutes)
        generation = self._replace_timeout_deadline(deadline)
        self._setup_timer(timeout_minutes, generation=generation)
        return SupervisorTimeout(
            status="timeout_reset",
            active=True,
            shutdown_time=deadline.isoformat(),
            timeout_minutes=timeout_minutes,
            remaining_seconds=(deadline - datetime.now()).total_seconds(),
        )

    async def extend_timeout(self, minutes: Optional[int] = 3) -> SupervisorTimeout:
        """传递指定的时长，延长超时销毁的时间，单默认延长3分钟"""
        # 1.获取超时分钟数
        if minutes is None:
            raise BadRequestException("超时时间未配置, 请核实后重试")
        extend_minutes = self._validate_timeout_minutes(minutes)
        now = datetime.now()
        with self._timeout_lock:
            if not self.timeout_active or self.shutdown_time is None:
                raise BadRequestException("超时销毁未激活, 请先激活后重试")
            deadline = max(self.shutdown_time, now) + timedelta(
                minutes=extend_minutes
            )
            self._timeout_generation += 1
            generation = self._timeout_generation
            self._timeout_fired_generation = None
            self.shutdown_time = deadline
            timeout_minutes = (deadline - now).total_seconds() / 60

        # 2.更新超时配置
        self.timeout_active = True

        # 3.创建一个新的定时器
        self._setup_timer(timeout_minutes, generation=generation)

        return SupervisorTimeout(
            status="timeout_extended",
            active=True,
            shutdown_time=self.shutdown_time.isoformat(),
            timeout_minutes=timeout_minutes,
            remaining_seconds=(self.shutdown_time - datetime.now()).total_seconds(),
        )

    async def cancel_timeout(self) -> SupervisorTimeout:
        """取消超时销毁设置"""
        # 1.判断是否设置了超时销毁
        if not self.timeout_active:
            return SupervisorTimeout(status="no_timeout_active", activate=False)

        # 2.先换代并清状态；已经醒来的旧timer随后只能no-op。
        with self._timeout_lock:
            self._timeout_generation += 1
            self._timeout_fired_generation = None
            shutdown_task = self.shutdown_task
            shutdown_timer = self.shutdown_timer
            self.shutdown_task = None
            self.shutdown_timer = None
            self.timeout_active = False
            self.shutdown_time = None
            self._expand_enabled = True

        if shutdown_task:
            try:
                shutdown_task.cancel()
            except Exception as e:
                logger.warning(f"取消shutdown任务失败: {str(e)}")

        # 3.同步检查是否存在定时器
        if shutdown_timer:
            shutdown_timer.cancel()

        return SupervisorTimeout(status="timeout_cancelled", active=False)

    async def get_timeout_status(self) -> SupervisorTimeout:
        """获取当前supervisor的超时状态"""
        # 1.判断是否开启超时销毁功能
        if not self.timeout_active:
            return SupervisorTimeout(active=False)

        # 2.统计剩余秒数
        remaining_seconds = 0
        if self.shutdown_time:
            remaining = self.shutdown_time - datetime.now()
            remaining_seconds = max(0, remaining.total_seconds())

        return SupervisorTimeout(
            active=self.timeout_active,
            shutdown_time=(
                self.shutdown_time.isoformat() if self.shutdown_time else None
            ),
            remaining_seconds=remaining_seconds,
        )
