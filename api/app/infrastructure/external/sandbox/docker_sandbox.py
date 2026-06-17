import asyncio
import io
import logging
import re
import socket
import time
import uuid
from pathlib import Path
from typing import BinaryIO, Optional, Self

# UUID v4 / 结构化 id 白名单。user_id 会被直接拼进 bind mount 路径，
# 必须限定为安全字符以防止 ``../`` / 绝对路径注入。
_SAFE_USER_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


def _is_within(path: Path, root: Path) -> bool:
    """True iff ``path`` 规范化后严格位于 ``root`` 之内（含 root 本身）。"""
    try:
        resolved = path.resolve(strict=False)
        resolved_root = root.resolve(strict=False)
    except OSError:
        return False
    try:
        resolved.relative_to(resolved_root)
    except ValueError:
        return False
    return True

import docker
import httpx
from app.domain.external.browser import Browser
from app.domain.external.sandbox import Sandbox
from app.domain.models.tool_result import ToolResult
from app.infrastructure.external.browser.playwright_browser import PlaywrightBrowser
from async_lru import alru_cache
from core.config import get_settings
from docker.errors import APIError, NotFound
from docker.models.resource import Model
from docker.types import Mount

logger = logging.getLogger(__name__)


class DockerSandbox(Sandbox):
    """基于Docker的沙箱服务"""

    def __init__(
        self, ip: Optional[str] = None, container_name: Optional[str] = None
    ) -> None:
        """构造函数，完成Docker沙箱扩展创建"""
        self.client = httpx.AsyncClient(timeout=600)
        self._ip = ip
        self._container_name = container_name
        self._base_url = f"http://{ip}:8080"
        self._shell_ws_url = f"ws://{ip}:8080/api/shell/ws"
        self._vnc_url = f"ws://{ip}:5901"
        self._cdp_url = f"http://{ip}:9222"

    @property
    def id(self) -> str:
        """获取沙箱的唯一id，使用容器名字作为唯一id"""
        if not self._container_name:
            return "mooc-manus-sandbox"
        return self._container_name

    @property
    def vnc_url(self) -> str:
        return self._vnc_url

    @property
    def cdp_url(self) -> str:
        return self._cdp_url

    @property
    def shell_ws_url(self) -> str:
        return self._shell_ws_url

    @classmethod
    @alru_cache(maxsize=128, typed=True)
    async def _resolve_hostname_to_ip(cls, hostname: str) -> Optional[str]:
        """将docker容器主机/地址转换成ipv4格式数据

        Note: @alru_cache is intentional and permanent for process lifetime.
        DNS resolution for sandbox_address rarely changes. If the sandbox
        address changes (e.g., container restart with new IP), a process
        restart is required to clear the cache. async_lru does not support
        TTL; consider switching to a TTL-capable cache if DNS volatility
        becomes an issue.
        """
        try:
            # 1.首先解析传递的hostname是不是ip
            try:
                socket.inet_pton(socket.AF_INET, hostname)
                return hostname
            except OSError:
                pass

            # 2.使用socket获取地址信息
            addr_info = socket.getaddrinfo(hostname, None, family=socket.AF_INET)

            # 3.判断地址信息是否存在，如果存在则返回第一个ipv4地址
            if addr_info and len(addr_info) > 0:
                return addr_info[0][4][0]

            return None
        except Exception as e:
            logger.error(f"解析Docker容器主机地址{hostname}失败: {str(e)}")
            return None

    @classmethod
    def _get_container_ip(cls, container: Model) -> Optional[str]:
        """根据传递的容器获取ip信息"""
        # 1.获取inspect网络设置（不同 docker 版本字段可能不完整，统一走安全读取）
        network_settings = container.attrs.get("NetworkSettings", {}) or {}
        ip_address = network_settings.get("IPAddress")
        if ip_address:
            return ip_address

        # 2.从Networks中读取每个网络下的IP
        networks = network_settings.get("Networks", {}) or {}
        for network_config in networks.values():
            network_ip = network_config.get("IPAddress")
            if network_ip:
                return network_ip

        return None

    @classmethod
    def _wait_for_container_ip(
        cls,
        container: Model,
        retries: int = 20,
        interval_seconds: float = 0.5,
    ) -> Optional[str]:
        """等待容器网络就绪并获取IP。"""
        for _ in range(retries):
            container.reload()
            ip = cls._get_container_ip(container)
            if ip:
                return ip
            time.sleep(interval_seconds)
        return None

    @classmethod
    def _create_docker_client(cls) -> docker.DockerClient:
        """创建 Docker 客户端，兼容 Docker Desktop 的用户目录 socket。"""
        try:
            client = docker.from_env()
            client.ping()
            return client
        except Exception as first_error:
            desktop_socket = Path.home() / ".docker" / "run" / "docker.sock"
            if desktop_socket.exists():
                try:
                    client = docker.DockerClient(base_url=f"unix://{desktop_socket}")
                    client.ping()
                    logger.warning(
                        "docker.from_env() 连接失败，回退使用 Docker Desktop socket: %s",
                        desktop_socket,
                    )
                    return client
                except Exception:
                    pass
            raise first_error

    @classmethod
    def _create_task(cls, user_id: Optional[str] = None) -> Self:
        """创建沙箱容器的异步任务。

        ``user_id`` 为 M1 引入：传入时为 sandbox 注入 read-only bind mount，
        ``${memory_root_host}/{user_id}`` → ``${memory_root_container}/{user_id}``。
        见 docs/superpowers/specs/2026-04-17-m0-sandbox-memory-mount-spike.md。
        """
        # 1.获取系统配置信息
        settings = get_settings()

        # 2.构建容器的名字
        image = settings.sandbox_image
        name_prefix = settings.sandbox_name_prefix
        container_name = f"{name_prefix}-{str(uuid.uuid4())[:8]}"

        try:
            # 3.创建一个docker客户端
            docker_client = cls._create_docker_client()

            # 4.预配置容器信息
            container_config = {
                "image": image,
                "name": container_name,
                "detach": True,
                "remove": True,
                "environment": {
                    "SERVICE_TIMEOUT_MINUTES": settings.sandbox_ttl_minutes,
                    "CHROME_ARGS": settings.sandbox_chrome_args,
                    "HTTPS_PROXY": settings.sandbox_https_proxy,
                    "HTTP_PROXY": settings.sandbox_http_proxy,
                    "NO_PROXY": settings.sandbox_no_proxy,
                    "TZ": settings.container_timezone,
                },
                # 容器级资源上限，防止 Chromium 失控导致宿主机 OOM
                "mem_limit": settings.sandbox_mem_limit,
            }

            # 5.判断是否传递了网络
            if settings.sandbox_network:
                container_config["network"] = settings.sandbox_network

            # 5b.M1 memory bind mount：user_id 传入时把用户私有 memory 目录
            # 以 read-only 形式挂进 sandbox。api 容器通过 FsMemoryWriter 负责
            # 写入（PR-5A），sandbox 只消费最新快照。
            memory_mount = cls._build_memory_mount(settings, user_id)
            if memory_mount is not None:
                container_config["mounts"] = [memory_mount]

            # 6.调用docker客户端容器运行参数创建沙箱
            container = docker_client.containers.run(**container_config)

            # 7.等待容器网络初始化完成后再获取IP
            ip = cls._wait_for_container_ip(container)
            if not ip:
                networks = (
                    (container.attrs.get("NetworkSettings", {}) or {}).get("Networks", {})
                    or {}
                )
                raise Exception(
                    f"容器已创建但未获取到IP地址，容器网络: {list(networks.keys())}"
                )

            return DockerSandbox(ip=ip, container_name=container_name)
        except Exception as e:
            logger.error(f"创建Docker沙箱容器失败: {str(e)}")
            raise Exception(f"创建Docker沙箱容器失败: {str(e)}")
        finally:
            if "docker_client" in locals():
                docker_client.close()

    @classmethod
    async def create(cls, user_id: Optional[str] = None) -> Self:
        """类方法，创建沙箱容器。

        ``user_id`` 为 M1 memory 系统引入。不传时容器按旧行为启动，
        传入时通过 bind mount 挂载该用户的 memory 目录（只读）。
        """
        # 1.获取系统配置信息
        settings = get_settings()

        # 2.判断是否使用现成的沙箱
        if settings.sandbox_address:
            # 3.将沙箱主机/地址解析成ip
            ip = await cls._resolve_hostname_to_ip(settings.sandbox_address)
            return DockerSandbox(ip=ip)

        # 4.使用子线程创建一个容器后返回
        return await asyncio.to_thread(cls._create_task, user_id)

    @staticmethod
    def _build_memory_mount(settings, user_id: Optional[str]) -> Optional[Mount]:
        """构造 user memory 目录的 read-only bind mount。

        - ``user_id`` 为空 / 非白名单字符 → 返回 None
        - ``sandbox_memory_mount_enabled=False`` → 返回 None，适用于 host 侧
          MEMORY_ROOT_HOST 路径未就绪的情形（临时降级，agent 退化为只走
          memory_search；M1 PR-6A 起默认 True）
        - 挂载采用 M0 spike 约定：
            source = 宿主机侧 ``${memory_root_host}/{user_id}``
            target = sandbox 固定路径 ``sandbox_memory_mount_target``
                    （默认 ``/workspace/.memory``，agent 工具按此路径读取）
        - 启用后会先在 api 容器视图下 ``${memory_root_container}/{user_id}``
          mkdir 一次；需要 api 容器本身已经把该路径 bind 到 ``memory_root_host``
          （PR-6A docker-compose.yml api service volume），host 侧同时必须
          真实存在才会成功启动 sandbox
        """
        if not user_id:
            return None

        if not _SAFE_USER_ID_RE.fullmatch(user_id):
            logger.warning(
                "拒绝构造 memory bind mount：user_id 不符合安全白名单 user_id=%r",
                user_id,
            )
            return None

        # Feature gate：PR-0 引入默认 False，M1 PR-6A 起默认 True。``False``
        # 时返回 None（DockerSandbox 按旧行为启动），部署者可在 host bind 尚未
        # 就绪时临时降级到不挂载。getattr 仍用默认 False 兜底，防止 legacy
        # settings dataclass / 单测 namespace 漏字段时崩溃。
        if not getattr(settings, "sandbox_memory_mount_enabled", False):
            return None

        # 两条 root 都要求绝对路径；这里不做 expanduser，避免把 ``~`` 在 api
        # 容器里误展开成 ``/root/...`` 后再传给宿主机 docker daemon。
        container_root = Path(settings.memory_root_container)
        host_root = Path(settings.memory_root_host)
        if not container_root.is_absolute() or not host_root.is_absolute():
            logger.warning(
                "拒绝构造 memory bind mount：root 不是绝对路径 "
                "host_root=%r container_root=%r",
                str(host_root),
                str(container_root),
            )
            return None

        container_mem_dir = container_root / user_id
        host_mem_dir = host_root / user_id
        # 最后兜底：resolve 后必须仍在各自 root 内（防御 symlink + 不规范 user_id）
        if not _is_within(container_mem_dir, container_root) or not _is_within(
            host_mem_dir, host_root
        ):
            logger.warning(
                "拒绝构造 memory bind mount：路径越界 user_id=%r",
                user_id,
            )
            return None

        try:
            container_mem_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            # 开启 mount 但 mkdir 失败 → bind source 大概率也不在，直接
            # 拒绝挂载而不是让 Docker daemon 抛一个更难诊断的启动错误。
            logger.warning(
                "memory 目录 mkdir 失败，跳过 memory bind mount: user_id=%s err=%s",
                user_id,
                exc,
            )
            return None

        return Mount(
            target=str(settings.sandbox_memory_mount_target),
            source=str(host_mem_dir),
            type="bind",
            read_only=True,
        )

    async def destroy(self) -> bool:
        """销毁当前的DockerSandbox实例.

        Returns ``True`` on success **including the case where the container
        is already gone** (``docker.errors.NotFound``) — gone-is-gone, this
        is the idempotent terminal-success contract C3 PR-1 requires.
        Returns ``False`` only on genuine Docker remove/close failures the
        caller should retry.

        C3 PR-1 (codex round 10 P2): NotFound is terminal success, not
        failure. Without this distinction, an externally removed container
        (e.g. ``docker rm`` from ops, or a parallel cleanup) would be
        classified as a retryable failure by the registry, leaving the
        binding stuck in DESTROYING and causing an infinite reconcile loop
        for a container that no longer exists.
        """
        try:
            # 1.关闭httpx客户端
            if self.client:
                await self.client.aclose()

            # 2.关闭并移除容器
            if self._container_name:
                docker_client = self._create_docker_client()
                try:
                    try:
                        container = docker_client.containers.get(
                            self._container_name
                        )
                    except NotFound:
                        # 容器已被外部删除 → 终态成功，符合 C3 PR-1
                        # destroy() 的 idempotent 语义。
                        logger.info(
                            "销毁时容器 %s 已不存在，按终态成功处理",
                            self._container_name,
                        )
                        return True
                    try:
                        container.remove(force=True)
                    except NotFound:
                        # 与上面的 get() NotFound 同义——race condition 下
                        # 容器在 get 之后、remove 之前被移除也算成功。
                        logger.info(
                            "移除容器 %s 时已不存在，按终态成功处理",
                            self._container_name,
                        )
                        return True
                finally:
                    docker_client.close()
            return True
        except Exception as e:
            logger.error(f"销毁当前Docker沙箱[{self._container_name}]失败: {str(e)}")
            return False

    @classmethod
    async def get(cls, id: str) -> Optional[Self]:
        """根据传递的id获取沙箱实例

        Note: @alru_cache removed per I4 — caching correctness hazard.
        Replaced by SandboxRegistry state-indexed lookup.
        """
        # 1.先获取系统配置并判断是否直连沙箱
        settings = get_settings()
        if settings.sandbox_address:
            try:
                ip = await cls._resolve_hostname_to_ip(settings.sandbox_address)
                return DockerSandbox(ip=ip, container_name=id)
            except Exception as e:
                logger.error(f"解析沙箱地址失败: {str(e)}")
                return None

        try:
            # 2.创建docker客户端并根据容器名字获取容器
            docker_client = cls._create_docker_client()

            try:
                # 3.根据id获取容器
                container = docker_client.containers.get(id)
                container.reload()

                # 4.检查容器是否正常运行
                if container.status != "running":
                    logger.warning(f"容器存在但未运行, 容器名字: {id}")
                    return None

                # 4.获取容器的ip地址
                ip = cls._get_container_ip(container)
                if not ip:
                    return None

                return DockerSandbox(ip=ip, container_name=id)
            except NotFound:
                # 5.找不到容器(容器被销毁)
                logger.warning(f"该容器找不到可能被销毁: {str(id)}")
                return None
            except APIError as e:
                # 6.Docker容器守护进程出错
                logger.error(f"Docker API出错: {str(e)}")
                return None
            finally:
                # 7.显示关闭docker client
                docker_client.close()
        except Exception as e:
            # 8.其他错误统一捕获
            logger.error(f"获取沙箱发生未知错误: {str(e)}")
            return None

    async def get_browser(self) -> Browser:
        """获取沙箱中的浏览器实例"""
        return PlaywrightBrowser(self.cdp_url)

    async def ensure_sandbox(self) -> None:
        """确保沙箱一定存在/服务全部都开启了才执行后续步骤"""
        # 1.定义最大重试次数+重试间隔
        max_retries = 30
        retry_interval = 2

        # 2.循环请求获取supervisor状态并判断服务是否正常
        for attempt in range(max_retries):
            try:
                # 3.调用client客户端向沙箱发起api请求获取状态
                response = await self.client.get(
                    f"{self._base_url}/api/supervisor/status"
                )
                response.raise_for_status()

                # 4.将响应结果转换为ToolResult
                tool_result = ToolResult.from_sandbox(**response.json())

                # 5.判断是否执行成功
                if not tool_result.success:
                    logger.warning(f"Supervisor进程状态监测失败: {tool_result.message}")
                    await asyncio.sleep(retry_interval)
                    continue

                # 6.读取services数据并判断
                services = tool_result.data or []
                if not services:
                    logger.warning(f"Supervisor进程中未发现任何服务")
                    await asyncio.sleep(retry_interval)
                    continue

                # 7.循环遍历所有服务并判断是否全部正常运行
                all_running = True
                non_running_services = []
                for service in services:
                    service_name = service.get("name", "unknown")
                    state_name = service.get("statename", "")

                    # 8.判断state_name是不是RUNNING
                    if state_name != "RUNNING":
                        all_running = False
                        non_running_services.append(f"{service_name}({state_name})")

                # 9.判断是否所有服务都启动
                if all_running:
                    logger.info("Sandbox Supervisor所有进程服务运行正常")
                    return
                else:
                    logger.info(
                        f"正在等待Sandbox Supervisor进程服务运行, 还未运行的服务列表: {non_running_services}"
                    )
                    await asyncio.sleep(retry_interval)
            except Exception as e:
                logger.warning(f"无法确认Sandbox Supervisor进程状态: {str(e)}")
                await asyncio.sleep(retry_interval)

        # 经过max_retries次监测后还无法确认则抛出异常
        logger.error(f"在经过{max_retries}次尝试后仍无法确认Sandbox Supervisor状态信息")
        raise Exception(
            f"在经过{max_retries}次尝试后仍无法确认Sandbox Supervisor状态信息"
        )

    async def read_file(
        self,
        filepath: str,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
        sudo: bool = False,
        max_length: int = 10000,
    ) -> ToolResult:
        """读取沙箱中指定路径的文件内容"""
        response = await self.client.post(
            f"{self._base_url}/api/file/read-file",
            json={
                "filepath": filepath,
                "start_line": start_line,
                "end_line": end_line,
                "sudo": sudo,
                "max_length": max_length,
            },
        )
        return ToolResult.from_sandbox(**response.json())

    async def write_file(
        self,
        filepath: str,
        content: str,
        append: bool = False,
        leading_newline: bool = False,
        trailing_newline: bool = False,
        sudo: bool = False,
    ) -> ToolResult:
        """向沙箱中指定文件写入内容"""
        response = await self.client.post(
            f"{self._base_url}/api/file/write-file",
            json={
                "filepath": filepath,
                "content": content,
                "append": append,
                "leading_newline": leading_newline,
                "trailing_newline": trailing_newline,
                "sudo": sudo,
            },
        )
        return ToolResult.from_sandbox(**response.json())

    async def replace_in_file(
        self,
        filepath: str,
        old_str: str,
        new_str: str,
        sudo: bool = False,
    ) -> ToolResult:
        """替换沙箱中文件的旧内容为指定内容"""
        response = await self.client.post(
            f"{self._base_url}/api/file/replace-in-file",
            json={
                "filepath": filepath,
                "old_str": old_str,
                "new_str": new_str,
                "sudo": sudo,
            },
        )
        return ToolResult.from_sandbox(**response.json())

    async def search_in_file(
        self, filepath: str, regex: str, sudo: bool = False
    ) -> ToolResult:
        """搜索沙箱中指定文件的内容"""
        response = await self.client.post(
            f"{self._base_url}/api/file/search-in-file",
            json={
                "filepath": filepath,
                "regex": regex,
                "sudo": sudo,
            },
        )
        return ToolResult.from_sandbox(**response.json())

    async def find_files(self, dir_path: str, glob_pattern: str) -> ToolResult:
        """查找沙箱中指定目录的文件列表"""
        response = await self.client.post(
            f"{self._base_url}/api/file/find-files",
            json={
                "dir_path": dir_path,
                "glob_pattern": glob_pattern,
            },
        )
        return ToolResult.from_sandbox(**response.json())

    async def list_files(self, dir_path: str) -> ToolResult:
        """传递目录列出沙箱指定目录下的所有文件"""
        return await self.find_files(dir_path, "*")

    async def check_file_exists(self, filepath: str) -> ToolResult:
        """传递指定路径检查沙箱中指定文件是否存在"""
        response = await self.client.post(
            f"{self._base_url}/api/file/check-file-exists",
            json={
                "filepath": filepath,
            },
        )
        return ToolResult.from_sandbox(**response.json())

    async def delete_file(self, filepath: str) -> ToolResult:
        """传递路径删除指定的文件"""
        response = await self.client.post(
            f"{self._base_url}/api/file/delete-file",
            json={
                "filepath": filepath,
            },
        )
        return ToolResult.from_sandbox(**response.json())

    async def upload_file(
        self,
        file_data: BinaryIO,
        filepath: str,
        filename: str = None,
        *,
        refuse_special: bool = False,
    ) -> ToolResult:
        """将文件源上传至沙箱指定位置

        ``refuse_special=True``（仅 coordinator apply/seed 路径）随多部分表单
        透传给沙箱端, 令底层原子写在 special 文件目标处 raise 而非 D12 写穿透。
        非 coordinator 调用方保持默认 ``False``。
        """
        # 1.预配置上传数据
        files = {"file": (filename or "upload", file_data, "application/octet-stream")}
        data = {
            "filepath": filepath,
            "refuse_special": str(refuse_special).lower(),  # "true"/"false"
        }

        # 2.发起请求上传数据获取响应
        response = await self.client.post(
            f"{self._base_url}/api/file/upload-file",
            files=files,
            data=data,
        )
        return ToolResult.from_sandbox(**response.json())

    async def download_file(self, filepath: str) -> BinaryIO:
        """从沙箱中下载文件"""
        response = await self.client.get(
            f"{self._base_url}/api/file/download-file", params={"filepath": filepath}
        )
        response.raise_for_status()

        return io.BytesIO(response.content)

    async def exec_command(
        self,
        session_id: str,
        exec_dir: str,
        command: str,
        wait_seconds: Optional[int] = None,
    ) -> ToolResult:
        """在沙箱中执行命令

        ``wait_seconds`` 透传给沙箱端, 控制同步等待窗口。未指定时使用沙箱
        默认 (5 秒)。
        """
        payload: dict = {
            "session_id": session_id,
            "exec_dir": exec_dir,
            "command": command,
        }
        if wait_seconds is not None:
            payload["wait_seconds"] = wait_seconds
        response = await self.client.post(
            f"{self._base_url}/api/shell/exec-command",
            json=payload,
        )
        return ToolResult.from_sandbox(**response.json())

    async def read_shell_output(
        self, session_id: str, console: bool = False
    ) -> ToolResult:
        """读取沙箱中shell的输出"""
        response = await self.client.post(
            f"{self._base_url}/api/shell/read-shell-output",
            json={
                "session_id": session_id,
                "console": console,
            },
        )
        return ToolResult.from_sandbox(**response.json())

    async def write_shell_input(
        self,
        session_id: str,
        input_text: str,
        press_enter: bool = True,
    ) -> ToolResult:
        """向沙箱的Shell进程写入数据"""
        response = await self.client.post(
            f"{self._base_url}/api/shell/write-shell-input",
            json={
                "session_id": session_id,
                "input_text": input_text,
                "press_enter": press_enter,
            },
        )
        return ToolResult.from_sandbox(**response.json())

    async def resize_shell_session(
        self,
        session_id: str,
        cols: int,
        rows: int,
    ) -> ToolResult:
        """调整沙箱中PTY会话窗口大小"""
        response = await self.client.post(
            f"{self._base_url}/api/shell/resize-shell",
            json={
                "session_id": session_id,
                "cols": cols,
                "rows": rows,
            },
        )
        return ToolResult.from_sandbox(**response.json())

    async def wait_process(
        self, session_id: str, seconds: Optional[int] = None
    ) -> ToolResult:
        """等待沙箱中进程的执行"""
        response = await self.client.post(
            f"{self._base_url}/api/shell/wait-process",
            json={
                "session_id": session_id,
                "seconds": seconds,
            },
        )
        return ToolResult.from_sandbox(**response.json())

    async def kill_process(self, session_id: str) -> ToolResult:
        """杀死沙箱中指定进程"""
        response = await self.client.post(
            f"{self._base_url}/api/shell/kill-process",
            json={
                "session_id": session_id,
            },
        )
        return ToolResult.from_sandbox(**response.json())
