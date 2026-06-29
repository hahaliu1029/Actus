from typing import TYPE_CHECKING, BinaryIO, Optional, Protocol, Self

from app.domain.external.browser import Browser
from app.domain.models.tool_result import ToolResult

if TYPE_CHECKING:
    from app.domain.models.sandbox_policy import ContainerRuntimePolicy


class Sandbox(Protocol):
    """沙箱底层服务协议（Docker/本地/云沙箱适配面）。

    外部代码不应直接持有此类型——应通过 SandboxHandle 访问。
    只有 SandboxLifecycleService / SandboxRegistry / DockerSandbox 内部使用。
    """

    async def exec_command(
        self,
        session_id: str,
        exec_dir: str,
        command: str,
        wait_seconds: Optional[int] = None,
    ) -> ToolResult:
        """根据传递的会话id+目录+命令执行对应的命令

        ``wait_seconds`` 控制同步等待命令结束的最长秒数，超过后 ToolResult
        的 data 中 ``status`` 为 ``"running"``，调用方需要用
        ``wait_process``/``read_shell_output`` 轮询。
        """
        ...

    async def read_shell_output(
        self, session_id: str, console: bool = False
    ) -> ToolResult:
        """根据传递的会话id+是否返回控制台记录获取shell结果"""
        ...

    async def wait_process(
        self, session_id: str, seconds: Optional[int] = None
    ) -> ToolResult:
        """根据传递的会话id+秒数等待程序执行"""
        ...

    async def write_shell_input(
        self,
        session_id: str,
        input_text: str,
        press_enter: bool = True,
    ) -> ToolResult:
        """根据传递会话id+文本内容+是否回车键写入内容到进程中"""
        ...

    async def resize_shell_session(
        self,
        session_id: str,
        cols: int,
        rows: int,
    ) -> ToolResult:
        """根据传递会话id+终端列行参数调整PTY窗口大小"""
        ...

    async def kill_process(self, session_id: str) -> ToolResult:
        """根据传递的会话id杀死对应的进程"""
        ...

    async def write_file(
        self,
        filepath: str,
        content: str,
        append: bool = False,
        leading_newline: bool = False,
        trailing_newline: bool = False,
        sudo: bool = False,
    ) -> ToolResult:
        """根据传递的文件路径+写入内容+追加模式+前后内容新行+超级权限写入对应的文件"""
        ...

    async def read_file(
        self,
        filepath: str,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
        sudo: bool = False,
        max_length: int = 10000,
    ) -> ToolResult:
        """根据传递的文件路径+起点终点行数+超级权限读取对应的文件内容"""
        ...

    async def check_file_exists(self, filepath: str) -> ToolResult:
        """根据传递的文件路径判断文件是否存在"""
        ...

    async def delete_file(self, filepath: str) -> ToolResult:
        """根据传递的文件路径删除指定文件"""
        ...

    async def list_files(self, dir_path: str) -> ToolResult:
        """根据传递的文件夹路径列出该路径下的所有文件"""
        ...

    async def replace_in_file(
        self,
        filepath: str,
        old_str: str,
        new_str: str,
        sudo: bool = False,
    ) -> ToolResult:
        """根据传递文件路径+新旧内容+超级权限完成文件内容替换"""
        ...

    async def search_in_file(
        self, filepath: str, regex: str, sudo: bool = False
    ) -> ToolResult:
        """根据传递的文件路径+正则+超级权限完成文件内容检索"""
        ...

    async def find_files(self, dir_path: str, glob_pattern: str) -> ToolResult:
        """根据传递的文件夹路径+匹配规则查找文件"""
        ...

    async def snapshot_workspace(
        self,
        root: str = "/home/ubuntu",
        *,
        max_paths: int,
        max_files: int,
        max_total_bytes: int,
        max_seconds: float,
    ) -> ToolResult:
        """S2 §3.1 — capped directory-excluded content snapshot of ``root``.

        Returns a ToolResult whose ``data`` carries ``{entries, truncated}``;
        the ParentSandboxAdapter decodes it into a domain WorkspaceScan.
        """
        ...

    async def kill_all_shell_sessions(self) -> ToolResult:
        """[S2 PR-4 §3.2] os.killpg every tracked shell session's process group
        so a backgrounded descendant cannot keep writing during the POST scan,
        then run the §3.2(c) /proc survivor check. ``success == False`` (a
        survivor still holds the workspace) ⇒ the adapter raises ⇒ finalizer
        FAILED."""
        ...

    async def upload_file(
        self,
        file_data: BinaryIO,
        filepath: str,
        filename: Optional[str] = None,
        *,
        refuse_special: bool = False,
    ) -> ToolResult:
        """根据文件源数据+路径+文件名将文件上传到沙箱中。

        ``refuse_special=True``（仅 coordinator apply/seed 路径）令底层原子写
        在 special 文件目标处 raise 而非 D12 写穿透。
        """
        ...

    async def download_file(self, filepath: str) -> BinaryIO:
        """根据传递的文件路径下载沙箱中的文件"""
        ...

    async def ensure_sandbox(self) -> None:
        """确保当前沙箱存在，如果不存在会创建"""
        ...

    async def destroy(self) -> bool:
        """销毁当前沙箱实例"""
        ...

    async def get_browser(self) -> Browser:
        """获取沙箱中的浏览器实例"""
        ...

    @property
    def id(self) -> str:
        """只读属性，返回沙箱的id"""
        ...

    @property
    def cdp_url(self) -> str:
        """只读属性，返回沙箱的cdp链接(操控浏览器的)"""
        ...

    @property
    def shell_ws_url(self) -> str:
        """只读属性，返回沙箱Shell WebSocket基础链接"""
        ...

    @property
    def vnc_url(self) -> str:
        """只读属性，获取沙箱的vnc链接(远程桌面链接)"""
        ...

    @property
    def applied_runtime_policy(self) -> "ContainerRuntimePolicy | None":
        """C5c: the applied ContainerRuntimePolicy assembled inside ``_create_task``
        from the real docker kwargs (capture_kind='applied'). ``None`` on the
        hardening-OFF path, the external_address path, and fakes that do not set it.
        Callers MUST read it via ``getattr(sandbox, "applied_runtime_policy", None)``.
        """
        ...

    @classmethod
    async def create(
        cls, user_id: Optional[str] = None, *,
        runtime_policy: "ContainerRuntimePolicy | None" = None,
    ) -> Self:
        """类方法，用于快速创建一个沙箱。

        ``user_id`` 在 M1 引入：不传时容器只挂载基础目录（向后兼容），
        传入时实现方应同时 bind mount 该用户的 memory 目录
        （``${MEMORY_ROOT_HOST}/{user_id}`` → ``${MEMORY_ROOT_CONTAINER}/{user_id}``）。

        ``runtime_policy`` 为 C5c 引入：传入（仅 hardening 开启时）令实现方把
        compiled hardening kwargs 合并进 container_config 并构建 applied 快照；
        不传时按旧行为创建（INV-0 byte-identical）。
        """
        ...

    @classmethod
    async def get(cls, id: str) -> Optional[Self]:
        """类方法，根据传递的id获取沙箱实例"""
        ...


class SandboxHandle(Protocol):
    """Lifecycle-aware sandbox wrapper（I7）。

    Holder 持有此类型而非 Sandbox。所有 async method 在调用前校验
    generation，stale 立即 raise SandboxPoisonedError。
    支持 async context manager 自动释放。

    NOT @runtime_checkable — __getattr__ 代理下 isinstance 检查无意义。
    Caller 变量始终注解为 SandboxHandle（Protocol），不 import SandboxHandleImpl。
    """

    # ── Properties (cached at acquire time, no generation check) ──

    @property
    def id(self) -> str: ...

    @property
    def cdp_url(self) -> str: ...

    @property
    def shell_ws_url(self) -> str: ...

    @property
    def vnc_url(self) -> str: ...

    @property
    def generation(self) -> int: ...

    # ── Forwarded async methods (all check generation before dispatch) ──

    async def exec_command(
        self,
        session_id: str,
        exec_dir: str,
        command: str,
        wait_seconds: Optional[int] = None,
    ) -> ToolResult: ...

    async def read_shell_output(
        self, session_id: str, console: bool = False
    ) -> ToolResult: ...

    async def wait_process(
        self, session_id: str, seconds: Optional[int] = None
    ) -> ToolResult: ...

    async def write_shell_input(
        self, session_id: str, input_text: str, press_enter: bool = True
    ) -> ToolResult: ...

    async def resize_shell_session(
        self, session_id: str, cols: int, rows: int
    ) -> ToolResult: ...

    async def kill_process(self, session_id: str) -> ToolResult: ...

    async def write_file(
        self,
        filepath: str,
        content: str,
        append: bool = False,
        leading_newline: bool = False,
        trailing_newline: bool = False,
        sudo: bool = False,
    ) -> ToolResult: ...

    async def read_file(
        self,
        filepath: str,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
        sudo: bool = False,
        max_length: int = 10000,
    ) -> ToolResult: ...

    async def check_file_exists(self, filepath: str) -> ToolResult: ...

    async def delete_file(self, filepath: str) -> ToolResult: ...

    async def list_files(self, dir_path: str) -> ToolResult: ...

    async def replace_in_file(
        self, filepath: str, old_str: str, new_str: str, sudo: bool = False
    ) -> ToolResult: ...

    async def search_in_file(
        self, filepath: str, regex: str, sudo: bool = False
    ) -> ToolResult: ...

    async def find_files(self, dir_path: str, glob_pattern: str) -> ToolResult: ...

    async def snapshot_workspace(
        self,
        root: str = "/home/ubuntu",
        *,
        max_paths: int,
        max_files: int,
        max_total_bytes: int,
        max_seconds: float,
    ) -> ToolResult:
        """S2 §3.1 forwarded RPC (see SANDBOX_FORWARDED_METHODS)."""
        ...

    async def kill_all_shell_sessions(self) -> ToolResult:
        """[S2 PR-4 §3.2] os.killpg every tracked shell session's process group
        so a backgrounded descendant cannot keep writing during the POST scan,
        then run the §3.2(c) /proc survivor check. ``success == False`` (a
        survivor still holds the workspace) ⇒ the adapter raises ⇒ finalizer
        FAILED."""
        ...

    async def upload_file(
        self,
        file_data: BinaryIO,
        filepath: str,
        filename: Optional[str] = None,
        *,
        refuse_special: bool = False,
    ) -> ToolResult: ...

    async def download_file(self, filepath: str) -> BinaryIO: ...

    async def ensure_sandbox(self) -> None: ...

    async def get_browser(self) -> Browser: ...

    # ── Lifecycle ──

    def release(self) -> None:
        """Release this handle. Removes from registry's open-handle set. Idempotent."""
        ...

    async def __aenter__(self) -> "SandboxHandle": ...

    async def __aexit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None: ...
