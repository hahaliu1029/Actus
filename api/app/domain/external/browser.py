from typing import Optional, Protocol

from app.domain.models.tool_result import ToolResult


class Browser(Protocol):
    """浏览器服务扩展，涵盖：访问页面、URL跳转、输入、移动鼠标、滚动、截图、执行js代码等"""

    async def view_page(self) -> ToolResult:
        """浏览获取当前浏览器的页面内容"""
        ...

    async def navigate(self, url: str) -> ToolResult:
        """传递对应的url使用浏览器导航到该页面"""
        ...

    async def restart(self, url: str) -> ToolResult:
        """重启浏览器并访问对应的URL"""
        ...

    async def click(
        self,
        index: Optional[int] = None,
        coordinate_x: Optional[float] = None,
        coordinate_y: Optional[float] = None,
    ) -> ToolResult:
        """传递对应的元素索引或者xy坐标实现点击功能"""
        ...

    async def input(
        self,
        text: str,
        press_enter: bool,
        index: Optional[int] = None,
        coordinate_x: Optional[float] = None,
        coordinate_y: Optional[float] = None,
    ) -> ToolResult:
        """传递文本+回车标识+索引+xy坐标实现在网页输入框中输入对应内容"""
        ...

    async def move_mouse(self, coordinate_x: float, coordinate_y: float) -> ToolResult:
        """传递对应的xy坐标移动鼠标"""
        ...

    async def press_key(self, key: str) -> ToolResult:
        """传递按键标识实现浏览器模拟按键"""
        ...

    async def select_option(self, index: int, option: int) -> ToolResult:
        """传递索引+选项标识在下拉菜单中选择指定的选项"""
        ...

    async def scroll_up(self, to_top: Optional[bool] = None) -> ToolResult:
        """向上滚动浏览器，如果没传递to_top=True则向上滚动一屏"""
        ...

    async def scroll_down(self, to_down: Optional[bool] = None) -> ToolResult:
        """向下滚动浏览器，如果没传递to_down=True则向下滚动一屏"""
        ...

    async def screenshot(self, full_page: Optional[bool] = None) -> bytes:
        """对当前浏览器的页面进行截图，传递full_page=True则意味着整页截图"""
        ...

    async def console_exec(self, javascript: str) -> ToolResult:
        """传递对应的js脚本在浏览器的控制台执行"""
        ...

    async def console_view(self, max_lines: Optional[int] = None) -> ToolResult:
        """传递最大输出行数，获取控制台的输出结果，如果不传递则获取所有结果"""
        ...

    async def aclose(self) -> None:
        """best-effort 关闭底层连接（幂等；无连接时 no-op）。

        SPM PR-1b：``BrowserAccessor`` 终态释放委托到本方法。既有实现方
        （``PlaywrightBrowser``）委托其现有 ``cleanup()``；``Browser`` 是 Protocol
        而非 ABC，结构化子类型下未实现本方法的第三方类不会在运行期破裂
        （无 isinstance 校验），故此处无需提供默认实现。"""
        ...


class BrowserAccessor(Protocol):
    """SPM PR-1b：tool 层访问 Browser 的显式契约（typing-only；镜像 SandboxAccessor）。

    Eager 实现（``EagerBrowserAccessor``）包裹既有 Browser，``get()`` 零 IO；
    OnDemand 变体留 PR-1c。NOT @runtime_checkable（结构化协议）。
    """

    async def get(self) -> "Browser": ...

    def peek(self) -> "Browser | None": ...

    async def aclose(self) -> None: ...
