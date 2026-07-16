import asyncio
import json
import logging
from typing import Any, List, Optional

from app.domain.external.browser import Browser as BrowserProtocol
from app.domain.models.tool_result import ToolResult
from langchain_core.language_models import BaseChatModel
from markdownify import markdownify
from playwright.async_api import (
    Browser,
    Page,
    Playwright,
    TimeoutError as PlaywrightTimeoutError,
    async_playwright,
)

from app.infrastructure.external.browser.aria_extractor import (
    parse_aria_snapshot,
    format_descriptors_for_llm,
)
from app.infrastructure.external.browser.grounded_click import (
    GroundedClickEngine,
    resolve_bboxes_for_descriptors,
)

from .playwright_browser_fun import (
    GET_VISIBLE_CONTENT_FUNC,
    INJECT_CONSOLE_LOGS_FUNC,
)

logger = logging.getLogger(__name__)


class PlaywrightBrowser(BrowserProtocol):
    """基础Playwright管理的浏览器扩展"""

    def __init__(
        self,
        cdp_url: str,  # CDP的连接地址
        llm: Optional[
            BaseChatModel
        ] = None,  # 可选参数，传递LLM，如果传递了则会使用LLM对页面内容进行整理变成markdown格式
    ) -> None:
        """构造函数，完成Playwright浏览器的初始化"""
        # LLM相关
        self.llm: Optional[BaseChatModel] = llm

        # 浏览器相关
        self.cdp_url: str = cdp_url
        self.playwright: Optional[Playwright] = None
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None

        # GroundedClick 引擎：descriptor 缓存 + L1/L2/L3 fresh resolve
        self._grounded = GroundedClickEngine()

    async def _ensure_browser(self) -> None:
        """确保浏览器存在，如果不存在则初始化"""
        if not self.browser or not self.page:
            if not await self.initialize():
                raise Exception("初始化Playwright浏览器失败")

    async def _ensure_page(self) -> None:
        """确保浏览器页面存在，如果不存在则新建"""
        # 1.先保证浏览器存在
        await self._ensure_browser()

        # 2.如果页面不存在则创建新上下文+页面
        if not self.page:
            self.page = (
                await self.browser.new_page()
            )  # 等同于self.browser.new_context().new_page()
        else:
            # 3.如果页面存在则提取所有上下文
            contexts = self.browser.contexts
            if contexts:
                # 4.获取默认上下文及页面
                default_context = contexts[0]
                pages = default_context.pages

                # 5.判断页面是否存在
                if pages:
                    # 6.获取当前最新页面(Chrome浏览器新增页面时，默认往右侧添加，相当于pages中序号较大的页面)
                    latest_page = pages[-1]

                    # 7.判断当前页面是否为最新页面，如果不是则更新
                    if self.page != latest_page:
                        self.page = latest_page

    async def _extract_content(self) -> str:
        """提取当前页面内容"""
        # 1.使用js代码获取当前页面可见元素内容
        visible_content = await self.page.evaluate(GET_VISIBLE_CONTENT_FUNC)

        # 2.使用markdownify这个库将html文档转换为markdown
        markdown_content = markdownify(visible_content)

        # 3.模型上下文长度有限，提取最大不超过50k个字符
        max_content_length = min(len(markdown_content), 50000)

        # 4.判断是否传递了llm，如果传递了，还可以使用llm对markdown_content进行整理
        if self.llm:
            # 5.调用llm对markdown_content内容进行二次整理
            from langchain_core.messages import HumanMessage, SystemMessage
            response = await self.llm.ainvoke(
                [
                    SystemMessage(content="您是一名专业的网页信息提取助手。请从当前页面内容中提取所有信息并将其转换为markdown格式。"),
                    HumanMessage(content=markdown_content[:max_content_length]),
                ]
            )
            return response.content if hasattr(response, "content") else str(response)
        else:
            return markdown_content[:max_content_length]

    async def _extract_interactive_elements(self) -> List[str]:
        """ARIA-based extraction: locator('body').aria_snapshot() → descriptors → bbox 填充 → LLM 字符串。

        Clears the engine descriptor cache up-front so a mid-extraction failure
        (e.g. `aria_snapshot()` raises after navigation) cannot leave stale
        descriptors from a previous page available to T12+ click(index=...) flows.
        """
        await self._ensure_page()
        self._grounded.set_descriptors([])
        snapshot_text = await self.page.locator("body").aria_snapshot()
        descriptors = parse_aria_snapshot(snapshot_text)
        descriptors = await resolve_bboxes_for_descriptors(self.page, descriptors)
        self._grounded.set_descriptors(descriptors)
        return format_descriptors_for_llm(descriptors)

    async def click(
        self,
        index: Optional[int] = None,
        coordinate_x: Optional[float] = None,
        coordinate_y: Optional[float] = None,
    ) -> ToolResult:
        """index 路径走 GroundedClickEngine fresh resolve；coordinate 路径直接 mouse.click."""
        await self._ensure_page()

        if coordinate_x is not None and coordinate_y is not None:
            try:
                await self.page.mouse.click(coordinate_x, coordinate_y)
                return ToolResult(success=True)
            except Exception as exc:
                return ToolResult(success=False, message=f"坐标点击出错: {exc}")

        if index is None:
            return ToolResult(success=False, message="必须提供 index 或 coordinate_x/y")

        try:
            descriptor = self._grounded.get_descriptor(index)
        except IndexError:
            return ToolResult(
                success=False,
                message=f"使用索引{index}查找该元素无效, 未找到",
            )

        before_url = self.page.url if isinstance(self.page.url, str) else ""
        await self._grounded.install_mutation_observer(self.page)
        trace = await self._grounded.resolve_and_click(
            self.page,
            intent=f"点击元素 {index}",
            descriptor=descriptor,
        )

        if trace.success_level is None:
            # explicit_failure trace recorded for eval harness to inspect.
            self._grounded.record_trace(trace)
            return ToolResult(success=False, message=trace.error or "定位失败")

        postcondition = await self._grounded.verify_postcondition(
            self.page,
            before_url=before_url,
        )
        trace.postcondition = postcondition
        self._grounded.record_trace(trace)

        if postcondition == "no_observable_change":
            return ToolResult(
                success=True,
                message="操作已执行，未检测到页面变化。如果预期应有变化，请重试或检查目标元素。",
            )
        return ToolResult(success=True)

    async def initialize(self) -> bool:
        """初始化并确保资源是可用的"""
        # 1.定义重试次数+重试延迟确保资源存在
        max_retries = 5
        retry_interval = 1

        # 2.循环开始资源构建
        for attempt in range(max_retries):
            try:
                # 3.创建playwright上下文并连接到cdp浏览器
                self.playwright = await async_playwright().start()
                self.browser = await self.playwright.chromium.connect_over_cdp(
                    self.cdp_url
                )

                # 4.获取浏览器的所有上下文
                contexts = self.browser.contexts

                # 5.如果上下文存在且有页面，则复用已有页面
                if contexts and contexts[0].pages:
                    # 6.获取当前上下文的最新页面(最右侧标签页)
                    pages = contexts[0].pages
                    self.page = pages[-1]
                else:
                    # 7.上下文不存在或者没有页面，新建一个页面
                    context = (
                        contexts[0] if contexts else await self.browser.new_context()
                    )
                    self.page = await context.new_page()

                return True
            except Exception as e:
                # 10.清除所有资源
                await self.cleanup()

                # 11.判断重试次数是否等于最大重试次数
                if attempt == max_retries - 1:
                    logger.error(
                        f"初始化Playwright浏览器失败(已重试{max_retries}次): {str(e)}"
                    )
                    return False

                # 12.使用指数级增长进行休眠，最大休眠时间为10s
                retry_interval = min(retry_interval * 2, 10)
                logger.warning(
                    f"初始化Playwright浏览器失败, 即将进行第{attempt + 1}次重试: {str(e)}"
                )
                await asyncio.sleep(retry_interval)

    async def cleanup(self) -> None:
        """清除Playwright资源，包含浏览器+页面+Playwright"""
        try:
            # 1.检测浏览器是否存在，如果存在则删除该浏览器下的所有tabs页面
            if self.browser:
                # 2.获取该浏览器的所有上下文
                contexts = self.browser.contexts
                if contexts:
                    # 3.循环遍历所有上下文逐个处理
                    for context in contexts:
                        # 4.获取每个上下文的所有页面
                        pages = context.pages
                        for page in pages:
                            # 5.循环遍历清除所有页面
                            if not page.is_closed():
                                await page.close()

            # 6.判断self.page是否关闭
            if self.page and not self.page.is_closed():
                await self.page.close()

            # 7.关闭浏览器
            if self.browser:
                await self.browser.close()

            # 8.停止playwright
            if self.playwright:
                await self.playwright.stop()
        except Exception as e:
            # 9记录错误日志
            logger.error(f"清理Playwright浏览器资源出错: {str(e)}")
        finally:
            # 10.重置所有资源
            self.page = None
            self.browser = None
            self.playwright = None

    async def aclose(self) -> None:
        """SPM Task 8：best-effort 关闭底层 Playwright/CDP 连接（幂等；无连接时 no-op）。

        委托既有 ``cleanup()``——它已关闭 pages/browser、``stop()`` playwright、吞掉
        内部 Exception，并在 ``finally`` 把三个句柄重置为 ``None``，因此第二次调用
        自然 no-op（所有句柄已 ``None`` → 各 ``if`` 分支跳过）。外层再包一层
        best-effort try/except 以满足 ``BrowserAccessor.aclose`` 合同——即便
        ``cleanup`` 的语义未来改变也不外抛。

        NEW method（INV-SPM-2：不触碰任何既有方法/行为）。
        """
        try:
            await self.cleanup()
        except Exception:  # best-effort terminal close — never propagate
            logger.warning(
                "PlaywrightBrowser.aclose best-effort close failed", exc_info=True
            )

    async def wait_for_page_load(self, timeout: int = 15) -> bool:
        """传递超时时间，等待当前页面是否加载完毕"""
        # 1.确保当前页面存在
        await self._ensure_page()

        # 2.使用异步任务事件循环中的时间来作为开始时间(只和异步任务相关)
        start_time = asyncio.get_event_loop().time()
        check_interval = 0.5

        # 3.循环检测网页是否加载成功
        while asyncio.get_event_loop().time() - start_time < timeout:
            # 4.使用js代码判断网页是否加载成功
            is_completed = await self.page.evaluate(
                """() => document.readyState === 'complete'"""
            )
            if is_completed:
                return True

            # 5.未加载成功则休眠对应时间
            await asyncio.sleep(check_interval)

        return False

    async def navigate(self, url: str) -> ToolResult:
        """根据传递的url跳转到指定页面"""
        # 1.确保页面存在
        await self._ensure_page()

        try:
            # 2.使用强等待策略进行跳转，提升SPA页面可见内容提取稳定性
            await self.page.goto(url, wait_until="load", timeout=30000)

            warnings: list[str] = []
            try:
                await self.page.wait_for_load_state("networkidle", timeout=10000)
            except PlaywrightTimeoutError:
                warnings.append("页面等待networkidle超时，已继续提取当前可见内容。")
            except Exception as e:
                logger.warning(f"等待networkidle状态异常: {str(e)}")
                warnings.append("页面等待networkidle异常，已继续提取当前可见内容。")

            page_loaded = await self.wait_for_page_load(timeout=15)
            if not page_loaded:
                warnings.append("页面未完全加载，返回当前可见内容。")

            return ToolResult(
                success=True,
                message=" ".join(warnings),
                data={
                    "interactive_elements": await self._extract_interactive_elements()
                },
            )
        except Exception as e:
            # 返回错误的工具结果
            return ToolResult(
                success=False, message=f"浏览器导航到[{url}]失败: {str(e)}"
            )

    async def view_page(self) -> ToolResult:
        """获取当前网页的内容(内容+可交互元素列表)"""
        # 1.确保页面存在
        await self._ensure_page()

        # 2.等待页面加载完成
        await self.wait_for_page_load()

        # 3.更新页面的可交互元素
        interactive_elements = await self._extract_interactive_elements()

        # 4.返回工具结果
        return ToolResult(
            success=True,
            data={
                "content": await self._extract_content(),
                "interactive_elements": interactive_elements,
            },
        )

    async def input(
        self,
        text: str,
        press_enter: bool,
        index: Optional[int] = None,
        coordinate_x: Optional[float] = None,
        coordinate_y: Optional[float] = None,
    ) -> ToolResult:
        """index 路径走 GroundedClickEngine fresh resolve (L1+L2)；coordinate 路径走 mouse.click + keyboard.type。"""
        await self._ensure_page()

        if coordinate_x is not None and coordinate_y is not None:
            await self.page.mouse.click(coordinate_x, coordinate_y)
            await self.page.keyboard.type(text)
            if press_enter:
                await self.page.keyboard.press("Enter")
            return ToolResult(success=True)

        if index is None:
            return ToolResult(success=False, message="必须提供 index 或 coordinate_x/y")

        try:
            descriptor = self._grounded.get_descriptor(index)
        except IndexError:
            return ToolResult(success=False, message=f"使用索引{index}查找该元素无效, 未找到")

        # input 不走 L3 bbox 兜底：bbox.click + keyboard.type 在 coordinate 路径已覆盖。
        locator = await self._grounded.fresh_resolve_l1(self.page, descriptor)
        if locator is None:
            locator = await self._grounded.fresh_resolve_l2(self.page, descriptor)
        if locator is None:
            return ToolResult(success=False, message=f"输入框 index={index} 定位失败")

        # `fill()` works for native `<input>` / `<textarea>` / `[contenteditable]`.
        # ARIA textboxes (`<div role="textbox" tabindex="0">`) are not in that
        # set — Playwright raises "Element is not an <input>, <textarea> or
        # [contenteditable] element". Fall back to focus-via-click + keyboard
        # typing, mirroring the coordinate path's strategy.
        try:
            await locator.fill(text)
        except Exception:
            try:
                await locator.click(timeout=5000)
                await self.page.keyboard.type(text)
            except Exception as exc:
                return ToolResult(success=False, message=f"输入出错: {exc}")

        if press_enter:
            try:
                await self.page.keyboard.press("Enter")
            except Exception as exc:
                return ToolResult(success=False, message=f"按 Enter 出错: {exc}")

        return ToolResult(success=True)

    async def move_mouse(self, coordinate_x: float, coordinate_y: float) -> ToolResult:
        """传递xy坐标移动鼠标"""
        await self._ensure_page()
        await self.page.mouse.move(coordinate_x, coordinate_y)
        return ToolResult(success=True)

    async def press_key(self, key: str) -> ToolResult:
        """传递按键进行模拟"""
        await self._ensure_page()
        await self.page.keyboard.press(key)
        return ToolResult(success=True)

    async def select_option(self, index: int, option: int) -> ToolResult:
        """index 路径走 GroundedClickEngine fresh resolve (L1+L2 only)。

        Native `<select>` and ARIA combobox both surface as `role="combobox"`
        in `aria_snapshot()`, so the LLM-facing tag is the same. We dispatch
        per element type at runtime:

        - **Native `<select>`**: use `locator.select_option(index=option)`.
        - **Custom combobox (e.g. `<button role="combobox">`)**: click the
          combobox to expand, then click the Nth visible `[role="option"]`.
          Playwright `select_option()` raises "Element is not a <select>"
          on these, so we must detect via `tagName` and fall back.
        """
        await self._ensure_page()
        try:
            descriptor = self._grounded.get_descriptor(index)
        except IndexError:
            return ToolResult(success=False, message=f"使用索引{index}查找该元素无效, 未找到")

        locator = await self._grounded.fresh_resolve_l1(self.page, descriptor)
        if locator is None:
            locator = await self._grounded.fresh_resolve_l2(self.page, descriptor)
        if locator is None:
            return ToolResult(success=False, message=f"下拉框 index={index} 定位失败")

        try:
            tag_name = await locator.evaluate("el => el.tagName")
        except Exception:
            tag_name = ""

        if isinstance(tag_name, str) and tag_name.upper() == "SELECT":
            try:
                await locator.select_option(index=option)
                return ToolResult(success=True)
            except Exception as exc:
                return ToolResult(success=False, message=f"选择选项出错: {exc}")

        # Custom combobox: route via either aria-controls (deterministic) or a
        # visibility-diff heuristic that detects the listbox/menu newly opened
        # by the click. Naive "filter visible .first" was retired (codex audit)
        # because it silently picks an unrelated already-visible listbox when
        # one exists earlier in the DOM.
        aria_scope = await self._scope_via_aria_controls(locator)
        if aria_scope is not None:
            try:
                await locator.click(timeout=5000)
                options = aria_scope.get_by_role("option")
                await options.nth(option).click(timeout=5000)
                return ToolResult(success=True)
            except Exception as exc:
                return ToolResult(success=False, message=f"自定义下拉选择出错: {exc}")

        try:
            diff_scope = await self._scope_via_visible_diff(locator)
        except Exception as exc:
            return ToolResult(success=False, message=f"自定义下拉选择出错: {exc}")
        if diff_scope is None:
            # Ambiguous — 0 or multiple newly-visible popups after the click.
            # Failing loudly is safer than `.nth()` silently landing on an
            # unrelated listbox elsewhere on the page.
            return ToolResult(
                success=False,
                message=(
                    "自定义下拉框作用域识别失败：点击后未检测到唯一新打开的弹窗 "
                    "(0 或 ≥2 个 listbox/menu 同时可见)。请检查 combobox 是否声明 "
                    "`aria-controls` 指向其 popup。"
                ),
            )
        try:
            options = diff_scope.get_by_role("option")
            await options.nth(option).click(timeout=5000)
            return ToolResult(success=True)
        except Exception as exc:
            return ToolResult(success=False, message=f"自定义下拉选择出错: {exc}")

    async def _scope_via_aria_controls(self, combobox_locator: Any) -> Optional[Any]:
        """Resolve the combobox's owned listbox via `aria-controls` / `aria-owns`.

        Returns a `Locator` scoped to the referenced element, or `None` when
        neither attribute exposes an ID. Uses CSS attribute-selector form
        `[id="..."]` (with `json.dumps`-escaped value) rather than `#<id>`
        because framework-generated ids like Radix UI's `radix-:r1:` contain
        characters that break naive `#<id>` CSS parsing.
        """
        for attr in ("aria-controls", "aria-owns"):
            try:
                value = await combobox_locator.get_attribute(attr)
            except Exception:
                value = None
            if value:
                # `aria-controls` may carry a space-separated ID list; the
                # owned listbox is conventionally the first one.
                controls_id = value.split()[0] if isinstance(value, str) else None
                if controls_id:
                    return self.page.locator(f"css=[id={json.dumps(controls_id)}]")
        return None

    # Pre-click: tag every currently-visible `[role=listbox]` / `[role=menu]`
    # with a stable marker attribute. Identity-based (not index-based) so a
    # subsequent DOM mutation that inserts a new popup BEFORE existing visible
    # popups won't be mistaken for "new index appeared at end" — which broke
    # the prior set-of-indices diff (codex audit round 5 follow-up).
    _LISTBOX_TAG_BASELINE_SCRIPT = """() => {
        // Wipe any stale markers (from a prior call that crashed mid-flight)
        // so the baseline is clean before tagging.
        document.querySelectorAll('[data-br1-listbox-baseline]').forEach(el => {
            el.removeAttribute('data-br1-listbox-baseline');
        });
        const isVisible = (el) =>
            !el.hidden
            && el.getClientRects().length > 0
            && getComputedStyle(el).visibility !== 'hidden'
            && getComputedStyle(el).visibility !== 'collapse';
        document.querySelectorAll('[role="listbox"], [role="menu"]').forEach((el) => {
            if (isVisible(el)) el.setAttribute('data-br1-listbox-baseline', '1');
        });
    }"""

    # Post-click: enumerate current `[role=listbox]` / `[role=menu]` elements
    # and return the DOM-order indices of those that are visible AND lack the
    # baseline marker (= newly-visible since the pre-click tag). The indices
    # describe the CURRENT DOM order, which matches `page.locator(...).nth()`
    # semantics. Markers are wiped at the end so subsequent calls start clean.
    _LISTBOX_NEWLY_VISIBLE_SCRIPT = """() => {
        const isVisible = (el) =>
            !el.hidden
            && el.getClientRects().length > 0
            && getComputedStyle(el).visibility !== 'hidden'
            && getComputedStyle(el).visibility !== 'collapse';
        const all = [...document.querySelectorAll('[role="listbox"], [role="menu"]')];
        const newlyVisible = [];
        all.forEach((el, i) => {
            if (isVisible(el) && !el.hasAttribute('data-br1-listbox-baseline')) {
                newlyVisible.push(i);
            }
        });
        document.querySelectorAll('[data-br1-listbox-baseline]').forEach(el => {
            el.removeAttribute('data-br1-listbox-baseline');
        });
        return newlyVisible;
    }"""

    async def _scope_via_visible_diff(self, combobox_locator: Any) -> Optional[Any]:
        """Click `combobox_locator` and return a `Locator` scoped to the
        listbox/menu the click newly revealed.

        Side-effect: clicks the combobox. Caller MUST NOT click separately.

        Algorithm (marker-based identity, not index-based set diff):
        1. Pre-click: tag every currently-visible `[role=listbox]` /
           `[role=menu]` element with `data-br1-listbox-baseline`.
        2. Click the combobox.
        3. Post-click: enumerate the current DOM and collect indices of
           visible elements that LACK the marker. Markers travel with
           elements through DOM mutations, so this correctly identifies
           the new popup even when the click inserts it between existing
           visible popups (which would shift index-based diff results).
        4. If exactly one newly-visible index, scope to
           `page.locator(LISTBOX_QUERY).nth(idx)`. Otherwise return `None`
           to signal ambiguity — caller must fail loudly.
        """
        try:
            await self.page.evaluate(self._LISTBOX_TAG_BASELINE_SCRIPT)
        except Exception:
            # Pre-tag failed → can't reliably diff; fail rather than guess.
            return None

        await combobox_locator.click(timeout=5000)

        try:
            newly_visible = await self.page.evaluate(self._LISTBOX_NEWLY_VISIBLE_SCRIPT)
        except Exception:
            newly_visible = []
        if not isinstance(newly_visible, list):
            newly_visible = []

        if len(newly_visible) != 1:
            return None
        [idx] = newly_visible
        return self.page.locator('[role="listbox"], [role="menu"]').nth(idx)

    async def restart(self, url: str) -> ToolResult:
        """重启并跳转到指定URL"""
        await self.cleanup()
        return await self.navigate(url)

    async def scroll_up(self, to_top: Optional[bool] = None) -> ToolResult:
        """向上滚动浏览器一个屏幕或者整个页面"""
        # 1.确保页面存在
        await self._ensure_page()

        # 2.判断是否滚动到最顶部
        if to_top:
            await self.page.evaluate("window.scrollTo(0, 0)")
        else:
            await self.page.evaluate("window.scrollBy(0, -window.innerHeight)")

        return ToolResult(success=True)

    async def scroll_down(self, to_down: Optional[bool] = None) -> ToolResult:
        """向下滚动浏览器一个屏幕或者到最底部"""
        # 1.确保页面存在
        await self._ensure_page()

        # 2.判断是否滚动到最底部
        if to_down:
            await self.page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        else:
            await self.page.evaluate("window.scrollBy(0, window.innerHeight)")

        return ToolResult(success=True)

    async def screenshot(self, full_page: Optional[bool] = None) -> bytes:
        """传递full_page完成网页截图"""
        # 1.确保页面存在
        await self._ensure_page()

        # 2.创建一个截图配置
        screenshot_options = {"full_page": full_page, "type": "png"}

        return await self.page.screenshot(**screenshot_options)

    async def console_exec(self, javascript: str) -> ToolResult:
        """传递js代码在当前页面控制台执行"""
        # 1.确保页面存在
        await self._ensure_page()

        # 2.在正式开始执行代码之前先注入logs
        try:
            await self.page.evaluate(INJECT_CONSOLE_LOGS_FUNC)
        except Exception as e:
            logger.warning(f"注入window.console.logs失败: {str(e)}")

        # 3.正式执行js脚本
        result = await self.page.evaluate(javascript)
        return ToolResult(success=True, data={"result": result})

    async def console_view(self, max_lines: Optional[int] = None) -> ToolResult:
        """根据传递的行数查看控制台的日志"""
        # 1.确保页面存在
        await self._ensure_page()

        # 2.在查看日志之前确保日志注入已启用
        try:
            await self.page.evaluate(INJECT_CONSOLE_LOGS_FUNC)
        except Exception as e:
            logger.warning(f"注入window.console.logs失败: {str(e)}")

        # 3.可以指定另外一段js代码查看控制台的内容
        logs = await self.page.evaluate(
            """() => {
            return window.console.logs || [];
        }"""
        )

        if max_lines is not None:
            logs = logs[-max_lines:]

        return ToolResult(success=True, data={"logs": logs})
