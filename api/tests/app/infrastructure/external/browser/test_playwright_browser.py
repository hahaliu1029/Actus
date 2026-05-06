from unittest.mock import AsyncMock

import pytest
from playwright.async_api import TimeoutError as PlaywrightTimeoutError

from app.infrastructure.external.browser.playwright_browser import PlaywrightBrowser
from app.infrastructure.external.browser.playwright_browser_fun import (
    GET_VISIBLE_CONTENT_FUNC,
    INJECT_CONSOLE_LOGS_FUNC,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class _FakePage:
    def __init__(self) -> None:
        self.goto_calls: list[tuple[str, dict[str, object]]] = []
        self.wait_calls: list[tuple[str, int]] = []
        self.evaluate_calls: list[str] = []
        self.aria_snapshot_text = '- button "提交"\n- textbox "搜索"'
        self.aria_snapshot_raises: bool = False
        self.logs = ["[INFO] start", "[WARN] retry", "[ERROR] failed"]
        self.raise_networkidle_timeout = False
        # Recorded calls — let tests verify the production path actually hit
        # the right Playwright surface, not just any locator.
        self.locator_calls: list[str] = []
        self.role_calls: list[tuple[str, object, bool]] = []

    @property
    def url(self) -> str:
        return "https://example.com/"

    def locator(self, selector: str) -> "_FakeLocator":
        self.locator_calls.append(selector)
        return _FakeLocator(
            self.aria_snapshot_text, raises=self.aria_snapshot_raises
        )

    def get_by_role(self, role: str, *, name: object = None, exact: bool = False) -> "_FakeLocator":
        self.role_calls.append((role, name, exact))
        return _FakeLocator(self.aria_snapshot_text)

    async def goto(self, url: str, **kwargs) -> None:
        self.goto_calls.append((url, kwargs))

    async def wait_for_load_state(self, state: str, timeout: int) -> None:
        self.wait_calls.append((state, timeout))
        if self.raise_networkidle_timeout:
            raise PlaywrightTimeoutError("networkidle timeout")

    async def evaluate(self, script: str, *_args):
        self.evaluate_calls.append(script)
        if script == INJECT_CONSOLE_LOGS_FUNC:
            return True
        if "window.console.logs || []" in script:
            return list(self.logs)
        if "MutationObserver" in script or "__br1" in script:
            return None
        return None


class _FakeContext:
    def __init__(self, pages: list[_FakePage]) -> None:
        self.pages = pages


class _FakeBrowser:
    def __init__(self, page: _FakePage) -> None:
        self.contexts = [_FakeContext([page])]


class _FakeLocator:
    def __init__(self, snapshot_text: str = "", raises: bool = False) -> None:
        self._snapshot_text = snapshot_text
        self._raises = raises
        self.nth_calls: list[int] = []
        self.click_calls: list[dict[str, object]] = []

    @property
    def first(self) -> "_FakeLocator":
        return self

    def nth(self, i: int) -> "_FakeLocator":
        self.nth_calls.append(i)
        return self

    async def aria_snapshot(self) -> str:
        if self._raises:
            raise RuntimeError("simulated mid-extraction failure")
        return self._snapshot_text

    async def bounding_box(self) -> "dict[str, float] | None":
        return {"x": 10.0, "y": 20.0, "width": 100.0, "height": 30.0}

    async def count(self) -> int:
        return 1

    async def is_visible(self) -> bool:
        return True

    async def click(self, **kwargs: object) -> None:
        # L1/L2 path in GroundedClickEngine awaits locator.click(timeout=...).
        self.click_calls.append(dict(kwargs))


async def test_navigate_waits_load_then_networkidle_then_extracts_elements() -> None:
    browser = PlaywrightBrowser(cdp_url="ws://example")
    page = _FakePage()
    browser.page = page
    browser.browser = _FakeBrowser(page)  # type: ignore[assignment]
    browser._extract_interactive_elements = AsyncMock(return_value=["0:<button>提交</button>"])  # type: ignore[method-assign]
    browser.wait_for_page_load = AsyncMock(return_value=True)  # type: ignore[method-assign]

    result = await browser.navigate("https://example.com")

    assert result.success is True
    assert page.goto_calls == [
        ("https://example.com", {"wait_until": "load", "timeout": 30000})
    ]
    assert page.wait_calls == [("networkidle", 10000)]
    browser.wait_for_page_load.assert_awaited_once_with(timeout=15)
    assert result.data == {"interactive_elements": ["0:<button>提交</button>"]}


async def test_navigate_keeps_success_when_networkidle_timeout() -> None:
    browser = PlaywrightBrowser(cdp_url="ws://example")
    page = _FakePage()
    page.raise_networkidle_timeout = True
    browser.page = page
    browser.browser = _FakeBrowser(page)  # type: ignore[assignment]
    browser._extract_interactive_elements = AsyncMock(return_value=[])  # type: ignore[method-assign]
    browser.wait_for_page_load = AsyncMock(return_value=True)  # type: ignore[method-assign]

    result = await browser.navigate("https://example.com")

    assert result.success is True
    assert "networkidle" in str(result.message or "")


async def test_console_view_injects_logger_and_respects_max_lines() -> None:
    browser = PlaywrightBrowser(cdp_url="ws://example")
    page = _FakePage()
    browser.page = page
    browser.browser = _FakeBrowser(page)  # type: ignore[assignment]

    result = await browser.console_view(max_lines=2)

    assert result.success is True
    assert result.data == {"logs": ["[WARN] retry", "[ERROR] failed"]}
    assert page.evaluate_calls[0] == INJECT_CONSOLE_LOGS_FUNC
    assert "window.console.logs || []" in page.evaluate_calls[1]


def test_console_inject_script_is_idempotent_and_multilevel() -> None:
    assert "__manusConsoleHooked" in INJECT_CONSOLE_LOGS_FUNC
    assert "MAX_LOGS = 1000" in INJECT_CONSOLE_LOGS_FUNC
    assert "'log'" in INJECT_CONSOLE_LOGS_FUNC
    assert "'info'" in INJECT_CONSOLE_LOGS_FUNC
    assert "'warn'" in INJECT_CONSOLE_LOGS_FUNC
    assert "'error'" in INJECT_CONSOLE_LOGS_FUNC
    assert "'debug'" in INJECT_CONSOLE_LOGS_FUNC


def test_visible_content_script_uses_viewport_width_and_dedup() -> None:
    assert "viewportWidth" in GET_VISIBLE_CONTENT_FUNC
    assert "viewportWeight" not in GET_VISIBLE_CONTENT_FUNC
    assert "seenContentKeys" in GET_VISIBLE_CONTENT_FUNC
    assert "normalizeText" in GET_VISIBLE_CONTENT_FUNC
    assert "MAX_TEXT_LENGTH = 300" in GET_VISIBLE_CONTENT_FUNC


async def test_extract_interactive_elements_uses_aria_path() -> None:
    browser = PlaywrightBrowser(cdp_url="ws://example")
    page = _FakePage()
    page.aria_snapshot_text = '- button "提交"\n- textbox "搜索":\n  - /placeholder: 关键词'
    browser.page = page
    browser.browser = _FakeBrowser(page)  # type: ignore[assignment]

    out = await browser._extract_interactive_elements()

    assert out == ["0:<button>提交</button>", "1:<input>[Placeholder: 关键词]</input>"]
    assert len(browser._grounded.descriptors) == 2
    assert browser._grounded.descriptors[0].role == "button"
    assert browser._grounded.descriptors[0].name == "提交"
    assert browser._grounded.descriptors[1].role == "textbox"
    assert browser._grounded.descriptors[1].placeholder == "关键词"
    # Pin the call shape: ARIA snapshot must come from `body`, and bbox resolver
    # must fan out via `get_by_role` for each descriptor.
    assert page.locator_calls == ["body"]
    assert [(role, name) for role, name, _exact in page.role_calls] == [
        ("button", "提交"),
        ("textbox", "搜索"),
    ]


async def test_extract_interactive_elements_clears_cache_when_aria_snapshot_raises() -> None:
    """If aria_snapshot() raises mid-extraction, the engine must NOT keep the
    previous page's descriptors — otherwise click(index) on the new page would
    fresh-resolve a stale element."""
    browser = PlaywrightBrowser(cdp_url="ws://example")
    page = _FakePage()
    browser.page = page
    browser.browser = _FakeBrowser(page)  # type: ignore[assignment]

    # First extraction: seed cache with two descriptors.
    page.aria_snapshot_text = '- button "提交"\n- textbox "搜索"'
    await browser._extract_interactive_elements()
    assert len(browser._grounded.descriptors) == 2

    # Second extraction on the new page: aria_snapshot raises mid-call.
    page.aria_snapshot_raises = True
    with pytest.raises(RuntimeError, match="simulated mid-extraction failure"):
        await browser._extract_interactive_elements()

    # Engine cache must be empty — no stale descriptors leaked from page #1.
    assert browser._grounded.descriptors == []


async def test_click_index_uses_grounded_engine_l1_path() -> None:
    browser = PlaywrightBrowser(cdp_url="ws://example")
    page = _FakePage()
    browser.page = page
    browser.browser = _FakeBrowser(page)  # type: ignore[assignment]
    page.aria_snapshot_text = '- button "提交"'
    await browser._extract_interactive_elements()

    result = await browser.click(index=0)

    assert result.success is True
    # Postcondition trace recorded via engine.record_trace; eval harness reads via last_trace.
    assert browser._grounded.last_trace is not None
    assert browser._grounded.last_trace.success_level == 1


async def test_click_index_returns_failure_when_descriptor_out_of_range() -> None:
    browser = PlaywrightBrowser(cdp_url="ws://example")
    page = _FakePage()
    browser.page = page
    browser.browser = _FakeBrowser(page)  # type: ignore[assignment]
    page.aria_snapshot_text = ""
    await browser._extract_interactive_elements()

    result = await browser.click(index=10)

    assert result.success is False
    assert ("无效" in (result.message or "")) or ("未找到" in (result.message or ""))


async def test_click_coordinate_path_unchanged() -> None:
    """The coordinate_x/coordinate_y path must keep its legacy direct-mouse.click
    behavior — no engine, no postcondition. Pure passthrough."""
    browser = PlaywrightBrowser(cdp_url="ws://example")
    page = _FakePage()

    class _Mouse:
        def __init__(self) -> None:
            self.clicks: list[tuple[float, float]] = []

        async def click(self, x: float, y: float) -> None:
            self.clicks.append((x, y))

    page.mouse = _Mouse()  # type: ignore[attr-defined]
    browser.page = page
    browser.browser = _FakeBrowser(page)  # type: ignore[assignment]

    result = await browser.click(coordinate_x=100.0, coordinate_y=200.0)

    assert result.success is True
    assert page.mouse.clicks == [(100.0, 200.0)]  # type: ignore[attr-defined]


async def test_input_index_uses_grounded_engine() -> None:
    browser = PlaywrightBrowser(cdp_url="ws://example")
    page = _FakePage()
    fills: list[str] = []
    enters: list[bool] = []

    class _LocatorWithFill(_FakeLocator):
        async def fill(self, text: str) -> None:
            fills.append(text)

    class _Keyboard:
        async def press(self, key: str) -> None:
            enters.append(key == "Enter")

    # Override get_by_role to return a locator that supports .fill()
    def _fake_get_by_role(role: str, *, name: object = None, exact: bool = False) -> _FakeLocator:
        page.role_calls.append((role, name, exact))
        return _LocatorWithFill('- textbox "搜索"')

    page.get_by_role = _fake_get_by_role  # type: ignore[assignment]
    page.keyboard = _Keyboard()  # type: ignore[attr-defined]
    page.aria_snapshot_text = '- textbox "搜索"'
    browser.page = page
    browser.browser = _FakeBrowser(page)  # type: ignore[assignment]
    await browser._extract_interactive_elements()

    result = await browser.input(text="hello", press_enter=True, index=0)

    assert result.success is True
    assert fills == ["hello"]
    assert enters == [True]


async def test_select_option_uses_grounded_engine() -> None:
    browser = PlaywrightBrowser(cdp_url="ws://example")
    page = _FakePage()
    selects: list[int] = []

    class _LocatorSelect(_FakeLocator):
        async def select_option(self, *, index: int) -> None:
            selects.append(index)

    def _fake_get_by_role(role: str, *, name: object = None, exact: bool = False) -> _FakeLocator:
        page.role_calls.append((role, name, exact))
        return _LocatorSelect('- combobox "语言"')

    page.get_by_role = _fake_get_by_role  # type: ignore[assignment]
    page.aria_snapshot_text = '- combobox "语言"'
    browser.page = page
    browser.browser = _FakeBrowser(page)  # type: ignore[assignment]
    await browser._extract_interactive_elements()

    result = await browser.select_option(index=0, option=2)

    assert result.success is True
    assert selects == [2]
