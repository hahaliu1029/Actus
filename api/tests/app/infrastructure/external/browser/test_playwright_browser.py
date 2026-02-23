from unittest.mock import AsyncMock

import pytest
from playwright.async_api import TimeoutError as PlaywrightTimeoutError

from app.infrastructure.external.browser.playwright_browser import PlaywrightBrowser
from app.infrastructure.external.browser.playwright_browser_fun import (
    GET_INTERACTIVE_ELEMENTS_FUNC,
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
        self.interactive_elements_cache = ["stale"]
        self.logs = ["[INFO] start", "[WARN] retry", "[ERROR] failed"]
        self.raise_networkidle_timeout = False

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
        return None


class _FakeContext:
    def __init__(self, pages: list[_FakePage]) -> None:
        self.pages = pages


class _FakeBrowser:
    def __init__(self, page: _FakePage) -> None:
        self.contexts = [_FakeContext([page])]


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
    assert page.interactive_elements_cache == []
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


def test_interactive_selector_covers_balanced_aria_roles() -> None:
    for role in [
        "link",
        "menuitem",
        "menuitemcheckbox",
        "menuitemradio",
        "tab",
        "option",
        "checkbox",
        "radio",
        "switch",
        "textbox",
        "searchbox",
        "combobox",
        "slider",
        "spinbutton",
    ]:
        assert f'[role="{role}"]' in GET_INTERACTIVE_ELEMENTS_FUNC
    assert "aria-label" in GET_INTERACTIVE_ELEMENTS_FUNC


def test_visible_content_script_uses_viewport_width_and_dedup() -> None:
    assert "viewportWidth" in GET_VISIBLE_CONTENT_FUNC
    assert "viewportWeight" not in GET_VISIBLE_CONTENT_FUNC
    assert "seenContentKeys" in GET_VISIBLE_CONTENT_FUNC
    assert "normalizeText" in GET_VISIBLE_CONTENT_FUNC
    assert "MAX_TEXT_LENGTH = 300" in GET_VISIBLE_CONTENT_FUNC
