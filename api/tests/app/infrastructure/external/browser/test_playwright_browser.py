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
    """Native `<select>` path: tagName probe returns 'SELECT' → native
    `select_option(index=option)`. No combobox click + option click.

    Includes negative locks (per codex audit follow-up) so a regression that
    falls through to the custom-combobox branch would fail the test, not
    silently pass on the side-effect of `select_option` running."""
    browser = PlaywrightBrowser(cdp_url="ws://example")
    page = _FakePage()
    selects: list[int] = []
    locator_clicks: list[bool] = []

    class _LocatorSelect(_FakeLocator):
        async def select_option(self, *, index: int) -> None:
            selects.append(index)

        async def click(self, *, timeout: int = 5000) -> None:
            # Native path must NOT click the locator (that's the custom-combobox
            # fallback's first step). Recording lets the test fail loudly.
            locator_clicks.append(True)

        async def evaluate(self, script: str) -> object:
            # Native select detection probe.
            return "SELECT"

    def _fake_get_by_role(role: str, *, name: object = None, exact: bool = False) -> _FakeLocator:
        page.role_calls.append((role, name, exact))
        return _LocatorSelect('- combobox "语言"')

    page.get_by_role = _fake_get_by_role  # type: ignore[assignment]
    page.aria_snapshot_text = '- combobox "语言"'
    browser.page = page
    browser.browser = _FakeBrowser(page)  # type: ignore[assignment]
    await browser._extract_interactive_elements()

    role_calls_before_select = list(page.role_calls)
    result = await browser.select_option(index=0, option=2)

    assert result.success is True
    assert selects == [2]
    # Negative locks: native path must NOT take the custom-combobox fallback.
    assert locator_clicks == [], "native select must not click the locator"
    new_role_calls = page.role_calls[len(role_calls_before_select):]
    assert not any(role == "option" for role, _name, _exact in new_role_calls), (
        "native select must not request `page.get_by_role('option')`"
    )


async def test_select_option_visible_diff_picks_uniquely_new_listbox() -> None:
    """Codex audit round 5 (happy path): when the combobox has no
    `aria-controls`, production must compare DOM-ordered visible indices of
    `[role="listbox"]` / `[role="menu"]` BEFORE vs AFTER the click. If exactly
    one index becomes newly visible, scope to `page.locator(...).nth(idx)`.

    Critical regression lock: `.first` (= `.nth(0)`) would silently land on an
    unrelated already-visible listbox earlier in the DOM, returning success
    with the wrong option clicked. Production must use the diffed index.
    """
    browser = PlaywrightBrowser(cdp_url="ws://example")
    page = _FakePage()
    nth_calls_on_query: list[int] = []
    option_nth_calls: list[int] = []
    option_clicks: list[bool] = []
    combobox_clicks: list[bool] = []

    class _OptionLocator(_FakeLocator):
        def nth(self, i: int) -> "_OptionLocator":
            option_nth_calls.append(i)
            return self

        async def click(self, *, timeout: int = 5000) -> None:
            option_clicks.append(True)

    class _ScopedListbox:
        def get_by_role(self, role: str, *, name: object = None, exact: bool = False) -> _FakeLocator:
            return _OptionLocator('- option "..."')

    class _ListboxQuery:
        """Returned by `page.locator('[role="listbox"], [role="menu"]')`."""

        def nth(self, idx: int) -> _ScopedListbox:
            nth_calls_on_query.append(idx)
            return _ScopedListbox()

    class _NoAriaComboboxLocator(_FakeLocator):
        async def evaluate(self, script: str) -> object:
            return "BUTTON"

        async def click(self, *, timeout: int = 5000) -> None:
            combobox_clicks.append(True)

        async def get_attribute(self, name: str) -> "str | None":
            return None

    combobox = _NoAriaComboboxLocator('- combobox "Custom"')

    # Stage two listbox-script evaluate responses: pre-tag returns None
    # (production tags currently-visible elements, doesn't read indices);
    # post-script returns the CURRENT DOM index of the newly-visible popup.
    # Codex regression: in this test the new popup is at current idx=2 and
    # post-script returns [2] directly — the marker-based approach reads the
    # newly-visible index from the post-DOM, NOT a set-difference of indices.
    evaluate_responses: list[object] = [None, [2]]

    async def _evaluate_compat(script: str, *args: object) -> object:
        if "querySelectorAll" in script and "listbox" in script and "menu" in script:
            return evaluate_responses.pop(0)
        if script == INJECT_CONSOLE_LOGS_FUNC:
            return True
        if "window.console.logs || []" in script:
            return list(page.logs)
        if "MutationObserver" in script or "__br1" in script:
            return None
        return None

    def _fake_get_by_role(role: str, *, name: object = None, exact: bool = False) -> _FakeLocator:
        page.role_calls.append((role, name, exact))
        return combobox

    def _patched_locator(selector: str) -> object:
        page.locator_calls.append(selector)
        if "listbox" in selector and "menu" in selector:
            return _ListboxQuery()
        return _FakeLocator(page.aria_snapshot_text, raises=page.aria_snapshot_raises)

    page.get_by_role = _fake_get_by_role  # type: ignore[assignment]
    page.locator = _patched_locator  # type: ignore[assignment]
    page.evaluate = _evaluate_compat  # type: ignore[assignment]
    page.aria_snapshot_text = '- combobox "Custom"'
    browser.page = page
    browser.browser = _FakeBrowser(page)  # type: ignore[assignment]
    await browser._extract_interactive_elements()

    result = await browser.select_option(index=0, option=1)

    assert result.success is True, f"visible-diff happy path failed: {result.message}"
    # Combobox was clicked once between the two snapshots.
    assert combobox_clicks == [True]
    # Production scoped to nth(2) — the index returned by the post-script.
    # Hard regression lock against `.first` / `.nth(0)`.
    assert nth_calls_on_query == [2], (
        f"expected production to scope via nth(2) (the only newly-visible "
        f"listbox), got {nth_calls_on_query}"
    )
    assert option_nth_calls == [1]
    assert option_clicks == [True]


async def test_select_option_visible_diff_uses_marker_identity_not_index_set() -> None:
    """Codex audit round 5 BLOCK regression: when the click inserts a new
    popup BEFORE an existing visible popup in the DOM, an index-based
    set-difference would conclude the appended slot is the "new" element.
    With marker-based identity tracking, the post-script returns the index
    of the element that genuinely lacks the baseline marker — even after a
    DOM-shift.

    Test scenario:
    - Pre-click: visible listboxes at DOM positions [0, 1].
    - Click inserts a new popup at DOM position 1, shifting old 1 → 2.
    - Post-click DOM: [old_at_0 (marked), new_at_1 (no marker), old_at_2 (marked)].
    - Production's marker-aware post-script returns [1] — the genuinely-new index.
    - A naive `set(after_indices) - set(before_indices)` would have returned
      `{2}` (the appended slot), routing the click to the OLD popup.

    The fake encodes the marker-aware production behavior: post-script
    returns [1] regardless of how the indices shifted, because identity
    travelled with the marker.
    """
    browser = PlaywrightBrowser(cdp_url="ws://example")
    page = _FakePage()
    nth_calls_on_query: list[int] = []
    option_nth_calls: list[int] = []
    option_clicks: list[bool] = []

    class _OptionLocator(_FakeLocator):
        def nth(self, i: int) -> "_OptionLocator":
            option_nth_calls.append(i)
            return self

        async def click(self, *, timeout: int = 5000) -> None:
            option_clicks.append(True)

    class _ScopedListbox:
        def get_by_role(self, role: str, *, name: object = None, exact: bool = False) -> _FakeLocator:
            return _OptionLocator('- option "..."')

    class _ListboxQuery:
        def nth(self, idx: int) -> _ScopedListbox:
            nth_calls_on_query.append(idx)
            return _ScopedListbox()

    class _NoAriaComboboxLocator(_FakeLocator):
        async def evaluate(self, script: str) -> object:
            return "BUTTON"

        async def click(self, *, timeout: int = 5000) -> None:
            pass

        async def get_attribute(self, name: str) -> "str | None":
            return None

    combobox = _NoAriaComboboxLocator('- combobox "Inserted"')
    # Pre-tag returns None. Post-script returns [1] — the new popup's
    # current DOM index, which differs from where it'd be in a naive
    # set-difference (which would have said {2} = appended slot).
    evaluate_responses: list[object] = [None, [1]]

    async def _evaluate_compat(script: str, *args: object) -> object:
        if "querySelectorAll" in script and "listbox" in script and "menu" in script:
            return evaluate_responses.pop(0)
        return None

    def _fake_get_by_role(role: str, *, name: object = None, exact: bool = False) -> _FakeLocator:
        page.role_calls.append((role, name, exact))
        return combobox

    def _patched_locator(selector: str) -> object:
        page.locator_calls.append(selector)
        if "listbox" in selector and "menu" in selector:
            return _ListboxQuery()
        return _FakeLocator(page.aria_snapshot_text, raises=page.aria_snapshot_raises)

    page.get_by_role = _fake_get_by_role  # type: ignore[assignment]
    page.locator = _patched_locator  # type: ignore[assignment]
    page.evaluate = _evaluate_compat  # type: ignore[assignment]
    page.aria_snapshot_text = '- combobox "Inserted"'
    browser.page = page
    browser.browser = _FakeBrowser(page)  # type: ignore[assignment]
    await browser._extract_interactive_elements()

    result = await browser.select_option(index=0, option=0)

    assert result.success is True, f"DOM-shift case failed: {result.message}"
    # The post-script's [1] return value is what production uses for nth.
    # If production had used set-difference of pre-snapshot vs post-snapshot
    # indices, it would have routed to a different (wrong) index.
    assert nth_calls_on_query == [1], (
        f"expected nth(1) (marker-identified new popup), got {nth_calls_on_query}"
    )


async def test_select_option_visible_diff_fails_on_zero_new_popups() -> None:
    """Codex audit round 5 (ambiguous: nothing opened): if the click does
    not reveal a new listbox/menu, production must FAIL — silently picking
    `.nth(0)` of all-visible would land on an unrelated already-open popup.
    """
    browser = PlaywrightBrowser(cdp_url="ws://example")
    page = _FakePage()

    class _NoAriaComboboxLocator(_FakeLocator):
        async def evaluate(self, script: str) -> object:
            return "BUTTON"

        async def click(self, *, timeout: int = 5000) -> None:
            pass

        async def get_attribute(self, name: str) -> "str | None":
            return None

    combobox = _NoAriaComboboxLocator('- combobox "Stuck"')
    # Pre-tag returns None; post-script returns [] (no element lacks the
    # marker, i.e. nothing new became visible after the click).
    evaluate_responses: list[object] = [None, []]

    async def _evaluate_compat(script: str, *args: object) -> object:
        if "querySelectorAll" in script and "listbox" in script and "menu" in script:
            return evaluate_responses.pop(0)
        return None

    def _fake_get_by_role(role: str, *, name: object = None, exact: bool = False) -> _FakeLocator:
        page.role_calls.append((role, name, exact))
        return combobox

    page.get_by_role = _fake_get_by_role  # type: ignore[assignment]
    page.evaluate = _evaluate_compat  # type: ignore[assignment]
    page.aria_snapshot_text = '- combobox "Stuck"'
    browser.page = page
    browser.browser = _FakeBrowser(page)  # type: ignore[assignment]
    await browser._extract_interactive_elements()

    result = await browser.select_option(index=0, option=0)

    assert result.success is False, "must fail when no new listbox/menu opened"
    assert "唯一新打开的弹窗" in (result.message or ""), (
        f"expected explicit ambiguity message, got {result.message!r}"
    )


async def test_select_option_visible_diff_fails_on_multiple_new_popups() -> None:
    """Codex audit round 5 (ambiguous: many opened): if the click reveals
    more than one popup (e.g. a tooltip + listbox both becoming visible),
    production must FAIL rather than guess via `.nth(0)`.
    """
    browser = PlaywrightBrowser(cdp_url="ws://example")
    page = _FakePage()

    class _NoAriaComboboxLocator(_FakeLocator):
        async def evaluate(self, script: str) -> object:
            return "BUTTON"

        async def click(self, *, timeout: int = 5000) -> None:
            pass

        async def get_attribute(self, name: str) -> "str | None":
            return None

    combobox = _NoAriaComboboxLocator('- combobox "Ambiguous"')
    # Pre-tag returns None; post-script returns [3, 5] — TWO marker-less
    # elements newly visible. Production must fail rather than guess.
    evaluate_responses: list[object] = [None, [3, 5]]

    async def _evaluate_compat(script: str, *args: object) -> object:
        if "querySelectorAll" in script and "listbox" in script and "menu" in script:
            return evaluate_responses.pop(0)
        return None

    def _fake_get_by_role(role: str, *, name: object = None, exact: bool = False) -> _FakeLocator:
        page.role_calls.append((role, name, exact))
        return combobox

    page.get_by_role = _fake_get_by_role  # type: ignore[assignment]
    page.evaluate = _evaluate_compat  # type: ignore[assignment]
    page.aria_snapshot_text = '- combobox "Ambiguous"'
    browser.page = page
    browser.browser = _FakeBrowser(page)  # type: ignore[assignment]
    await browser._extract_interactive_elements()

    result = await browser.select_option(index=0, option=0)

    assert result.success is False, "must fail when multiple popups opened"
    assert "唯一新打开的弹窗" in (result.message or ""), (
        f"expected explicit ambiguity message, got {result.message!r}"
    )


async def test_select_option_custom_combobox_uses_aria_controls_scope() -> None:
    """Codex audit follow-up: page-level `get_by_role('option')` is unscoped.
    If the page has unrelated `<option>` elements (e.g. a different native
    `<select>` ahead of the target combobox), `.nth(option)` resolves to the
    wrong element. Production must read `aria-controls` from the combobox and
    scope option lookup to that listbox via `page.locator('#<id>')`.

    Lock: assert `page.locator('#lang-list')` was called AND the option click
    landed on the scoped locator's option (not the page-wide one).
    """
    browser = PlaywrightBrowser(cdp_url="ws://example")
    page = _FakePage()
    combobox_clicks: list[bool] = []
    option_nth_calls: list[int] = []
    option_clicks: list[bool] = []
    scope_get_by_role_calls: list[str] = []

    class _ScopedOptionLocator(_FakeLocator):
        def nth(self, i: int) -> "_ScopedOptionLocator":
            option_nth_calls.append(i)
            return self

        async def click(self, *, timeout: int = 5000) -> None:
            option_clicks.append(True)

    scoped_options = _ScopedOptionLocator('- option "..."')

    class _ScopedListbox:
        """Returned by `page.locator('#<id>')` — only exposes `get_by_role`."""

        def get_by_role(self, role: str, *, name: object = None, exact: bool = False) -> _FakeLocator:
            scope_get_by_role_calls.append(role)
            return scoped_options

    class _CustomComboboxLocator(_FakeLocator):
        async def evaluate(self, script: str) -> object:
            return "BUTTON"

        async def click(self, *, timeout: int = 5000) -> None:
            combobox_clicks.append(True)

        async def get_attribute(self, name: str) -> "str | None":
            return "lang-list" if name == "aria-controls" else None

    combobox = _CustomComboboxLocator('- combobox "Language"')

    def _fake_get_by_role(role: str, *, name: object = None, exact: bool = False) -> _FakeLocator:
        page.role_calls.append((role, name, exact))
        return combobox

    page.get_by_role = _fake_get_by_role  # type: ignore[assignment]

    def _patched_locator(selector: str) -> object:
        page.locator_calls.append(selector)
        # Production uses CSS attribute-selector form `css=[id="..."]` (NOT
        # `#<id>`) to handle framework-generated ids with special chars.
        if selector.startswith("css=[id=") or selector.startswith("[id="):
            return _ScopedListbox()
        return _FakeLocator(page.aria_snapshot_text, raises=page.aria_snapshot_raises)

    page.locator = _patched_locator  # type: ignore[assignment]
    page.aria_snapshot_text = '- combobox "Language"'
    browser.page = page
    browser.browser = _FakeBrowser(page)  # type: ignore[assignment]
    await browser._extract_interactive_elements()

    result = await browser.select_option(index=0, option=2)

    assert result.success is True, f"scoped fallback failed: {result.message}"
    # Combobox was clicked once to expand its listbox.
    assert combobox_clicks == [True]
    # `aria-controls=lang-list` → CSS attribute-selector with quoted value was
    # used to scope the option search. Naive `#<id>` form was abandoned in
    # favour of `[id="..."]` to handle framework-generated ids with colons.
    assert 'css=[id="lang-list"]' in page.locator_calls
    # The scoped listbox's `get_by_role('option').nth(2).click()` ran.
    assert scope_get_by_role_calls == ["option"]
    assert option_nth_calls == [2]
    assert option_clicks == [True]


async def test_select_option_aria_controls_id_with_special_chars() -> None:
    """Codex audit follow-up #1: framework-generated ids like Radix's
    `radix-:r1:` contain CSS-special characters (`:`). `page.locator('#<id>')`
    raises 'Unexpected token "" while parsing css selector "#radix-:r1:"'.
    Production must use the CSS attribute-selector form `[id="..."]` with
    properly-escaped string literal — `json.dumps` produces this.
    """
    browser = PlaywrightBrowser(cdp_url="ws://example")
    page = _FakePage()
    locator_selectors_used: list[str] = []

    class _ScopedListbox:
        def get_by_role(self, role: str, *, name: object = None, exact: bool = False) -> _FakeLocator:
            opt = _FakeLocator('- option "..."')
            return opt

    class _RadixComboboxLocator(_FakeLocator):
        async def evaluate(self, script: str) -> object:
            return "BUTTON"

        async def click(self, *, timeout: int = 5000) -> None:
            pass

        async def get_attribute(self, name: str) -> "str | None":
            # Radix-style id with colons.
            return "radix-:r1:" if name == "aria-controls" else None

    combobox = _RadixComboboxLocator('- combobox "Choice"')

    def _fake_get_by_role(role: str, *, name: object = None, exact: bool = False) -> _FakeLocator:
        page.role_calls.append((role, name, exact))
        return combobox

    original_locator_method = page.locator

    def _patched_locator(selector: str) -> object:
        locator_selectors_used.append(selector)
        page.locator_calls.append(selector)
        # Return a scoped listbox for any non-body selector so production can
        # chain `.get_by_role('option').nth(...).click()`.
        if selector != "body":
            return _ScopedListbox()
        return _FakeLocator(page.aria_snapshot_text, raises=page.aria_snapshot_raises)

    page.get_by_role = _fake_get_by_role  # type: ignore[assignment]
    page.locator = _patched_locator  # type: ignore[assignment]
    page.aria_snapshot_text = '- combobox "Choice"'
    browser.page = page
    browser.browser = _FakeBrowser(page)  # type: ignore[assignment]
    await browser._extract_interactive_elements()

    await browser.select_option(index=0, option=0)

    # Production must use attribute-selector form (NOT `#radix-:r1:` which
    # would crash Playwright's CSS engine).
    assert 'css=[id="radix-:r1:"]' in locator_selectors_used, (
        f"expected attribute-selector form for special-char id, got {locator_selectors_used}"
    )
    # Negative lock: the broken `#<id>` form must NOT appear.
    assert "#radix-:r1:" not in locator_selectors_used


async def test_input_falls_back_to_click_and_type_for_aria_textbox() -> None:
    """Codex audit #2: ARIA `<div role="textbox" tabindex="0">` is in the
    interactive-roles set but Playwright `fill()` raises 'Element is not an
    <input>, <textarea> or [contenteditable] element'. input() must fall back
    to: locator.click() → page.keyboard.type(text), preserving press_enter."""
    browser = PlaywrightBrowser(cdp_url="ws://example")
    page = _FakePage()
    typed: list[str] = []
    enters: list[bool] = []
    locator_clicks: list[bool] = []

    class _AriaTextboxLocator(_FakeLocator):
        async def fill(self, text: str) -> None:
            raise RuntimeError(
                "Element is not an <input>, <textarea> or [contenteditable] element"
            )

        async def click(self, *, timeout: int = 5000) -> None:
            locator_clicks.append(True)

    class _Keyboard:
        async def type(self, text: str) -> None:
            typed.append(text)

        async def press(self, key: str) -> None:
            enters.append(key == "Enter")

    def _fake_get_by_role(role: str, *, name: object = None, exact: bool = False) -> _FakeLocator:
        page.role_calls.append((role, name, exact))
        return _AriaTextboxLocator('- textbox "Search"')

    page.get_by_role = _fake_get_by_role  # type: ignore[assignment]
    page.keyboard = _Keyboard()  # type: ignore[attr-defined]
    page.aria_snapshot_text = '- textbox "Search"'
    browser.page = page
    browser.browser = _FakeBrowser(page)  # type: ignore[assignment]
    await browser._extract_interactive_elements()

    result = await browser.input(text="query", press_enter=True, index=0)

    assert result.success is True, f"fallback path failed: {result.message}"
    # Fallback path: click for focus, then keyboard.type, then keyboard.press("Enter").
    assert locator_clicks == [True]
    assert typed == ["query"]
    assert enters == [True]
