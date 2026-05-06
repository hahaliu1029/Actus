"""BR1 Phase 1 narrow eval — 5 ready cases + 15 stub cases (xfail-marked)."""

import pytest
from playwright.async_api import async_playwright

pytestmark = [pytest.mark.anyio, pytest.mark.integration, pytest.mark.browser_eval]


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _xfail_reason(case: dict) -> "str | None":
    if case.get("vision_required"):
        return "vision_required: deferred to BR1 Phase 1.5"
    if case.get("html_pending"):
        return "local HTML page not authored yet"
    if case.get("network"):
        return "network-gated case: requires live external site (GitHub) — flaky for narrow eval"
    return None


@pytest.mark.parametrize("case_id", range(1, 21))
async def test_grounded_click_case(case_id: int, cases: list[dict]) -> None:
    case = next((c for c in cases if c["id"] == case_id), None)
    if case is None:
        pytest.skip(f"case {case_id} fixture not present")

    reason = _xfail_reason(case)
    if reason:
        pytest.xfail(reason)

    from app.infrastructure.external.browser.playwright_browser import PlaywrightBrowser

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()
        await page.goto(case["url"])

        actus_browser = PlaywrightBrowser(cdp_url="")
        actus_browser.browser = browser  # type: ignore[assignment]
        actus_browser.page = page  # type: ignore[assignment]

        await actus_browser._extract_interactive_elements()
        target_idx = case["click_target_index_hint"]
        result = await actus_browser.click(index=target_idx)
        # Capture engine trace BEFORE browser.close — engine is in-process,
        # but reading any locator after close would fail. trace itself is plain data.
        trace = actus_browser._grounded.last_trace

        await browser.close()

    assert result.success is True, f"case {case_id} click failed: {result.message}"
    # Postcondition must MATCH the case's expected value. Without this assert,
    # a state_changed case would silently pass on no_observable_change.
    assert trace is not None, f"case {case_id} engine produced no trace"
    assert trace.postcondition == case["expected_postcondition"], (
        f"case {case_id} postcondition mismatch: "
        f"expected {case['expected_postcondition']!r}, got {trace.postcondition!r} "
        f"(levels_tried={trace.levels_tried}, success_level={trace.success_level})"
    )
