# api/app/infrastructure/external/browser/grounded_click.py
"""GroundedClick 引擎 — Phase 1 narrow 切（descriptor + L1/L2/L3 fresh resolve + postcondition）。

当前文件只放 skeleton + 数据类型；fresh_resolve / postcondition 在 Task 4-7 / 9-10 增补。
Vision Level 4 与 Action Recipe Memory 推迟到 Phase 1 eval 数据回来后另做。
"""

import asyncio
import math
import re
import time
from dataclasses import dataclass, replace
from typing import Any, Literal, Optional

from app.infrastructure.external.browser.snapshot_bundle import ElementDescriptor


PostconditionKind = Literal[
    "url_changed",
    "dom_mutated",
    "state_changed",
    "modal_opened",
    "no_observable_change",
    "explicit_failure",
]


@dataclass
class ClickTrace:
    """单次交互的结构化 trace，喂 eval harness 与未来 Phase 2 Action Recipe 输入。"""

    intent: str
    levels_tried: list[int]
    success_level: Optional[int]
    postcondition: PostconditionKind
    duration_ms: float
    error: Optional[str]
    snapshot_id: Optional[str]


class GroundedClickEngine:
    """服务器侧 ElementDescriptor 缓存 + fresh-resolve 引擎。"""

    def __init__(self) -> None:
        self._descriptors: list[ElementDescriptor] = []
        # 最近一次 resolve_and_click 的 trace（含 postcondition 覆盖）；eval harness 读取做断言。
        self._last_trace: Optional[ClickTrace] = None

    @property
    def descriptors(self) -> list[ElementDescriptor]:
        return list(self._descriptors)

    @property
    def last_trace(self) -> Optional[ClickTrace]:
        """最近一次交互的 trace；None 表示从未交互或上次失败前 clear。"""
        return self._last_trace

    def record_trace(self, trace: ClickTrace) -> None:
        """PlaywrightBrowser 在 postcondition 覆盖后回写最新 trace（不要从外部直写 `_last_trace`）。

        Defensive-copy `levels_tried` 以防 caller 在 record 后继续 append（trace 在
        eval harness / observability 路径里可能被多个消费者读取，aliasing 会导致
        断言飘）。
        """
        self._last_trace = replace(trace, levels_tried=list(trace.levels_tried))

    def set_descriptors(self, descriptors: list[ElementDescriptor]) -> None:
        """整体替换 descriptor 缓存。每次 view_page/navigate 后调用一次。"""
        self._descriptors = list(descriptors)

    def get_descriptor(self, index: int) -> ElementDescriptor:
        """按 LLM 看到的 index 取 descriptor；越界抛 IndexError，由 click() 转 ToolResult.

        显式拒绝负数 index——Python list 默认把 `-1` 当倒数第一个元素，但 LLM 输出
        负数通常是工具误用，应当报错而不是静默选错元素。
        """
        if index < 0 or index >= len(self._descriptors):
            raise IndexError(
                f"descriptor index {index} out of range [0, {len(self._descriptors)})"
            )
        return self._descriptors[index]

    async def fresh_resolve_l1(self, page: Any, descriptor: ElementDescriptor) -> Optional[Any]:
        """Level 1: Playwright get_by_role 精确匹配。

        成功条件：locator.count() == 1 且 locator.first.is_visible() == True。
        失败返 None，由 caller 进入 L2。

        `exact=True` is required: Playwright defaults `name=` to substring match,
        which would silently match e.g. "提交订单" when descriptor.name="提交" and
        let L1 succeed on the wrong element. L2 is the fuzzy/regex layer.

        `page.get_by_role(...)` is wrapped inside the try block too: while it's
        synchronous and usually only constructs a locator, a closed page or
        odd Playwright state can raise immediately, and the coordinator must
        still fall through to L2 in that case.
        """
        try:
            locator = page.get_by_role(descriptor.role, name=descriptor.name, exact=True)
            count = await locator.count()
            if count != 1:
                return None
            if not await locator.first.is_visible():
                return None
            return locator
        except Exception:
            return None

    async def fresh_resolve_l2(self, page: Any, descriptor: ElementDescriptor) -> Optional[Any]:
        r"""Level 2: fuzzy match — regex name + nth disambiguation.

        Uses `re.escape(name)` so descriptor names with regex metachars don't break;
        `re.IGNORECASE` to tolerate accessible-name casing drift between snapshots.
        Empty/whitespace descriptor.name falls back to `re.compile(r"^\s*$")` so the
        match stays consistent with T2's per-`(role, name)` nth bucket — using `.*`
        would mix named elements into the empty-name count and select the wrong nth.

        Wraps everything (including `re.compile` and the synchronous `page.get_by_role`)
        in try/except so the coordinator falls through to L3 on any error.
        """
        try:
            # `.strip()` catches whitespace-only accessible names (e.g. `aria-label="  "`)
            # that would otherwise compile to a useless three-space-literal pattern.
            #
            # Empty/whitespace fallback uses `^\s*$` (NOT `.*`): T2's parser assigns
            # `nth` per `(role, name)` bucket, so an empty-name descriptor's `nth`
            # only counts other empty-name elements. A `.*` pattern would match
            # named elements too, inflating the count and selecting the wrong nth.
            pattern = (
                re.compile(re.escape(descriptor.name), re.IGNORECASE)
                if descriptor.name.strip()
                else re.compile(r"^\s*$")
            )
            locator = page.get_by_role(descriptor.role, name=pattern)
            count = await locator.count()
            if count <= descriptor.nth:
                return None
            target = locator.nth(descriptor.nth)
            if not await target.is_visible():
                return None
            return target
        except Exception:
            return None

    async def fresh_resolve_l3_click(self, page: Any, descriptor: ElementDescriptor) -> bool:
        """Level 3: bbox center-point mouse click fallback.

        L3 has a different contract from L1/L2: it does not return a locator;
        the bbox drives `page.mouse.click(x, y)` directly, so the caller must
        immediately enter postcondition verification. Returns True on a click
        attempt; False when the bbox is degenerate (placeholder `(0, 0, 0, 0)`,
        non-positive width/height, or non-finite NaN/inf), or when
        `page.mouse.click` raises (closed page, etc.).

        T2's parser produces zero-bbox placeholders that T8 fills via
        `_resolve_bboxes_for_descriptors`. Any descriptor still carrying a
        zero/NaN bbox at L3 time means upstream resolution failed.
        """
        x, y, w, h = descriptor.bbox
        if (
            w <= 0.0
            or h <= 0.0
            or not math.isfinite(x)
            or not math.isfinite(y)
            or not math.isfinite(w)
            or not math.isfinite(h)
        ):
            return False
        center_x = x + w / 2.0
        center_y = y + h / 2.0
        try:
            await page.mouse.click(center_x, center_y)
            return True
        except Exception:
            return False

    async def resolve_and_click(
        self,
        page: Any,
        *,
        intent: str,
        descriptor: ElementDescriptor,
    ) -> ClickTrace:
        """Wire L1 → L2 → L3 fallback chain into a single trace.

        Phase 1 narrow scope: no Vision Level 4. `postcondition` field is the
        placeholder `"dom_mutated"` for any successful resolve — PlaywrightBrowser's
        `verify_postcondition` (T9/T10) overwrites it with the real observation.
        On all-fail the postcondition is set to `"explicit_failure"` here.
        """
        start = time.monotonic()
        levels_tried: list[int] = []
        error: Optional[str] = None

        # L1
        levels_tried.append(1)
        locator = await self.fresh_resolve_l1(page, descriptor)
        if locator is not None:
            try:
                await locator.click(timeout=5000)
                return ClickTrace(
                    intent=intent,
                    levels_tried=levels_tried,
                    success_level=1,
                    postcondition="dom_mutated",
                    duration_ms=(time.monotonic() - start) * 1000.0,
                    error=None,
                    snapshot_id=None,
                )
            except Exception as exc:
                error = f"L1 click error: {exc}"

        # L2
        levels_tried.append(2)
        locator = await self.fresh_resolve_l2(page, descriptor)
        if locator is not None:
            try:
                await locator.click(timeout=5000)
                return ClickTrace(
                    intent=intent,
                    levels_tried=levels_tried,
                    success_level=2,
                    postcondition="dom_mutated",
                    duration_ms=(time.monotonic() - start) * 1000.0,
                    error=None,
                    snapshot_id=None,
                )
            except Exception as exc:
                error = f"L2 click error: {exc}"

        # L3
        levels_tried.append(3)
        ok = await self.fresh_resolve_l3_click(page, descriptor)
        if ok:
            return ClickTrace(
                intent=intent,
                levels_tried=levels_tried,
                success_level=3,
                postcondition="dom_mutated",
                duration_ms=(time.monotonic() - start) * 1000.0,
                error=None,
                snapshot_id=None,
            )

        # All-fail explicit_failure trace
        return ClickTrace(
            intent=intent,
            levels_tried=levels_tried,
            success_level=None,
            postcondition="explicit_failure",
            duration_ms=(time.monotonic() - start) * 1000.0,
            error=error
            or f"使用索引查找该元素无效, 定位失败: role={descriptor.role}, name={descriptor.name}",
            snapshot_id=None,
        )

    async def install_mutation_observer(self, page: Any) -> None:
        """Inject MutationObserver + state baseline + prepare modal detection.

        Sets:
        - `window.__br1MutationDetected` — true on any DOM mutation
        - `window.__br1StateChanged` — true if aria-checked/expanded/selected/disabled
          on any element changed from the install-time baseline

        Errors are intentionally NOT caught: setup failure must surface.

        Caveat for callers: the flag captures ANY DOM activity in the
        install→verify window, not only click-induced mutations. Background
        XHRs, animations, or third-party iframe injections will register as
        `dom_mutated`. Keep the install→click→verify sequence tight (no extra
        `await asyncio.sleep` between install and the click).
        """
        await page.evaluate(
            """() => {
                window.__br1MutationDetected = false;
                window.__br1StateChanged = false;
                window.__br1StateBaseline = new Map();
                // Modal baseline: snapshot each existing dialog's VISIBILITY (not
                // just presence) so verify can detect both "new dialog appeared"
                // and "pre-rendered hidden dialog became visible". A common modal
                // pattern is a `<div role="dialog" hidden>` whose `hidden`
                // attribute is removed on click — without visibility tracking,
                // verify reads "element still in DOM, in baseline" → false negative.
                window.__br1ModalBaseline = new Map();
                // Robust visibility predicate covering: hidden attribute,
                // display:none (rects empty), visibility:hidden/collapse, AND
                // position:fixed (which is invisible to layout-parent tests).
                const isVisible = (el) => {
                    if (el.hidden) return false;
                    if (el.getClientRects().length === 0) return false;
                    const style = window.getComputedStyle(el);
                    return style.visibility !== 'hidden' && style.visibility !== 'collapse';
                };
                document.querySelectorAll('[role="dialog"], [aria-modal="true"]').forEach((el) => {
                    window.__br1ModalBaseline.set(el, isVisible(el));
                });
                document.querySelectorAll('[aria-checked],[aria-expanded],[aria-selected],[disabled]').forEach((el) => {
                    window.__br1StateBaseline.set(el, {
                        checked: el.getAttribute('aria-checked'),
                        expanded: el.getAttribute('aria-expanded'),
                        selected: el.getAttribute('aria-selected'),
                        disabled: el.hasAttribute('disabled'),
                    });
                });
                // Observer is configured WITHOUT `attributeFilter` so visibility-
                // affecting mutations (`hidden`, `style`, `class`) also fire
                // `__br1MutationDetected`. The state-change branch still filters
                // to known semantic attrs inside the callback.
                const observer = new MutationObserver((mutations) => {
                    window.__br1MutationDetected = true;
                    mutations.forEach(m => {
                        if (m.type === 'attributes' && ['aria-checked','aria-expanded','aria-selected','disabled'].includes(m.attributeName)) {
                            const baseline = window.__br1StateBaseline.get(m.target);
                            if (!baseline) return;
                            const current = {
                                checked: m.target.getAttribute('aria-checked'),
                                expanded: m.target.getAttribute('aria-expanded'),
                                selected: m.target.getAttribute('aria-selected'),
                                disabled: m.target.hasAttribute('disabled'),
                            };
                            if (JSON.stringify(current) !== JSON.stringify(baseline)) {
                                window.__br1StateChanged = true;
                            }
                        }
                    });
                });
                observer.observe(document.body, {
                    childList: true, subtree: true, attributes: true,
                });
            }"""
        )

    async def verify_postcondition(
        self,
        page: Any,
        *,
        before_url: str,
        wait_seconds: float = 2.0,
    ) -> PostconditionKind:
        """Wait `wait_seconds`, then evaluate priority:
        url_changed > state_changed > modal_opened > dom_mutated > no_observable_change.

        Caller must invoke `install_mutation_observer` BEFORE the click and this
        method AFTER. `wait_seconds=0` skips the sleep for tests.
        """
        if wait_seconds > 0:
            await asyncio.sleep(wait_seconds)

        try:
            current_url = page.url if isinstance(page.url, str) else ""
        except Exception:
            current_url = ""
        if current_url and current_url != before_url:
            return "url_changed"

        try:
            state_changed = await page.evaluate("() => Boolean(window.__br1StateChanged)")
        except Exception:
            state_changed = False
        if state_changed:
            return "state_changed"

        try:
            modal = await page.evaluate(
                """() => { /*br1ModalCheck*/
                    const baseline = window.__br1ModalBaseline;
                    if (!baseline) return false;
                    // Same predicate as install — covers `position: fixed`
                    // modals via `getClientRects()` (layout-aware). Computed
                    // style covers `visibility: hidden`, which has rects but
                    // is invisible to the user.
                    const isVisible = (el) => {
                        if (el.hidden) return false;
                        if (el.getClientRects().length === 0) return false;
                        const style = window.getComputedStyle(el);
                        return style.visibility !== 'hidden' && style.visibility !== 'collapse';
                    };
                    const current = document.querySelectorAll('[role=\"dialog\"], [aria-modal=\"true\"]');
                    for (const el of current) {
                        const wasVisible = baseline.get(el);
                        const nowVisible = isVisible(el);
                        if (wasVisible === undefined) {
                            // New dialog inserted since install → fires only if visible.
                            if (nowVisible) return true;
                        } else if (!wasVisible && nowVisible) {
                            // Pre-rendered hidden dialog became visible (e.g.
                            // `hidden` attr removed, `style.display` toggled).
                            return true;
                        }
                    }
                    return false;
                }"""
            )
        except Exception:
            modal = False
        if modal:
            return "modal_opened"

        try:
            mutated = await page.evaluate("() => Boolean(window.__br1MutationDetected)")
        except Exception:
            mutated = False
        if mutated:
            return "dom_mutated"

        return "no_observable_change"


async def resolve_bboxes_for_descriptors(
    page: Any,
    descriptors: list[ElementDescriptor],
) -> list[ElementDescriptor]:
    """Async second pass: use `page.get_by_role(role, name).[nth(n).]bounding_box()`
    to replace the (0, 0, 0, 0) placeholder bboxes that T2's parser emits.

    Returns a NEW list with each descriptor either bbox-updated (when bounding_box
    returned a dict) or carried through unchanged (when None or any error).
    Caller must call this BEFORE `engine.set_descriptors(...)` so L3 fallback
    has real coordinates to click.
    """
    out: list[ElementDescriptor] = []
    for d in descriptors:
        try:
            # `exact=True` is required: Playwright defaults `name=` to substring
            # match, so a descriptor for "提交" can pick up the bbox of a
            # neighboring "提交订单" button. T2's parser captured the descriptor
            # name verbatim from the snapshot, so the bbox lookup must demand
            # an exact match too — otherwise L3 fallback clicks the wrong pixel.
            locator = page.get_by_role(d.role, name=d.name, exact=True)
            # Always go through `.nth(...)` (even for nth=0): Playwright's
            # `Locator.bounding_box()` is strict, so calling it on a base locator
            # that resolves to multiple elements raises a strict-mode violation.
            # `nth(d.nth)` (== `.first` when d.nth == 0) avoids the strict check
            # and is the form T2's per-(role, name) nth bucket assumes.
            target = locator.nth(d.nth)
            box = await target.bounding_box()
        except Exception:
            box = None
        if box is None:
            out.append(d)
            continue
        new_bbox = (
            float(box.get("x", 0.0)),
            float(box.get("y", 0.0)),
            float(box.get("width", 0.0)),
            float(box.get("height", 0.0)),
        )
        # `replace` keeps us drift-safe if ElementDescriptor gains fields later.
        out.append(replace(d, bbox=new_bbox))
    return out
