# api/tests/app/infrastructure/external/browser/test_grounded_click.py
import re
from typing import get_args

import pytest

from app.infrastructure.external.browser.grounded_click import (
    GroundedClickEngine,
    ClickTrace,
    PostconditionKind,
)
from app.infrastructure.external.browser.snapshot_bundle import ElementDescriptor

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _desc(role: str = "button", name: str = "提交", nth: int = 0) -> ElementDescriptor:
    return ElementDescriptor(
        role=role,
        name=name,
        text=name,
        tag="button",
        placeholder="",
        bbox=(10.0, 20.0, 100.0, 40.0),
        nth=nth,
    )


def test_click_trace_is_dataclass_with_required_fields() -> None:
    trace = ClickTrace(
        intent="点击提交按钮",
        levels_tried=[1],
        success_level=1,
        postcondition="url_changed",
        duration_ms=42.0,
        error=None,
        snapshot_id=None,
    )
    assert trace.intent == "点击提交按钮"
    assert trace.levels_tried == [1]
    assert trace.success_level == 1
    assert trace.postcondition == "url_changed"


def test_postcondition_kind_enumerates_phase1_values() -> None:
    """Phase 1 postcondition 取值集冻结。"""
    expected = {
        "url_changed",
        "dom_mutated",
        "state_changed",
        "modal_opened",
        "no_observable_change",
        "explicit_failure",
    }
    assert set(get_args(PostconditionKind)) == expected


def test_engine_starts_with_empty_descriptor_cache() -> None:
    engine = GroundedClickEngine()
    assert engine.descriptors == []


def test_engine_last_trace_starts_as_none() -> None:
    engine = GroundedClickEngine()
    assert engine.last_trace is None


def test_engine_record_trace_round_trips_and_breaks_aliasing() -> None:
    """`record_trace` 写完后 `last_trace` 立即可读；levels_tried 必须 defensively-copied。"""
    engine = GroundedClickEngine()
    levels = [1, 2]
    trace = ClickTrace(
        intent="round trip",
        levels_tried=levels,
        success_level=2,
        postcondition="dom_mutated",
        duration_ms=10.0,
        error=None,
        snapshot_id=None,
    )
    engine.record_trace(trace)

    stored = engine.last_trace
    assert stored is not None
    assert stored.intent == "round trip"
    assert stored.levels_tried == [1, 2]

    # 来源 list 后续 append 不能影响已 record 的 trace（aliasing 防御）
    levels.append(3)
    assert engine.last_trace is not None
    assert engine.last_trace.levels_tried == [1, 2]


def test_engine_get_descriptor_negative_index_raises() -> None:
    """LLM 输出 -1 应当显式 IndexError，而不是静默返回最后一个 descriptor。"""
    engine = GroundedClickEngine()
    engine.set_descriptors([_desc(name="a"), _desc(name="b")])
    with pytest.raises(IndexError):
        engine.get_descriptor(-1)


def test_engine_set_descriptors_replaces_cache() -> None:
    engine = GroundedClickEngine()
    engine.set_descriptors([_desc(name="a"), _desc(name="b")])
    assert len(engine.descriptors) == 2
    assert engine.descriptors[0].name == "a"
    engine.set_descriptors([_desc(name="c")])
    assert len(engine.descriptors) == 1
    assert engine.descriptors[0].name == "c"


def test_engine_descriptors_property_returns_defensive_copy() -> None:
    """外部 mutate `engine.descriptors` 不能影响内部缓存。"""
    engine = GroundedClickEngine()
    engine.set_descriptors([_desc(name="a"), _desc(name="b")])
    returned = engine.descriptors
    returned.pop()
    assert len(engine.descriptors) == 2
    assert [d.name for d in engine.descriptors] == ["a", "b"]


def test_engine_get_descriptor_by_index_out_of_range_raises() -> None:
    engine = GroundedClickEngine()
    engine.set_descriptors([_desc()])
    with pytest.raises(IndexError):
        engine.get_descriptor(5)


def test_engine_get_descriptor_returns_cached() -> None:
    engine = GroundedClickEngine()
    target = _desc(name="目标")
    engine.set_descriptors([_desc(), target, _desc()])
    assert engine.get_descriptor(1) == target


class _FakeLocator:
    def __init__(self, count: int = 1, visible: bool = True) -> None:
        self._count = count
        self._visible = visible
        self.click_calls = 0
        self.first_accessed = False

    @property
    def first(self) -> "_FakeLocator":
        self.first_accessed = True
        return self

    async def count(self) -> int:
        return self._count

    async def is_visible(self) -> bool:
        return self._visible

    async def click(self, *, timeout: int = 5000) -> None:
        self.click_calls += 1


class _RaisingLocator(_FakeLocator):
    """Drives the L1 `except Exception` defensive branch."""

    async def count(self) -> int:
        raise RuntimeError("simulated playwright transient error")


class _FakePage:
    def __init__(self) -> None:
        # (role, name, exact) — name can be `re.Pattern` when L2 calls in.
        self.role_calls: list[tuple[str, "str | re.Pattern[str] | None", bool]] = []
        self._next_locator: _FakeLocator = _FakeLocator()
        self._locator_queue: list[_FakeLocator] | None = None

    def stage_locator(self, locator: _FakeLocator) -> None:
        """Single-call tests: every get_by_role returns the same staged locator."""
        self._next_locator = locator
        self._locator_queue = None

    def queue_locators(self, *locators: _FakeLocator) -> None:
        """Chained-call tests (T7 coordinator): each get_by_role pops the next staged
        locator. After the queue is exhausted, the last-popped locator is reused."""
        self._locator_queue = list(locators)

    def get_by_role(
        self,
        role: str,
        *,
        name: "str | re.Pattern[str] | None" = None,
        exact: bool = False,
    ) -> _FakeLocator:
        self.role_calls.append((role, name, exact))
        if self._locator_queue:
            self._next_locator = self._locator_queue.pop(0)
        return self._next_locator


async def test_resolve_l1_exact_match_succeeds() -> None:
    engine = GroundedClickEngine()
    descriptor = _desc(name="提交")
    engine.set_descriptors([descriptor])
    page = _FakePage()
    locator = _FakeLocator(count=1, visible=True)
    page.stage_locator(locator)

    resolved = await engine.fresh_resolve_l1(page, descriptor)

    assert resolved is locator
    # exact=True required: Playwright defaults to substring; L1 must demand exact match.
    assert page.role_calls == [("button", "提交", True)]
    # `.first` was actually accessed (regression guard against accidental drop).
    assert locator.first_accessed is True


async def test_resolve_l1_returns_none_when_count_not_one() -> None:
    engine = GroundedClickEngine()
    descriptor = _desc()
    page = _FakePage()
    page.stage_locator(_FakeLocator(count=3, visible=True))

    resolved = await engine.fresh_resolve_l1(page, descriptor)
    assert resolved is None


async def test_resolve_l1_returns_none_when_not_visible() -> None:
    engine = GroundedClickEngine()
    descriptor = _desc()
    page = _FakePage()
    page.stage_locator(_FakeLocator(count=1, visible=False))

    resolved = await engine.fresh_resolve_l1(page, descriptor)
    assert resolved is None


async def test_resolve_l1_returns_none_on_transient_exception() -> None:
    """L1 must swallow Playwright runtime errors (timeouts / detached frames)
    and return None so the coordinator can fall through to L2."""
    engine = GroundedClickEngine()
    page = _FakePage()
    page.stage_locator(_RaisingLocator())

    resolved = await engine.fresh_resolve_l1(page, _desc())
    assert resolved is None


async def test_resolve_l1_returns_none_when_get_by_role_raises_synchronously() -> None:
    """If `page.get_by_role` itself raises (closed page / Playwright state),
    L1 must still return None so the coordinator falls through to L2."""

    class _RaisingPage(_FakePage):
        def get_by_role(
            self,
            role: str,
            *,
            name: "str | re.Pattern[str] | None" = None,
            exact: bool = False,
        ) -> _FakeLocator:
            raise RuntimeError("simulated playwright state error before await")

    engine = GroundedClickEngine()
    resolved = await engine.fresh_resolve_l1(_RaisingPage(), _desc())
    assert resolved is None


class _FakeLocatorWithNth(_FakeLocator):
    def __init__(self, count: int = 1, visible: bool = True, nth_visible: bool = True) -> None:
        super().__init__(count=count, visible=visible)
        self.nth_calls: list[int] = []
        self._nth_visible = nth_visible

    def nth(self, index: int) -> _FakeLocator:
        self.nth_calls.append(index)
        return _FakeLocator(count=1, visible=self._nth_visible)


async def test_resolve_l2_uses_regex_name_and_nth() -> None:
    engine = GroundedClickEngine()
    descriptor = _desc(name="确认", nth=2)
    page = _FakePage()
    locator = _FakeLocatorWithNth(count=3, visible=True)
    page.stage_locator(locator)

    resolved = await engine.fresh_resolve_l2(page, descriptor)

    assert resolved is not None
    assert locator.nth_calls == [2]
    # role_calls is post-T4 3-tuple (role, name, exact)
    role, name, exact = page.role_calls[0]
    assert role == "button"
    # L2 passes a regex pattern (not a string) — exact kwarg is irrelevant for regex,
    # default False is fine.
    assert exact is False
    assert isinstance(name, re.Pattern)
    assert name.search("确认按钮") is not None


async def test_resolve_l2_returns_none_when_zero_matches() -> None:
    engine = GroundedClickEngine()
    descriptor = _desc()
    page = _FakePage()
    page.stage_locator(_FakeLocatorWithNth(count=0, visible=True))

    resolved = await engine.fresh_resolve_l2(page, descriptor)
    assert resolved is None


async def test_resolve_l2_returns_none_when_nth_out_of_range() -> None:
    engine = GroundedClickEngine()
    descriptor = _desc(nth=10)  # cached nth larger than fresh-page count
    page = _FakePage()
    page.stage_locator(_FakeLocatorWithNth(count=3, visible=True))

    resolved = await engine.fresh_resolve_l2(page, descriptor)
    assert resolved is None


async def test_resolve_l2_returns_none_when_nth_target_invisible() -> None:
    """Stale-snapshot scenario: element exists in DOM at the right nth but is hidden.
    L2 must return None so the coordinator falls through to L3."""
    engine = GroundedClickEngine()
    descriptor = _desc(nth=0)
    page = _FakePage()
    page.stage_locator(_FakeLocatorWithNth(count=1, nth_visible=False))

    resolved = await engine.fresh_resolve_l2(page, descriptor)
    assert resolved is None


async def test_resolve_l2_whitespace_only_name_falls_back_to_empty_match() -> None:
    """Whitespace-only descriptor.name must hit the `^\\s*$` fallback (NOT `.*`).

    The `.*` pattern would match named elements too, inflating the count beyond
    what T2's `(role, name)` nth bucket assumed and selecting the wrong nth.
    """
    engine = GroundedClickEngine()
    descriptor = _desc(name="   ", nth=0)
    page = _FakePage()
    page.stage_locator(_FakeLocatorWithNth(count=2, visible=True))

    resolved = await engine.fresh_resolve_l2(page, descriptor)

    assert resolved is not None
    role, name, _exact = page.role_calls[0]
    assert isinstance(name, re.Pattern)
    # The empty/whitespace pattern must only match empty/whitespace accessible names.
    assert name.pattern == r"^\s*$"
    assert name.search("") is not None
    assert name.search("   ") is not None
    assert name.search("Save") is None


class _FakeMouse:
    def __init__(self) -> None:
        self.click_calls: list[tuple[float, float]] = []

    async def click(self, x: float, y: float) -> None:
        self.click_calls.append((x, y))


class _FakePageWithMouse(_FakePage):
    def __init__(self) -> None:
        super().__init__()
        self.mouse = _FakeMouse()


async def test_resolve_l3_clicks_bbox_center() -> None:
    engine = GroundedClickEngine()
    descriptor = ElementDescriptor(
        role="button", name="x", text="x", tag="button", placeholder="",
        bbox=(100.0, 200.0, 50.0, 30.0),  # center (125, 215)
        nth=0,
    )
    page = _FakePageWithMouse()

    ok = await engine.fresh_resolve_l3_click(page, descriptor)

    assert ok is True
    assert page.mouse.click_calls == [(125.0, 215.0)]


async def test_resolve_l3_returns_false_when_bbox_zero() -> None:
    engine = GroundedClickEngine()
    descriptor = ElementDescriptor(
        role="button", name="x", text="x", tag="button", placeholder="",
        bbox=(0.0, 0.0, 0.0, 0.0), nth=0,
    )
    page = _FakePageWithMouse()

    ok = await engine.fresh_resolve_l3_click(page, descriptor)
    assert ok is False
    assert page.mouse.click_calls == []


async def test_resolve_l3_returns_false_on_non_finite_bbox() -> None:
    """NaN/inf bbox values must be rejected — `w <= 0.0` is False for NaN, so an
    explicit `math.isfinite` guard prevents `mouse.click(NaN, NaN)`."""
    engine = GroundedClickEngine()
    descriptor = ElementDescriptor(
        role="button", name="x", text="x", tag="button", placeholder="",
        bbox=(float("nan"), 0.0, 50.0, 30.0), nth=0,
    )
    page = _FakePageWithMouse()

    ok = await engine.fresh_resolve_l3_click(page, descriptor)
    assert ok is False
    assert page.mouse.click_calls == []


async def test_resolve_l3_returns_false_when_mouse_click_raises() -> None:
    """Defensive `except Exception: return False` is the production path for
    closed pages / detached frames — must be unit-covered so a future refactor
    can't silently drop it."""

    class _RaisingMouse:
        async def click(self, x: float, y: float) -> None:
            raise RuntimeError("simulated closed page")

    class _RaisingMousePage(_FakePageWithMouse):
        def __init__(self) -> None:
            super().__init__()
            self.mouse = _RaisingMouse()

    engine = GroundedClickEngine()
    descriptor = ElementDescriptor(
        role="button", name="x", text="x", tag="button", placeholder="",
        bbox=(10.0, 20.0, 50.0, 30.0), nth=0,
    )
    ok = await engine.fresh_resolve_l3_click(_RaisingMousePage(), descriptor)
    assert ok is False


async def test_resolve_and_click_l1_path_records_level_1() -> None:
    engine = GroundedClickEngine()
    descriptor = _desc(name="提交")
    engine.set_descriptors([descriptor])
    page = _FakePageWithMouse()
    locator = _FakeLocator(count=1, visible=True)
    page.stage_locator(locator)

    trace = await engine.resolve_and_click(page, intent="点击提交", descriptor=descriptor)

    assert trace.levels_tried == [1]
    assert trace.success_level == 1
    assert trace.error is None
    assert locator.click_calls == 1
    assert page.mouse.click_calls == []  # L1 用 locator.click，不走 mouse


async def test_resolve_and_click_falls_through_to_l3_when_l1_l2_miss() -> None:
    """L1 count=0 / L2 count=0 → L3 用 bbox 兜底。"""
    engine = GroundedClickEngine()
    descriptor = ElementDescriptor(
        role="button", name="x", text="x", tag="button", placeholder="",
        bbox=(50.0, 60.0, 20.0, 20.0), nth=0,
    )
    engine.set_descriptors([descriptor])
    page = _FakePageWithMouse()
    page.stage_locator(_FakeLocator(count=0, visible=True))

    trace = await engine.resolve_and_click(page, intent="点击", descriptor=descriptor)

    assert trace.levels_tried == [1, 2, 3]
    assert trace.success_level == 3
    assert page.mouse.click_calls == [(60.0, 70.0)]


async def test_resolve_and_click_all_levels_fail_records_explicit_failure() -> None:
    engine = GroundedClickEngine()
    descriptor = ElementDescriptor(
        role="button", name="x", text="x", tag="button", placeholder="",
        bbox=(0.0, 0.0, 0.0, 0.0), nth=0,
    )
    engine.set_descriptors([descriptor])
    page = _FakePageWithMouse()
    page.stage_locator(_FakeLocator(count=0, visible=False))

    trace = await engine.resolve_and_click(page, intent="点击", descriptor=descriptor)

    assert trace.levels_tried == [1, 2, 3]
    assert trace.success_level is None
    assert trace.postcondition == "explicit_failure"
    assert trace.error is not None and "定位失败" in trace.error


async def test_resolve_and_click_l1_click_raises_falls_through_to_l2() -> None:
    """L1 resolves a locator but `locator.click()` raises (transient stale frame).
    Coordinator must capture the error, fall through to L2, and on L2 success
    return a clean trace (`error=None`) — the L1 transient is intentionally
    discarded since the click ultimately succeeded.

    L2 calls `.nth(...)` on its locator, so the L2 fake must be `_FakeLocatorWithNth`.
    """

    class _ClickRaisingLocator(_FakeLocator):
        async def click(self, *, timeout: int = 5000) -> None:
            raise RuntimeError("transient click failure")

    engine = GroundedClickEngine()
    descriptor = _desc(name="提交")
    engine.set_descriptors([descriptor])
    page = _FakePageWithMouse()
    l1_locator = _ClickRaisingLocator(count=1, visible=True)
    l2_locator = _FakeLocatorWithNth(count=1, visible=True, nth_visible=True)
    page.queue_locators(l1_locator, l2_locator)

    trace = await engine.resolve_and_click(page, intent="点击提交", descriptor=descriptor)

    assert trace.levels_tried == [1, 2]
    assert trace.success_level == 2
    assert trace.error is None
    # L2 went through .nth(0) — proves the L2 path actually executed.
    assert l2_locator.nth_calls == [0]
    # Mouse path not used — L2 succeeded via locator (the .nth() target).
    assert page.mouse.click_calls == []


class _FakeLocatorWithBox(_FakeLocator):
    def __init__(self, box: "dict[str, float] | None" = None) -> None:
        super().__init__(count=1, visible=True)
        self._box = box
        self.nth_calls: list[int] = []

    def nth(self, index: int) -> "_FakeLocatorWithBox":
        self.nth_calls.append(index)
        return self

    async def bounding_box(self) -> "dict[str, float] | None":
        return self._box


class _FakePageBox:
    def __init__(self) -> None:
        # 3-tuple `(role, name, exact)` — matches `_FakePage.role_calls`. The
        # `exact` slot is required so tests can verify the bbox resolver passes
        # `exact=True` (codex audit P1 #2: substring match would let "提交"
        # pick up the bbox of a neighboring "提交订单" button).
        self.role_calls: list[tuple[str, "str | None", bool]] = []
        self._next_box: "dict[str, float] | None" = None
        self._next_locator: _FakeLocatorWithBox | None = None

    def stage_box(self, box: "dict[str, float] | None") -> None:
        self._next_box = box
        self._next_locator = None

    def stage_locator(self, locator: _FakeLocatorWithBox) -> None:
        """For strict-mode regression tests: caller controls the exact fake."""
        self._next_locator = locator

    def get_by_role(
        self,
        role: str,
        *,
        name: "str | None" = None,
        exact: bool = False,
    ) -> _FakeLocatorWithBox:
        self.role_calls.append((role, name, exact))
        if self._next_locator is not None:
            return self._next_locator
        return _FakeLocatorWithBox(box=self._next_box)


async def test_resolve_bboxes_fills_real_bbox() -> None:
    desc = ElementDescriptor(
        role="button", name="提交", text="提交", tag="button",
        placeholder="", bbox=(0.0, 0.0, 0.0, 0.0), nth=0,
    )
    page = _FakePageBox()
    page.stage_box({"x": 10.0, "y": 20.0, "width": 100.0, "height": 30.0})

    from app.infrastructure.external.browser.grounded_click import resolve_bboxes_for_descriptors
    out = await resolve_bboxes_for_descriptors(page, [desc])

    assert len(out) == 1
    assert out[0].bbox == (10.0, 20.0, 100.0, 30.0)
    assert out[0].role == "button" and out[0].name == "提交"


async def test_resolve_bboxes_keeps_zero_when_box_missing() -> None:
    desc = ElementDescriptor(
        role="button", name="x", text="x", tag="button",
        placeholder="", bbox=(0.0, 0.0, 0.0, 0.0), nth=0,
    )
    page = _FakePageBox()
    page.stage_box(None)  # element invisible / not present

    from app.infrastructure.external.browser.grounded_click import resolve_bboxes_for_descriptors
    out = await resolve_bboxes_for_descriptors(page, [desc])

    assert out[0].bbox == (0.0, 0.0, 0.0, 0.0)


async def test_resolve_bboxes_uses_nth_even_when_zero() -> None:
    """Real Playwright bounding_box() is strict: on multi-match, calling it on the
    base locator throws. We must always go through `.nth(d.nth)` (== `.first` when
    nth=0) to bypass the strict check. Regression guard for the codex finding."""

    class _StrictLocator(_FakeLocatorWithBox):
        """`bounding_box()` on the base raises strict; only after `.nth(...)` it works."""

        def __init__(self) -> None:
            super().__init__(box={"x": 5.0, "y": 6.0, "width": 7.0, "height": 8.0})
            self._nth_branched = False

        def nth(self, index: int) -> "_StrictLocator":
            self.nth_calls.append(index)
            self._nth_branched = True
            return self

        async def bounding_box(self) -> "dict[str, float] | None":
            if not self._nth_branched:
                raise RuntimeError(
                    "strict mode violation: locator resolved to multiple elements"
                )
            return self._box

    locator = _StrictLocator()
    page = _FakePageBox()
    page.stage_locator(locator)

    desc = ElementDescriptor(
        role="button", name="确认", text="确认", tag="button",
        placeholder="", bbox=(0.0, 0.0, 0.0, 0.0), nth=0,
    )

    from app.infrastructure.external.browser.grounded_click import resolve_bboxes_for_descriptors
    out = await resolve_bboxes_for_descriptors(page, [desc])

    assert locator.nth_calls == [0]
    assert out[0].bbox == (5.0, 6.0, 7.0, 8.0)


async def test_resolve_bboxes_passes_exact_true_to_get_by_role() -> None:
    """Codex audit P1 #2: Playwright `name=` defaults to substring match. Without
    `exact=True`, descriptor for `Submit` would pick up the bbox of a neighboring
    `Submit Order` button. T2's parser captured the descriptor name verbatim, so
    the bbox lookup must demand an exact match too."""
    desc = ElementDescriptor(
        role="button", name="Submit", text="Submit", tag="button",
        placeholder="", bbox=(0.0, 0.0, 0.0, 0.0), nth=0,
    )
    page = _FakePageBox()
    page.stage_box({"x": 1.0, "y": 2.0, "width": 3.0, "height": 4.0})

    from app.infrastructure.external.browser.grounded_click import resolve_bboxes_for_descriptors
    out = await resolve_bboxes_for_descriptors(page, [desc])

    # Lock the exact=True invariant: any future regression that drops the kwarg
    # would let substring matches silently corrupt L3 fallback coordinates.
    assert page.role_calls == [("button", "Submit", True)]
    assert out[0].bbox == (1.0, 2.0, 3.0, 4.0)


class _FakePagePostcondition:
    """Page fake for postcondition testing — exposes `url` property + `evaluate(script)`."""

    def __init__(
        self,
        url_after: str = "https://example.com/",
        mutation_after: bool = False,
    ) -> None:
        self._url = url_after
        self._mutation = mutation_after
        self.evaluate_calls: list[str] = []

    @property
    def url(self) -> str:
        return self._url

    async def evaluate(self, script: str, *args: object) -> object:
        self.evaluate_calls.append(script)
        if "MutationObserver" in script:
            return None
        if "__br1MutationDetected" in script:
            return self._mutation
        return None


async def test_postcondition_detects_url_change() -> None:
    engine = GroundedClickEngine()
    page = _FakePagePostcondition(url_after="https://example.com/next")

    kind = await engine.verify_postcondition(
        page,
        before_url="https://example.com/",
        wait_seconds=0.0,
    )
    assert kind == "url_changed"


async def test_postcondition_detects_dom_mutation_when_url_unchanged() -> None:
    engine = GroundedClickEngine()
    page = _FakePagePostcondition(
        url_after="https://example.com/", mutation_after=True
    )

    kind = await engine.verify_postcondition(
        page,
        before_url="https://example.com/",
        wait_seconds=0.0,
    )
    assert kind == "dom_mutated"


async def test_postcondition_returns_no_observable_change_when_neither() -> None:
    engine = GroundedClickEngine()
    page = _FakePagePostcondition(
        url_after="https://example.com/", mutation_after=False
    )

    kind = await engine.verify_postcondition(
        page,
        before_url="https://example.com/",
        wait_seconds=0.0,
    )
    assert kind == "no_observable_change"


async def test_install_mutation_observer_injects_detection_script() -> None:
    """Lock in the install-side spec: install_mutation_observer must call
    page.evaluate with a script that references both MutationObserver and the
    `__br1MutationDetected` flag. Without this, a refactor could no-op the
    install and the verify-side tests would still falsely pass."""
    engine = GroundedClickEngine()
    page = _FakePagePostcondition()

    await engine.install_mutation_observer(page)

    assert any("MutationObserver" in s for s in page.evaluate_calls)
    assert any("__br1MutationDetected" in s for s in page.evaluate_calls)


async def test_install_mutation_observer_tracks_modal_visibility_and_no_attribute_filter() -> None:
    """Codex audit P2 #3 (and follow-up): pre-rendered hidden dialogs that
    toggle `hidden` need visibility-aware modal detection. The install JS must:

    1. Record per-element visibility (`getClientRects().length > 0` +
       `visibility !== 'hidden'`) in `__br1ModalBaseline`, not just element
       identity. Note: `offsetParent !== null` is INSUFFICIENT — it returns
       false for `position: fixed` modals (a common pattern), classifying them
       as `dom_mutated` instead of `modal_opened`.
    2. Use `Map` (not `Set`) so `wasVisible` lookup returns a boolean.
    3. Configure the MutationObserver WITHOUT `attributeFilter` so `hidden` /
       `style` / `class` mutations also fire `__br1MutationDetected` (otherwise
       hidden→visible toggles register as `no_observable_change`).
    """
    engine = GroundedClickEngine()
    page = _FakePagePostcondition()

    await engine.install_mutation_observer(page)

    install_script = next(
        s for s in page.evaluate_calls if "MutationObserver" in s
    )
    # Visibility predicate: layout-aware, position:fixed-safe, visibility-aware.
    assert "el.hidden" in install_script
    assert "getClientRects" in install_script
    assert "visibility" in install_script
    # Lock the codex regression: ANY use of `offsetParent` (any comparison form,
    # any coercion) must be absent — the bare-substring ban catches all
    # variants because production code no longer mentions it in comments either.
    assert "offsetParent" not in install_script
    assert "__br1ModalBaseline = new Map()" in install_script
    # Observer config drops `attributeFilter:` (the config-key form, distinct
    # from the word in comments) so hidden/style/class mutations also fire
    # `__br1MutationDetected`.
    assert "attributeFilter:" not in install_script
    assert "attributeFilter :" not in install_script


async def test_verify_postcondition_modal_check_handles_visibility_transition() -> None:
    """The modal-check JS must distinguish three cases:
       - new dialog inserted AND visible → modal_opened
       - pre-rendered dialog hidden→visible → modal_opened
       - pre-rendered dialog still hidden / still visible → no modal_opened

    Lock the JS shape so a future refactor doesn't drop the visibility branch
    or regress to `offsetParent` (broken for `position: fixed` modals).
    """
    engine = GroundedClickEngine()
    page = _FakePagePostcondition()

    await engine.verify_postcondition(
        page, before_url="https://example.com/", wait_seconds=0.0
    )

    modal_script = next(
        s for s in page.evaluate_calls if "br1ModalCheck" in s
    )
    # Both branches required: new-element check AND visibility-transition check.
    assert "wasVisible === undefined" in modal_script
    assert "!wasVisible && nowVisible" in modal_script
    # Visibility predicate present (matches install-side baseline shape).
    assert "el.hidden" in modal_script
    assert "getClientRects" in modal_script
    assert "visibility" in modal_script
    # Codex regression lock: ANY `offsetParent` usage is banned (production
    # code must not mention it in comments either, which keeps this assertion
    # robust against rewordings of the predicate).
    assert "offsetParent" not in modal_script


class _FakePagePostcondition2(_FakePagePostcondition):
    def __init__(
        self,
        url_after: str = "https://example.com/",
        mutation_after: bool = False,
        state_changed: bool = False,
        modal_present: bool = False,
    ) -> None:
        super().__init__(url_after=url_after, mutation_after=mutation_after)
        self._state_changed = state_changed
        self._modal = modal_present

    async def evaluate(self, script: str, *args: object) -> object:
        self.evaluate_calls.append(script)
        if "MutationObserver" in script:
            return None
        if "__br1MutationDetected" in script:
            return self._mutation
        if "__br1StateChanged" in script:
            return self._state_changed
        if "br1ModalCheck" in script:
            return self._modal
        return None


async def test_postcondition_detects_state_changed() -> None:
    engine = GroundedClickEngine()
    page = _FakePagePostcondition2(state_changed=True)

    kind = await engine.verify_postcondition(
        page, before_url="https://example.com/", wait_seconds=0.0,
    )
    assert kind == "state_changed"


async def test_postcondition_detects_modal_opened() -> None:
    engine = GroundedClickEngine()
    page = _FakePagePostcondition2(modal_present=True)

    kind = await engine.verify_postcondition(
        page, before_url="https://example.com/", wait_seconds=0.0,
    )
    assert kind == "modal_opened"


async def test_postcondition_priority_url_over_state() -> None:
    engine = GroundedClickEngine()
    page = _FakePagePostcondition2(url_after="https://example.com/next", state_changed=True)

    kind = await engine.verify_postcondition(
        page, before_url="https://example.com/", wait_seconds=0.0,
    )
    assert kind == "url_changed"
