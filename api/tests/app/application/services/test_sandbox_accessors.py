"""SPM Task 8: ``EagerSandboxAccessor`` / ``EagerBrowserAccessor`` unit tests.

PR-1b decouples the tool/runner layer from raw sandbox handles via explicit
accessor protocols. The *Eager* implementations wrap an already-provisioned
concrete handle/browser: ``get()`` does ZERO IO / ZERO provisioning (byte-equiv
``always`` behavior), ``peek()`` is a non-provisioning probe, and the terminal
release / close paths are best-effort + idempotent.

Async runner note: this repo ships **pytest-anyio**, not pytest-asyncio. Tests
are marked via the module-level ``pytestmark = pytest.mark.anyio`` + a local
``anyio_backend`` fixture (mirroring
``test_sandbox_provision_flight.py`` / ``test_playwright_browser.py``). The
task brief's illustrative ``@pytest.mark.asyncio`` decorators are intentionally
dropped — under this repo's plugin set a bare ``asyncio`` marker would leave the
coroutine un-awaited (false-green).
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.application.services.sandbox_accessors import (
    EagerBrowserAccessor,
    EagerSandboxAccessor,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class _FakeHandle:
    """Duck-typed ``SandboxHandle`` — only the sync ``release()`` is exercised."""

    def __init__(self) -> None:
        self.release_calls = 0

    def release(self) -> None:
        self.release_calls += 1


class _BoomHandle:
    """A handle whose ``release()`` raises — proves the swallow path."""

    def __init__(self) -> None:
        self.release_calls = 0

    def release(self) -> None:
        self.release_calls += 1
        raise RuntimeError("release boom")


class _FakeBrowser:
    """Duck-typed ``Browser`` — records ``aclose()`` invocations."""

    def __init__(self) -> None:
        self.closed = 0

    async def aclose(self) -> None:
        self.closed += 1


# ── EagerSandboxAccessor ────────────────────────────────────────────────────


class TestEagerSandboxAccessor:
    async def test_get_and_peek_return_same_handle(self) -> None:
        h = _FakeHandle()
        acc = EagerSandboxAccessor(h)
        assert await acc.get() is h
        assert acc.peek() is h

    async def test_get_does_zero_io(self) -> None:
        """Eager ``get()`` is a pure return — no release / provisioning side effect,
        even across repeated calls (byte-equivalent ``always`` behavior)."""
        h = _FakeHandle()
        acc = EagerSandboxAccessor(h)
        await acc.get()
        await acc.get()
        assert h.release_calls == 0

    async def test_release_owned_releases_once_then_noop(self) -> None:
        h = _FakeHandle()
        acc = EagerSandboxAccessor(h)
        await acc.release_owned()
        assert h.release_calls == 1
        assert acc.peek() is None  # handle cleared → terminal
        await acc.release_owned()  # idempotent second call
        assert h.release_calls == 1  # not released again

    async def test_release_owned_swallows_exception_and_clears(self) -> None:
        h = _BoomHandle()
        acc = EagerSandboxAccessor(h)
        await acc.release_owned()  # release() raises → swallowed, does NOT propagate
        assert h.release_calls == 1
        assert acc.peek() is None  # still cleared despite the raise
        await acc.release_owned()  # idempotent
        assert h.release_calls == 1

    async def test_release_owned_quiet_when_handle_has_no_release(self) -> None:
        """SPM Task 17 fix #3: the AgentService no-lifecycle test fallback wraps a
        raw sandbox object with NO ``release()``. ``release_owned`` must clear the
        handle quietly (hasattr guard) — no AttributeError, no misleading warning."""
        h = SimpleNamespace(id="raw")  # no release() attribute
        acc = EagerSandboxAccessor(h)
        await acc.release_owned()  # must not raise
        assert acc.peek() is None  # cleared → terminal, idempotent
        await acc.release_owned()  # second call is a no-op


# ── EagerBrowserAccessor ────────────────────────────────────────────────────


class TestEagerBrowserAccessor:
    async def test_get_and_peek_return_same_browser(self) -> None:
        b = _FakeBrowser()
        acc = EagerBrowserAccessor(b)
        assert await acc.get() is b
        assert acc.peek() is b

    async def test_aclose_delegates_to_browser(self) -> None:
        b = _FakeBrowser()
        acc = EagerBrowserAccessor(b)
        await acc.aclose()
        assert b.closed == 1

    async def test_aclose_swallows_exception(self) -> None:
        class _Boom:
            async def aclose(self) -> None:
                raise RuntimeError("x")

        # best-effort: a raising underlying aclose() must not propagate.
        await EagerBrowserAccessor(_Boom()).aclose()


# ── PR-1b Task 9: tool-factory laziness contract ────────────────────────────


class _CountingAccessor:
    """Minimal ``SandboxAccessor`` / ``BrowserAccessor`` double that counts pulls.

    Wraps an arbitrary handle/browser target. ``get()`` is async (per the
    accessor protocol) and bumps ``gets`` on every call so a test can prove
    (a) factory *build* does ZERO pulls and (b) each tool *invocation* pulls
    exactly once (never cached at build time — that would regress to eager).
    """

    def __init__(self, target: object) -> None:
        self._target = target
        self.gets = 0

    async def get(self) -> object:
        self.gets += 1
        return self._target

    def peek(self) -> object:
        return self._target


class _RecordingShellHandle:
    """Duck-typed ``SandboxHandle`` exercised by the shell tool bodies."""

    def __init__(self) -> None:
        self.read_calls = 0

    async def read_shell_output(
        self, session_id: str = "default", console: bool = False
    ) -> SimpleNamespace:
        self.read_calls += 1
        return SimpleNamespace(success=True, data={"output": "ok"}, message="")


class _RecordingBrowser:
    """Duck-typed ``Browser`` exercised by the browser tool bodies."""

    def __init__(self) -> None:
        self.view_calls = 0

    async def view_page(self) -> SimpleNamespace:
        self.view_calls += 1
        return SimpleNamespace(success=True, data={"output": "page"}, message="")


def _tool_by_name(tools: list, name: str):
    for tool in tools:
        if tool.name == name:
            return tool
    raise AssertionError(f"tool {name!r} not found in {[t.name for t in tools]}")


class TestFactoryLaziness:
    """Task 9 invariant: factories capture the *accessor*, never a handle; the
    handle is pulled lazily inside each tool coroutine (one ``get()`` per call).
    """

    def test_make_shell_tools_does_not_call_get_at_build_time(self) -> None:
        from app.domain.services.tools.langchain_tools import _make_shell_tools

        acc = _CountingAccessor(_RecordingShellHandle())
        tools = _make_shell_tools(acc)
        assert acc.gets == 0
        assert len(tools) > 0

    def test_make_file_tools_does_not_call_get_at_build_time(self) -> None:
        from app.domain.services.tools.langchain_tools import _make_file_tools

        acc = _CountingAccessor(_RecordingShellHandle())
        tools = _make_file_tools(acc)
        assert acc.gets == 0
        assert len(tools) > 0

    def test_make_browser_tools_does_not_call_get_at_build_time(self) -> None:
        from app.domain.services.tools.langchain_tools import _make_browser_tools

        acc = _CountingAccessor(_RecordingBrowser())
        tools = _make_browser_tools(acc)
        assert acc.gets == 0
        assert len(tools) > 0

    async def test_shell_tool_invocation_pulls_handle_per_call(self) -> None:
        from app.domain.services.tools.langchain_tools import _make_shell_tools

        handle = _RecordingShellHandle()
        acc = _CountingAccessor(handle)
        shell_read = _tool_by_name(_make_shell_tools(acc), "shell_read_output")

        await shell_read.ainvoke({"session_id": "default"})
        assert acc.gets == 1
        assert handle.read_calls == 1

        await shell_read.ainvoke({"session_id": "default"})
        assert acc.gets == 2  # pulled again per call — never cached at build
        assert handle.read_calls == 2

    async def test_browser_tool_invocation_pulls_handle_per_call(self) -> None:
        from app.domain.services.tools.langchain_tools import _make_browser_tools

        browser = _RecordingBrowser()
        acc = _CountingAccessor(browser)
        view = _tool_by_name(_make_browser_tools(acc), "browser_view")

        await view.ainvoke({})
        assert acc.gets == 1
        assert browser.view_calls == 1

        await view.ainvoke({})
        assert acc.gets == 2
        assert browser.view_calls == 2

    def test_make_file_view_tools_rejects_both_lookup_and_factory(self) -> None:
        from app.domain.services.tools.langchain_tools import _make_file_view_tools

        acc = _CountingAccessor(_RecordingShellHandle())
        with pytest.raises(AssertionError):
            _make_file_view_tools(
                acc,
                file_processor_lookup=object(),
                file_processor_factory=lambda handle: object(),
            )

    def test_create_native_tools_off_assembly_skips_sandbox_family(self) -> None:
        """include_sandbox_tools=False + None accessors: no sandbox/browser family
        constructed, so None is never dereferenced (off-assembly future-proofing)."""
        from app.domain.services.tools.langchain_tools import create_native_tools

        tools = create_native_tools(
            sandbox_accessor=None,
            browser_accessor=None,
            search_engine=SimpleNamespace(invoke=lambda *a, **k: None),
            include_sandbox_tools=False,
        )
        names = {t.name for t in tools}
        assert "message_notify_user" in names  # always assembled
        assert "search_web" in names  # always assembled
        assert not any(
            n.startswith(("file_", "shell_", "browser_")) for n in names
        )

    def test_create_native_tools_asserts_accessors_when_including_sandbox(self) -> None:
        """include_sandbox_tools=True with a None accessor is a mis-assembly and
        must fail fast at construction (not defer a None-deref to first tool call)."""
        from app.domain.services.tools.langchain_tools import create_native_tools

        with pytest.raises(AssertionError):
            create_native_tools(
                sandbox_accessor=None,
                browser_accessor=None,
                search_engine=SimpleNamespace(invoke=lambda *a, **k: None),
                include_sandbox_tools=True,
            )
