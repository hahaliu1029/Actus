"""SPM PR-1b/1c: sandbox / browser accessor implementations.

PR-1b decouples the tool/runner layer from raw sandbox handles via the
``SandboxAccessor`` / ``BrowserAccessor`` protocols (``domain/external``). This
module ships the **Eager** implementations: they wrap an already-provisioned
concrete handle / browser so ``get()`` does ZERO IO / ZERO provisioning —
byte-equivalent ``always`` behavior.

PR-1c (Task 14) adds the **OnDemand** variants: ``get()`` delegates to a
``SandboxProvisioner`` so the FIRST call triggers lazy provisioning. The browser
variant keeps a handle-identity cache behind a single-flight lock (see
``OnDemandBrowserAccessor``).

INV-SPM-2: NEW code only — no existing behavior is altered.
"""
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from app.domain.external.browser import Browser
from app.domain.external.sandbox import SandboxHandle

if TYPE_CHECKING:
    from app.application.services.sandbox_provisioner import SandboxProvisioner

logger = logging.getLogger(__name__)


class EagerSandboxAccessor:
    """always 档 / 子会话：包裹既有 concrete handle，``get()`` 零供给零 IO。

    ``get()`` / ``peek()`` 都直接返回包裹的 handle（``get`` 从不做 IO）；
    ``release_owned()`` 走现状同步 ``SandboxHandle.release()`` 语义，幂等。
    """

    def __init__(self, handle: SandboxHandle) -> None:
        # release_owned() 后置 None → 字段必须为 optional 才能类型自洽。
        self._handle: SandboxHandle | None = handle

    async def get(self) -> SandboxHandle:
        # Eager：纯返回，零 IO / 零供给。release_owned() 是终态调用，其后不再 get()。
        return self._handle  # type: ignore[return-value]

    def peek(self) -> SandboxHandle | None:
        return self._handle

    async def release_owned(self) -> None:
        """r7/F3：终态释放包裹的 handle（幂等）。``release()`` 现状为同步方法
        （见 ``SandboxHandle`` 协议）——best-effort 吞异常，然后清空句柄使二次调用 no-op。

        SPM Task 17 fix #3: the AgentService no-lifecycle test fallback wraps a
        RAW ``DockerSandbox`` (no ``release()``) in this accessor. Guard with
        ``hasattr`` so shutdown's ``release_owned()`` stays quiet on those objects
        (the pre-accessor code's ``hasattr`` guard did the same) instead of logging
        a misleading warning+traceback for a benign missing method.
        """
        if self._handle is not None:
            if hasattr(self._handle, "release"):
                try:
                    self._handle.release()
                except Exception:
                    logger.warning("eager release_owned failed", exc_info=True)
            self._handle = None


class EagerBrowserAccessor:
    """镜像 ``EagerSandboxAccessor``：包裹既有 ``Browser``，``get()`` 零 IO。

    ``aclose()`` 委托到 ``Browser.aclose()`` 并 best-effort 吞异常；幂等性由底层
    ``Browser`` 实现负责（``PlaywrightBrowser.aclose`` 委托的 ``cleanup()`` 会把
    句柄重置为 None，故重复 close 天然 no-op）。
    """

    def __init__(self, browser: Browser) -> None:
        self._browser = browser

    async def get(self) -> Browser:
        return self._browser

    def peek(self) -> Browser | None:
        return self._browser

    async def aclose(self) -> None:
        try:
            await self._browser.aclose()
        except Exception:
            logger.warning(
                "EagerBrowserAccessor.aclose best-effort failed", exc_info=True
            )


class OnDemandSandboxAccessor:
    """on_demand 档：薄委托到 ``SandboxProvisioner``——首次 ``get()`` 触发懒供给。

    ``get()`` / ``peek()`` / ``release_owned()`` 一一转发给 provisioner
    （provision-if-needed / ready-only 探针 / owner-level 释放）。所有供给状态机
    语义（单飞、hooks_failed 保留、透传、超时 typed error）都在 provisioner 内。
    """

    def __init__(self, provisioner: "SandboxProvisioner") -> None:
        self._provisioner = provisioner

    async def get(self) -> SandboxHandle:
        return await self._provisioner.get()

    def peek(self) -> SandboxHandle | None:
        return self._provisioner.peek()

    async def release_owned(self) -> None:
        # r7/F3：= provisioner.release_held_handle()（覆盖 ready / hooks_failed，
        # peek()=None 也不漏）。同步方法，无 await。
        self._provisioner.release_held_handle()


class OnDemandBrowserAccessor:
    """handle-identity 缓存 + 单飞锁（r7 收敛：见 spec §5.2b / 设计 §4）。

    缓存 ``(self._handle, self._browser)``——持 handle 对象引用、用 ``is`` 比较
    （r7/codex R5 PART-E P2：不用 ``id(handle)``——旧对象 GC 后 id 可复用，新 handle
    命中旧 id 会漏重建；持对象引用则旧 handle 被缓存钉住不会 GC，``is`` 免疫 id 复用）。

    单 run on_demand 语义下 provisioner ready 后不存在 in-run 换代触发路径（handle
    缓存恒定 → browser 零重建，正确的一沙箱一浏览器）；identity 比较是**防御性**契约
    （覆盖「若 provisioner 曾 unprovisioned 后重 provision 产生新 handle」这一理论转换），
    不是生产热路径。CDP 断连由下一次沙箱工具调用经 ``sandbox_accessor.get()`` 的
    provisioner 层兜住（binding 非 ACTIVE → Suspended/Finalized 透传），与 always 一致。
    """

    def __init__(self, sandbox_accessor: OnDemandSandboxAccessor) -> None:
        self._sandbox_accessor = sandbox_accessor
        self._handle: SandboxHandle | None = None
        self._browser: Browser | None = None
        self._lock = asyncio.Lock()

    async def get(self) -> Browser:
        handle = await self._sandbox_accessor.get()
        async with self._lock:  # r7/codex R6-F4：单飞——两个并发 browser 首调
            if self._browser is not None and handle is self._handle:  # 各建一个实例会丢其一
                return self._browser
            if self._browser is not None:
                await self._browser.aclose()  # 换代（handle 身份变）→ aclose 旧
            self._browser = await handle.get_browser()
            self._handle = handle
            return self._browser

    def peek(self) -> Browser | None:
        return self._browser

    async def aclose(self) -> None:
        """幂等关闭 + 清缓存（锁内）。二次调用 no-op（``_browser`` 已 None）。"""
        async with self._lock:
            if self._browser is not None:
                try:
                    await self._browser.aclose()
                except Exception:
                    logger.warning(
                        "OnDemandBrowserAccessor.aclose best-effort failed",
                        exc_info=True,
                    )
                self._browser = None
            self._handle = None
