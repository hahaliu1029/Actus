"""SPM PR-1b: sandbox / browser accessor implementations.

PR-1b decouples the tool/runner layer from raw sandbox handles via the
``SandboxAccessor`` / ``BrowserAccessor`` protocols (``domain/external``). This
module ships the **Eager** implementations: they wrap an already-provisioned
concrete handle / browser so ``get()`` does ZERO IO / ZERO provisioning —
byte-equivalent ``always`` behavior. The *OnDemand* variants (first ``get()``
triggers provisioning) land in PR-1c.

INV-SPM-2: NEW code only — no existing behavior is altered.
"""
from __future__ import annotations

import logging

from app.domain.external.browser import Browser
from app.domain.external.sandbox import SandboxHandle

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
        （见 ``SandboxHandle`` 协议）——best-effort 吞异常，然后清空句柄使二次调用 no-op。"""
        if self._handle is not None:
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
