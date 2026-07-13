"""D1a §3.6 identity 锁单一进程单例结构门（R4#1/R4#2/R5#1）。

断言：
 (a) ``get_identity_locks()`` 幂等——同一进程同一实例。
 (b) 三消费类源码的锁取用路径引用 ``get_identity_locks``（文本；SkillService +
     ExtensionInstallService + PluginInstallService（T24 起全部已落地——分阶段门补全）。
 (c) ``api/app`` 源码树零 ``app.state.extension_identity_locks`` 引用（锁不经 app.state
     传递——R4#1；断言对象=**源码树**，非 plan 文档，避免扫说明文字自打脸 R5#1）。
行为半：共享单例同 key 序列化——持锁期间另一取用式（模拟第二 service）阻塞等待。
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.application.services.extension_identity_locks import get_identity_locks

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
API_APP = REPO_ROOT / "api" / "app"

# 三消费类源码文件（相对 api/app）——每个的锁取用路径必须引用 get_identity_locks
_CONSUMER_FILES = [
    "application/services/skill_service.py",           # T16
    "application/services/extension_install_service.py",  # T19（本任务）
    "application/services/plugin_install_service.py",  # T24（本任务起必须已落地）
]

# 统一取用式（R4#2）——self._identity_locks or get_identity_locks()
_FALLBACK_SNIPPET = "or get_identity_locks()"


def test_get_identity_locks_is_process_singleton():
    """(a) 幂等：同一进程同一实例。"""
    first = get_identity_locks()
    second = get_identity_locks()
    assert first is second


def test_consumers_reference_get_identity_locks():
    """(b) 三消费类源码锁取用路径引用 get_identity_locks（存在的文件必须引用）。"""
    checked = 0
    for rel in _CONSUMER_FILES:
        path = API_APP / rel
        if not path.exists():
            continue                                   # T24 未落地 = 天然合规
        source = path.read_text()
        assert "get_identity_locks" in source, f"{rel} 未引用 get_identity_locks"
        assert _FALLBACK_SNIPPET in source, (
            f"{rel} 未使用统一取用式 `{_FALLBACK_SNIPPET}`")
        checked += 1
    # SkillService + ExtensionInstallService + PluginInstallService（T24 起）——三者必须已落地
    assert checked >= 3, f"预期至少 3 个消费类已落地，实际 {checked}"


def test_plugin_install_service_references_get_identity_locks():
    """(b) T24 分阶段门补全：PluginInstallService 锁取用路径引用 get_identity_locks
    统一取用式（非 app.state），与 T19 结构门 (b) 同式——类此时已存在，显式断言。"""
    path = API_APP / "application/services/plugin_install_service.py"
    assert path.exists(), "PluginInstallService（T24）必须已落地"
    source = path.read_text()
    assert "get_identity_locks" in source
    assert _FALLBACK_SNIPPET in source, (
        f"plugin_install_service.py 未使用统一取用式 `{_FALLBACK_SNIPPET}`")


def test_no_app_state_identity_lock_reference_in_source():
    """(c) api/app 源码树零 app.state.extension_identity_locks 引用（锁不经 app.state）。"""
    offenders: list[str] = []
    for py in API_APP.rglob("*.py"):
        if "app.state.extension_identity_locks" in py.read_text():
            offenders.append(py.relative_to(API_APP).as_posix())
    assert not offenders, (
        "锁不得经 app.state 传递（R4#1）——命中:\n" + "\n".join(offenders))


@pytest.mark.anyio
async def test_shared_singleton_serializes_same_key(anyio_backend):
    """行为半：共享单例同 key 序列化——持锁期间第二个取用式（模拟第二 service）阻塞等待。"""
    import asyncio

    reg = get_identity_locks()
    entered = asyncio.Event()
    async with reg.acquire_all([("mcp", "singleton-contention-probe")]):
        async def _second_service():
            # 另一 service 经同一进程工厂拿到**同一** registry → 同 key 争用
            async with get_identity_locks().acquire_all(
                [("mcp", "singleton-contention-probe")]
            ):
                entered.set()

        task = asyncio.create_task(_second_service())
        await asyncio.sleep(0.02)
        assert not entered.is_set()                    # 共享单例 → 同 key 阻塞
    await asyncio.wait_for(task, timeout=1.0)
    assert entered.is_set()                            # 释放后完成


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"
