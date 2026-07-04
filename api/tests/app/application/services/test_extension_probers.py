"""B9 Task 13：DefaultExtensionProber（MCP 临时 manager / A2A httpx）+ error mapper。

覆盖三个关注面：
1. error mapper 逐值断言（8 code 全覆盖：timeout/spawn_failed/connect_failed/auth_failed/protocol_error）
2. probe_mcp——monkeypatch fake MCPClientManager 验证：同协程 init/cleanup 顺序、
   errors→outcome 映射、tool_count（成功）、cleanup 必被调（含异常路径）、结构性 finally 断言
3. probe_a2a——httpx.MockTransport 验证：成功取 display_name、raise_for_status→auth/protocol、
   timeout/connect→对应 code、JSON 解析失败→protocol_error
"""
import inspect

import httpx
import pytest

from app.application.services.extension_probe_service import (
    A2A_AGENT_CARD_PATH,
    A2A_PROBE_TIMEOUT_SECONDS,
    DefaultExtensionProber,
    ProbeOutcome,
    _map_a2a_exception,
    _map_mcp_error,
    _map_mcp_exception,
)
from app.domain.models.app_config import A2AServerConfig, MCPServerConfig, MCPTransport

MODULE = "app.application.services.extension_probe_service"


# ======================================================================
# 1. error mapper 逐值断言
# ======================================================================


class TestMapMcpError:
    """_map_mcp_error：MCPClientManager.errors 是中/英混合字符串，按前缀/关键字归类。"""

    def test_timeout_chinese(self):
        assert _map_mcp_error("连接MCP服务器[srv]超时(5s)") == "timeout"

    @pytest.mark.parametrize(
        "msg",
        [
            "FileNotFoundError: npx",
            "No such file or directory: npx",
            "failed to spawn subprocess",
            "executable not found",
        ],
    )
    def test_spawn_failed(self, msg):
        assert _map_mcp_error(msg) == "spawn_failed"

    @pytest.mark.parametrize(
        "msg",
        [
            "connect call failed",
            "Connection refused",
            "host unreachable",
            "ConnectionError to remote",
        ],
    )
    def test_connect_failed(self, msg):
        assert _map_mcp_error(msg) == "connect_failed"

    @pytest.mark.parametrize(
        "msg",
        [
            "HTTP 401 returned",
            "403 Forbidden",
            "Unauthorized access",
            "auth token invalid",
        ],
    )
    def test_auth_failed(self, msg):
        assert _map_mcp_error(msg) == "auth_failed"

    def test_protocol_error_fallback(self):
        assert _map_mcp_error("some unexpected weird failure") == "protocol_error"


class TestMapMcpException:
    def test_timeout(self):
        import asyncio

        assert _map_mcp_exception(asyncio.TimeoutError()) == "timeout"
        assert _map_mcp_exception(TimeoutError()) == "timeout"

    def test_spawn_failed(self):
        assert _map_mcp_exception(FileNotFoundError()) == "spawn_failed"
        assert _map_mcp_exception(PermissionError()) == "spawn_failed"

    def test_connect_failed(self):
        assert _map_mcp_exception(ConnectionError()) == "connect_failed"
        assert _map_mcp_exception(OSError()) == "connect_failed"

    def test_protocol_error_fallback(self):
        assert _map_mcp_exception(ValueError("weird")) == "protocol_error"

    def test_filenotfound_precedes_oserror(self):
        # FileNotFoundError 是 OSError 子类——顺序必须 spawn 先于 connect
        assert _map_mcp_exception(FileNotFoundError()) == "spawn_failed"


class TestMapA2aException:
    def test_timeout(self):
        exc = httpx.TimeoutException("timed out")
        assert _map_a2a_exception(exc) == "timeout"

    def test_connect_failed(self):
        exc = httpx.ConnectError("connection failed")
        assert _map_a2a_exception(exc) == "connect_failed"

    @pytest.mark.parametrize("code", [401, 403])
    def test_auth_failed(self, code):
        request = httpx.Request("GET", "http://x/.well-known/agent-card.json")
        response = httpx.Response(code, request=request)
        exc = httpx.HTTPStatusError("auth", request=request, response=response)
        assert _map_a2a_exception(exc) == "auth_failed"

    @pytest.mark.parametrize("code", [404, 500, 503])
    def test_http_status_non_auth_protocol_error(self, code):
        request = httpx.Request("GET", "http://x/.well-known/agent-card.json")
        response = httpx.Response(code, request=request)
        exc = httpx.HTTPStatusError("boom", request=request, response=response)
        assert _map_a2a_exception(exc) == "protocol_error"

    def test_value_error_fallback(self):
        # JSON 解析失败（resp.json() 抛 ValueError）→ protocol_error
        assert _map_a2a_exception(ValueError("bad json")) == "protocol_error"


# ======================================================================
# 2. probe_mcp（fake MCPClientManager）
# ======================================================================


class FakeMCPManager:
    """记录 init/cleanup 调用顺序与 asyncio Task（验证 INV-B9-5 同协程闭环）。"""

    events: list[str] = []
    init_tasks: list = []
    cleanup_tasks: list = []

    def __init__(self, mcp_config=None):
        import asyncio  # noqa: F401

        self.mcp_config = mcp_config
        self.errors: dict[str, str] = {}
        self.tools: dict[str, list] = {}
        self.raise_on_init: BaseException | None = None

    async def initialize(self):
        import asyncio

        type(self).events.append("init")
        type(self).init_tasks.append(asyncio.current_task())
        if self.raise_on_init is not None:
            raise self.raise_on_init

    async def cleanup(self):
        import asyncio

        type(self).events.append("cleanup")
        type(self).cleanup_tasks.append(asyncio.current_task())


def _install_fake_manager(monkeypatch, *, errors=None, tools=None, raise_on_init=None):
    FakeMCPManager.events = []
    FakeMCPManager.init_tasks = []
    FakeMCPManager.cleanup_tasks = []
    holder: dict[str, FakeMCPManager] = {}

    def _factory(mcp_config=None):
        mgr = FakeMCPManager(mcp_config=mcp_config)
        if errors:
            mgr.errors = dict(errors)
        if tools:
            mgr.tools = dict(tools)
        mgr.raise_on_init = raise_on_init
        holder["mgr"] = mgr
        return mgr

    monkeypatch.setattr(f"{MODULE}.MCPClientManager", _factory)
    return holder


@pytest.mark.anyio
async def test_probe_mcp_success_returns_tool_count(monkeypatch):
    _install_fake_manager(monkeypatch, tools={"srv": [object(), object()]})
    prober = DefaultExtensionProber()
    config = MCPServerConfig(url="http://x", enabled=True)

    outcome = await prober.probe_mcp("srv", config)

    assert outcome.ok is True
    assert outcome.tool_count == 2
    assert outcome.error_code is None
    assert outcome.latency_ms >= 0
    # init 与 cleanup 都在同一 Task（INV-B9-5）
    assert FakeMCPManager.events == ["init", "cleanup"]
    assert FakeMCPManager.init_tasks[0] is FakeMCPManager.cleanup_tasks[0]


@pytest.mark.anyio
async def test_probe_mcp_no_tools_zero_count(monkeypatch):
    _install_fake_manager(monkeypatch, tools={})
    prober = DefaultExtensionProber()
    outcome = await prober.probe_mcp("srv", MCPServerConfig(url="http://x"))
    assert outcome.ok is True
    assert outcome.tool_count == 0


@pytest.mark.anyio
async def test_probe_mcp_manager_error_maps_to_code(monkeypatch):
    _install_fake_manager(
        monkeypatch, errors={"srv": "连接MCP服务器[srv]超时(5s)"}
    )
    prober = DefaultExtensionProber()
    outcome = await prober.probe_mcp("srv", MCPServerConfig(url="http://x"))
    assert outcome.ok is False
    assert outcome.error_code == "timeout"
    assert outcome.error_message == "连接MCP服务器[srv]超时(5s)"
    # 失败路径 cleanup 仍被调
    assert FakeMCPManager.events == ["init", "cleanup"]


@pytest.mark.anyio
async def test_probe_mcp_init_exception_maps_and_cleans_up(monkeypatch):
    _install_fake_manager(
        monkeypatch, raise_on_init=FileNotFoundError("npx missing")
    )
    prober = DefaultExtensionProber()
    outcome = await prober.probe_mcp("srv", MCPServerConfig(url="http://x"))
    assert outcome.ok is False
    assert outcome.error_code == "spawn_failed"
    assert outcome.error_message == "npx missing"
    # 异常路径 cleanup 必被调（finally 闭环）
    assert FakeMCPManager.events == ["init", "cleanup"]
    assert FakeMCPManager.init_tasks[0] is FakeMCPManager.cleanup_tasks[0]


@pytest.mark.anyio
async def test_probe_mcp_builds_single_server_config(monkeypatch):
    holder = _install_fake_manager(monkeypatch, tools={"srv": []})
    prober = DefaultExtensionProber()
    config = MCPServerConfig(url="http://x", enabled=True)
    await prober.probe_mcp("srv", config)
    # 临时 manager 只装单项 config
    mcp_config = holder["mgr"].mcp_config
    assert set(mcp_config.mcpServers.keys()) == {"srv"}
    assert mcp_config.mcpServers["srv"] is config


# —— 结构性断言：源码级 finally + 同帧 init/cleanup ——


def test_probe_mcp_source_has_finally_cleanup():
    src = inspect.getsource(DefaultExtensionProber.probe_mcp)
    assert "initialize" in src
    assert "finally" in src
    # cleanup 在 finally 之后出现（同帧闭环）
    finally_idx = src.index("finally")
    cleanup_idx = src.index("cleanup", finally_idx)
    assert cleanup_idx > finally_idx


# ======================================================================
# 3. probe_a2a（httpx.MockTransport）
# ======================================================================


def _patch_async_client(monkeypatch, handler):
    """让 DefaultExtensionProber.probe_a2a 内部 httpx.AsyncClient 走 MockTransport。"""
    real_async_client = httpx.AsyncClient

    def _factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        # timeout 参数保留，验证 A2A_PROBE_TIMEOUT_SECONDS 被传入
        return real_async_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", _factory)


@pytest.mark.anyio
async def test_probe_a2a_success_display_name(monkeypatch):
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return httpx.Response(200, json={"name": "Remote Helper", "version": "1"})

    _patch_async_client(monkeypatch, handler)
    prober = DefaultExtensionProber()
    config = A2AServerConfig(id="a2a-1", base_url="http://remote:9000", enabled=True)

    outcome = await prober.probe_a2a(config)

    assert outcome.ok is True
    assert outcome.display_name == "Remote Helper"
    assert outcome.error_code is None
    assert captured["url"] == f"http://remote:9000{A2A_AGENT_CARD_PATH}"


@pytest.mark.anyio
async def test_probe_a2a_missing_name_none(monkeypatch):
    def handler(request):
        return httpx.Response(200, json={"version": "1"})

    _patch_async_client(monkeypatch, handler)
    prober = DefaultExtensionProber()
    outcome = await prober.probe_a2a(
        A2AServerConfig(base_url="http://remote:9000")
    )
    assert outcome.ok is True
    assert outcome.display_name is None


@pytest.mark.anyio
@pytest.mark.parametrize("code", [401, 403])
async def test_probe_a2a_auth_failed(monkeypatch, code):
    def handler(request):
        return httpx.Response(code)

    _patch_async_client(monkeypatch, handler)
    prober = DefaultExtensionProber()
    outcome = await prober.probe_a2a(
        A2AServerConfig(base_url="http://remote:9000")
    )
    assert outcome.ok is False
    assert outcome.error_code == "auth_failed"


@pytest.mark.anyio
async def test_probe_a2a_http_500_protocol_error(monkeypatch):
    def handler(request):
        return httpx.Response(500)

    _patch_async_client(monkeypatch, handler)
    prober = DefaultExtensionProber()
    outcome = await prober.probe_a2a(
        A2AServerConfig(base_url="http://remote:9000")
    )
    assert outcome.ok is False
    assert outcome.error_code == "protocol_error"


@pytest.mark.anyio
async def test_probe_a2a_invalid_json_protocol_error(monkeypatch):
    def handler(request):
        return httpx.Response(200, content=b"not json{{{")

    _patch_async_client(monkeypatch, handler)
    prober = DefaultExtensionProber()
    outcome = await prober.probe_a2a(
        A2AServerConfig(base_url="http://remote:9000")
    )
    assert outcome.ok is False
    assert outcome.error_code == "protocol_error"


@pytest.mark.anyio
async def test_probe_a2a_connect_error(monkeypatch):
    def handler(request):
        raise httpx.ConnectError("connection refused", request=request)

    _patch_async_client(monkeypatch, handler)
    prober = DefaultExtensionProber()
    outcome = await prober.probe_a2a(
        A2AServerConfig(base_url="http://remote:9000")
    )
    assert outcome.ok is False
    assert outcome.error_code == "connect_failed"


@pytest.mark.anyio
async def test_probe_a2a_timeout(monkeypatch):
    def handler(request):
        raise httpx.TimeoutException("timed out", request=request)

    _patch_async_client(monkeypatch, handler)
    prober = DefaultExtensionProber()
    outcome = await prober.probe_a2a(
        A2AServerConfig(base_url="http://remote:9000")
    )
    assert outcome.ok is False
    assert outcome.error_code == "timeout"


@pytest.mark.anyio
async def test_probe_a2a_error_message_redacted(monkeypatch):
    """error_message 是 raw（redaction 由 ExtensionProbeService 落 record 时做）——
    但 prober 层返回 str(exc) 供上层脱敏。此处只断言携带原始信息。"""
    def handler(request):
        raise httpx.ConnectError("connection refused to remote:9000", request=request)

    _patch_async_client(monkeypatch, handler)
    prober = DefaultExtensionProber()
    outcome = await prober.probe_a2a(
        A2AServerConfig(base_url="http://remote:9000")
    )
    assert outcome.error_message is not None
    assert "connection refused" in outcome.error_message


# —— DefaultExtensionProber 满足 ExtensionProber Protocol ——


def test_default_prober_satisfies_protocol():
    """结构性对齐 ExtensionProber Protocol（Task 12 未标 @runtime_checkable，
    故用签名对齐而非 isinstance）。"""
    import inspect

    prober = DefaultExtensionProber()
    # 两个必需的 async 方法存在且为协程函数
    assert inspect.iscoroutinefunction(prober.probe_mcp)
    assert inspect.iscoroutinefunction(prober.probe_a2a)
    # 签名参数名与 Protocol 一致
    mcp_params = list(inspect.signature(prober.probe_mcp).parameters)
    a2a_params = list(inspect.signature(prober.probe_a2a).parameters)
    assert mcp_params == ["server_name", "config"]
    assert a2a_params == ["config"]


def test_constants_frozen():
    assert A2A_PROBE_TIMEOUT_SECONDS == 8
    assert A2A_AGENT_CARD_PATH == "/.well-known/agent-card.json"


def test_probe_outcome_shape():
    outcome = ProbeOutcome(ok=True, latency_ms=1, tool_count=3, display_name="X")
    assert outcome.ok and outcome.tool_count == 3 and outcome.display_name == "X"
