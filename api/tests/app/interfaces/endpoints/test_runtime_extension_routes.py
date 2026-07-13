"""B9 端点：GET 聚合 + 角色投影 wire 断言（spec §6/§13）。

端点测试只验 HTTP 面（路由 / auth / 序列化 / is_admin 透传）——投影语义
已在 Task 6 (RuntimeExtensionService) 单测锁定。fake service 为 MagicMock，
``get_extensions`` 是 AsyncMock，返回 Task 6 同款 snapshot；断言 ``get_extensions``
被调用时的 ``is_admin`` 与登录用户 ``is_admin()`` 一致。

隔离约定（对齐 test_notification_routes.py:94）：fixture teardown 精确 pop
自己设置的 override key，禁用 clear()。匿名 client 不设 get_current_user
override，走真实 auth 依赖 → 401。
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from app.application.errors.exceptions import NotFoundError
from app.application.services.extension_probe_service import (
    ProbeBusyError,
    ProbeDisabledError,
    ProbeGoneError,
    ProbeRecord,
)
from app.application.services.app_config_service import AppConfigService
from app.application.services.runtime_extension_service import (
    RuntimeExtensionService,
)
from app.application.services.skill_service import SkillService
from app.domain.models.runtime_extension import (
    ExtensionConfigInfo,
    ExtensionHealthInfo,
    ExtensionItemInfo,
    ExtensionLivenessInfo,
    ExtensionStatsInfo,
    RuntimeExtensionsSnapshot,
)
from app.domain.models.user import User, UserRole, UserStatus
from app.interfaces.dependencies.auth import get_current_user
from app.interfaces.dependencies.rate_limit import rate_limit_write
from app.interfaces.endpoints import runtime_extension_routes
from app.interfaces.service_dependencies import (
    get_app_config_service,
    get_extension_probe_service,
    get_runtime_extension_service,
)
from app.main import app

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


# --- fixtures: fake users (真实 User 模型) --------------------------------- #


def _admin_user() -> User:
    return User(
        id="u1",
        username="admin",
        role=UserRole.SUPER_ADMIN,
        status=UserStatus.ACTIVE,
    )


def _normal_user() -> User:
    return User(
        id="u1",
        username="tester",
        role=UserRole.USER,
        status=UserStatus.ACTIVE,
    )


# --- fixtures: fake snapshots --------------------------------------------- #


def _admin_snapshot() -> RuntimeExtensionsSnapshot:
    item = ExtensionItemInfo(
        kind="mcp",
        id="server-a",
        name="server-a",
        description="a mcp server",
        config=ExtensionConfigInfo(
            enabled_global=True,
            enabled_user=True,
            effective_enabled=True,
            reason_code="enabled",
        ),
        health=ExtensionHealthInfo(kind="probe", state="unknown"),
        liveness=ExtensionLivenessInfo(state="unknown", active_run_count=0),
        stats=ExtensionStatsInfo(available=False, unavailable_reason="disabled"),
        details={"transport": "stdio", "tool_count": None},
    )
    return RuntimeExtensionsSnapshot(
        items=(item,),
        snapshot_at=datetime(2026, 7, 4, 12, 0, 0, tzinfo=timezone.utc),
        probe_enabled=False,
        stats_enabled=False,
    )


def _projected_snapshot() -> RuntimeExtensionsSnapshot:
    """非 Admin 投影 snapshot：stats.unavailable_reason == admin_only。"""
    item = ExtensionItemInfo(
        kind="mcp",
        id="server-a",
        name="server-a",
        description="a mcp server",
        config=ExtensionConfigInfo(
            enabled_global=True,
            enabled_user=True,
            effective_enabled=True,
            reason_code="enabled",
        ),
        health=ExtensionHealthInfo(kind="probe", state="unknown"),
        liveness=ExtensionLivenessInfo(state="unknown", active_run_count=0),
        stats=ExtensionStatsInfo(available=False, unavailable_reason="admin_only"),
        details={"transport": "stdio"},
    )
    return RuntimeExtensionsSnapshot(
        items=(item,),
        snapshot_at=datetime(2026, 7, 4, 12, 0, 0, tzinfo=timezone.utc),
        probe_enabled=False,
        stats_enabled=False,
    )


def _make_service(snapshot: RuntimeExtensionsSnapshot) -> MagicMock:
    service = MagicMock(spec=RuntimeExtensionService)
    service.get_extensions = AsyncMock(return_value=snapshot)
    return service


async def _noop_rate_limit() -> None:
    return None


# --- clients -------------------------------------------------------------- #


@pytest.fixture
def client_admin():
    fake_service = _make_service(_admin_snapshot())
    app.dependency_overrides[get_current_user] = _admin_user
    app.dependency_overrides[get_runtime_extension_service] = lambda: fake_service
    app.dependency_overrides[rate_limit_write] = _noop_rate_limit
    transport = httpx.ASGITransport(app=app)
    client = httpx.AsyncClient(transport=transport, base_url="http://test")
    client.fake_service = fake_service  # type: ignore[attr-defined]
    try:
        yield client
    finally:
        for dep in (get_current_user, get_runtime_extension_service, rate_limit_write):
            app.dependency_overrides.pop(dep, None)


@pytest.fixture
def client_user():
    fake_service = _make_service(_projected_snapshot())
    app.dependency_overrides[get_current_user] = _normal_user
    app.dependency_overrides[get_runtime_extension_service] = lambda: fake_service
    app.dependency_overrides[rate_limit_write] = _noop_rate_limit
    transport = httpx.ASGITransport(app=app)
    client = httpx.AsyncClient(transport=transport, base_url="http://test")
    client.fake_service = fake_service  # type: ignore[attr-defined]
    try:
        yield client
    finally:
        for dep in (get_current_user, get_runtime_extension_service, rate_limit_write):
            app.dependency_overrides.pop(dep, None)


@pytest.fixture
def client_anonymous():
    # 不设 get_current_user override → 走真实 auth 依赖 → 401（无 credentials）。
    transport = httpx.ASGITransport(app=app)
    client = httpx.AsyncClient(transport=transport, base_url="http://test")
    try:
        yield client
    finally:
        pass


# --- tests ---------------------------------------------------------------- #


async def test_get_extensions_200_and_shape(client_admin):
    resp = await client_admin.get("/api/v1/runtime/extensions")
    assert resp.status_code == 200
    body = resp.json()
    assert set(body.keys()) == {"items", "snapshot_at", "probe_enabled", "stats_enabled"}
    assert body["snapshot_at"].endswith("Z")
    # is_admin 透传：admin 用户 → get_extensions(is_admin=True)
    client_admin.fake_service.get_extensions.assert_awaited_once_with(
        user_id="u1", is_admin=True
    )


async def test_get_extensions_requires_auth(client_anonymous):
    resp = await client_anonymous.get("/api/v1/runtime/extensions")
    assert resp.status_code in (401, 403)


async def test_non_admin_gets_projected_view(client_user):
    resp = await client_user.get("/api/v1/runtime/extensions")
    assert resp.status_code == 200
    body = resp.json()
    for item in body["items"]:
        assert item["stats"]["unavailable_reason"] == "admin_only"
    # is_admin 透传：普通用户 → get_extensions(is_admin=False)
    client_user.fake_service.get_extensions.assert_awaited_once_with(
        user_id="u1", is_admin=False
    )


async def test_get_catalog_200(client_user):
    resp = await client_user.get("/api/v1/runtime/extensions/catalog")
    assert resp.status_code == 200
    assert isinstance(resp.json()["items"], list)      # R11#8 外形


# --- Task 15: enabled façade -------------------------------------------- #
#
# façade handler = 写委托（既有 service）→ probe 快照 invalidate →
# 重组装取更新后条目 → 返回 wire item（四段缺一不可，R3#1）。端点测试只验
# HTTP 面 + 委托契约（INV-B9-7 不双写）+ invalidate fail-open；service 内部
# 语义已在各自单测锁定。


def _facade_item(kind: str, ext_id: str) -> ExtensionItemInfo:
    return ExtensionItemInfo(
        kind=kind,  # type: ignore[arg-type]
        id=ext_id,
        name=ext_id,
        description=None,
        config=ExtensionConfigInfo(
            enabled_global=True,
            enabled_user=True,
            effective_enabled=True,
            reason_code="enabled",
        ),
        health=ExtensionHealthInfo(kind="probe", state="unknown"),
        liveness=ExtensionLivenessInfo(state="unknown", active_run_count=0),
        stats=ExtensionStatsInfo(available=False, unavailable_reason="disabled"),
        details={},
    )


def _facade_snapshot() -> RuntimeExtensionsSnapshot:
    """重组装快照：含所有 façade 测试用到的 (kind, id)，让 handler 取到更新后条目。"""
    return RuntimeExtensionsSnapshot(
        items=(
            _facade_item("mcp", "srv-on"),
            _facade_item("skill", "skill-good"),
            _facade_item("a2a", "a2a-1111-2222"),
        ),
        snapshot_at=datetime(2026, 7, 4, 12, 0, 0, tzinfo=timezone.utc),
        probe_enabled=False,
        stats_enabled=False,
    )


@pytest.fixture
def fake_app_config_service() -> MagicMock:
    svc = MagicMock(spec=AppConfigService)
    svc.set_mcp_server_enabled = AsyncMock(return_value=None)
    svc.set_a2a_server_enabled = AsyncMock(return_value=None)
    return svc


@pytest.fixture
def fake_skill_service() -> MagicMock:
    svc = MagicMock(spec=SkillService)
    svc.set_skill_enabled = AsyncMock(return_value=None)
    return svc


@pytest.fixture
def fake_probe_service() -> MagicMock:
    # invalidate 是同步方法（P-5）→ 普通 MagicMock 属性即可。
    svc = MagicMock()
    svc.invalidate = MagicMock(return_value=None)
    return svc


def _facade_overrides(
    user_factory,
    fake_app_config_service: MagicMock,
    fake_skill_service: MagicMock,
    fake_probe_service: MagicMock | None,
):
    """安装 façade 端点所需 overrides，返回 (client, teardown)。

    probe service 走 app.state（handler 用 request.app.state 读，非 DI）；
    None 时不设置 → 走 fail-open 分支。
    """
    fake_runtime = _make_service(_facade_snapshot())
    app.dependency_overrides[get_current_user] = user_factory
    app.dependency_overrides[get_app_config_service] = lambda: fake_app_config_service
    app.dependency_overrides[get_runtime_extension_service] = lambda: fake_runtime
    app.dependency_overrides[rate_limit_write] = _noop_rate_limit
    # _build_skill_service 是路由模块本地函数（非 DI 通道）——用 monkeypatch 语义
    # 直接替换模块属性；teardown 恢复原值。
    orig_build_skill = runtime_extension_routes._build_skill_service
    runtime_extension_routes._build_skill_service = lambda: fake_skill_service
    # probe service 挂 app.state（handler 用 request.app.state 读）。
    had_probe = hasattr(app.state, "extension_probe_service")
    prev_probe = getattr(app.state, "extension_probe_service", None)
    app.state.extension_probe_service = fake_probe_service

    def teardown() -> None:
        for dep in (
            get_current_user,
            get_app_config_service,
            get_runtime_extension_service,
            rate_limit_write,
        ):
            app.dependency_overrides.pop(dep, None)
        runtime_extension_routes._build_skill_service = orig_build_skill
        if had_probe:
            app.state.extension_probe_service = prev_probe
        else:
            delattr(app.state, "extension_probe_service")

    return teardown


@pytest.fixture
def facade_client_admin(
    fake_app_config_service, fake_skill_service, fake_probe_service
):
    teardown = _facade_overrides(
        _admin_user, fake_app_config_service, fake_skill_service, fake_probe_service
    )
    transport = httpx.ASGITransport(app=app)
    client = httpx.AsyncClient(transport=transport, base_url="http://test")
    try:
        yield client
    finally:
        teardown()


@pytest.fixture
def facade_client_admin_no_probe(
    fake_app_config_service, fake_skill_service
):
    """probe service = None（fail-open 分支）——不挂 app.state。"""
    teardown = _facade_overrides(
        _admin_user, fake_app_config_service, fake_skill_service, None
    )
    transport = httpx.ASGITransport(app=app)
    client = httpx.AsyncClient(transport=transport, base_url="http://test")
    try:
        yield client
    finally:
        teardown()


@pytest.fixture
def facade_client_user(
    fake_app_config_service, fake_skill_service, fake_probe_service
):
    teardown = _facade_overrides(
        _normal_user, fake_app_config_service, fake_skill_service, fake_probe_service
    )
    transport = httpx.ASGITransport(app=app)
    client = httpx.AsyncClient(transport=transport, base_url="http://test")
    try:
        yield client
    finally:
        teardown()


async def test_enabled_facade_delegates_not_duplicates(
    facade_client_admin, fake_app_config_service
):
    """INV-B9-7：mcp 启停必须且只调用既有 set_mcp_server_enabled（验证不双写）。"""
    resp = await facade_client_admin.post(
        "/api/v1/runtime/extensions/mcp/srv-on/enabled", json={"enabled": False}
    )
    assert resp.status_code == 200
    fake_app_config_service.set_mcp_server_enabled.assert_awaited_once_with(
        "srv-on", False
    )
    body = resp.json()
    assert body["kind"] == "mcp" and body["id"] == "srv-on"


async def test_enabled_facade_invalidates_probe_snapshot(
    facade_client_admin, fake_probe_service
):
    await facade_client_admin.post(
        "/api/v1/runtime/extensions/mcp/srv-on/enabled", json={"enabled": False}
    )
    fake_probe_service.invalidate.assert_called_once_with("mcp", "srv-on", "disable")


async def test_enabled_facade_enable_invalidate_reason(
    facade_client_admin, fake_probe_service
):
    """enabled=True → invalidate reason "enable"（P-5 语义对称）。"""
    await facade_client_admin.post(
        "/api/v1/runtime/extensions/mcp/srv-on/enabled", json={"enabled": True}
    )
    fake_probe_service.invalidate.assert_called_once_with("mcp", "srv-on", "enable")


async def test_enabled_facade_skill_and_a2a_routes(
    facade_client_admin, fake_skill_service, fake_app_config_service
):
    r1 = await facade_client_admin.post(
        "/api/v1/runtime/extensions/skill/skill-good/enabled", json={"enabled": True}
    )
    assert r1.status_code == 200
    fake_skill_service.set_skill_enabled.assert_awaited_once_with("skill-good", True)
    r2 = await facade_client_admin.post(
        "/api/v1/runtime/extensions/a2a/a2a-1111-2222/enabled", json={"enabled": True}
    )
    assert r2.status_code == 200
    fake_app_config_service.set_a2a_server_enabled.assert_awaited_once_with(
        "a2a-1111-2222", True
    )


async def test_enabled_facade_fail_open_when_probe_none(
    facade_client_admin_no_probe, fake_app_config_service
):
    """probe service 未启动（app.state 无句柄）→ invalidate 跳过，端点仍 200。"""
    resp = await facade_client_admin_no_probe.post(
        "/api/v1/runtime/extensions/mcp/srv-on/enabled", json={"enabled": False}
    )
    assert resp.status_code == 200
    fake_app_config_service.set_mcp_server_enabled.assert_awaited_once_with(
        "srv-on", False
    )


async def test_enabled_facade_fail_open_when_invalidate_raises(
    fake_app_config_service, fake_skill_service
):
    """Fix 3：半坏的 probe 单例 invalidate 抛错 → 写已成功，端点仍 200（不翻 500）。"""
    raising_probe = MagicMock()
    raising_probe.invalidate = MagicMock(side_effect=RuntimeError("probe 半坏"))
    teardown = _facade_overrides(
        _admin_user, fake_app_config_service, fake_skill_service, raising_probe
    )
    transport = httpx.ASGITransport(app=app)
    client = httpx.AsyncClient(transport=transport, base_url="http://test")
    try:
        resp = await client.post(
            "/api/v1/runtime/extensions/mcp/srv-on/enabled", json={"enabled": False}
        )
        assert resp.status_code == 200                       # fail-open：写已成功不翻 500
        fake_app_config_service.set_mcp_server_enabled.assert_awaited_once_with(
            "srv-on", False
        )
        raising_probe.invalidate.assert_called_once()        # invalidate 确实被调（并抛错被吞）
    finally:
        teardown()


async def test_enabled_facade_admin_only(facade_client_user):
    resp = await facade_client_user.post(
        "/api/v1/runtime/extensions/mcp/srv-on/enabled", json={"enabled": False}
    )
    assert resp.status_code == 403


async def test_enabled_facade_unknown_id_404(
    facade_client_admin, fake_app_config_service
):
    fake_app_config_service.set_mcp_server_enabled.side_effect = NotFoundError("不存在")
    resp = await facade_client_admin.post(
        "/api/v1/runtime/extensions/mcp/ghost/enabled", json={"enabled": False}
    )
    assert resp.status_code == 404


async def test_enabled_facade_invalid_kind_422(facade_client_admin):
    resp = await facade_client_admin.post(
        "/api/v1/runtime/extensions/bogus/x/enabled", json={"enabled": False}
    )
    assert resp.status_code == 422


# --- Task 16: manual probe 端点 ---------------------------------------- #
#
# 端点序列（spec §3.1/§6）：rate limit → flag 检查（有效 probe 能力 False →
# 409 probe_disabled）→ 目标存在性/enabled 预检（404 / 409 extension_disabled）
# → 冷却窗（进程内 (kind,id) map，5s → 429）→ kind 分派：mcp/a2a 走
# probe_one_manual（预算内化 service；Busy→503 / Disabled→409 / Gone→404），
# skill 短路重扫（0 调用 service）→ 审计日志（每个实际执行的探测）→ 重组装返回。
# 端点测试只验 HTTP 面 + 映射 + 冷却 + 审计；预算/退避语义已在 Task 12 单测锁定。


@pytest.fixture(autouse=True)
def _clean_probe_cooldowns():
    """模块级 cooldown map 在同文件多用例间顺序耦合——前后各 reset 一次。

    对齐 tests/conftest.py:77 对全局可变注册表的 autouse reset 惯例。
    """
    from app.interfaces.endpoints.runtime_extension_routes import (
        _reset_probe_cooldowns,
    )

    _reset_probe_cooldowns()
    yield
    _reset_probe_cooldowns()


def _probe_item(
    kind: str,
    ext_id: str,
    *,
    health_state: str = "reachable",
    enabled_global: bool = True,
    error_code: str | None = None,
    latency_ms: int | None = 12,
) -> ExtensionItemInfo:
    health_kind = "integrity" if kind == "skill" else "probe"
    return ExtensionItemInfo(
        kind=kind,  # type: ignore[arg-type]
        id=ext_id,
        name=ext_id,
        description=None,
        config=ExtensionConfigInfo(
            enabled_global=enabled_global,
            enabled_user=True,
            effective_enabled=enabled_global,
            reason_code="enabled" if enabled_global else "disabled_global",
        ),
        health=ExtensionHealthInfo(
            kind=health_kind,  # type: ignore[arg-type]
            state=health_state,  # type: ignore[arg-type]
            latency_ms=latency_ms,
            error_code=error_code,
        ),
        liveness=ExtensionLivenessInfo(state="unknown", active_run_count=0),
        stats=ExtensionStatsInfo(available=False, unavailable_reason="disabled"),
        details={},
    )


def _probe_snapshot(
    *items: ExtensionItemInfo, probe_enabled: bool = True
) -> RuntimeExtensionsSnapshot:
    return RuntimeExtensionsSnapshot(
        items=tuple(items),
        snapshot_at=datetime(2026, 7, 4, 12, 0, 0, tzinfo=timezone.utc),
        probe_enabled=probe_enabled,
        stats_enabled=False,
    )


def _probe_overrides(
    user_factory,
    snapshot: RuntimeExtensionsSnapshot,
    probe_service: MagicMock | None,
):
    """安装 probe 端点 overrides，返回 teardown。

    runtime service（重组装/pre-check 权威）走 get_runtime_extension_service DI；
    probe service（mcp/a2a probe_one_manual）走 get_extension_probe_service DI。
    """
    fake_runtime = _make_service(snapshot)
    app.dependency_overrides[get_current_user] = user_factory
    app.dependency_overrides[get_runtime_extension_service] = lambda: fake_runtime
    app.dependency_overrides[get_extension_probe_service] = lambda: probe_service
    app.dependency_overrides[rate_limit_write] = _noop_rate_limit

    def teardown() -> None:
        for dep in (
            get_current_user,
            get_runtime_extension_service,
            get_extension_probe_service,
            rate_limit_write,
        ):
            app.dependency_overrides.pop(dep, None)

    return fake_runtime, teardown


def _make_probe_client(
    user_factory,
    snapshot: RuntimeExtensionsSnapshot,
    probe_service: MagicMock | None,
):
    fake_runtime, teardown = _probe_overrides(user_factory, snapshot, probe_service)
    transport = httpx.ASGITransport(app=app)
    client = httpx.AsyncClient(transport=transport, base_url="http://test")
    client.fake_runtime = fake_runtime  # type: ignore[attr-defined]
    client.fake_probe = probe_service  # type: ignore[attr-defined]
    client._teardown = teardown  # type: ignore[attr-defined]
    return client


@pytest.fixture
def fake_prober() -> MagicMock:
    svc = MagicMock()
    svc.probe_one_manual = AsyncMock(
        return_value=ProbeRecord(state="reachable", latency_ms=12)
    )
    return svc


async def test_probe_flag_off_409_probe_disabled(fake_prober):
    """有效 probe 能力 False（snapshot.probe_enabled=False）→ 409 probe_disabled。"""
    snap = _probe_snapshot(_probe_item("mcp", "srv-a"), probe_enabled=False)
    client = _make_probe_client(_admin_user, snap, fake_prober)
    try:
        resp = await client.post("/api/v1/runtime/extensions/mcp/srv-a/probe")
        assert resp.status_code == 409
        assert resp.json()["data"]["reason"] == "probe_disabled"
        fake_prober.probe_one_manual.assert_not_called()
    finally:
        client._teardown()  # type: ignore[attr-defined]


async def test_probe_disabled_extension_409(fake_prober):
    """目标全局 disabled → 409 extension_disabled。"""
    snap = _probe_snapshot(
        _probe_item("mcp", "srv-off", enabled_global=False), probe_enabled=True
    )
    client = _make_probe_client(_admin_user, snap, fake_prober)
    try:
        resp = await client.post("/api/v1/runtime/extensions/mcp/srv-off/probe")
        assert resp.status_code == 409
        assert resp.json()["data"]["reason"] == "extension_disabled"
        fake_prober.probe_one_manual.assert_not_called()
    finally:
        client._teardown()  # type: ignore[attr-defined]


async def test_probe_unknown_target_404(fake_prober):
    """snapshot 不含该 (kind,id) → 404。"""
    snap = _probe_snapshot(_probe_item("mcp", "srv-a"), probe_enabled=True)
    client = _make_probe_client(_admin_user, snap, fake_prober)
    try:
        resp = await client.post("/api/v1/runtime/extensions/mcp/ghost/probe")
        assert resp.status_code == 404
        fake_prober.probe_one_manual.assert_not_called()
    finally:
        client._teardown()  # type: ignore[attr-defined]


async def test_probe_cooldown_429_with_retry_after(fake_prober):
    """连发两次：第二次 429 + data.retry_after ∈ [1,5]。"""
    snap = _probe_snapshot(_probe_item("mcp", "srv-a"), probe_enabled=True)
    client = _make_probe_client(_admin_user, snap, fake_prober)
    try:
        r1 = await client.post("/api/v1/runtime/extensions/mcp/srv-a/probe")
        assert r1.status_code == 200
        r2 = await client.post("/api/v1/runtime/extensions/mcp/srv-a/probe")
        assert r2.status_code == 429
        retry_after = r2.json()["data"]["retry_after"]
        assert 1 <= retry_after <= 5
    finally:
        client._teardown()  # type: ignore[attr-defined]


async def test_probe_budget_exhausted_503_probe_busy(fake_prober):
    """probe_one_manual 抛 ProbeBusyError → 503 + data.reason == probe_busy。"""
    fake_prober.probe_one_manual = AsyncMock(side_effect=ProbeBusyError())
    snap = _probe_snapshot(_probe_item("mcp", "srv-a"), probe_enabled=True)
    client = _make_probe_client(_admin_user, snap, fake_prober)
    try:
        resp = await client.post("/api/v1/runtime/extensions/mcp/srv-a/probe")
        assert resp.status_code == 503
        assert resp.json()["data"]["reason"] == "probe_busy"
    finally:
        client._teardown()  # type: ignore[attr-defined]


async def test_probe_success_returns_updated_item(fake_prober):
    """fake 返回 reachable record → 200 + health.state == reachable。"""
    snap = _probe_snapshot(
        _probe_item("mcp", "srv-a", health_state="reachable"), probe_enabled=True
    )
    client = _make_probe_client(_admin_user, snap, fake_prober)
    try:
        resp = await client.post("/api/v1/runtime/extensions/mcp/srv-a/probe")
        assert resp.status_code == 200
        body = resp.json()
        assert body["kind"] == "mcp" and body["id"] == "srv-a"
        assert body["health"]["state"] == "reachable"
        fake_prober.probe_one_manual.assert_awaited_once_with("mcp", "srv-a")
    finally:
        client._teardown()  # type: ignore[attr-defined]


async def test_probe_midflight_deleted_404(fake_prober):
    """probe_one_manual 抛 ProbeGoneError（探测中被删）→ 404。"""
    fake_prober.probe_one_manual = AsyncMock(side_effect=ProbeGoneError())
    snap = _probe_snapshot(_probe_item("mcp", "srv-a"), probe_enabled=True)
    client = _make_probe_client(_admin_user, snap, fake_prober)
    try:
        resp = await client.post("/api/v1/runtime/extensions/mcp/srv-a/probe")
        assert resp.status_code == 404
    finally:
        client._teardown()  # type: ignore[attr-defined]


async def test_probe_disabled_error_from_service_409(fake_prober):
    """probe_one_manual 二次复核抛 ProbeDisabledError(reason) → 409 + reason 透传。"""
    fake_prober.probe_one_manual = AsyncMock(
        side_effect=ProbeDisabledError("extension_disabled")
    )
    snap = _probe_snapshot(_probe_item("mcp", "srv-a"), probe_enabled=True)
    client = _make_probe_client(_admin_user, snap, fake_prober)
    try:
        resp = await client.post("/api/v1/runtime/extensions/mcp/srv-a/probe")
        assert resp.status_code == 409
        assert resp.json()["data"]["reason"] == "extension_disabled"
    finally:
        client._teardown()  # type: ignore[attr-defined]


async def test_probe_audit_log_fields(fake_prober, caplog):
    """审计 record 含 7 字段 + raw ext_id（含换行）不出现在任何 record 属性值中。"""
    import logging as _logging
    from urllib.parse import quote

    raw_id = "srv-a\nINJECTED"
    snap = _probe_snapshot(
        _probe_item("mcp", raw_id, health_state="reachable"), probe_enabled=True
    )
    client = _make_probe_client(_admin_user, snap, fake_prober)
    try:
        with caplog.at_level(_logging.INFO, logger="actus.extension_probe"):
            resp = await client.post(
                f"/api/v1/runtime/extensions/mcp/{quote(raw_id, safe='')}/probe"
            )
        assert resp.status_code == 200
        records = [
            r for r in caplog.records if r.name == "actus.extension_probe"
        ]
        assert len(records) == 1
        rec = records[0]
        for field in (
            "admin_user_id",
            "kind",
            "id_display",
            "id_hash",
            "outcome",
            "latency_ms",
            "error_code",
        ):
            assert hasattr(rec, field), f"缺字段 {field}"
        assert rec.kind == "mcp"
        assert rec.outcome == "success"
        # raw ext_id（含换行）不得出现在任何 record 属性值中。
        for value in vars(rec).values():
            assert raw_id not in str(value)
    finally:
        client._teardown()  # type: ignore[attr-defined]


async def test_probe_cooldown_rejected_not_audited(fake_prober, caplog):
    """429 冷却拒绝路径：caplog 无 info 级审计记录（仅第一次实际探测有）。"""
    import logging as _logging

    snap = _probe_snapshot(_probe_item("mcp", "srv-a"), probe_enabled=True)
    client = _make_probe_client(_admin_user, snap, fake_prober)
    try:
        with caplog.at_level(_logging.INFO, logger="actus.extension_probe"):
            # 第一次实际探测会审计——清掉后只看 429 冷却拒绝路径是否留审计记录。
            await client.post("/api/v1/runtime/extensions/mcp/srv-a/probe")
            caplog.clear()
            r2 = await client.post("/api/v1/runtime/extensions/mcp/srv-a/probe")
        assert r2.status_code == 429
        audited = [
            r
            for r in caplog.records
            if r.name == "actus.extension_probe" and r.levelno == _logging.INFO
        ]
        assert audited == []
    finally:
        client._teardown()  # type: ignore[attr-defined]


async def test_probe_admin_only_403(fake_prober):
    snap = _probe_snapshot(_probe_item("mcp", "srv-a"), probe_enabled=True)
    client = _make_probe_client(_normal_user, snap, fake_prober)
    try:
        resp = await client.post("/api/v1/runtime/extensions/mcp/srv-a/probe")
        assert resp.status_code == 403
    finally:
        client._teardown()  # type: ignore[attr-defined]


async def test_probe_skill_short_circuits_service(fake_prober, caplog):
    """kind=skill：probe service 0 调用 + 200 返回该 skill 条目 + 审计 outcome 与 health.state 一致。"""
    import logging as _logging

    snap = _probe_snapshot(
        _probe_item("skill", "skill-good", health_state="ok"), probe_enabled=True
    )
    client = _make_probe_client(_admin_user, snap, fake_prober)
    try:
        with caplog.at_level(_logging.INFO, logger="actus.extension_probe"):
            resp = await client.post(
                "/api/v1/runtime/extensions/skill/skill-good/probe"
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["kind"] == "skill" and body["id"] == "skill-good"
        # probe service 从未被调用（skill 无网络探测语义）。
        fake_prober.probe_one_manual.assert_not_called()
        records = [
            r
            for r in caplog.records
            if r.name == "actus.extension_probe" and r.levelno == _logging.INFO
        ]
        assert len(records) == 1
        # health.state == "ok" → outcome == "success"。
        assert records[0].outcome == "success"
        assert records[0].kind == "skill"
    finally:
        client._teardown()  # type: ignore[attr-defined]


async def test_probe_skill_error_outcome_failure(fake_prober, caplog):
    """kind=skill 且 health.state == error → 审计 outcome == failure。"""
    import logging as _logging

    snap = _probe_snapshot(
        _probe_item(
            "skill",
            "skill-bad",
            health_state="error",
            error_code="parse_error",
            latency_ms=None,
        ),
        probe_enabled=True,
    )
    client = _make_probe_client(_admin_user, snap, fake_prober)
    try:
        with caplog.at_level(_logging.INFO, logger="actus.extension_probe"):
            resp = await client.post(
                "/api/v1/runtime/extensions/skill/skill-bad/probe"
            )
        assert resp.status_code == 200
        records = [
            r
            for r in caplog.records
            if r.name == "actus.extension_probe" and r.levelno == _logging.INFO
        ]
        assert len(records) == 1
        assert records[0].outcome == "failure"
        assert records[0].error_code == "parse_error"
    finally:
        client._teardown()  # type: ignore[attr-defined]


async def test_probe_service_raises_valueerror_on_skill():
    """服务层结构性拒绝：probe_one_manual("skill", ...) 直接调用抛 ValueError（防误用）。"""
    from app.application.services.extension_probe_service import (
        ExtensionProbeService,
    )

    svc = ExtensionProbeService(
        config_provider=lambda: MagicMock(),
        skill_repository=MagicMock(),
        prober=MagicMock(),
        probe_flag_provider=lambda: True,
    )
    with pytest.raises(ValueError):
        await svc.probe_one_manual("skill", "skill-good")


async def test_probe_invalid_kind_422(fake_prober):
    snap = _probe_snapshot(_probe_item("mcp", "srv-a"), probe_enabled=True)
    client = _make_probe_client(_admin_user, snap, fake_prober)
    try:
        resp = await client.post("/api/v1/runtime/extensions/bogus/x/probe")
        assert resp.status_code == 422
    finally:
        client._teardown()  # type: ignore[attr-defined]


async def test_facade_plugin_path_422(client_admin):
    """D1a §9.1 F2 + R6#C1：两条 façade path 的 kind Literal 各自内联三值
    （mcp/a2a/skill），**不含** plugin——POST /extensions/plugin/x/{enabled,probe} 均
    422（path Literal 不扩，语义=plugin 不可执行/不支持启停探测）；聚合 service **零调用**
    （422 由 path 参数校验触发，handler 从不运行）。"""
    r_enabled = await client_admin.post(
        "/api/v1/runtime/extensions/plugin/x/enabled", json={"enabled": True}
    )
    assert r_enabled.status_code == 422
    r_probe = await client_admin.post("/api/v1/runtime/extensions/plugin/x/probe")
    assert r_probe.status_code == 422
    # service 零调用（enabled step 3 / probe step 1 的 get_extensions 均未触达）。
    client_admin.fake_service.get_extensions.assert_not_awaited()
