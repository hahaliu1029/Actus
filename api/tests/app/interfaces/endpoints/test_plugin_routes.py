"""D1a Task 24 §9.2 Plugin 路由端点测试（fake service/store + ASGITransport）。

只验 HTTP 面（PR-6 收口）：
- off 门四条（install / delete / enabled / list 一律 409 ``governance_disabled``）；
- install dry_run → 200 preview 透传（dry_run=True 穿透）；
- install 422 族（manifest_version>1 / version 非法 / zip）——preflight 拒绝异常 → 422；
- **install 终态三条（R4#3）**：fake ``install`` 返回 completed/compensated/failed 的
  ``InstallResult`` → 200 / 422 ``plugin_install_failed_compensated``（body 含
  collided_targets）/ 500 ``plugin_install_failed_requires_admin``；
- DELETE 缺 ``expected_row_revision`` → 422；DELETE 透传 uninstall；
- enabled=true 对 pending operation → 409 ``operation_pending``（R47#1 断言①——与 generic
  governance-enable **同一 ``set_enabled`` 方法**被调）；enabled 透传；
- GET /v2/plugins 形状含 ``last_operation`` 与 ``members``；非 Admin 403。

service/store 内部语义已在 saga/preflight 单测锁定。双 client fixture 对齐
test_extension_governance_routes.py：teardown 精确 pop 自己的 override key，禁 clear()。
"""
from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from pydantic import ValidationError as PydanticValidationError

from app.application.errors.exceptions import ValidationError
from app.application.services.plugin_install_service import (
    InstallResult,
    PluginIdentityCollisionError,
    PluginInstallPreview,
)
from app.domain.models.extension_governance import OperationPendingError
from app.domain.models.plugin_manifest import (
    PluginManifest,
    UnsupportedManifestVersionError,
)
from app.domain.models.user import User, UserRole, UserStatus
from app.infrastructure.external.governance.plugin_saga_store import (
    PluginDetailRow,
    PluginLastOperationRow,
    PluginMemberDetailRow,
)
from app.interfaces.dependencies.auth import get_current_user
from app.interfaces.endpoints.plugin_routes import (
    get_plugin_install_service,
    get_plugin_saga_store,
)
from app.interfaces.service_dependencies import get_extension_governance_service
from app.main import app

pytestmark = pytest.mark.anyio

_BASE = "/api/v2/plugins"


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _admin_user() -> User:
    return User(id="u1", username="admin",
                role=UserRole.SUPER_ADMIN, status=UserStatus.ACTIVE)


def _normal_user() -> User:
    return User(id="u1", username="tester",
                role=UserRole.USER, status=UserStatus.ACTIVE)


def _pydantic_manifest_error() -> PydanticValidationError:
    """真实 manifest pydantic 校验错误（id 非法 charset → version 非法族代表）。"""
    try:
        PluginManifest.parse(
            {"manifest_version": 1, "id": "bad id!", "name": "n", "version": "1.0"})
    except PydanticValidationError as exc:
        return exc
    raise AssertionError("expected pydantic ValidationError")


def _detail_row() -> PluginDetailRow:
    return PluginDetailRow(
        ext_id="my-plugin", name=None, version="1.0.0", status="active",
        artifact_hash="sha256:abc", row_revision=3,
        last_operation=PluginLastOperationRow(
            type="plugin_install", state="completed", error=None,
            updated_at="2026-01-01T00:00:00+00:00"),
        members=[PluginMemberDetailRow(
            declared_component_id="s1", kind="skill", ext_id="skill-x",
            expected_hash=None, installed_version="1.0.0", managed_by_plugin=True,
            status="active", scan_verdict="safe", scan_report=None)])


def _make_plugin_service() -> MagicMock:
    svc = MagicMock()
    svc.install = AsyncMock(return_value=InstallResult(
        status="completed", operation_id=uuid.uuid4(),
        plugin_ext_id="my-plugin", error=None, collided_targets=[]))
    svc.uninstall = AsyncMock(return_value=None)
    return svc


def _make_governance_service() -> MagicMock:
    svc = MagicMock()
    svc.set_enabled = AsyncMock(return_value=7)
    return svc


def _make_store() -> MagicMock:
    store = MagicMock()
    store.list_plugin_details = AsyncMock(return_value=[_detail_row()])
    return store


class _Deps:
    """三 provider override 句柄（install service / governance service / saga store）。"""

    def __init__(self, plugin, governance, store):
        self.plugin = plugin
        self.governance = governance
        self.store = store


def _install_overrides(user_factory, plugin, governance, store):
    app.dependency_overrides[get_current_user] = user_factory
    app.dependency_overrides[get_plugin_install_service] = lambda: plugin
    app.dependency_overrides[get_extension_governance_service] = lambda: governance
    app.dependency_overrides[get_plugin_saga_store] = lambda: store


def _pop_overrides():
    for dep in (get_current_user, get_plugin_install_service,
                get_extension_governance_service, get_plugin_saga_store):
        app.dependency_overrides.pop(dep, None)


def _client() -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


@pytest.fixture
def client_on():
    plugin, governance, store = (
        _make_plugin_service(), _make_governance_service(), _make_store())
    _install_overrides(_admin_user, plugin, governance, store)
    client = _client()
    client.deps = _Deps(plugin, governance, store)  # type: ignore[attr-defined]
    try:
        yield client
    finally:
        _pop_overrides()


@pytest.fixture
def client_off():
    _install_overrides(_admin_user, None, None, None)
    client = _client()
    try:
        yield client
    finally:
        _pop_overrides()


@pytest.fixture
def client_user():
    plugin, governance, store = (
        _make_plugin_service(), _make_governance_service(), _make_store())
    _install_overrides(_normal_user, plugin, governance, store)
    client = _client()
    try:
        yield client
    finally:
        _pop_overrides()


# ------------------------------------------------- off 门（四条）--------------


async def test_off_all_four_endpoints_409(client_off):
    for method, path, body in (
        ("post", f"{_BASE}/install", {"source_type": "local", "source_ref": "/x"}),
        ("delete", f"{_BASE}/my-plugin", {"expected_row_revision": 1}),
        ("post", f"{_BASE}/my-plugin/enabled", {"enabled": True, "expected_row_revision": 1}),
        ("get", f"{_BASE}", None),
    ):
        if method == "delete":
            resp = await client_off.request("DELETE", path, json=body)
        elif method == "get":
            resp = await client_off.get(path)
        else:
            resp = await client_off.post(path, json=body)
        assert resp.status_code == 409, path
        assert resp.json().get("code") == "governance_disabled", path


# ------------------------------------------------- install dry_run -----------


async def test_install_dry_run_returns_preview_200(client_on):
    client_on.deps.plugin.install = AsyncMock(return_value=PluginInstallPreview(
        plugin_id="my-plugin", name="My Plugin", version="1.0.0",
        aggregate_verdict="safe", install_policy_decision="allow", members=[]))
    resp = await client_on.post(
        f"{_BASE}/install",
        json={"source_type": "local", "source_ref": "/x", "dry_run": True})
    assert resp.status_code == 200
    body = resp.json()
    assert body["plugin_id"] == "my-plugin"
    assert body["aggregate_verdict"] == "safe"
    # dry_run=True 穿透
    _, kwargs = client_on.deps.plugin.install.await_args
    assert kwargs["dry_run"] is True


# ------------------------------------------------- install 422 族 ------------


@pytest.mark.parametrize("exc", [
    UnsupportedManifestVersionError(2),          # manifest_version>1
    _pydantic_manifest_error(),                  # version/id 非法（pydantic）
    ValidationError(msg="plugin 不支持压缩包输入（zip/tar）"),  # zip（application）
])
async def test_install_422_family(client_on, exc):
    client_on.deps.plugin.install = AsyncMock(side_effect=exc)
    resp = await client_on.post(
        f"{_BASE}/install", json={"source_type": "local", "source_ref": "/x"})
    assert resp.status_code == 422


async def test_install_identity_collision_409(client_on):
    client_on.deps.plugin.install = AsyncMock(
        side_effect=PluginIdentityCollisionError(msg="plugin[my-plugin] 已存活"))
    resp = await client_on.post(
        f"{_BASE}/install", json={"source_type": "local", "source_ref": "/x"})
    assert resp.status_code == 409


# ------------------------------------------------- install 终态三条 ----------


async def test_install_completed_200(client_on):
    op_id = uuid.uuid4()
    client_on.deps.plugin.install = AsyncMock(return_value=InstallResult(
        status="completed", operation_id=op_id, plugin_ext_id="my-plugin",
        error=None, collided_targets=[]))
    resp = await client_on.post(
        f"{_BASE}/install", json={"source_type": "local", "source_ref": "/x"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["plugin_ext_id"] == "my-plugin"
    assert body["status"] == "completed"
    assert body["operation_id"] == str(op_id)


async def test_install_compensated_422(client_on):
    op_id = uuid.uuid4()
    client_on.deps.plugin.install = AsyncMock(return_value=InstallResult(
        status="compensated", operation_id=op_id, plugin_ext_id="my-plugin",
        error="publish reverify failed", collided_targets=["mcp_config:srv"]))
    resp = await client_on.post(
        f"{_BASE}/install", json={"source_type": "local", "source_ref": "/x"})
    assert resp.status_code == 422
    body = resp.json()
    assert body["code"] == "plugin_install_failed_compensated"
    assert body["operation_id"] == str(op_id)
    assert body["error"] == "publish reverify failed"
    assert body["collided_targets"] == ["mcp_config:srv"]


async def test_install_failed_500(client_on):
    op_id = uuid.uuid4()
    client_on.deps.plugin.install = AsyncMock(return_value=InstallResult(
        status="failed", operation_id=op_id, plugin_ext_id="my-plugin",
        error="compensation failed", collided_targets=[]))
    resp = await client_on.post(
        f"{_BASE}/install", json={"source_type": "local", "source_ref": "/x"})
    assert resp.status_code == 500
    body = resp.json()
    assert body["code"] == "plugin_install_failed_requires_admin"
    assert body["operation_id"] == str(op_id)


# ------------------------------------------------- DELETE (uninstall) --------


async def test_delete_missing_revision_422(client_on):
    resp = await client_on.request("DELETE", f"{_BASE}/my-plugin", json={})
    assert resp.status_code == 422


async def test_delete_passthrough(client_on):
    resp = await client_on.request(
        "DELETE", f"{_BASE}/my-plugin", json={"expected_row_revision": 4})
    assert resp.status_code == 200
    client_on.deps.plugin.uninstall.assert_awaited_once_with(
        "my-plugin", actor_id="u1", expected_row_revision=4)


# ------------------------------------------------- enabled (reuse set_enabled) --


async def test_enabled_pending_operation_409(client_on):
    # R47#1 断言①：与 generic governance-enable 同一 set_enabled 方法；pending → 409
    client_on.deps.governance.set_enabled = AsyncMock(
        side_effect=OperationPendingError("plugin has in_progress operation"))
    resp = await client_on.post(
        f"{_BASE}/my-plugin/enabled",
        json={"enabled": True, "expected_row_revision": 2})
    assert resp.status_code == 409
    assert resp.json()["code"] == "operation_pending"
    # 同一方法被调（专用端点不绕过 T20 迁移服务前置）
    client_on.deps.governance.set_enabled.assert_awaited_once_with(
        "plugin", "my-plugin", enabled=True, expected_row_revision=2, actor_id="u1")


async def test_enabled_passthrough_true_and_false(client_on):
    resp = await client_on.post(
        f"{_BASE}/my-plugin/enabled",
        json={"enabled": True, "expected_row_revision": 2})
    assert resp.status_code == 200
    assert resp.json()["row_revision"] == 7
    client_on.deps.governance.set_enabled.assert_awaited_with(
        "plugin", "my-plugin", enabled=True, expected_row_revision=2, actor_id="u1")
    resp = await client_on.post(
        f"{_BASE}/my-plugin/enabled",
        json={"enabled": False, "expected_row_revision": 2})
    assert resp.status_code == 200
    client_on.deps.governance.set_enabled.assert_awaited_with(
        "plugin", "my-plugin", enabled=False, expected_row_revision=2, actor_id="u1")


# ------------------------------------------------- GET list ------------------


async def test_list_plugins_shape(client_on):
    resp = await client_on.get(f"{_BASE}")
    assert resp.status_code == 200
    items = resp.json()
    assert isinstance(items, list) and len(items) == 1
    item = items[0]
    assert item["ext_id"] == "my-plugin"
    assert item["status"] == "active"
    assert item["row_revision"] == 3
    # 形状含 last_operation 与 members（membership/operations 声明读者）
    assert item["last_operation"]["type"] == "plugin_install"
    assert item["last_operation"]["state"] == "completed"
    assert item["members"][0]["declared_component_id"] == "s1"
    assert item["members"][0]["kind"] == "skill"
    assert item["members"][0]["managed_by_plugin"] is True
    client_on.deps.store.list_plugin_details.assert_awaited_once()


# ------------------------------------------------- 403 ----------------------


async def test_non_admin_403(client_user):
    resp = await client_user.post(
        f"{_BASE}/install", json={"source_type": "local", "source_ref": "/x"})
    assert resp.status_code == 403


# --------------------- 真实 reject audit sink 合同（关闭 T21 Protocol 占位）-----


async def test_reject_audit_sink_writes_install_rejected(monkeypatch):
    """``DbPluginRejectAuditSink.record_install_rejected`` → 自持 ``install_rejected``
    audit（extension_id NULL + ext_id NULL）+ ``source_ref`` 合并进 details（R13#5/INV-D1-7）。"""
    import app.infrastructure.external.governance.plugin_saga_store as store_mod

    captured: dict = {}

    async def _fake_insert_audit(session, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(store_mod, "insert_audit", _fake_insert_audit)

    class _Ctx:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

    class _Session:
        def begin(self):
            return _Ctx()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

    sink = store_mod.DbPluginRejectAuditSink(lambda: _Session())
    await sink.record_install_rejected(
        source_type="local", source_ref="/x",
        details={"category": "identity_collision"})

    assert captured["event"] == "install_rejected"
    assert captured["kind"] == "plugin"
    assert captured["extension_id"] is None
    assert captured["ext_id"] is None
    assert captured["details"]["category"] == "identity_collision"
    assert captured["details"]["source_ref"] == "/x"  # source_ref 合并进 details
