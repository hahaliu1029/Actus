"""B9 wire schema：spec §2 契约锁定 + ISO UTC 序列化（R6#1/R8#6）。"""
import json
from datetime import datetime, timezone

from app.domain.models.runtime_extension import (
    ExtensionConfigInfo, ExtensionHealthInfo, ExtensionItemInfo,
    ExtensionLivenessInfo, ExtensionStatsInfo, RuntimeExtensionsSnapshot,
)
from app.interfaces.schemas.runtime_extensions import (
    CatalogItem, ExtensionHealth, ExtensionItem,
    RuntimeExtensionCatalogResponse, RuntimeExtensionsResponse,
    to_wire_item, to_wire_response,
)


def _item_info(**over) -> ExtensionItemInfo:
    base = dict(
        kind="mcp", id="srv-a", name="srv-a", description=None,
        config=ExtensionConfigInfo(enabled_global=True, enabled_user=True,
                                   effective_enabled=True, reason_code="enabled"),
        health=ExtensionHealthInfo(kind="probe", state="unknown"),
        liveness=ExtensionLivenessInfo(state="unknown"),
        stats=ExtensionStatsInfo(available=False, unavailable_reason="disabled"),
        details={"transport": "stdio"},
    )
    base.update(over)
    return ExtensionItemInfo(**base)


def test_health_vocabulary_has_no_connected():
    """INV-B9-1：词表禁 connected/disconnected。"""
    ann = ExtensionHealth.model_fields["state"].annotation
    literals = set(getattr(ann, "__args__", ()))
    assert "connected" not in literals and "disconnected" not in literals
    assert {"reachable", "unreachable", "ok", "error", "unknown", "skipped"} == literals


def test_datetime_serializes_iso_utc():
    ts = datetime(2026, 7, 4, 12, 0, 0, tzinfo=timezone.utc)
    health = ExtensionHealth(kind="probe", state="reachable", last_checked_at=ts)
    payload = json.loads(health.model_dump_json())
    assert payload["last_checked_at"] == "2026-07-04T12:00:00Z"


def test_to_wire_roundtrip():
    snap = RuntimeExtensionsSnapshot(
        items=(_item_info(),),
        snapshot_at=datetime(2026, 7, 4, tzinfo=timezone.utc),
        probe_enabled=False, stats_enabled=False,
    )
    wire = to_wire_response(snap)
    assert isinstance(wire, RuntimeExtensionsResponse)
    assert wire.items[0].kind == "mcp"
    assert wire.items[0].stats.unavailable_reason == "disabled"
    assert wire.probe_enabled is False


def test_catalog_response_shape():
    """R11#8：catalog 响应外形冻结（非裸 list）+ 九字段必填。"""
    item = CatalogItem(
        id="filesystem", name="Filesystem", description="本地文件系统",
        transport="stdio", config_template={"transport": "stdio", "command": "npx"},
        homepage="https://example.com", tags=["official"],
        source="https://github.com/modelcontextprotocol/servers", reviewed_at="2026-07-04",
    )
    resp = RuntimeExtensionCatalogResponse(items=[item])
    assert resp.items[0].reviewed_at == "2026-07-04"
