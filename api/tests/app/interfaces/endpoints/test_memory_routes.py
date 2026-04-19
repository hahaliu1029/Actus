"""Integration tests for /v2/memories routes.

使用 FastAPI 的 ``app.dependency_overrides`` 将 ``MemoryManagementService``
和 ``get_current_user`` 替换成 AsyncMock/伪造实例，避免接触真实 DB/Redis。
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock

import httpx
import pytest

from app.application.errors.exceptions import ConflictError
from app.domain.models.memory_chunk import MemoryChunk
from app.domain.models.user import User, UserRole, UserStatus
from app.interfaces.dependencies.auth import get_current_user
from app.interfaces.dependencies.rate_limit import rate_limit_read, rate_limit_write
from app.interfaces.service_dependencies import get_memory_management_service
from app.main import app

from tests.conftest import TEST_USER_ID_FIXED

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


# --- fixtures ---------------------------------------------------------------


def _fake_user() -> User:
    return User(
        id=TEST_USER_ID_FIXED,
        username="tester",
        role=UserRole.USER,
        status=UserStatus.ACTIVE,
    )


def _make_chunk(
    *,
    chunk_id: str = "mem-1",
    user_id: str = TEST_USER_ID_FIXED,
    content: str = "记忆内容",
    content_hash: str = "abc123",
    source: str = "manual",
    session_id: str | None = None,
    metadata: dict | None = None,
) -> MemoryChunk:
    now = datetime(2026, 4, 16, 12, 0, 0)
    return MemoryChunk(
        id=chunk_id,
        user_id=user_id,
        content=content,
        content_hash=content_hash,
        source=source,
        metadata=metadata or {},
        created_at=now,
        updated_at=now,
        session_id=session_id,
    )


@pytest.fixture
def mock_service() -> AsyncMock:
    svc = AsyncMock()
    svc.list_memories = AsyncMock(return_value=([], 0))
    svc.get_memory = AsyncMock(return_value=None)
    svc.update_memory_content = AsyncMock(return_value=None)
    svc.update_memory_pinned = AsyncMock(return_value=None)
    svc.delete_memory = AsyncMock(return_value=False)
    svc.bulk_delete_memories = AsyncMock(return_value=0)
    svc.delete_all_memories = AsyncMock(return_value=0)
    svc.delete_legacy_memories = AsyncMock(return_value=0)
    svc.reindex_memory = AsyncMock()
    return svc


async def _noop_rate_limit() -> None:
    """测试中跳过限流（底层需要 Redis，不在 TestClient 环境里初始化）。"""
    return None


@pytest.fixture
def client_app(mock_service: AsyncMock):
    app.dependency_overrides[get_memory_management_service] = lambda: mock_service
    app.dependency_overrides[get_current_user] = _fake_user
    app.dependency_overrides[rate_limit_read] = _noop_rate_limit
    app.dependency_overrides[rate_limit_write] = _noop_rate_limit
    try:
        yield app
    finally:
        app.dependency_overrides.pop(get_memory_management_service, None)
        app.dependency_overrides.pop(get_current_user, None)
        app.dependency_overrides.pop(rate_limit_read, None)
        app.dependency_overrides.pop(rate_limit_write, None)


async def _request(
    client_app,
    method: str,
    url: str,
    *,
    json: dict | None = None,
    params: dict | None = None,
) -> httpx.Response:
    transport = httpx.ASGITransport(app=client_app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test"
    ) as client:
        return await client.request(method, url, json=json, params=params)


# --- list_memories ----------------------------------------------------------


async def test_list_memories_empty_returns_200(
    client_app, mock_service: AsyncMock
) -> None:
    response = await _request(client_app, "GET", "/api/v2/memories")

    assert response.status_code == 200
    body = response.json()
    assert body["code"] == 200
    data = body["data"]
    assert data["items"] == []
    assert data["total"] == 0
    assert data["page"] == 1
    assert data["page_size"] == 20
    assert data["has_next"] is False
    mock_service.list_memories.assert_awaited_once()


async def test_list_memories_with_items_has_next_flag(
    client_app, mock_service: AsyncMock
) -> None:
    chunks = [_make_chunk(chunk_id=f"mem-{i}") for i in range(3)]
    mock_service.list_memories = AsyncMock(return_value=(chunks, 25))

    response = await _request(
        client_app,
        "GET",
        "/api/v2/memories",
        params={"page": 1, "page_size": 3},
    )

    assert response.status_code == 200
    data = response.json()["data"]
    assert len(data["items"]) == 3
    assert data["total"] == 25
    # page=1, page_size=3 → 3 < 25 → has_next True
    assert data["has_next"] is True
    # 列表视图不应暴露 content_hash / metadata
    assert "content_hash" not in data["items"][0]
    assert "metadata" not in data["items"][0]


async def test_list_memories_rejects_page_size_above_50(client_app) -> None:
    response = await _request(
        client_app,
        "GET",
        "/api/v2/memories",
        params={"page_size": 100},
    )
    # FastAPI Query(le=50) → 422 Unprocessable Entity
    assert response.status_code == 422


async def test_list_memories_rejects_query_too_short(client_app) -> None:
    response = await _request(
        client_app,
        "GET",
        "/api/v2/memories",
        params={"query": "a"},
    )
    assert response.status_code == 422


# --- get_memory -------------------------------------------------------------


async def test_get_memory_404_when_missing(
    client_app, mock_service: AsyncMock
) -> None:
    mock_service.get_memory = AsyncMock(return_value=None)

    response = await _request(client_app, "GET", "/api/v2/memories/mem-absent")

    assert response.status_code == 404
    body = response.json()
    # 全局异常处理器把 NotFoundError 转成结构化响应
    assert body["code"] == 404


async def test_get_memory_returns_detail_on_success(
    client_app, mock_service: AsyncMock
) -> None:
    chunk = _make_chunk(
        chunk_id="mem-42",
        content="hello",
        content_hash="hash-42",
        metadata={"tag": "demo"},
    )
    mock_service.get_memory = AsyncMock(return_value=chunk)

    response = await _request(client_app, "GET", "/api/v2/memories/mem-42")

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["id"] == "mem-42"
    assert data["content"] == "hello"
    assert data["content_hash"] == "hash-42"
    assert data["metadata"] == {"tag": "demo"}


# --- update_memory ----------------------------------------------------------


async def test_update_memory_rejects_empty_content_via_pydantic(
    client_app,
) -> None:
    response = await _request(
        client_app, "PATCH", "/api/v2/memories/mem-1", json={"content": ""}
    )
    # Pydantic min_length=1 → 422
    assert response.status_code == 422


async def test_update_memory_converts_value_error_to_400(
    client_app, mock_service: AsyncMock
) -> None:
    # service 对空白内容抛 ValueError；路由转为 400 BadRequestError
    mock_service.update_memory_content = AsyncMock(
        side_effect=ValueError("content must not be empty")
    )

    response = await _request(
        client_app,
        "PATCH",
        "/api/v2/memories/mem-1",
        json={"content": "   "},
    )

    assert response.status_code == 400
    assert response.json()["code"] == 400


async def test_update_memory_conflict_becomes_409(
    client_app, mock_service: AsyncMock
) -> None:
    mock_service.update_memory_content = AsyncMock(
        side_effect=ConflictError("相同内容的长期记忆已存在")
    )

    response = await _request(
        client_app,
        "PATCH",
        "/api/v2/memories/mem-1",
        json={"content": "new content"},
    )

    assert response.status_code == 409
    assert response.json()["code"] == 409


async def test_update_memory_404_when_missing(
    client_app, mock_service: AsyncMock
) -> None:
    mock_service.update_memory_content = AsyncMock(return_value=None)

    response = await _request(
        client_app,
        "PATCH",
        "/api/v2/memories/mem-absent",
        json={"content": "new content"},
    )

    assert response.status_code == 404


async def test_update_memory_200_on_success(
    client_app, mock_service: AsyncMock
) -> None:
    updated = _make_chunk(
        chunk_id="mem-1",
        content="updated content",
        content_hash="new-hash",
        metadata={"tag": "x"},
    )
    mock_service.update_memory_content = AsyncMock(return_value=updated)

    response = await _request(
        client_app,
        "PATCH",
        "/api/v2/memories/mem-1",
        json={"content": "updated content"},
    )

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["id"] == "mem-1"
    assert data["content"] == "updated content"
    assert data["content_hash"] == "new-hash"
    assert data["metadata"] == {"tag": "x"}
    mock_service.update_memory_content.assert_awaited_once_with(
        TEST_USER_ID_FIXED, "mem-1", "updated content"
    )


# --- PATCH pinned（pin/unpin 扩展）-------------------------------------------


async def test_patch_pinned_true_routes_to_update_memory_pinned(
    client_app, mock_service: AsyncMock
) -> None:
    """``{"pinned": true}`` → 走 update_memory_pinned，不碰 content 路径。"""
    import dataclasses

    chunk = _make_chunk(chunk_id="mem-1", content="profile")
    chunk = dataclasses.replace(chunk, category="user", pinned=True)
    mock_service.update_memory_pinned = AsyncMock(return_value=chunk)

    response = await _request(
        client_app, "PATCH", "/api/v2/memories/mem-1", json={"pinned": True},
    )

    assert response.status_code == 200
    assert response.json()["data"]["pinned"] is True
    mock_service.update_memory_pinned.assert_awaited_once_with(
        TEST_USER_ID_FIXED, "mem-1", True
    )
    # content 路径**绝不**被调（互斥契约）
    mock_service.update_memory_content.assert_not_called()


async def test_patch_pinned_false_unpin_routes_correctly(
    client_app, mock_service: AsyncMock
) -> None:
    """``{"pinned": false}`` 正常路由到 pinned 分支（False 也是 pinned 分支，
    schema exactly-one 断言靠的是"是否传了"，不是值）。"""
    import dataclasses

    chunk = _make_chunk(chunk_id="mem-1")
    chunk = dataclasses.replace(chunk, category="user", pinned=False)
    mock_service.update_memory_pinned = AsyncMock(return_value=chunk)

    response = await _request(
        client_app, "PATCH", "/api/v2/memories/mem-1", json={"pinned": False},
    )
    assert response.status_code == 200
    mock_service.update_memory_pinned.assert_awaited_once_with(
        TEST_USER_ID_FIXED, "mem-1", False
    )


async def test_patch_both_fields_rejected_by_schema_422(
    client_app, mock_service: AsyncMock
) -> None:
    """互斥契约：同时传 content + pinned → Pydantic validator 422。"""
    response = await _request(
        client_app,
        "PATCH",
        "/api/v2/memories/mem-1",
        json={"content": "x", "pinned": True},
    )
    assert response.status_code == 422
    # service 未被调（schema 前置拦截）
    mock_service.update_memory_content.assert_not_called()
    mock_service.update_memory_pinned.assert_not_called()


async def test_patch_empty_body_rejected_by_schema_422(
    client_app, mock_service: AsyncMock
) -> None:
    """互斥契约：两字段都不传 → 422。"""
    response = await _request(
        client_app, "PATCH", "/api/v2/memories/mem-1", json={},
    )
    assert response.status_code == 422


async def test_patch_pin_non_user_category_400(
    client_app, mock_service: AsyncMock
) -> None:
    """service 抛 BadRequestError（category!=user + pin=True）→ 400。"""
    from app.application.errors.exceptions import BadRequestError

    mock_service.update_memory_pinned = AsyncMock(
        side_effect=BadRequestError("pinned=True 仅允许 category='user'")
    )
    response = await _request(
        client_app, "PATCH", "/api/v2/memories/rule-1", json={"pinned": True},
    )
    assert response.status_code == 400


async def test_patch_pinned_404_when_chunk_missing(
    client_app, mock_service: AsyncMock
) -> None:
    """service 返 None（chunk 不存在）→ 404。"""
    mock_service.update_memory_pinned = AsyncMock(return_value=None)
    response = await _request(
        client_app, "PATCH", "/api/v2/memories/absent", json={"pinned": True},
    )
    assert response.status_code == 404


# --- delete_memory ----------------------------------------------------------


async def test_delete_memory_404_when_missing(
    client_app, mock_service: AsyncMock
) -> None:
    mock_service.delete_memory = AsyncMock(return_value=False)

    response = await _request(client_app, "DELETE", "/api/v2/memories/mem-absent")

    assert response.status_code == 404


async def test_delete_memory_200_on_success(
    client_app, mock_service: AsyncMock
) -> None:
    mock_service.delete_memory = AsyncMock(return_value=True)

    response = await _request(client_app, "DELETE", "/api/v2/memories/mem-1")

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["deleted_count"] == 1


# --- bulk_delete ------------------------------------------------------------


async def test_bulk_delete_rejects_empty_ids(client_app) -> None:
    response = await _request(
        client_app,
        "POST",
        "/api/v2/memories/bulk-delete",
        json={"ids": []},
    )
    # Pydantic Field(min_length=1) → 422
    assert response.status_code == 422


async def test_bulk_delete_success_returns_count(
    client_app, mock_service: AsyncMock
) -> None:
    mock_service.bulk_delete_memories = AsyncMock(return_value=2)

    response = await _request(
        client_app,
        "POST",
        "/api/v2/memories/bulk-delete",
        json={"ids": ["mem-1", "mem-2", "mem-3"]},
    )

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["deleted_count"] == 2
    mock_service.bulk_delete_memories.assert_awaited_once_with(
        TEST_USER_ID_FIXED, ["mem-1", "mem-2", "mem-3"]
    )


# --- delete_all -------------------------------------------------------------


async def test_delete_all_returns_count(
    client_app, mock_service: AsyncMock
) -> None:
    mock_service.delete_all_memories = AsyncMock(return_value=7)

    response = await _request(client_app, "POST", "/api/v2/memories/delete-all")

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["deleted_count"] == 7
    mock_service.delete_all_memories.assert_awaited_once_with(TEST_USER_ID_FIXED)


# --- delete_legacy (M3-A) ---------------------------------------------------


async def test_delete_legacy_returns_count(
    client_app, mock_service: AsyncMock
) -> None:
    """M3-A: DELETE /v2/memories/legacy → 清理旧 session_flush 数据。

    条件：source='session_flush' AND category IS NULL AND auto_promoted_at IS NULL。
    Service 负责过滤；route 只传递 user_id + 返 count。
    """
    mock_service.delete_legacy_memories = AsyncMock(return_value=5)

    response = await _request(client_app, "DELETE", "/api/v2/memories/legacy")

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["deleted_count"] == 5
    mock_service.delete_legacy_memories.assert_awaited_once_with(TEST_USER_ID_FIXED)


async def test_delete_legacy_returns_zero_when_empty(
    client_app, mock_service: AsyncMock
) -> None:
    """无 legacy 行 → 200 + deleted_count=0（不是 404）。"""
    mock_service.delete_legacy_memories = AsyncMock(return_value=0)

    response = await _request(client_app, "DELETE", "/api/v2/memories/legacy")

    assert response.status_code == 200
    assert response.json()["data"]["deleted_count"] == 0


async def test_delete_legacy_route_not_shadowed_by_chunk_id(
    client_app, mock_service: AsyncMock
) -> None:
    """确保 ``/legacy`` 的 DELETE 走 delete_legacy，而非 delete_memory({chunk_id='legacy'})。

    FastAPI 按注册顺序匹配；具体路径必须在 parametric ``/{chunk_id}`` 之前注册，
    否则 'legacy' 会被当成 chunk_id 参数。本测试钉死路由顺序。
    """
    mock_service.delete_legacy_memories = AsyncMock(return_value=0)
    mock_service.delete_memory = AsyncMock(return_value=False)

    response = await _request(client_app, "DELETE", "/api/v2/memories/legacy")

    assert response.status_code == 200
    mock_service.delete_legacy_memories.assert_awaited_once()
    # delete_memory 绝不应被调用——'legacy' 不是 chunk_id
    mock_service.delete_memory.assert_not_awaited()


# --- cleanup-config (M3-A codex fix P1) -------------------------------------


async def test_get_cleanup_config_exposes_rollout_at(
    client_app, monkeypatch
) -> None:
    """``GET /v2/memories/cleanup-config`` 返 settings 里的 rollout_at。

    前端在显示 "清理旧记忆" 对话框前拉此 endpoint：设了 → 显 cutoff，没设 →
    显警告。codex fix P1 的 UI 配套。
    """
    from datetime import datetime, timezone
    from types import SimpleNamespace

    from app.interfaces.endpoints import memory_routes

    cutoff = datetime(2026, 4, 1, 0, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(
        memory_routes, "get_settings",
        lambda: SimpleNamespace(memory_gate_rollout_at=cutoff),
    )

    response = await _request(client_app, "GET", "/api/v2/memories/cleanup-config")

    assert response.status_code == 200
    data = response.json()["data"]
    # ISO 串（FastAPI json encoder 把 datetime 序列化为 ISO 8601）
    assert data["rollout_at"].startswith("2026-04-01T00:00:00")


async def test_get_cleanup_config_null_when_not_set(
    client_app, monkeypatch
) -> None:
    """未配置 rollout_at → 返 null，前端 dialog 展示警告。"""
    from types import SimpleNamespace

    from app.interfaces.endpoints import memory_routes

    monkeypatch.setattr(
        memory_routes, "get_settings",
        lambda: SimpleNamespace(memory_gate_rollout_at=None),
    )

    response = await _request(client_app, "GET", "/api/v2/memories/cleanup-config")
    assert response.status_code == 200
    assert response.json()["data"]["rollout_at"] is None


async def test_get_cleanup_config_route_not_shadowed_by_chunk_id(
    client_app, monkeypatch
) -> None:
    """``/cleanup-config`` GET 必须在 ``/{chunk_id}`` GET 之前注册；否则会被
    当成 get_memory(chunk_id='cleanup-config') 走到 404 或 500。"""
    from types import SimpleNamespace

    from app.interfaces.endpoints import memory_routes

    monkeypatch.setattr(
        memory_routes, "get_settings",
        lambda: SimpleNamespace(memory_gate_rollout_at=None),
    )

    response = await _request(client_app, "GET", "/api/v2/memories/cleanup-config")
    assert response.status_code == 200
    # 若被 /{chunk_id} 吞了，get_memory 会被调 → 这里应该没 await
    # 注意：mock_service 默认 get_memory 返 None → NotFoundError 404
    assert "rollout_at" in response.json()["data"]


# --- reindex_memory (post-M3 Option A) --------------------------------------


async def test_reindex_returns_200_with_reindexed_fields(
    client_app, mock_service: AsyncMock
) -> None:
    """happy path：service 返回 ReindexResult → 200 + reindexed_fields。"""
    from app.application.services.memory_management_service import ReindexResult

    mock_service.reindex_memory = AsyncMock(
        return_value=ReindexResult(
            reindexed_fields=["content"],
            warnings=[],
            fs_synced=True,
        )
    )

    response = await _request(
        client_app, "POST", "/api/v2/memories/mem-123/reindex"
    )

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["reindexed_fields"] == ["content"]
    assert data["warnings"] == []
    assert data["fs_synced"] is True
    mock_service.reindex_memory.assert_awaited_once_with(TEST_USER_ID_FIXED, "mem-123")


async def test_reindex_noop_returns_200_with_empty_fields(
    client_app, mock_service: AsyncMock
) -> None:
    """no-op 场景（盘与 DB 一致）也返 200——幂等契约。"""
    from app.application.services.memory_management_service import ReindexResult

    mock_service.reindex_memory = AsyncMock(
        return_value=ReindexResult(
            reindexed_fields=[], warnings=[], fs_synced=True
        )
    )

    response = await _request(
        client_app, "POST", "/api/v2/memories/mem-noop/reindex"
    )
    assert response.status_code == 200
    assert response.json()["data"]["reindexed_fields"] == []


async def test_reindex_surfaces_warnings(
    client_app, mock_service: AsyncMock
) -> None:
    """service 返的 warnings 原样透给前端。Option A 的核心 UX 契约。"""
    from app.application.services.memory_management_service import ReindexResult

    warnings = [
        "frontmatter.category='rule' 与 DB 'user' 不一致；reindex 不支持",
        "frontmatter.pinned=True 与 DB False 不一致",
    ]
    mock_service.reindex_memory = AsyncMock(
        return_value=ReindexResult(
            reindexed_fields=["content"], warnings=warnings, fs_synced=True
        )
    )

    response = await _request(
        client_app, "POST", "/api/v2/memories/mem-w/reindex"
    )
    assert response.status_code == 200
    assert response.json()["data"]["warnings"] == warnings


async def test_reindex_404_when_not_found(
    client_app, mock_service: AsyncMock
) -> None:
    """service 抛 NotFoundError → 404。"""
    from app.application.errors.exceptions import NotFoundError

    mock_service.reindex_memory = AsyncMock(
        side_effect=NotFoundError("记忆不存在")
    )

    response = await _request(
        client_app, "POST", "/api/v2/memories/missing/reindex"
    )
    assert response.status_code == 404


async def test_reindex_409_when_file_missing(
    client_app, mock_service: AsyncMock
) -> None:
    """file 不存在 / id 不匹配 → 409（service 抛 ConflictError）。"""
    mock_service.reindex_memory = AsyncMock(
        side_effect=ConflictError("磁盘上不存在此记忆文件")
    )

    response = await _request(
        client_app, "POST", "/api/v2/memories/ghost/reindex"
    )
    assert response.status_code == 409


async def test_reindex_400_on_parse_failure(
    client_app, mock_service: AsyncMock
) -> None:
    """frontmatter YAML 解析失败 → 400。"""
    from app.application.errors.exceptions import BadRequestError

    mock_service.reindex_memory = AsyncMock(
        side_effect=BadRequestError("memory 文件 frontmatter 解析失败：...")
    )

    response = await _request(
        client_app, "POST", "/api/v2/memories/bad-yaml/reindex"
    )
    assert response.status_code == 400


async def test_reindex_400_on_empty_body(
    client_app, mock_service: AsyncMock
) -> None:
    """codex round-4 P2：hand-edit 删光正文 → 400，与 create/update 契约对齐。"""
    from app.application.errors.exceptions import BadRequestError

    mock_service.reindex_memory = AsyncMock(
        side_effect=BadRequestError("reindex 结果 body 为空；与 create/update 契约对齐")
    )
    response = await _request(
        client_app, "POST", "/api/v2/memories/empty/reindex"
    )
    assert response.status_code == 400


async def test_reindex_503_when_file_store_not_configured(
    client_app, mock_service: AsyncMock
) -> None:
    """codex round-4 P2：deployment 未配置 file_store → 503 而非 400。"""
    from app.application.errors.exceptions import ServiceUnavailableError

    mock_service.reindex_memory = AsyncMock(
        side_effect=ServiceUnavailableError(
            "reindex 需要 file_store 后端；当前 deployment 运行在 DB-only 模式"
        )
    )
    response = await _request(
        client_app, "POST", "/api/v2/memories/any/reindex"
    )
    assert response.status_code == 503


# --- create_memory (PR-2) ---------------------------------------------------


async def test_create_memory_returns_201(
    client_app, mock_service: AsyncMock
) -> None:
    """POST /v2/memories 成功 → 201 + MemoryDetail payload（含 M1 新字段）。"""
    import dataclasses

    created = _make_chunk(
        chunk_id="mem-new",
        user_id=TEST_USER_ID_FIXED,
        content="user prefers dark mode",
        source="manual",
    )
    created = dataclasses.replace(
        created, category="user", pinned=True, fs_synced=True
    )
    mock_service.create_memory = AsyncMock(return_value=created)

    response = await _request(
        client_app,
        "POST",
        "/api/v2/memories",
        json={"content": "user prefers dark mode", "category": "user", "pinned": True},
    )

    assert response.status_code == 201
    data = response.json()["data"]
    assert data["id"] == "mem-new"
    assert data["category"] == "user"
    assert data["pinned"] is True
    assert data["fs_synced"] is True
    mock_service.create_memory.assert_awaited_once()
    kwargs = mock_service.create_memory.call_args.kwargs
    assert kwargs["category"] == "user"
    assert kwargs["pinned"] is True
    assert kwargs["source"] == "manual"
    # tags 默认没传 → None（service 侧可选参数）
    assert kwargs.get("tags") is None


async def test_create_memory_with_tags_passes_cleaned_list(
    client_app, mock_service: AsyncMock
) -> None:
    """PR-7 tags 路径：输入 strip/dedupe 后交给 service。"""
    import dataclasses

    created = _make_chunk(
        chunk_id="mem-tagged",
        user_id=TEST_USER_ID_FIXED,
        content="Go preference",
        source="manual",
    )
    created = dataclasses.replace(created, category="user", fs_synced=True)
    mock_service.create_memory = AsyncMock(return_value=created)

    response = await _request(
        client_app,
        "POST",
        "/api/v2/memories",
        json={
            "content": "Go preference",
            "category": "user",
            "tags": [" Go ", "", "backend", "Go"],  # 冗余 + 空 + 重复
        },
    )
    assert response.status_code == 201
    kwargs = mock_service.create_memory.call_args.kwargs
    # pydantic _normalize_tags 清洗后：strip + 丢空 + 保序去重
    assert kwargs["tags"] == ["Go", "backend"]


async def test_create_memory_tag_too_long_422(
    client_app, mock_service: AsyncMock
) -> None:
    """单个 tag 超过 64 字符 → pydantic 422，不到 service。"""
    mock_service.create_memory = AsyncMock()
    response = await _request(
        client_app,
        "POST",
        "/api/v2/memories",
        json={
            "content": "x",
            "category": "fact",
            "tags": ["x" * 65],
        },
    )
    assert response.status_code == 422
    mock_service.create_memory.assert_not_called()


async def test_create_memory_too_many_tags_422(
    client_app, mock_service: AsyncMock
) -> None:
    """tags 数量超过 20 → pydantic 422。"""
    mock_service.create_memory = AsyncMock()
    response = await _request(
        client_app,
        "POST",
        "/api/v2/memories",
        json={
            "content": "x",
            "category": "fact",
            "tags": [f"t{i}" for i in range(21)],
        },
    )
    assert response.status_code == 422
    mock_service.create_memory.assert_not_called()


async def test_create_memory_pinned_rule_rejected_by_schema(
    client_app, mock_service: AsyncMock
) -> None:
    """pinned=True + category=rule → pydantic 422，不到 service。"""
    # 显式把 create_memory attr 设上，保证 AsyncMock 可追踪其调用状态。
    mock_service.create_memory = AsyncMock()

    response = await _request(
        client_app,
        "POST",
        "/api/v2/memories",
        json={"content": "x", "category": "rule", "pinned": True},
    )
    assert response.status_code == 422
    mock_service.create_memory.assert_not_called()


async def test_create_memory_whitespace_only_content_400(
    client_app, mock_service: AsyncMock
) -> None:
    """单个空格通过 pydantic min_length=1，服务层 strip 后报 ValueError → 400。"""
    mock_service.create_memory = AsyncMock(
        side_effect=ValueError("content must not be empty")
    )
    response = await _request(
        client_app,
        "POST",
        "/api/v2/memories",
        json={"content": " ", "category": "user"},
    )
    assert response.status_code == 400


async def test_create_memory_invalid_category_422(
    client_app, mock_service: AsyncMock
) -> None:
    """非枚举 category → 422（Literal 校验在 pydantic 层）。"""
    response = await _request(
        client_app,
        "POST",
        "/api/v2/memories",
        json={"content": "x", "category": "nope"},
    )
    assert response.status_code == 422


async def test_create_memory_quota_exceeded_maps_to_429(
    client_app, mock_service: AsyncMock
) -> None:
    """service 抛 QuotaExceededError → 全局 handler 转 429。"""
    from app.application.errors.exceptions import QuotaExceededError

    mock_service.create_memory = AsyncMock(
        side_effect=QuotaExceededError(
            msg="每日 memory 写入上限（500）已达到",
            limit=500,
            bucket="memory_user_daily",
        )
    )
    response = await _request(
        client_app,
        "POST",
        "/api/v2/memories",
        json={"content": "hit cap", "category": "user"},
    )
    assert response.status_code == 429


async def test_create_memory_conflict_maps_to_409(
    client_app, mock_service: AsyncMock
) -> None:
    """service 抛 ConflictError（同 hash 已存在）→ 409。"""
    from app.application.errors.exceptions import ConflictError

    mock_service.create_memory = AsyncMock(
        side_effect=ConflictError("相同内容的长期记忆已存在")
    )
    response = await _request(
        client_app,
        "POST",
        "/api/v2/memories",
        json={"content": "dup", "category": "fact"},
    )
    assert response.status_code == 409


async def test_list_memories_forwards_category_filter(
    client_app, mock_service: AsyncMock
) -> None:
    """?category=user → service.list_memories 收到 category="user"。"""
    response = await _request(
        client_app, "GET", "/api/v2/memories", params={"category": "user"}
    )
    assert response.status_code == 200
    kwargs = mock_service.list_memories.call_args.kwargs
    assert kwargs["category"] == "user"


async def test_list_memories_invalid_category_returns_422(
    client_app, mock_service: AsyncMock
) -> None:
    """GET ?category=garbage → 422（与 POST path 的 Literal 校验对齐）。"""
    response = await _request(
        client_app, "GET", "/api/v2/memories", params={"category": "garbage"}
    )
    assert response.status_code == 422
    mock_service.list_memories.assert_not_called()


async def test_list_memories_forwards_auto_promoted_after(
    client_app, mock_service: AsyncMock
) -> None:
    """design doc §777：?source=session_flush&auto_promoted_after=<iso> →
    service.list_memories 收到等价 datetime。审阅最近 N 天 LLM gate 收录路径。"""
    from datetime import datetime, timezone

    iso = "2026-04-12T00:00:00+00:00"
    response = await _request(
        client_app,
        "GET",
        "/api/v2/memories",
        params={"source": "session_flush", "auto_promoted_after": iso},
    )
    assert response.status_code == 200
    kwargs = mock_service.list_memories.call_args.kwargs
    assert kwargs["source"] == "session_flush"
    assert kwargs["auto_promoted_after"] == datetime(
        2026, 4, 12, tzinfo=timezone.utc
    )


async def test_list_memories_invalid_auto_promoted_after_returns_422(
    client_app, mock_service: AsyncMock
) -> None:
    """非 datetime 字符串 → FastAPI 422，service 不被调用。"""
    response = await _request(
        client_app,
        "GET",
        "/api/v2/memories",
        params={"auto_promoted_after": "not-a-date"},
    )
    assert response.status_code == 422
    mock_service.list_memories.assert_not_called()


async def test_list_memories_naive_auto_promoted_after_returns_422(
    client_app, mock_service: AsyncMock
) -> None:
    """codex round-3 [P2]: naive datetime（无 timezone 后缀）必须 422 拒绝，
    避免不同部署节点对同一 cutoff 字符串按 host TZ 解释而产生不同结果。
    AwareDatetime 强制要求 ISO 8601 带时区。"""
    response = await _request(
        client_app,
        "GET",
        "/api/v2/memories",
        params={"auto_promoted_after": "2026-04-12T00:00:00"},  # 缺 TZ 后缀
    )
    assert response.status_code == 422
    mock_service.list_memories.assert_not_called()
