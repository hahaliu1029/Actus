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
    svc.delete_memory = AsyncMock(return_value=False)
    svc.bulk_delete_memories = AsyncMock(return_value=0)
    svc.delete_all_memories = AsyncMock(return_value=0)
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
