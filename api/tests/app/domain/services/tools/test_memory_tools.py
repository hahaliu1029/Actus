"""Unit tests for memory_search, memory_get, and memory_save tools."""
from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.application.errors.exceptions import ConflictError, QuotaExceededError
from app.domain.external.embedding_provider import EmbeddingUnavailableError
from app.domain.models.memory_chunk import MemoryChunk

from tests.conftest import TEST_OTHER_USER_ID_FIXED, TEST_USER_ID_FIXED

pytestmark = pytest.mark.anyio

_TEST_SESSION_ID = "sess-test-01"


def _make_session_redis(*, incr_result: int = 1) -> AsyncMock:
    """Mock redis with pipeline(transaction=True) CM for session_save counter."""
    redis = AsyncMock()
    pipe = MagicMock()
    pipe.incr = MagicMock(return_value=pipe)
    pipe.expire = MagicMock(return_value=pipe)
    pipe.execute = AsyncMock(return_value=[incr_result, True])

    @asynccontextmanager
    async def pipeline_cm(transaction: bool = True):
        yield pipe

    redis.pipeline = pipeline_cm
    redis._pipe = pipe
    return redis


def _make_chunk(**overrides) -> MemoryChunk:
    defaults = dict(
        id="chunk-1",
        user_id=TEST_USER_ID_FIXED,
        content="test content for memory chunk",
        content_hash="hash1",
        source="session_flush",
        metadata={},
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        updated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        embedding=(0.1, 0.2),  # C7: ranker needs non-None embedding
    )
    defaults.update(overrides)
    return MemoryChunk(**defaults)


def _make_tools(**overrides):
    """Create memory tools with mock dependencies.

    Pass ``session_id``/``memory_write_service``/``session_redis`` together to
    also build ``memory_save``. Omit any of them to keep the legacy 2-tool
    shape used by ``test_returns_two_tools_when_save_deps_missing``.
    """
    from app.domain.services.tools.memory_tools import create_memory_tools

    provider = overrides.get("embedding_provider", AsyncMock())
    if not hasattr(provider, "embed") or not isinstance(provider.embed, AsyncMock):
        provider.embed = AsyncMock(return_value=[[0.1, 0.2]])

    mock_repo = overrides.get("_mock_repo", AsyncMock())
    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    session_factory = MagicMock(return_value=mock_session)
    repo_factory = MagicMock(return_value=mock_repo)

    kwargs = dict(
        embedding_provider=provider,
        session_factory=session_factory,
        repo_factory=repo_factory,
        user_id=overrides.get("user_id", TEST_USER_ID_FIXED),
        half_life_days=overrides.get("half_life_days", 30),
        mmr_lambda=overrides.get("mmr_lambda", 0.7),
    )
    # memory_save 仅在 3 个依赖全部提供时才构建
    for k in ("session_id", "memory_write_service", "session_redis", "session_save_cap"):
        if k in overrides:
            kwargs[k] = overrides[k]

    tools = create_memory_tools(**kwargs)
    return tools, mock_repo, provider


def _make_save_tools(
    *,
    memory_write_service: AsyncMock | None = None,
    session_redis: AsyncMock | None = None,
    session_save_cap: int = 20,
    user_id: str = TEST_USER_ID_FIXED,
):
    """Build all 3 tools (search/get/save). Returns (tools, service, redis)."""
    if memory_write_service is None:
        memory_write_service = AsyncMock()
        memory_write_service.create_memory = AsyncMock(
            return_value=_make_chunk(id="saved-1", content="saved content"),
        )
    if session_redis is None:
        session_redis = _make_session_redis(incr_result=1)

    tools, _, _ = _make_tools(
        user_id=user_id,
        session_id=_TEST_SESSION_ID,
        memory_write_service=memory_write_service,
        session_redis=session_redis,
        session_save_cap=session_save_cap,
    )
    return tools, memory_write_service, session_redis


class TestCreateMemoryTools:
    def test_returns_two_tools_when_save_deps_missing(self) -> None:
        """Backward compat: old call sites without session_id get 2 tools only."""
        tools, _, _ = _make_tools()
        assert len(tools) == 2
        names = {t.name for t in tools}
        assert "memory_search" in names
        assert "memory_get" in names
        assert "memory_save" not in names

    def test_returns_three_tools_when_save_deps_provided(self) -> None:
        tools, _, _ = _make_save_tools()
        assert len(tools) == 3
        names = {t.name for t in tools}
        assert names == {"memory_search", "memory_get", "memory_save"}

    def test_save_missing_session_id_still_two_tools(self) -> None:
        """session_id 单独缺失也不该半拉子构建 memory_save。"""
        tools, _, _ = _make_tools(
            memory_write_service=AsyncMock(),
            session_redis=_make_session_redis(),
            # session_id intentionally omitted
        )
        assert {t.name for t in tools} == {"memory_search", "memory_get"}


class TestMemorySearch:
    async def test_success_returns_formatted_results(self) -> None:
        mock_repo = AsyncMock()
        mock_repo.search_by_vector.return_value = [
            _make_chunk(id="c1", content="first result"),
            _make_chunk(id="c2", content="second result"),
        ]
        tools, _, _ = _make_tools(_mock_repo=mock_repo)
        search = next(t for t in tools if t.name == "memory_search")

        result = await search.ainvoke({"query": "test query", "max_results": 5})

        assert "[1] (id: c1" in result
        assert "first result" in result
        assert "[2] (id: c2" in result

    async def test_empty_results(self) -> None:
        mock_repo = AsyncMock()
        mock_repo.search_by_vector.return_value = []
        tools, _, _ = _make_tools(_mock_repo=mock_repo)
        search = next(t for t in tools if t.name == "memory_search")

        result = await search.ainvoke({"query": "nothing", "max_results": 5})
        assert "未找到相关记忆" in result

    async def test_embedding_unavailable_degrades(self) -> None:
        provider = AsyncMock()
        provider.embed.side_effect = EmbeddingUnavailableError("disabled")
        tools, _, _ = _make_tools(embedding_provider=provider)
        search = next(t for t in tools if t.name == "memory_search")

        result = await search.ainvoke({"query": "test", "max_results": 5})
        assert "暂不可用" in result

    async def test_max_results_passed_to_repo(self) -> None:
        mock_repo = AsyncMock()
        mock_repo.search_by_vector.return_value = []
        tools, _, _ = _make_tools(_mock_repo=mock_repo)
        search = next(t for t in tools if t.name == "memory_search")

        await search.ainvoke({"query": "test", "max_results": 3})
        mock_repo.search_by_vector.assert_awaited_once()
        call_kwargs = mock_repo.search_by_vector.call_args.kwargs
        assert call_kwargs["top_k"] == 9  # 3 * CANDIDATE_MULTIPLIER

    async def test_3x_overfetch_with_max_results_5(self) -> None:
        """max_results=5 should query repo with top_k=15."""
        mock_repo = AsyncMock()
        mock_repo.search_by_vector.return_value = []
        tools, _, _ = _make_tools(_mock_repo=mock_repo)
        search = next(t for t in tools if t.name == "memory_search")

        await search.ainvoke({"query": "test", "max_results": 5})
        call_kwargs = mock_repo.search_by_vector.call_args.kwargs
        assert call_kwargs["top_k"] == 15

    async def test_search_calls_ranker(self) -> None:
        """memory_search should call rank_memory_results with correct args."""
        mock_repo = AsyncMock()
        chunk = _make_chunk(embedding=(0.1, 0.2))
        mock_repo.search_by_vector.return_value = [chunk]
        tools, _, provider = _make_tools(
            _mock_repo=mock_repo,
            half_life_days=60,
            mmr_lambda=0.5,
        )
        search = next(t for t in tools if t.name == "memory_search")

        with patch(
            "app.domain.services.tools.memory_tools.rank_memory_results",
            return_value=[chunk],
        ) as mock_ranker:
            await search.ainvoke({"query": "test", "max_results": 2})
            mock_ranker.assert_called_once()
            call_kwargs = mock_ranker.call_args.kwargs
            assert call_kwargs["half_life_days"] == 60
            assert call_kwargs["mmr_lambda"] == 0.5
            assert call_kwargs["top_k"] == 2

    async def test_long_content_truncated_with_ellipsis(self) -> None:
        long_content = "x" * 600
        mock_repo = AsyncMock()
        mock_repo.search_by_vector.return_value = [_make_chunk(content=long_content)]
        tools, _, _ = _make_tools(_mock_repo=mock_repo)
        search = next(t for t in tools if t.name == "memory_search")

        result = await search.ainvoke({"query": "test", "max_results": 5})
        assert "..." in result
        assert len(result) < 600


class TestMemoryGet:
    async def test_success_returns_full_content(self) -> None:
        mock_repo = AsyncMock()
        mock_repo.get_by_id.return_value = _make_chunk(content="full content here")
        tools, _, _ = _make_tools(_mock_repo=mock_repo)
        get = next(t for t in tools if t.name == "memory_get")

        result = await get.ainvoke({"chunk_id": "chunk-1"})
        assert result == "full content here"

    async def test_not_found_returns_message(self) -> None:
        mock_repo = AsyncMock()
        mock_repo.get_by_id.return_value = None
        tools, _, _ = _make_tools(_mock_repo=mock_repo)
        get = next(t for t in tools if t.name == "memory_get")

        result = await get.ainvoke({"chunk_id": "nonexistent"})
        assert "不存在" in result

    async def test_passes_user_id_to_repo(self) -> None:
        mock_repo = AsyncMock()
        mock_repo.get_by_id.return_value = None
        tools, _, _ = _make_tools(_mock_repo=mock_repo, user_id=TEST_OTHER_USER_ID_FIXED)
        get = next(t for t in tools if t.name == "memory_get")

        await get.ainvoke({"chunk_id": "c1"})
        mock_repo.get_by_id.assert_awaited_once_with("c1", user_id=TEST_OTHER_USER_ID_FIXED)


class TestMemorySave:
    async def test_success_calls_service_with_memory_save_source(self) -> None:
        tools, service, redis = _make_save_tools()
        save = next(t for t in tools if t.name == "memory_save")

        result = await save.ainvoke(
            {"content": "remember this fact", "category": "fact"}
        )

        service.create_memory.assert_awaited_once_with(
            user_id=TEST_USER_ID_FIXED,
            content="remember this fact",
            category="fact",
            source="memory_save",
            session_id=_TEST_SESSION_ID,
        )
        assert "已保存" in result or "saved" in result.lower()
        # session counter 自增过
        redis._pipe.incr.assert_called_once()

    async def test_session_cap_exceeded_returns_friendly_and_skips_service(self) -> None:
        redis = _make_session_redis(incr_result=21)  # over cap=20
        service = AsyncMock()
        service.create_memory = AsyncMock(
            return_value=_make_chunk(id="nope", content=""),
        )
        tools, service, _ = _make_save_tools(
            memory_write_service=service,
            session_redis=redis,
            session_save_cap=20,
        )
        save = next(t for t in tools if t.name == "memory_save")

        result = await save.ainvoke(
            {"content": "hit cap content", "category": "fact"}
        )

        assert "上限" in result or "cap" in result.lower()
        service.create_memory.assert_not_awaited()

    async def test_duplicate_content_refunds_session_counter(self) -> None:
        service = AsyncMock()
        service.create_memory = AsyncMock(
            side_effect=ConflictError(msg="dup hash"),
        )
        redis = _make_session_redis(incr_result=1)
        tools, _, redis = _make_save_tools(
            memory_write_service=service, session_redis=redis
        )
        save = next(t for t in tools if t.name == "memory_save")

        result = await save.ainvoke(
            {"content": "already saved", "category": "fact"}
        )

        assert "已经保存" in result or "跳过" in result or "重复" in result
        # refund DECR 被调用（还原 session counter）
        redis.decr.assert_awaited_once()

    async def test_user_daily_quota_exceeded_refunds_and_returns_friendly(self) -> None:
        service = AsyncMock()
        service.create_memory = AsyncMock(
            side_effect=QuotaExceededError(
                msg="daily cap hit", limit=500, bucket="memory_user_daily"
            )
        )
        redis = _make_session_redis(incr_result=1)
        tools, _, redis = _make_save_tools(
            memory_write_service=service, session_redis=redis
        )
        save = next(t for t in tools if t.name == "memory_save")

        result = await save.ainvoke(
            {"content": "over daily cap", "category": "fact"}
        )

        assert "今日" in result or "daily" in result.lower()
        redis.decr.assert_awaited_once()

    async def test_unexpected_exception_refunds_and_returns_error_message(
        self,
    ) -> None:
        service = AsyncMock()
        service.create_memory = AsyncMock(side_effect=RuntimeError("db down"))
        redis = _make_session_redis(incr_result=1)
        tools, _, redis = _make_save_tools(
            memory_write_service=service, session_redis=redis
        )
        save = next(t for t in tools if t.name == "memory_save")

        result = await save.ainvoke({"content": "x", "category": "fact"})

        assert "失败" in result or "error" in result.lower()
        redis.decr.assert_awaited_once()

    async def test_empty_content_rejected_by_schema(self) -> None:
        """Pydantic schema 应该在执行前拦截空 content。"""
        tools, service, _ = _make_save_tools()
        save = next(t for t in tools if t.name == "memory_save")

        # LangChain tool invoke with invalid args — should raise ValidationError
        # or be surfaced as a tool error. Either way, service must not be called.
        from pydantic import ValidationError
        with pytest.raises((ValidationError, ValueError, Exception)):
            await save.ainvoke({"content": "", "category": "fact"})

        service.create_memory.assert_not_awaited()

    async def test_invalid_category_rejected_by_schema(self) -> None:
        tools, service, _ = _make_save_tools()
        save = next(t for t in tools if t.name == "memory_save")

        from pydantic import ValidationError
        with pytest.raises((ValidationError, ValueError, Exception)):
            await save.ainvoke(
                {"content": "ok", "category": "garbage_category"}
            )

        service.create_memory.assert_not_awaited()

    async def test_fail_open_then_conflict_does_not_refund(self) -> None:
        """P2 regression: ``check_and_increment_session_save`` fail-opens →
        returns 0 (no INCR happened). If create_memory then raises, we must
        NOT DECR — otherwise we'd turn a non-existent key into a TTL-less
        negative counter and long-term under-count this session's writes.
        """
        # incr_result=0 mimics the fail-open path (pipeline.execute errored,
        # helper swallowed it and returned 0).
        redis = _make_session_redis(incr_result=0)
        service = AsyncMock()
        service.create_memory = AsyncMock(side_effect=ConflictError(msg="dup"))

        tools, _, redis = _make_save_tools(
            memory_write_service=service, session_redis=redis
        )
        save = next(t for t in tools if t.name == "memory_save")

        result = await save.ainvoke({"content": "abc", "category": "fact"})

        assert "已经保存" in result or "重复" in result or "跳过" in result
        redis.decr.assert_not_awaited()

    async def test_fail_open_then_unexpected_error_does_not_refund(self) -> None:
        redis = _make_session_redis(incr_result=0)
        service = AsyncMock()
        service.create_memory = AsyncMock(side_effect=RuntimeError("kaboom"))

        tools, _, redis = _make_save_tools(
            memory_write_service=service, session_redis=redis
        )
        save = next(t for t in tools if t.name == "memory_save")

        result = await save.ainvoke({"content": "abc", "category": "fact"})

        assert "失败" in result
        redis.decr.assert_not_awaited()

    async def test_tool_source_registered(self) -> None:
        """memory_save 必须通过 annotate_and_register_tool_source 注册。"""
        from app.domain.services.tools.tool_source_resolver import (
            resolve_tool_source,
        )

        tools, _, _ = _make_save_tools()
        save = next(t for t in tools if t.name == "memory_save")

        ts = resolve_tool_source(save.name)
        assert ts.source == "native"
        assert ts.category == "memory"
        assert ts.canonical_name == "memory_save"
