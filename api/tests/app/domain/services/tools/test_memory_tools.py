"""Unit tests for memory_search and memory_get tools."""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.domain.external.embedding_provider import EmbeddingUnavailableError
from app.domain.models.memory_chunk import MemoryChunk

pytestmark = pytest.mark.anyio


def _make_chunk(**overrides) -> MemoryChunk:
    defaults = dict(
        id="chunk-1",
        user_id="user-1",
        content="test content for memory chunk",
        content_hash="hash1",
        source="session_flush",
        metadata={},
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        updated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    defaults.update(overrides)
    return MemoryChunk(**defaults)


def _make_tools(**overrides):
    """Create memory tools with mock dependencies."""
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

    tools = create_memory_tools(
        embedding_provider=provider,
        session_factory=session_factory,
        repo_factory=repo_factory,
        user_id=overrides.get("user_id", "user-1"),
    )
    return tools, mock_repo, provider


class TestCreateMemoryTools:
    def test_returns_two_tools(self) -> None:
        tools, _, _ = _make_tools()
        assert len(tools) == 2
        names = {t.name for t in tools}
        assert "memory_search" in names
        assert "memory_get" in names


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
        assert call_kwargs["top_k"] == 3

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
        tools, _, _ = _make_tools(_mock_repo=mock_repo, user_id="user-42")
        get = next(t for t in tools if t.name == "memory_get")

        await get.ainvoke({"chunk_id": "c1"})
        mock_repo.get_by_id.assert_awaited_once_with("c1", user_id="user-42")
