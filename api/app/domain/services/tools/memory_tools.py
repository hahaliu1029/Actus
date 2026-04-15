"""Memory search and retrieval tools for Agent execution."""
from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.tools import BaseTool, tool as lc_tool
from pydantic import BaseModel, Field

from app.domain.external.embedding_provider import EmbeddingUnavailableError
from app.domain.services.memory_ranker import rank_memory_results
from app.domain.services.tools.tool_source_resolver import (
    annotate_and_register_tool_source,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from app.domain.external.embedding_provider import EmbeddingProvider
    from app.domain.repositories.memory_chunk_repository import MemoryChunkRepository


class MemorySearchInput(BaseModel):
    query: str = Field(description="搜索查询，自然语言描述要查找的内容")
    max_results: int = Field(default=5, description="最大返回条数", ge=1, le=20)


def create_memory_tools(
    embedding_provider: EmbeddingProvider,
    session_factory: async_sessionmaker[AsyncSession],
    repo_factory: Callable[[AsyncSession], MemoryChunkRepository],
    user_id: str,
    half_life_days: int = 30,
    mmr_lambda: float = 0.7,
) -> list[BaseTool]:
    """Create memory search and get tools with closed-over dependencies."""
    CANDIDATE_MULTIPLIER = 3

    @lc_tool(args_schema=MemorySearchInput)
    async def memory_search(query: str, max_results: int = 5) -> str:
        """搜索历史记忆。当需要回忆之前的对话、查找历史上下文时使用。"""
        try:
            vectors = await embedding_provider.embed([query])
            embedding = vectors[0]
        except EmbeddingUnavailableError as e:
            return f"记忆检索暂不可用（embedding 服务异常: {e}）"
        except Exception as e:
            return f"记忆检索失败（{type(e).__name__}: {e}）"

        async with session_factory() as session:
            repo = repo_factory(session)
            chunks = await repo.search_by_vector(
                user_id=user_id,
                embedding=embedding,
                top_k=max_results * CANDIDATE_MULTIPLIER,
            )

        if not chunks:
            return "未找到相关记忆。"

        chunks = rank_memory_results(
            chunks,
            query_embedding=embedding,
            half_life_days=half_life_days,
            mmr_lambda=mmr_lambda,
            top_k=max_results,
        )

        if not chunks:
            return "未找到相关记忆。"

        lines = []
        for i, c in enumerate(chunks, 1):
            preview = c.content[:500] + ("..." if len(c.content) > 500 else "")
            lines.append(f"[{i}] (id: {c.id}, source: {c.source})\n{preview}")
        return "\n\n".join(lines)

    @lc_tool
    async def memory_get(chunk_id: str) -> str:
        """读取记忆片段完整内容。在 memory_search 后用 ID 获取完整文本。"""
        async with session_factory() as session:
            repo = repo_factory(session)
            chunk = await repo.get_by_id(chunk_id, user_id=user_id)

        if not chunk:
            return f"记忆片段 {chunk_id} 不存在。"
        return chunk.content

    # Optional safety net (main error handling is in react_graph.py tool_node)
    memory_search.handle_tool_error = True
    memory_get.handle_tool_error = True

    tools = [memory_search, memory_get]
    for t in tools:
        annotate_and_register_tool_source(t, source="native", category="memory")
    return tools
