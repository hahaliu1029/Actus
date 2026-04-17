"""Memory search / get / save tools for Agent execution.

Three tools share one factory so the closure can capture ``user_id``,
``session_id``, and the service dependencies — Agent-visible tools can't
carry auth context themselves (LLM picks arg values, so anything passed
through a tool arg is untrusted).

``memory_save`` is optional: only built when ``session_id``,
``memory_write_service`` and ``session_redis`` are all provided. Call sites
that don't have those (e.g. legacy unit tests, planner contexts before the
session wiring lands) keep the 2-tool shape.

Reverse-dep note: this domain module imports from ``application.services``
(``memory_session_limits``). Justified precedent: the same shape appears in
``agent_task_runner._handle_tool_event`` per the R4 CS3 migration; see
CLAUDE.md "Clean Architecture 已知例外".
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal, Protocol

from langchain_core.tools import BaseTool, tool as lc_tool
from pydantic import BaseModel, Field

from app.application.errors.exceptions import ConflictError, QuotaExceededError
from app.application.services.memory_session_limits import (
    check_and_increment_session_save,
    refund_session_save,
)
from app.domain.external.embedding_provider import EmbeddingUnavailableError
from app.domain.models.tool_result import AllowSuccess, ToolOutcome
from app.domain.services.memory_ranker import rank_memory_results
from app.domain.services.tools.tool_source_resolver import (
    annotate_and_register_tool_source,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from redis.asyncio import Redis
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from app.domain.external.embedding_provider import EmbeddingProvider
    from app.domain.models.memory_chunk import MemoryChunk
    from app.domain.repositories.memory_chunk_repository import MemoryChunkRepository

logger = logging.getLogger(__name__)

_MEMORY_CATEGORY = Literal["user", "rule", "fact"]


class MemorySearchInput(BaseModel):
    query: str = Field(description="搜索查询，自然语言描述要查找的内容")
    max_results: int = Field(default=5, description="最大返回条数", ge=1, le=20)


class MemorySaveInput(BaseModel):
    content: str = Field(
        description="要保存的记忆内容。必须是具体、可复用的事实/偏好/规则；"
        "避免保存无意义的短语或对话片段。",
        min_length=1,
        max_length=50000,
    )
    category: _MEMORY_CATEGORY = Field(
        description=(
            "记忆分类：user（用户偏好/身份/工作习惯）、"
            "rule（操作约束/永久性规则）、"
            "fact（可验证的世界事实或项目事实）。"
        ),
    )


class _MemoryWriteService(Protocol):
    """MemoryManagementService 的最小写入契约——duck-typed。

    仅接受 kwargs，避免 service 实现改位置参数顺序把 tool 吞掉。
    """

    async def create_memory(
        self,
        user_id: str,
        content: str,
        category: str,
        *,
        source: str = "manual",
        pinned: bool = False,
        session_id: str | None = None,
    ) -> "MemoryChunk": ...


def create_memory_tools(
    embedding_provider: EmbeddingProvider,
    session_factory: async_sessionmaker[AsyncSession],
    repo_factory: Callable[[AsyncSession], MemoryChunkRepository],
    user_id: str,
    *,
    session_id: str | None = None,
    memory_write_service: _MemoryWriteService | None = None,
    session_redis: Redis | None = None,
    session_save_cap: int = 20,
    half_life_days: int = 30,
    mmr_lambda: float = 0.7,
) -> list[BaseTool]:
    """Create memory tools with closed-over dependencies.

    ``memory_save`` is only added when all 3 save-path deps are present:
    ``session_id`` + ``memory_write_service`` + ``session_redis``. Partial
    wiring (e.g. planner context without a session) keeps the 2-tool shape.
    """
    CANDIDATE_MULTIPLIER = 3

    @lc_tool(args_schema=MemorySearchInput, response_format="content_and_artifact")
    async def memory_search(query: str, max_results: int = 5) -> tuple[str, ToolOutcome]:
        """搜索历史记忆。当需要回忆之前的对话、查找历史上下文时使用。"""
        try:
            vectors = await embedding_provider.embed([query])
            embedding = vectors[0]
        except EmbeddingUnavailableError as e:
            outcome = AllowSuccess(content=f"记忆检索暂不可用（embedding 服务异常: {e}）")
            return outcome.content, outcome
        except Exception as e:
            outcome = AllowSuccess(content=f"记忆检索失败（{type(e).__name__}: {e}）")
            return outcome.content, outcome

        async with session_factory() as session:
            repo = repo_factory(session)
            chunks = await repo.search_by_vector(
                user_id=user_id,
                embedding=embedding,
                top_k=max_results * CANDIDATE_MULTIPLIER,
            )

        if not chunks:
            outcome = AllowSuccess(content="未找到相关记忆。")
            return outcome.content, outcome

        chunks = rank_memory_results(
            chunks,
            query_embedding=embedding,
            half_life_days=half_life_days,
            mmr_lambda=mmr_lambda,
            top_k=max_results,
        )

        if not chunks:
            outcome = AllowSuccess(content="未找到相关记忆。")
            return outcome.content, outcome

        lines = []
        for i, c in enumerate(chunks, 1):
            preview = c.content[:500] + ("..." if len(c.content) > 500 else "")
            lines.append(f"[{i}] (id: {c.id}, source: {c.source})\n{preview}")
        outcome = AllowSuccess(content="\n\n".join(lines))
        return outcome.content, outcome

    @lc_tool(response_format="content_and_artifact")
    async def memory_get(chunk_id: str) -> tuple[str, ToolOutcome]:
        """读取记忆片段完整内容。在 memory_search 后用 ID 获取完整文本。"""
        async with session_factory() as session:
            repo = repo_factory(session)
            chunk = await repo.get_by_id(chunk_id, user_id=user_id)

        if not chunk:
            outcome = AllowSuccess(content=f"记忆片段 {chunk_id} 不存在。")
            return outcome.content, outcome
        outcome = AllowSuccess(content=chunk.content)
        return outcome.content, outcome

    tools: list[BaseTool] = [memory_search, memory_get]

    if session_id is not None and memory_write_service is not None and session_redis is not None:
        # 三个依赖齐全才构建 memory_save——防止 "半拉子" 的 save 进入 bound tools
        # 让 LLM 以为能用结果调用时 NoneType error。
        _session_id = session_id
        _service = memory_write_service
        _redis = session_redis
        _cap = session_save_cap

        @lc_tool(args_schema=MemorySaveInput, response_format="content_and_artifact")
        async def memory_save(
            content: str,
            category: _MEMORY_CATEGORY,
        ) -> tuple[str, ToolOutcome]:
            """保存一条新记忆到长期库。

            category：
            - user：用户偏好、身份、工作习惯（"我用中文"、"团队用 Go"）
            - rule：操作约束、永久性规则（"永远不直接 push main"）
            - fact：可验证的事实（"项目 DB 用 PostgreSQL 17"）

            同一 session 最多 20 次；重复内容会被自动去重。"""
            # 第 1 层：session 写入上限（memory_save 工具专用）
            #
            # ``check_and_increment_session_save`` 在 Redis 异常时 fail-open
            # 返回 0——意味着 INCR 根本没发生。下面的 refund 路径必须用
            # ``session_was_incremented`` 闸住，否则会对不存在的 key 做 DECR，
            # 把当天 session counter 打成负值（TTL 是首次 INCR 才写的，此时也
            # 不存在），Redis 恢复后该 session 当天额度会长期少算。参照
            # ``MemoryManagementService.create_memory`` 里的 ``quota_was_incremented``。
            try:
                current = await check_and_increment_session_save(
                    _redis, _session_id, cap=_cap
                )
            except QuotaExceededError as e:
                outcome = AllowSuccess(
                    content=(
                        f"当前 session memory_save 次数已达上限"
                        f"（{e.data['limit']} 次）。本次任务内不再保存，"
                        f"如需继续可结束任务后开启新 session。"
                    )
                )
                return outcome.content, outcome
            session_was_incremented = current > 0

            # 第 2 层：service 内部 user_daily quota + embedding + FS sync
            try:
                chunk = await _service.create_memory(
                    user_id=user_id,
                    content=content,
                    category=category,
                    source="memory_save",
                    session_id=_session_id,
                )
            except ConflictError:
                # 同 hash 已存在 → 实际写入未发生，仅当真的 INCR 过才退还
                if session_was_incremented:
                    await refund_session_save(_redis, _session_id)
                outcome = AllowSuccess(
                    content="这条内容之前已经保存过，跳过重复写入。"
                )
                return outcome.content, outcome
            except QuotaExceededError as e:
                # User daily cap 触发（session cap 上面已过）→ 仅当真的 INCR 过才退还
                if session_was_incremented:
                    await refund_session_save(_redis, _session_id)
                outcome = AllowSuccess(
                    content=(
                        f"今日全局 memory 写入已达上限"
                        f"（{e.data['limit']}），请明日再试。"
                    )
                )
                return outcome.content, outcome
            except Exception as e:
                if session_was_incremented:
                    await refund_session_save(_redis, _session_id)
                logger.warning(
                    "memory_save 意外失败: %s: %s", type(e).__name__, e
                )
                outcome = AllowSuccess(
                    content=f"保存失败（{type(e).__name__}: {e}）"
                )
                return outcome.content, outcome

            outcome = AllowSuccess(
                content=f"已保存到 {category} 记忆（id={chunk.id}）"
            )
            return outcome.content, outcome

        tools.append(memory_save)

    for t in tools:
        annotate_and_register_tool_source(t, source="native", category="memory")
    return tools
