"""Three-view consistency integration test (Memory System Redesign M1).

Design doc L671 承诺：API 写 memory 后，**同时**对三条视图可见：

1. **DB** (``memory_search`` / ``memory_get`` 工具的底层) —— 立即一致
2. **文件系统** (sandbox ``file_read /workspace/.memory/...`` 的底层) —— 落盘后一致
3. **向量检索** (``memory_recall`` 工具的底层，基于 embedding + pgvector) —— embedding
   计算完成后一致

单元测试分别覆盖了每条路径，但没有一个测试把三视图串起来。这个 gap 被
`/plan-eng-review` 标注为 M1 ship-readiness 的唯一 P1 阻塞。本文件就是答案。

测试用真 Postgres + pgvector（走 tests/integration/conftest.py）+ tmp fs +
stub EmbeddingProvider（deterministic 向量以便 search_by_vector 可断言）+
NoopFileMemoryStore 对比版本 + 真 FsMemoryWriter 主版本。

**不 mock 的**：Postgres 行为（CHECK constraint / UNIQUE / RETURNING）、
FsMemoryWriter 原子 rename、frontmatter 序列化、pgvector 距离算子。

**mock 的**：OpenAI embedding 调用（换成 stub，0.75 类别确定性向量对齐 top_k=1）、
Redis（配额路径禁用，redis=None + user_daily_quota=None）。
"""
from __future__ import annotations

import uuid
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.application.services.memory_management_service import MemoryManagementService
from app.domain.external.embedding_provider import EmbeddingProvider
from app.domain.models.memory_chunk import MemoryChunk
from app.infrastructure.external.memory import (
    FsMemoryWriter,
    parse_memory_file,
)
from app.infrastructure.models.memory_chunk_orm import MEMORY_EMBEDDING_DIM
from app.infrastructure.repositories.db_memory_chunk_repository import (
    DBMemoryChunkRepository,
)

pytestmark = pytest.mark.anyio


class _DeterministicEmbedding(EmbeddingProvider):
    """固定维度向量 stub。三视图测试需要 search_by_vector 命中，所以
    create_memory 写入的 embedding 和查询 embedding 必须对齐。
    实战里 OpenAI 对同文本返回稳定向量；本 stub 就是这个不变量的最小化复现。"""

    def __init__(self) -> None:
        self._canonical = tuple(0.1 for _ in range(MEMORY_EMBEDDING_DIM))

    async def embed(self, texts: list[str]) -> list[list[float]]:
        # 对任意文本返回同一向量——让 search_by_vector 以 cosine~=1 召回本测试写入的 chunk
        return [list(self._canonical) for _ in texts]

    @property
    def dimensions(self) -> int:
        return MEMORY_EMBEDDING_DIM

    @property
    def model_name(self) -> str:
        return "deterministic-stub"


@pytest.fixture
async def memory_root(tmp_path: Path) -> Path:
    root = tmp_path / "memory"
    root.mkdir(parents=True, exist_ok=True)
    return root


@pytest.fixture
def fs_writer(memory_root: Path) -> FsMemoryWriter:
    # base_backoff_seconds=0 让重试路径在测试里不增加 wall time
    return FsMemoryWriter(
        memory_root=memory_root, max_retries=2, base_backoff_seconds=0.0
    )


@pytest.fixture
async def test_user_id(async_engine) -> str:
    """Insert a fresh user row（memory_chunks FK 依赖）并 COMMIT ——随机 UUID 防
    止跨测试污染。必须 commit 而不是走 rollback-on-teardown 的 ``db_session``
    fixture，因为 ``MemoryManagementService`` 用的是独立 session_factory，READ
    COMMITTED 隔离级别下看不到未提交的 INSERT。Teardown 时级联删 memory_chunks
    由 FK ON DELETE CASCADE 承担。"""
    session_factory = async_sessionmaker(bind=async_engine, expire_on_commit=False)
    uid = str(uuid.uuid4())
    async with session_factory() as session:
        await session.execute(
            text("INSERT INTO users (id) VALUES (:uid) ON CONFLICT DO NOTHING"),
            {"uid": uid},
        )
        await session.commit()
    yield uid
    async with session_factory() as session:
        await session.execute(
            text("DELETE FROM users WHERE id = :uid"), {"uid": uid}
        )
        await session.commit()


@pytest.fixture
async def service(
    async_engine, fs_writer: FsMemoryWriter
) -> MemoryManagementService:
    """MemoryManagementService 走真 Postgres session_factory + FsMemoryWriter。"""
    session_factory = async_sessionmaker(
        bind=async_engine, expire_on_commit=False
    )
    return MemoryManagementService(
        repo_factory=DBMemoryChunkRepository,
        embedding_provider=_DeterministicEmbedding(),
        session_factory=session_factory,
        file_store=fs_writer,
        # redis + quota 禁用——quota 路径由 test_memory_quota.py 覆盖，
        # 本测试专注三视图一致性，不重复验证额度
        redis=None,
        user_daily_quota=None,
    )


class TestThreeViewConsistencyOnCreate:
    """create_memory → 三视图同时可见（design L671 core promise）。"""

    async def test_db_fs_vector_all_reflect_new_chunk(
        self,
        service: MemoryManagementService,
        test_user_id: str,
        memory_root: Path,
        async_engine,
    ) -> None:
        # --- Act: 单条写入
        chunk = await service.create_memory(
            test_user_id,
            content="用户偏好：深色主题 + 下午 3 点后不接新任务",
            category="user",
            source="manual",
            tags=["preference", "schedule"],
        )

        # --- Assert view 1: DB via get_memory
        fetched = await service.get_memory(test_user_id, chunk.id)
        assert fetched is not None, "DB 视图未看到新写入"
        assert fetched.content == chunk.content
        assert fetched.category == "user"
        assert fetched.fs_synced is True, (
            "FsMemoryWriter 成功落盘后 fs_synced 应翻为 True（design L431-432 承诺）"
        )
        assert fetched.metadata.get("tags") == ["preference", "schedule"]

        # --- Assert view 2: filesystem (sandbox file_read 的底层)
        fs_path = memory_root / test_user_id / "user" / f"{chunk.id}.md"
        assert fs_path.exists(), f"FS 视图未看到文件：{fs_path}"
        raw = fs_path.read_text(encoding="utf-8")
        fm, body = parse_memory_file(raw)
        assert fm["id"] == chunk.id
        assert fm["category"] == "user"
        assert fm["source"] == "manual"
        assert fm["tags"] == ["preference", "schedule"]
        assert "深色主题" in body

        # --- Assert view 3: 向量检索 (memory_recall 底层)
        session_factory = async_sessionmaker(bind=async_engine, expire_on_commit=False)
        async with session_factory() as session:
            repo = DBMemoryChunkRepository(session)
            # _DeterministicEmbedding 对任意输入返回同一向量 → 写入 chunk 的 embedding
            # 与 query embedding cosine similarity = 1.0，稳定命中 threshold。
            query_vec = [0.1] * MEMORY_EMBEDDING_DIM
            results = await repo.search_by_vector(
                test_user_id, query_vec, top_k=5, threshold=0.3
            )
        hit_ids = {c.id for c in results}
        assert chunk.id in hit_ids, (
            f"向量检索视图未命中新 chunk_id={chunk.id}；返回: {hit_ids}"
        )

    async def test_fs_writer_failure_leaves_fs_synced_false_but_db_and_vector_visible(
        self,
        async_engine,
        memory_root: Path,
        test_user_id: str,
    ) -> None:
        """design L430-432：fs 写失败不阻塞 DB/vector 视图——DB-first 的核心承诺。

        场景：FsMemoryWriter 构造时指向不可写目录 → write 抛 PermissionError →
        service 捕获并留 fs_synced=false + 审计 → DB 和 vector 视图仍一致。
        这样 FsReconciler 下轮 scan_pending 能补。
        """
        # 指向一个不可写路径：memory_root 下的文件（不是目录）→ mkdir 会失败
        blocker = memory_root / "not-a-dir"
        blocker.write_text("x")  # 普通文件，mkdir(parents=True) 会撞见
        bad_writer = FsMemoryWriter(
            memory_root=blocker, max_retries=1, base_backoff_seconds=0.0
        )

        session_factory = async_sessionmaker(bind=async_engine, expire_on_commit=False)
        failing_service = MemoryManagementService(
            repo_factory=DBMemoryChunkRepository,
            embedding_provider=_DeterministicEmbedding(),
            session_factory=session_factory,
            file_store=bad_writer,
            redis=None,
            user_daily_quota=None,
        )

        chunk = await failing_service.create_memory(
            test_user_id,
            content="这条会写进 DB 但 FS 写不出",
            category="rule",
            source="manual",
        )

        # DB 视图：可见，fs_synced=False（design 承诺）
        fetched = await failing_service.get_memory(test_user_id, chunk.id)
        assert fetched is not None, "DB 视图应该独立于 FS 失败 / DB-first 承诺"
        assert fetched.fs_synced is False, (
            "FS 写失败时 fs_synced 必须留 False，FsReconciler 后续拾回"
        )

        # Vector 视图：可见（embedding 成功，search_by_vector 不依赖 fs_synced）
        async with session_factory() as session:
            repo = DBMemoryChunkRepository(session)
            results = await repo.search_by_vector(
                test_user_id, [0.1] * MEMORY_EMBEDDING_DIM, top_k=5, threshold=0.3
            )
        assert chunk.id in {c.id for c in results}


class TestThreeViewConsistencyOnDelete:
    """delete_memory → 三视图同时清空（design L454-458）。"""

    async def test_delete_removes_from_db_fs_and_vector(
        self,
        service: MemoryManagementService,
        test_user_id: str,
        memory_root: Path,
        async_engine,
    ) -> None:
        chunk = await service.create_memory(
            test_user_id,
            content="待会要删的 memory",
            category="fact",
            source="manual",
        )
        fs_path = memory_root / test_user_id / "fact" / f"{chunk.id}.md"
        assert fs_path.exists(), "前置：创建后 fs 应已落盘"

        # Act: 删除（service.delete_memory 返回 bool——True 表示命中一行）
        assert await service.delete_memory(test_user_id, chunk.id) is True

        # View 1: DB 已删
        assert await service.get_memory(test_user_id, chunk.id) is None

        # View 2: FS 已删（best-effort；design L454-458 明说 writer.delete 幂等）
        assert not fs_path.exists(), f"FS 视图未清理：{fs_path}"

        # View 3: vector 视图已删
        session_factory = async_sessionmaker(bind=async_engine, expire_on_commit=False)
        async with session_factory() as session:
            repo = DBMemoryChunkRepository(session)
            results = await repo.search_by_vector(
                test_user_id, [0.1] * MEMORY_EMBEDDING_DIM, top_k=5, threshold=0.3
            )
        assert chunk.id not in {c.id for c in results}


class TestThreeViewConsistencyOnUpdate:
    """update_memory_content → DB + FS 视图更新；向量视图重算 embedding（design L440-445）。"""

    async def test_update_content_repropagates_to_all_views(
        self,
        service: MemoryManagementService,
        test_user_id: str,
        memory_root: Path,
        async_engine,
    ) -> None:
        chunk = await service.create_memory(
            test_user_id,
            content="原始内容：偏好 Vim",
            category="user",
            source="manual",
        )
        fs_path = memory_root / test_user_id / "user" / f"{chunk.id}.md"
        original_bytes = fs_path.read_bytes()

        # Act: 改内容
        updated = await service.update_memory_content(
            test_user_id, chunk.id, "修订内容：改用 Neovim"
        )
        assert updated is not None
        assert "Neovim" in updated.content

        # View 1: DB 已更新，fs_synced=True（writer 回写翻旗）
        fetched = await service.get_memory(test_user_id, chunk.id)
        assert fetched is not None
        assert "Neovim" in fetched.content
        assert fetched.fs_synced is True
        # updated_at 必须严格大于 created_at——否则 ranker 时间衰减看不出变化
        assert fetched.updated_at > fetched.created_at

        # View 2: FS 文件重写（内容不等于原始 bytes）
        new_bytes = fs_path.read_bytes()
        assert new_bytes != original_bytes
        fm, body = parse_memory_file(new_bytes.decode("utf-8"))
        assert "Neovim" in body
        assert fm["id"] == chunk.id

        # View 3: vector 视图 — embedding 已重算（_DeterministicEmbedding 对所有输入
        # 返回同向量，这里做的是"写入后查得到"的 smoke，真实 prod 下 embedding 会不同
        session_factory = async_sessionmaker(bind=async_engine, expire_on_commit=False)
        async with session_factory() as session:
            repo = DBMemoryChunkRepository(session)
            results = await repo.search_by_vector(
                test_user_id, [0.1] * MEMORY_EMBEDDING_DIM, top_k=5, threshold=0.3
            )
        hit = next((c for c in results if c.id == chunk.id), None)
        assert hit is not None
        assert "Neovim" in hit.content
