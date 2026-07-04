"""B8 PR-3: recall provider 核心行为（P-7）。

fixture 模式：镜像 test_planner_react_collect_tools 的 _make_flow +
test_memory_tools 的 AsyncMock provider/repo 管线。Task 10 在本文件追加
故障矩阵组（§6 每行一测）。
"""
import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.domain.models.app_config import MemoryConfig
from app.domain.models.memory_chunk import MemoryChunk
from app.domain.models.memory_recall import RecallCachePayload, RecallQueryMaterial
from app.domain.services.flows import planner_react as planner_react_mod
from app.domain.services.flows.planner_react import PlannerReActFlow
from app.domain.services.memory_recall import (
    build_params_version,
    build_recall_query,
    compute_query_hash,
    normalize_recall_query,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


# ---- fixtures ----------------------------------------------------------- #


class _AsyncCM:
    def __init__(self, obj):
        self._obj = obj

    async def __aenter__(self):
        return self._obj

    async def __aexit__(self, *args):
        return False


class _FakeCache:
    """内存版 RecallCache——shadow→on 连续性/空结果缓存断言用。"""

    def __init__(self):
        self.store: dict[tuple[str, str], RecallCachePayload] = {}
        self.set_calls: list[tuple[str, str, RecallCachePayload]] = []

    async def get(self, session_id, query_hash):
        return self.store.get((session_id, query_hash))

    async def set(self, session_id, query_hash, payload):
        self.set_calls.append((session_id, query_hash, payload))
        self.store[(session_id, query_hash)] = payload


def _chunk(cid="c1", content="数据库是 PostgreSQL 17", category="fact", days_old=0):
    now = datetime.now(timezone.utc)
    return MemoryChunk(
        id=cid, user_id="u1", content=content, content_hash=f"h-{cid}",
        source="manual", metadata={},
        created_at=now - timedelta(days=days_old), updated_at=now,
        embedding=(1.0, 0.0), category=category,
    )


def _memory_cfg(**overrides):
    defaults = dict(recall_mode="on")
    defaults.update(overrides)
    return MemoryConfig(**defaults)


def _make_flow(**overrides):
    """镜像 test_planner_react_collect_tools._make_flow 的最小构造。"""
    defaults = dict(
        uow_factory=MagicMock(),
        llm=MagicMock(),
        agent_config=MagicMock(memory=_memory_cfg()),
        session_id="sess-1",
        browser=MagicMock(),
        sandbox=MagicMock(),
        search_engine=MagicMock(),
        mcp_tool=MagicMock(),
        a2a_tool=MagicMock(),
        skill_tool=MagicMock(),
    )
    defaults.update(overrides)
    return PlannerReActFlow(**defaults)


def _make_recall_flow(*, cfg=None, cache=None, chunks=None, title="季度数据分析", **overrides):
    """带全部 recall 依赖的 flow + 可断言的 embed/repo/uow 句柄。"""
    cfg = cfg or _memory_cfg()
    embed = MagicMock()
    embed.embed = AsyncMock(return_value=[[1.0, 0.0]])
    repo = MagicMock()
    repo.search_by_vector = AsyncMock(return_value=list(chunks or []))
    session_factory = MagicMock(return_value=_AsyncCM(MagicMock()))
    repo_factory = MagicMock(return_value=repo)
    uow = MagicMock()
    uow.session.get_by_id = AsyncMock(return_value=MagicMock(title=title))
    uow_factory = MagicMock(return_value=_AsyncCM(uow))
    flow = _make_flow(
        uow_factory=uow_factory,
        agent_config=MagicMock(memory=cfg),
        user_id="u1",
        memory_embedding_provider=embed,
        memory_session_factory=session_factory,
        memory_repo_factory=repo_factory,
        recall_cache=cache,
        **overrides,
    )
    return flow, embed, repo


def _material(message="继续", original_request=None, entry="graph"):
    return RecallQueryMaterial(
        message=message, original_request=original_request,
        session_title=None, entry=entry,
    )


def _expected_hash(cfg, *, message, original_request=None, title=None):
    material = RecallQueryMaterial(
        message=message, original_request=original_request,
        session_title=title, entry="graph",
    )
    q = build_recall_query(material, max_chars=cfg.recall_query_max_chars)
    return compute_query_hash(
        normalize_recall_query(q), params_version=build_params_version(cfg),
    )


@pytest.fixture
def telemetry_spy(monkeypatch):
    events: list[dict] = []
    monkeypatch.setattr(
        planner_react_mod, "_emit_recall_telemetry", lambda payload: events.append(payload),
    )
    return events


# ---- ctor 契约 ----------------------------------------------------------- #


class TestCtorContract:
    def test_default_recall_cache_is_none(self):
        flow = _make_flow()
        assert flow._recall_cache is None

    def test_recall_cache_stored(self):
        cache = _FakeCache()
        assert _make_flow(recall_cache=cache)._recall_cache is cache

    def test_provider_attr_initialized_none(self):
        assert _make_flow()._memory_recall_provider is None


# ---- builder 退化 -------------------------------------------------------- #


class TestBuilderDegradation:
    def test_mode_off_returns_none(self):
        flow, _, _ = _make_recall_flow(cfg=_memory_cfg(recall_mode="off"))
        assert flow._build_memory_recall_provider() is None

    def test_garbage_mode_fails_closed(self):
        # MagicMock/脏配置 → None（fail-closed，文档化偏差 #4）
        flow = _make_flow(agent_config=MagicMock(), user_id="u1")
        assert flow._build_memory_recall_provider() is None

    @pytest.mark.parametrize("missing", ["user_id", "embed", "session_factory", "repo_factory"])
    def test_missing_dependency_returns_none(self, missing):
        # Deviation #A: the brief passed user_id="" through _make_recall_flow's
        # **overrides, which collides with its hardcoded user_id="u1" positional
        # kwarg (TypeError: multiple values). We mutate the attribute post-
        # construction like the sibling branches below — identical intent
        # (falsy user_id degrades the builder to None), zero collision.
        flow, _, _ = _make_recall_flow()
        if missing == "user_id":
            flow._user_id = ""
        elif missing == "embed":
            flow._memory_embedding_provider = None
        elif missing == "session_factory":
            flow._memory_session_factory = None
        elif missing == "repo_factory":
            flow._memory_repo_factory = None
        assert flow._build_memory_recall_provider() is None


# ---- 核心行为 ------------------------------------------------------------ #


class TestProviderCore:
    async def test_on_mode_happy_path(self, telemetry_spy):
        cfg = _memory_cfg()
        flow, embed, repo = _make_recall_flow(cfg=cfg, chunks=[_chunk()])
        provider = flow._build_memory_recall_provider()
        recalled = await provider(_material())

        assert recalled is not None and recalled.cache_hit is False
        item = recalled.items[0]
        assert item.chunk_id == "c1" and item.category == "fact"
        assert len(item.content) <= 120
        # query 素材：title 进 query（embed 收到 normalize 后的 query）
        embed.embed.assert_awaited_once()
        sent_query = embed.embed.await_args.args[0][0]
        assert "季度数据分析" in sent_query and "继续" in sent_query
        # 检索参数：top_k×3 + threshold
        repo.search_by_vector.assert_awaited_once()
        call = repo.search_by_vector.await_args.kwargs
        assert call["top_k"] == 15 and call["threshold"] == 0.35

    async def test_query_hash_matches_pure_pipeline(self, telemetry_spy):
        cfg = _memory_cfg()
        flow, _, _ = _make_recall_flow(cfg=cfg, chunks=[_chunk()])
        provider = flow._build_memory_recall_provider()
        recalled = await provider(_material())
        assert recalled.query_hash == _expected_hash(cfg, message="继续", title="季度数据分析")

    async def test_placeholder_title_excluded(self, telemetry_spy):
        flow, embed, _ = _make_recall_flow(title="新对话", chunks=[_chunk()])
        provider = flow._build_memory_recall_provider()
        await provider(_material())
        assert "新对话" not in embed.embed.await_args.args[0][0]

    async def test_title_fetch_failure_degrades(self, telemetry_spy):
        flow, embed, _ = _make_recall_flow(chunks=[_chunk()])
        uow = MagicMock()
        uow.session.get_by_id = AsyncMock(side_effect=RuntimeError("db down"))
        flow._uow_factory = MagicMock(return_value=_AsyncCM(uow))
        provider = flow._build_memory_recall_provider()
        recalled = await provider(_material())
        assert recalled is not None  # 少一路素材照常检索
        assert embed.embed.await_count == 1

    async def test_cache_miss_writes_then_hit_skips_retrieval(self, telemetry_spy):
        cache = _FakeCache()
        flow, embed, repo = _make_recall_flow(cache=cache, chunks=[_chunk()])
        provider = flow._build_memory_recall_provider()
        first = await provider(_material())
        assert len(cache.set_calls) == 1
        second = await provider(_material())
        assert second.cache_hit is True
        assert second.items == first.items
        assert second.recall_id != first.recall_id  # cache 命中也新生成
        assert embed.embed.await_count == 1 and repo.search_by_vector.await_count == 1

    async def test_shadow_mode_retrieves_caches_but_returns_none(self, telemetry_spy):
        cache = _FakeCache()
        flow, embed, repo = _make_recall_flow(
            cfg=_memory_cfg(recall_mode="shadow"), cache=cache, chunks=[_chunk()],
        )
        provider = flow._build_memory_recall_provider()
        assert await provider(_material()) is None       # 不注入
        assert embed.embed.await_count == 1               # 检索照跑
        assert len(cache.set_calls) == 1                  # 缓存照写
        assert telemetry_spy and telemetry_spy[0]["mode"] == "shadow"

    async def test_no_cache_still_works(self, telemetry_spy):
        flow, embed, _ = _make_recall_flow(cache=None, chunks=[_chunk()])
        provider = flow._build_memory_recall_provider()
        assert (await provider(_material())).items
        # 二次调用无缓存 → 再检索
        await provider(_material())
        assert embed.embed.await_count == 2

    async def test_original_request_deduped_in_query(self, telemetry_spy):
        flow, embed, _ = _make_recall_flow(title=None, chunks=[_chunk()])
        provider = flow._build_memory_recall_provider()
        await provider(_material(message="继续", original_request="继续"))
        assert embed.embed.await_args.args[0][0] == "继续"


# ---- 故障矩阵（spec §6 每行一测） ---------------------------------------- #

from app.domain.external.embedding_provider import EmbeddingUnavailableError


class TestFailureMatrix:
    async def test_timeout_returns_none_with_telemetry(self, telemetry_spy):
        flow, embed, _ = _make_recall_flow(
            cfg=_memory_cfg(recall_timeout_seconds=0.1), chunks=[_chunk()],
        )

        async def _slow(texts):
            await asyncio.sleep(5)

        embed.embed = AsyncMock(side_effect=_slow)
        provider = flow._build_memory_recall_provider()
        assert await provider(_material()) is None
        assert telemetry_spy[0]["result"] == "timeout"

    async def test_embed_unavailable_returns_none(self, telemetry_spy):
        flow, embed, _ = _make_recall_flow(chunks=[_chunk()])
        embed.embed = AsyncMock(side_effect=EmbeddingUnavailableError("breaker open"))
        provider = flow._build_memory_recall_provider()
        assert await provider(_material()) is None
        assert telemetry_spy[0]["result"] == "embed_unavailable"

    async def test_repo_error_returns_none(self, telemetry_spy):
        flow, _, repo = _make_recall_flow(chunks=[_chunk()])
        repo.search_by_vector = AsyncMock(side_effect=RuntimeError("pg down"))
        provider = flow._build_memory_recall_provider()
        assert await provider(_material()) is None
        assert telemetry_spy[0]["result"] == "error"

    async def test_cache_get_error_falls_through_to_retrieval(self, telemetry_spy):
        class _BrokenCache(_FakeCache):
            async def get(self, session_id, query_hash):
                raise RuntimeError("cache broken")

        flow, embed, _ = _make_recall_flow(cache=_BrokenCache(), chunks=[_chunk()])
        provider = flow._build_memory_recall_provider()
        recalled = await provider(_material())
        assert recalled is not None and recalled.items      # 继续检索
        assert embed.embed.await_count == 1

    async def test_cache_set_error_swallowed(self, telemetry_spy):
        class _WriteBrokenCache(_FakeCache):
            async def set(self, session_id, query_hash, payload):
                raise RuntimeError("cache broken")

        flow, _, _ = _make_recall_flow(cache=_WriteBrokenCache(), chunks=[_chunk()])
        provider = flow._build_memory_recall_provider()
        recalled = await provider(_material())
        assert recalled is not None and recalled.items      # 放弃写但结果照常

    async def test_cancelled_error_passes_through(self, telemetry_spy):
        flow, embed, _ = _make_recall_flow(
            cfg=_memory_cfg(recall_timeout_seconds=15.0), chunks=[_chunk()],
        )

        async def _hang(texts):
            await asyncio.sleep(60)

        embed.embed = AsyncMock(side_effect=_hang)
        provider = flow._build_memory_recall_provider()
        task = asyncio.ensure_future(provider(_material()))
        await asyncio.sleep(0.05)   # 让 provider 进入 embed 挂起点
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert telemetry_spy == []  # 透传路径不发 telemetry

    async def test_telemetry_failure_does_not_break_result(self, monkeypatch):
        flow, _, _ = _make_recall_flow(chunks=[_chunk()])
        monkeypatch.setattr(
            planner_react_mod._RECALL_TELEMETRY_LOGGER, "info",
            MagicMock(side_effect=RuntimeError("logging broken")),
        )
        provider = flow._build_memory_recall_provider()
        recalled = await provider(_material())     # helper 吞掉，不抛
        assert recalled is not None and recalled.items

    async def test_empty_result_cached_and_second_call_hits(self, telemetry_spy):
        cache = _FakeCache()
        flow, embed, repo = _make_recall_flow(cache=cache, chunks=[])   # 零命中
        provider = flow._build_memory_recall_provider()
        first = await provider(_material())
        assert first is not None and first.items == ()
        assert len(cache.set_calls) == 1                    # 空 payload 也写
        assert cache.set_calls[0][2].items == ()
        assert telemetry_spy[0]["result"] == "empty"
        second = await provider(_material())
        assert second.cache_hit is True and second.items == ()
        assert embed.embed.await_count == 1                 # 二次零检索
        assert repo.search_by_vector.await_count == 1

    async def test_all_embeddings_none_yields_empty_items(self, telemetry_spy):
        none_chunk = _chunk()
        object.__setattr__(none_chunk, "embedding", None)   # frozen dataclass 绕写
        flow, _, _ = _make_recall_flow(chunks=[none_chunk])
        provider = flow._build_memory_recall_provider()
        recalled = await provider(_material())
        assert recalled is not None and recalled.items == ()


class TestCacheContinuityAndVersioning:
    async def test_shadow_then_on_hits_same_cache(self, telemetry_spy):
        cache = _FakeCache()
        flow_shadow, embed_s, repo_s = _make_recall_flow(
            cfg=_memory_cfg(recall_mode="shadow"), cache=cache, chunks=[_chunk()],
        )
        assert await flow_shadow._build_memory_recall_provider()(_material()) is None
        assert len(cache.set_calls) == 1

        flow_on, embed_o, repo_o = _make_recall_flow(
            cfg=_memory_cfg(recall_mode="on"), cache=cache, chunks=[_chunk()],
        )
        recalled = await flow_on._build_memory_recall_provider()(_material())
        assert recalled is not None and recalled.cache_hit is True
        assert embed_o.embed.await_count == 0               # on 直接命中 shadow 写的缓存
        assert repo_o.search_by_vector.await_count == 0

    async def test_params_version_change_invalidates_cache(self, telemetry_spy):
        cache = _FakeCache()
        flow_a, _, _ = _make_recall_flow(
            cfg=_memory_cfg(recall_top_k=5), cache=cache, chunks=[_chunk()],
        )
        await flow_a._build_memory_recall_provider()(_material())
        flow_b, embed_b, repo_b = _make_recall_flow(
            cfg=_memory_cfg(recall_top_k=6), cache=cache, chunks=[_chunk()],
        )
        recalled = await flow_b._build_memory_recall_provider()(_material())
        assert recalled.cache_hit is False                  # 不同 params_version → 不同 key
        assert repo_b.search_by_vector.await_count == 1
        assert len(cache.store) == 2


class TestTelemetryPayload:
    async def test_field_superset_and_no_plaintext(self, telemetry_spy):
        import json as _json

        flow, _, _ = _make_recall_flow(chunks=[_chunk(content="机密内容XYZ" * 10)])
        provider = flow._build_memory_recall_provider()
        await provider(_material(message="含机密查询词ABC"))
        payload = telemetry_spy[0]
        assert set(payload) >= {
            "recall_id", "session_id", "entry", "query_hash", "mode", "result",
            "cache_hit", "latency_ms", "top_k", "threshold", "candidate_count",
            "item_count", "chunk_ids", "categories", "score_values",
        }
        dumped = _json.dumps(payload, ensure_ascii=False, default=str)
        assert "机密内容" not in dumped and "机密查询词" not in dumped   # 不记原文
        assert payload["entry"] == "graph" and payload["result"] == "items"

    async def test_logger_channel_emits_info_json(self, caplog):
        import json as _json
        import logging as _logging

        flow, _, _ = _make_recall_flow(chunks=[_chunk()])
        provider = flow._build_memory_recall_provider()
        with caplog.at_level(_logging.INFO, logger="actus.memory_recall"):
            await provider(_material())
        records = [r for r in caplog.records if r.name == "actus.memory_recall"]
        assert len(records) == 1
        parsed = _json.loads(records[0].getMessage())
        assert parsed["result"] == "items"
        assert isinstance(records[0].recall_event, dict)    # extra 通道（PR-4 handler 消费）
