"""B9 Task 19 — RedisExtensionStats：per-extension 调用统计的有界队列 + Redis flusher。

P-9 钉子 infra 实现：结构化满足 ``ExtensionStatsRecorder`` / ``ExtensionStatsReader``
两个 domain Protocol（不继承——domain 端口无 Redis 概念，infra 侧鸭子类型对接）。

**INV-B9-6 热路径零 await**：``record()`` / ``delete_key()`` 是同步 ``def``，只做
``queue.put_nowait``——绝不 await 网络。队列有界（``STATS_QUEUE_MAXSIZE``）；满则丢弃
+ ``dropped_count`` 自增 + debug 日志（永不阻塞 event loop，永不背压工具执行热路径）。
真正的落盘由后台 ``run_flusher`` 批量 drain → ``redis.pipeline()`` 完成。

flusher 语义（spec §5.2）：
- 每批至多 ``STATS_FLUSH_BATCH_SIZE`` 条，或每 ``STATS_FLUSH_INTERVAL_SECONDS`` 触发一次；
- per (kind, id) 聚合：``HINCRBY key call_count/success_count/failure_count`` +
  ``HSET key last_active_at/last_success_at/last_failure_at``；
- pipeline 异常 → warn + **丢批**（不重排队，避免毒批无限重试卡死 flusher）。

read 语义（spec §5.2，语义简化冻结）：``read_many`` 任一 Redis 异常**原样冒泡**
（不做局部吞噬）——由 ``RuntimeExtensionService._load_stats`` broad-catch 统一降级
``redis_unavailable``（避免半可用歧义）。

shutdown 语义（spec §9）：``_closed=True``（``record`` 拒收）→ 限时 drain ≤2s → cancel。
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from app.domain.external.extension_stats import ExtensionStatsData
from app.domain.services.tools.extension_attribution import resolve_extension

if TYPE_CHECKING:
    from redis.asyncio import Redis

logger = logging.getLogger(__name__)

# ── 冻结契约常量（pins P-11）─────────────────────────────────────────────
STATS_QUEUE_MAXSIZE = 1000
STATS_FLUSH_INTERVAL_SECONDS = 2.0
STATS_FLUSH_BATCH_SIZE = 50

# shutdown drain 上限（spec §9：flusher 限时 drain ≤2s）。
STATS_SHUTDOWN_DRAIN_SECONDS = 2.0

# shutdown 期间等待后台 flusher 在途批次落盘完成的轮询间隔（P2 修复）。
# 仅在"队列已空但 _flush_inflight > 0"时使用；受 STATS_SHUTDOWN_DRAIN_SECONDS 死线兜底。
_SHUTDOWN_INFLIGHT_POLL_SECONDS = 0.01

# 队列条目判别标记（第 0 位）：记录 vs 删除。
_OP_RECORD = "record"
_OP_DELETE = "delete"


def build_stats_key(kind: str, extension_id: str) -> str:
    """构造 Redis hash key：``ext_stats:{kind}:{sha1_hex(id)[:16]}``（pins P-11）。

    extension_id 经 SHA-1 截断 16 hex——把敌意 id（超长 / 含冒号换行的 server_name，
    R16#5）压成固定 16 字符十六进制，杜绝 key 注入 / 长度爆炸；同 id 幂等。
    """
    digest = hashlib.sha1(extension_id.encode("utf-8")).hexdigest()[:16]  # noqa: S324 — 非密码学用途，仅作 key 派生
    return f"ext_stats:{kind}:{digest}"


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class RedisExtensionStats:
    """Redis-backed 统计记录器 + 读取器（结构化满足 P-9 两个 Protocol，不继承）。"""

    def __init__(self, redis_client: "Redis") -> None:
        self._redis = redis_client
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=STATS_QUEUE_MAXSIZE)
        self._closed = False
        # queue-full 丢弃计数（供测试断言 / 运维日志观测背压）。
        self.dropped_count = 0
        # 在途批次计数器（P2 修复；PR3-R2#1 由 bool 改 counter）：每次 _flush_once 在
        # 出队前 +1、_apply_batch 完成后在 finally -1；"有在途批次" ⇔ counter > 0。
        # 必须是计数器而非共享布尔——后台 flusher 与 shutdown 自身的 drain 可能并发各持
        # 一批（后台 block 在 pipeline.execute 里，shutdown 又 drain 了队列里的下一批）：
        # 共享布尔会被后完成的那次 flush 的 finally 清成 False，令 shutdown 误判 drain
        # 干净提前返回，main.py 随即 cancel 后台 flusher，其在途批（≤STATS_FLUSH_BATCH_SIZE
        # 条）随 CancelledError 丢失且无死线 warn（违反停收→限时 drain→cancel 合同）。
        self._flush_inflight: int = 0

    # ── 记录端口（ExtensionStatsRecorder）：同步、绝不 await ────────────────
    def record(self, tool_name: str, success: bool, latency_ms: float) -> None:
        """把一次工具调用统计入队（INV-B9-6：同步、零 await）。

        - 未归因（``resolve_extension`` → None：native / A2A / 未注册工具）→ 直接 return。
        - shutdown 后（``_closed``）→ 拒收，直接 return。
        - 队列满（``QueueFull``）→ 丢弃 + ``dropped_count`` 自增 + debug 日志。

        ``latency_ms`` 当前不落盘（spec §5.2 只统计 count + 时间戳），保留在签名里以
        对齐 P-9 Protocol 形状 + 未来 P95 扩展。
        """
        if self._closed:
            return
        attribution = resolve_extension(tool_name)
        if attribution is None:
            return
        kind, extension_id = attribution
        item = (_OP_RECORD, kind, extension_id, bool(success), _utcnow_iso())
        try:
            self._queue.put_nowait(item)
        except asyncio.QueueFull:
            self.dropped_count += 1
            logger.debug(
                "扩展统计队列已满（maxsize=%d）；丢弃 %s/%s 记录（累计丢弃 %d）",
                STATS_QUEUE_MAXSIZE,
                kind,
                extension_id,
                self.dropped_count,
            )

    # ── 读取端口（ExtensionStatsReader）─────────────────────────────────────
    def delete_key(self, kind: str, extension_id: str) -> None:
        """入队一条 DEL（fire-and-forget；reconcile 剔除后调用，绝不 await）。

        与 ``record`` 同队列——保证删除与在途写入按入队顺序被同一 flusher 处理，
        不会因并发路径乱序把一次刚落盘的写覆盖掉。队列满 / shutdown 后静默丢弃
        （删除是尽力而为——留下的孤儿 key 由后续 reconcile 再次投递或 TTL 兜底）。
        """
        if self._closed:
            return
        item = (_OP_DELETE, kind, extension_id)
        try:
            self._queue.put_nowait(item)
        except asyncio.QueueFull:
            self.dropped_count += 1
            logger.debug(
                "扩展统计队列已满（maxsize=%d）；丢弃 %s/%s DEL",
                STATS_QUEUE_MAXSIZE,
                kind,
                extension_id,
            )

    async def read_many(
        self, keys: list[tuple[str, str]]
    ) -> dict[tuple[str, str], ExtensionStatsData]:
        """批量读取多个扩展的统计聚合值（GET 聚合器消费）。

        语义简化冻结（spec §5.2）：**任一** ``HGETALL`` 抛错则整体失败——异常
        原样冒泡（不包 RuntimeError、不局部吞），由
        ``RuntimeExtensionService._load_stats`` broad-catch 统一降级
        ``redis_unavailable``（避免"部分键成功、部分失败"的半可用歧义）。
        """
        if not keys:
            return {}
        result: dict[tuple[str, str], ExtensionStatsData] = {}
        for kind, extension_id in keys:
            redis_key = build_stats_key(kind, extension_id)
            raw = await self._redis.hgetall(redis_key)
            result[(kind, extension_id)] = _parse_stats_hash(raw)
        return result

    # ── flusher（后台循环）─────────────────────────────────────────────────
    async def run_flusher(self) -> None:
        """后台批量落盘循环：每 ≤2s 或凑满一批就 drain 一次，直至 shutdown。"""
        while not self._closed:
            try:
                await asyncio.sleep(STATS_FLUSH_INTERVAL_SECONDS)
                await self._flush_once()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 — flusher 自愈，单批异常不得终止循环
                logger.warning("扩展统计 flusher 单轮异常（继续循环）", exc_info=True)

    async def _flush_once(self) -> None:
        """从队列 drain 至多一批 → 聚合 → pipeline 落盘（单批语义，测试直接驱动）。

        pipeline 异常 → warn + 丢批（不重排队）。

        P2 修复：``_flush_inflight`` 计数器在**出队前** +1、``_apply_batch`` 完成后在
        ``finally`` -1——让 shutdown 的限时 drain 能看到"批次已离队、pipeline 未完成"
        这段窗口，从而不会在 flusher mid-pipeline 时误判 drain 干净就放行 cancel。
        +1/-1 成对包住 drain+apply 全程（两个调用方——后台 run_flusher 与 shutdown 自身
        的 drain——共用此单一入口，并发各持一批也互不清对方的在途标记，PR3-R2#1）；
        空批（未出队任何条目）在 finally 归还计数、直接返回。
        """
        self._flush_inflight += 1
        try:
            batch = self._drain_batch()
            if not batch:
                return
            await self._apply_batch(batch)
        finally:
            self._flush_inflight -= 1

    def _drain_batch(self) -> list[tuple]:
        """非阻塞地从队列取出至多 ``STATS_FLUSH_BATCH_SIZE`` 条。"""
        batch: list[tuple] = []
        while len(batch) < STATS_FLUSH_BATCH_SIZE:
            try:
                batch.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return batch

    async def _apply_batch(self, batch: list[tuple]) -> None:
        """把一批队列条目聚合成 per-key 增量，经单个 pipeline 落盘。

        pipeline 执行异常 → warn + 丢批（不重排队；毒批无限重试会永久卡住 flusher）。
        """
        # per (kind, id) 聚合计数 + 各时间戳（取批内最新——批内已按入队顺序，末条最新）。
        counts: dict[tuple[str, str], dict[str, int]] = {}
        active_at: dict[tuple[str, str], str] = {}
        success_at: dict[tuple[str, str], str] = {}
        failure_at: dict[tuple[str, str], str] = {}
        deletes: set[tuple[str, str]] = set()

        for entry in batch:
            op = entry[0]
            if op == _OP_DELETE:
                _, kind, extension_id = entry
                key = (kind, extension_id)
                deletes.add(key)
                # DEL 覆盖同批之前的写聚合——删除是最终意图。
                counts.pop(key, None)
                active_at.pop(key, None)
                success_at.pop(key, None)
                failure_at.pop(key, None)
                continue
            # _OP_RECORD
            _, kind, extension_id, success, now_iso = entry
            key = (kind, extension_id)
            # 若同批内该 key 之前被 DEL 又出现新写入，则撤销删除、重新计入。
            deletes.discard(key)
            agg = counts.setdefault(
                key, {"call_count": 0, "success_count": 0, "failure_count": 0}
            )
            agg["call_count"] += 1
            active_at[key] = now_iso
            if success:
                agg["success_count"] += 1
                success_at[key] = now_iso
            else:
                agg["failure_count"] += 1
                failure_at[key] = now_iso

        if not counts and not deletes:
            return

        pipe = self._redis.pipeline(transaction=False)
        for key, agg in counts.items():
            kind, extension_id = key
            redis_key = build_stats_key(kind, extension_id)
            for field, amount in agg.items():
                if amount:
                    pipe.hincrby(redis_key, field, amount)
            ts_mapping: dict[str, str] = {"last_active_at": active_at[key]}
            if key in success_at:
                ts_mapping["last_success_at"] = success_at[key]
            if key in failure_at:
                ts_mapping["last_failure_at"] = failure_at[key]
            pipe.hset(redis_key, mapping=ts_mapping)
        for key in deletes:
            kind, extension_id = key
            pipe.delete(build_stats_key(kind, extension_id))

        try:
            await pipe.execute()
        except Exception:  # noqa: BLE001 — 落盘失败丢批（不重排队），warn 后继续
            logger.warning(
                "扩展统计 pipeline 落盘失败——丢弃本批 %d 条（不重排队）",
                len(batch),
                exc_info=True,
            )

    # ── shutdown ────────────────────────────────────────────────────────────
    async def shutdown(self) -> None:
        """停机：拒收新记录 → 限时 drain ≤2s → 剩余丢弃（spec §9）。

        P2 修复：drain 完成判据从"队列空"收紧为"队列空 **且** 无在途批次"
        （``self._flush_inflight > 0``，计数器语义见 __init__/PR3-R2#1）。否则若后台
        flusher 已 ``get_nowait`` 出一批、正 mid-pipeline，队列瞬时为空，shutdown 会误判
        drain 干净直接返回，main.py 随即 cancel flusher，在途那批 ≤STATS_FLUSH_BATCH_SIZE
        条随 CancelledError 丢失。本方法自身调用 ``_flush_once`` 时同样会计入/归还
        计数器（单一入口），与后台 flusher 的在途批互不清除。
        仍受 ``STATS_SHUTDOWN_DRAIN_SECONDS`` 死线兜底——绝不无界等待；若在途批次到死线仍
        未落盘，按既有 drop-after-deadline 语义放弃并 warn（此时丢失可接受且已记日志）。
        """
        self._closed = True
        deadline = asyncio.get_running_loop().time() + STATS_SHUTDOWN_DRAIN_SECONDS
        while not self._queue.empty() or self._flush_inflight > 0:
            if asyncio.get_running_loop().time() >= deadline:
                remaining = self._queue.qsize()
                logger.warning(
                    "扩展统计 shutdown drain 超 %.1fs——丢弃剩余 %d 条（在途批次数=%d）",
                    STATS_SHUTDOWN_DRAIN_SECONDS,
                    remaining,
                    self._flush_inflight,
                )
                break
            # 后台 flusher 正在途（队列已空但 pipeline 未完成）——本 loop 不再自行 drain，
            # 小步轮询等它落盘完成（在途计数归零），死线兜底防无界等待。
            # 用 10ms 轮询而非 sleep(0) 忙等：既让出事件循环给在途 flusher 推进，
            # 又避免 2s 死线窗口内 CPU 紧转（最多约 200 次轮询）。
            if self._queue.empty() and self._flush_inflight > 0:
                await asyncio.sleep(_SHUTDOWN_INFLIGHT_POLL_SECONDS)
                continue
            try:
                await self._flush_once()
            except Exception:  # noqa: BLE001 — 停机 drain 异常不阻断关闭
                logger.warning("扩展统计 shutdown drain 异常（放弃剩余）", exc_info=True)
                break


def _parse_stats_hash(raw: dict) -> ExtensionStatsData:
    """把 ``HGETALL`` 原始 hash（str|bytes 混合）解析为 ``ExtensionStatsData``。

    生产 ``RedisClient`` 用 ``decode_responses=True``（返回 str），但兼容 bytes 以防
    注入方言不同——统一先 decode。缺字段取默认（0 / None）。
    """

    def _decode_key(k) -> str:
        return k.decode("utf-8") if isinstance(k, (bytes, bytearray)) else k

    def _decode_val(v):
        return v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else v

    data = {_decode_key(k): _decode_val(v) for k, v in raw.items()}

    def _int(field: str) -> int:
        val = data.get(field)
        try:
            return int(val) if val is not None else 0
        except (TypeError, ValueError):
            return 0

    def _dt(field: str) -> datetime | None:
        val = data.get(field)
        if not val:
            return None
        try:
            return datetime.fromisoformat(val)
        except (TypeError, ValueError):
            return None

    return ExtensionStatsData(
        call_count=_int("call_count"),
        success_count=_int("success_count"),
        failure_count=_int("failure_count"),
        last_active_at=_dt("last_active_at"),
        last_success_at=_dt("last_success_at"),
        last_failure_at=_dt("last_failure_at"),
    )
