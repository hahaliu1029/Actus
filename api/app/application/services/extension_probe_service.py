"""B9 扩展健康探测服务（P-5）——状态机 / 退避 / 并发防护。

本模块实现 spec §3.3（状态转移表）与 §3.4（并发五条）的服务层。核心不变式：

- **预算内化（R1#1/R2#1）**：20s 手动预算全部在 ``probe_one_manual`` 内部实现，
  用事件循环真实时间（``asyncio.get_running_loop().time()``）作 deadline。等待阶段
  （per-key lock / 全局 slot）耗尽预算 → ``ProbeBusyError``；探测一旦开始，锁内段
  ``_probe_locked`` 全程零取消源（无 wait_for/timeout/shield/create_task）。
- **注入的 ``clock``（返回 ``datetime``）只用于 record 时间戳与退避调度计算，绝不参与预算**
  （R4#2）——预算用真实事件循环时间。
- **并发额度单一机制（R6#3）**：``_inflight: int`` 是唯一权威 + ``asyncio.Condition``
  做 manual 等待/唤醒。三个 helper（``_acquire_slot_nowait`` / ``_acquire_slot_wait`` /
  ``_release_slot``）是唯一入口——``asyncio.Semaphore`` 禁用。后台 tick 走 nowait
  （同步检查+自增，无 await），结构上不可能落入排队。

分层：application 层，禁止 import interfaces / infrastructure.external.runtime_stats。
error_message 落 record 前统一过 ``redact_text``（INV-B9-2/R5#5）——本项目允许
application 复用 ``infrastructure.logging`` 的公共脱敏 API（单源同规则）。
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import random
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Callable, Literal, Protocol

import httpx

from app.domain.models.runtime_extension import HealthState
from app.domain.services.tools.mcp import MCPClientManager
from app.infrastructure.logging.redaction import redact_text

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from app.domain.models.app_config import (
        A2AServerConfig,
        AppConfig,
        MCPServerConfig,
    )
    from app.domain.repositories.skill_repository import SkillRepository

    from app.application.services.runtime_extension_service import LivenessView


# —— P-5 冻结常量 ——
PROBE_BACKOFF_BASE_SECONDS = 5
PROBE_BACKOFF_CAP_SECONDS = 300
PROBE_JITTER_RANGE = (0.75, 1.25)
PROBE_SUCCESS_INTERVAL_SECONDS = 60
PROBE_STALE_AFTER_SECONDS = 120
PROBE_TICK_INTERVAL_SECONDS = 30
PROBE_SEMAPHORE_SIZE = 3
MANUAL_PROBE_BUDGET_SECONDS = 20
MANUAL_PROBE_COOLDOWN_SECONDS = 5
# skill integrity 后台刷新纯防御超时（design.md:216）——挂死的文件扫描不拖垮整 tick。
SKILL_DIAG_REFRESH_TIMEOUT_SECONDS = 1.0


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _safe_log_id(value: str) -> str:
    """日志注入防御：仅保留可打印字符并截断（对齐审计路径 id_display 语义）。"""
    return "".join(ch for ch in str(value) if ch.isprintable())[:64]


@dataclass
class ProbeRecord:
    state: HealthState = "unknown"
    last_checked_at: datetime | None = None
    latency_ms: int | None = None
    error_code: str | None = None
    error_message: str | None = None          # 存储时已过 redact_text
    consecutive_failures: int = 0
    next_probe_at: datetime | None = None
    generation: int = 0
    config_fingerprint: str = ""
    tool_count: int | None = None             # mcp：成功探测的工具数（→ details.tool_count）
    display_name: str | None = None           # a2a：agent_card.name（→ 条目 name；
                                              # 失败探测保留上次值，仅成功时覆盖）


@dataclass(frozen=True)
class ProbeOutcome:                            # prober 返回值（Task 13 生产，Task 12 消费）
    ok: bool
    latency_ms: int
    error_code: str | None = None
    error_message: str | None = None           # raw；落 record 前过 redact_text
    tool_count: int | None = None              # mcp 专用
    display_name: str | None = None            # a2a 专用 = agent_card.name


class ProbeBusyError(Exception):
    """20s 预算耗尽（endpoint 映射 503 probe_busy）。"""


class ProbeGoneError(Exception):
    """目标已删（endpoint 映射 404）。"""


class ProbeDisabledError(Exception):
    """二次复核发现 flag 关 / 目标 disabled。"""

    def __init__(self, reason: Literal["probe_disabled", "extension_disabled"]) -> None:
        super().__init__(reason)
        self.reason = reason


class ExtensionProber(Protocol):               # Task 13 实现 DefaultExtensionProber
    async def probe_mcp(
        self, server_name: str, config: "MCPServerConfig"
    ) -> ProbeOutcome: ...
    async def probe_a2a(self, config: "A2AServerConfig") -> ProbeOutcome: ...


class ExtensionProbeService:
    """扩展健康探测状态机（mcp/a2a）。"""

    def __init__(
        self,
        config_provider: Callable[[], "AppConfig"],
        skill_repository: "SkillRepository",
        prober: "ExtensionProber",
        probe_flag_provider: Callable[[], bool],
        liveness_view: "LivenessView | None" = None,
        clock: Callable[[], datetime] = _utcnow,
        rng: Callable[[], float] = random.random,
    ) -> None:
        self._config_provider = config_provider
        self._skill_repository = skill_repository
        self._prober = prober
        self._probe_flag_provider = probe_flag_provider
        self._liveness_view = liveness_view
        self._clock = clock
        self._rng = rng

        self._records: dict[tuple[str, str], ProbeRecord] = {}
        self._locks: dict[tuple[str, str], asyncio.Lock] = {}

        # skill integrity 后台刷新去抖状态（key=skill_key → error_code；ok→None）。
        # 仅供"状态变化才打日志"的 diff 用——绝不进 probe 快照（GET 权威=repo 直读，
        # spec-R9#1 单源防漂移）。None 表示"尚未建基线"。
        self._last_skill_diag_state: dict[str, str | None] | None = None

        # 并发额度单一机制（R6#3）：_inflight 唯一权威 + Condition 做 manual 等待/唤醒。
        self._inflight: int = 0
        self._slot_cond = asyncio.Condition()

        self._stopping = False

    # —— 只读快照 ——

    def snapshot(self) -> dict[tuple[str, str], ProbeRecord]:
        """浅拷贝——外层不可原地改内部记录。"""
        return dict(self._records)

    # —— 并发 slot 三 helper（P-5 冻结；asyncio.Semaphore 禁用）——

    def _acquire_slot_nowait(self) -> bool:
        """同步检查+自增，无任何 await——后台 tick 专用，绝不排队。

        满额（含 manual 排队占满）→ 返回 False，后台跳过本项。
        """
        if self._inflight >= PROBE_SEMAPHORE_SIZE:
            return False
        self._inflight += 1
        return True

    async def _acquire_slot_wait(self, deadline: float) -> None:
        """manual 专用：带剩余预算 timeout 等空位，超时→ProbeBusyError。"""
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise ProbeBusyError()
        async with self._slot_cond:
            try:
                await asyncio.wait_for(
                    self._slot_cond.wait_for(
                        lambda: self._inflight < PROBE_SEMAPHORE_SIZE
                    ),
                    timeout=remaining,
                )
            except (asyncio.TimeoutError, TimeoutError) as exc:
                raise ProbeBusyError() from exc
            self._inflight += 1

    async def _release_slot(self) -> None:
        async with self._slot_cond:
            self._inflight -= 1
            self._slot_cond.notify()

    # —— 目标解析 ——

    def _fingerprint(self, config_obj) -> str:
        return hashlib.sha256(
            json.dumps(
                config_obj.model_dump(mode="json"), sort_keys=True
            ).encode("utf-8")
        ).hexdigest()

    def _resolve_target(self, kind: str, ext_id: str):
        """返回 (exists, enabled, config_obj, fingerprint)。

        二次复核与回写复核复用；config_obj 为 None 时 exists=False。
        """
        app_config = self._config_provider()
        if kind == "mcp":
            config_obj = app_config.mcp_config.mcpServers.get(ext_id)
        elif kind == "a2a":
            config_obj = next(
                (s for s in app_config.a2a_config.a2a_servers if s.id == ext_id),
                None,
            )
        else:  # pragma: no cover - probe_one_manual 已拒 skill
            config_obj = None
        if config_obj is None:
            return False, False, None, ""
        return True, bool(config_obj.enabled), config_obj, self._fingerprint(config_obj)

    # —— 退避 ——

    def _backoff_delay(self, failures: int) -> float:
        base = min(
            PROBE_BACKOFF_BASE_SECONDS * (2 ** (failures - 1)),
            PROBE_BACKOFF_CAP_SECONDS,
        )
        low, high = PROBE_JITTER_RANGE
        factor = low + (high - low) * self._rng()
        return base * factor

    # —— 回写 helper（成功/失败）——

    def _write_success(
        self, record: ProbeRecord, outcome: ProbeOutcome, now: datetime
    ) -> None:
        record.state = "reachable"
        record.consecutive_failures = 0
        record.last_checked_at = now
        record.latency_ms = outcome.latency_ms
        record.error_code = None
        record.error_message = None
        if outcome.tool_count is not None:
            record.tool_count = outcome.tool_count
        if outcome.display_name is not None:
            record.display_name = outcome.display_name
        record.next_probe_at = now + timedelta(
            seconds=PROBE_SUCCESS_INTERVAL_SECONDS
        )

    def _write_failure(
        self, record: ProbeRecord, outcome: ProbeOutcome, now: datetime
    ) -> None:
        record.state = "unreachable"
        record.consecutive_failures += 1
        record.last_checked_at = now
        record.latency_ms = outcome.latency_ms
        record.error_code = outcome.error_code
        if outcome.error_message is not None:
            try:
                record.error_message = redact_text(outcome.error_message)
            except Exception:
                record.error_message = None  # 脱敏抛错→只留 code
        else:
            record.error_message = None
        # display_name 保留上次值不清（失败探测不覆盖身份）
        if outcome.error_code == "auth_failed":
            record.next_probe_at = None  # 终态不自动重试
        else:
            record.next_probe_at = now + timedelta(
                seconds=self._backoff_delay(record.consecutive_failures)
            )

    # —— 手动探测（20s 预算内化）——

    async def probe_one_manual(self, kind: str, ext_id: str) -> ProbeRecord:
        if kind == "skill":
            raise ValueError(
                "skill 无网络探测语义——probe_one_manual 只服务 mcp/a2a"
            )
        if kind not in ("mcp", "a2a"):
            raise ValueError(f"未知扩展类型: {kind}")

        key = (kind, ext_id)
        deadline = (
            asyncio.get_running_loop().time() + MANUAL_PROBE_BUDGET_SECONDS
        )

        lock = self._locks.setdefault(key, asyncio.Lock())
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise ProbeBusyError()
        try:
            await asyncio.wait_for(lock.acquire(), timeout=remaining)
        except (asyncio.TimeoutError, TimeoutError) as exc:
            raise ProbeBusyError() from exc

        try:
            await self._acquire_slot_wait(deadline)  # 超时→ProbeBusyError
            try:
                return await self._probe_locked(kind, ext_id)
            finally:
                await self._release_slot()
        finally:
            lock.release()

    async def _probe_locked(self, kind: str, ext_id: str) -> ProbeRecord:
        """锁内段：二次复核 → 探测 → 回写复核 → 回写。全程零取消包装。

        前置条件：调用方已持有 per-key lock 与并发额度。
        """
        key = (kind, ext_id)

        # —— 二次复核（R19#1 四分支）——
        if not self._probe_flag_provider():
            raise ProbeDisabledError("probe_disabled")
        exists, enabled, config_obj, fingerprint = self._resolve_target(kind, ext_id)
        if not exists:
            self._records.pop(key, None)
            raise ProbeGoneError()
        if not enabled:
            raise ProbeDisabledError("extension_disabled")

        record = self._records.setdefault(key, ProbeRecord())
        # 手动 probe 无条件先解除终态并清零失败计数（含 auth_failed）
        record.consecutive_failures = 0
        record.next_probe_at = None
        record.config_fingerprint = fingerprint
        gen_before = record.generation

        # —— 探测（本体有 per-server 5s / A2A 8s 内建超时兜底）——
        if kind == "mcp":
            outcome = await self._prober.probe_mcp(ext_id, config_obj)
        else:
            outcome = await self._prober.probe_a2a(config_obj)

        now = self._clock()

        # —— 回写前复核（R18#3 / generation fencing）——
        exists_after, _, _, _ = self._resolve_target(kind, ext_id)
        if not exists_after:
            self._records.pop(key, None)  # 探测期间被删 → 丢弃 + 剔除
            raise ProbeGoneError()
        current = self._records.get(key)
        if current is None or current.generation != gen_before:
            # generation 已变（midflight invalidate）→ 丢弃结果，返回当前 record
            return current if current is not None else ProbeRecord()

        if outcome.ok:
            self._write_success(record, outcome, now)
        else:
            self._write_failure(record, outcome, now)
        return record

    # —— 内存同步：invalidate / reconcile ——

    def invalidate(
        self,
        kind: str,
        ext_id: str,
        reason: Literal["disable", "enable", "create", "delete", "update"],
    ) -> None:
        key = (kind, ext_id)
        if reason == "delete":
            self._records.pop(key, None)
            return

        record = self._records.setdefault(key, ProbeRecord())
        record.generation += 1
        if reason == "disable":
            record.state = "skipped"
            record.next_probe_at = None
        else:  # enable / create / update → unknown（字段重置）
            self._reset_to_unknown(record)

    def reconcile(
        self,
        live_keys: set[tuple[str, str]],
        fingerprints: dict[tuple[str, str], str],
    ) -> list[tuple[str, str]]:
        """纯内存同步：config 无 → 剔除+返回该 key；指纹变化 → generation+1 + 重置 unknown。"""
        evicted: list[tuple[str, str]] = []
        for key in list(self._records.keys()):
            if key not in live_keys:
                self._records.pop(key, None)
                evicted.append(key)
                continue
            new_fp = fingerprints.get(key)
            record = self._records[key]
            if new_fp is not None and new_fp != record.config_fingerprint:
                record.generation += 1
                self._reset_to_unknown(record)
                record.config_fingerprint = new_fp
        return evicted

    def _reset_to_unknown(self, record: ProbeRecord) -> None:
        record.state = "unknown"
        record.last_checked_at = None
        record.latency_ms = None
        record.error_code = None
        record.error_message = None
        record.consecutive_failures = 0
        record.next_probe_at = None
        record.tool_count = None
        record.display_name = None

    # —— 后台循环单步（可测纯逻辑）——

    async def tick_once(self) -> None:
        """扫描到期项，逐个尝试探测。后台绝不排队（nowait helper 结构保证）。

        枚举 config：为尚未跟踪的 enabled 条目播种一条 ``unknown`` 记录
        （``next_probe_at=now+60s``——首次见到条目不立即探测，遵循低频后台节奏，
        避免启动瞬间对全部条目同时开探）。已跟踪条目按到期条件（``next_probe_at<=now``）
        选取；stdio 且 in-use 的条目跳过探测并顺延（spec §3.3 状态表末行）。
        """
        if not self._probe_flag_provider():
            return
        # skill integrity 后台刷新（R9#4：同 flag 门控——flag off 时上面已 return，
        # skill 扫描与 mcp/a2a 探测一体不跑，default OFF 绝不每 30s 扫 skill 文件/打日志）。
        await self._refresh_skill_diagnostics()
        now = self._clock()
        liveness_active: dict[tuple[str, str], frozenset[str]] = {}
        if self._liveness_view is not None:
            liveness_active = self._liveness_view.snapshot().active

        app_config = self._config_provider()
        config_index = self._index_config(app_config)

        for key, (config_obj, enabled) in config_index.items():
            kind, ext_id = key
            if not enabled:
                continue
            if key not in self._records:
                # 首见 enabled 条目 → 播种（不在本轮探测）
                self._records[key] = ProbeRecord(
                    config_fingerprint=self._fingerprint(config_obj),
                    next_probe_at=now + timedelta(
                        seconds=PROBE_SUCCESS_INTERVAL_SECONDS
                    ),
                )
                continue
            record = self._records[key]
            if record.state == "skipped":
                continue
            if not self._is_due(record, now):
                continue

            # stdio + in-use → 跳过（保持原态 + next_probe_at 顺延 60s）
            if self._is_stdio_in_use(kind, config_obj, key, liveness_active):
                record.next_probe_at = now + timedelta(
                    seconds=PROBE_SUCCESS_INTERVAL_SECONDS
                )
                continue

            lock = self._locks.setdefault(key, asyncio.Lock())
            if lock.locked():
                continue  # 被 manual 占用 → 跳过
            if not self._acquire_slot_nowait():
                continue  # 满额（含 manual 排队占满）→ 跳过，后台不排队

            await lock.acquire()
            try:
                await self._background_probe(kind, ext_id, config_obj)
            finally:
                lock.release()
                await self._release_slot()

    async def _background_probe(self, kind: str, ext_id: str, config_obj) -> None:
        key = (kind, ext_id)
        record = self._records.setdefault(key, ProbeRecord())
        record.config_fingerprint = self._fingerprint(config_obj)
        gen_before = record.generation

        if kind == "mcp":
            outcome = await self._prober.probe_mcp(ext_id, config_obj)
        else:
            outcome = await self._prober.probe_a2a(config_obj)

        now = self._clock()

        exists_after, _, _, _ = self._resolve_target(kind, ext_id)
        if not exists_after:
            self._records.pop(key, None)
            return
        current = self._records.get(key)
        if current is None or current.generation != gen_before:
            return  # midflight invalidate → 丢弃结果

        if outcome.ok:
            self._write_success(record, outcome, now)
        else:
            self._write_failure(record, outcome, now)

    def _index_config(
        self, app_config: "AppConfig"
    ) -> dict[tuple[str, str], tuple[object, bool]]:
        index: dict[tuple[str, str], tuple[object, bool]] = {}
        for name, cfg in app_config.mcp_config.mcpServers.items():
            index[("mcp", name)] = (cfg, bool(cfg.enabled))
        for cfg in app_config.a2a_config.a2a_servers:
            index[("a2a", cfg.id)] = (cfg, bool(cfg.enabled))
        return index

    def _is_due(self, record: ProbeRecord, now: datetime) -> bool:
        """到期条件（spec §3.4 调度）。

        - ``next_probe_at`` 到点（``<=now``）→ 到期（退避/成功间隔调度主路径）；
        - 无 ``next_probe_at`` 的 ``unknown``（如 reconcile/invalidate 重置后）→ 到期
          （spec §3.3"state=unknown 且非 skipped"补漏，让重置条目下一轮即被重探）。
        播种时 ``unknown`` 携带 ``next_probe_at=now+60``，故首见条目不会立即命中本条。
        """
        if record.next_probe_at is not None:
            return record.next_probe_at <= now
        return record.state == "unknown"

    def _is_stdio_in_use(
        self,
        kind: str,
        config_obj,
        key: tuple[str, str],
        liveness_active: dict[tuple[str, str], frozenset[str]],
    ) -> bool:
        if kind != "mcp":
            return False
        transport = getattr(config_obj, "transport", None)
        transport_value = getattr(transport, "value", transport)
        if transport_value != "stdio":
            return False
        return bool(liveness_active.get(key))

    # —— skill integrity 后台刷新（Task 14；不进 probe 快照）——

    async def _refresh_skill_diagnostics(self) -> None:
        """扫 ``list_with_diagnostics()``，与上次结果 diff，仅状态变化时打日志。

        与上轮（``self._last_skill_diag_state``：skill_key→error_code，ok→None）比对：
        新损坏（ok→err 或 err_a→err_b）→ warn 一次；修复（err→ok）→ info 一次；
        无变化 → 静默。首轮只建基线，不打变化日志（避免启动即刷屏）。

        本方法**不写 probe 快照、不被 GET 消费**（GET 权威=repo 直读，spec-R9#1
        单源防漂移）；扫描自身异常吞掉不外抛（run_loop 顶层 try 亦兜底，双闸门）。
        """
        try:
            diagnostics = await asyncio.wait_for(
                self._skill_repository.list_with_diagnostics(),
                timeout=SKILL_DIAG_REFRESH_TIMEOUT_SECONDS,
            )
        except (asyncio.TimeoutError, TimeoutError):
            # 1s 纯防御（design.md:216）：挂死的文件扫描不拖垮整 tick——本轮跳过，
            # 不更新基线（下轮重试）。mcp/a2a 探测在 tick_once 后续段照常执行。
            logger.warning("skill integrity 后台扫描超时（1s 防御，下轮重试）")
            return
        except Exception:  # noqa: BLE001 - 扫描失败不影响 mcp/a2a 探测；下轮重试
            logger.warning("skill integrity 后台扫描失败（非致命，下轮重试）", exc_info=True)
            return

        current: dict[str, str | None] = {
            diag.skill_key: (None if diag.ok else diag.error_code)
            for diag in diagnostics
        }

        previous = self._last_skill_diag_state
        self._last_skill_diag_state = current
        if previous is None:
            return  # 首轮：只建基线，不打变化日志

        for skill_key, code in current.items():
            prev_code = previous.get(skill_key, None)
            if code == prev_code:
                continue
            if code is not None:
                # 新损坏（None→err 或 err_a→err_b）——skill_key 攻击者可控，过脱敏。
                logger.warning(
                    "skill integrity 变化：%s 损坏（error_code=%s）",
                    _safe_log_id(skill_key),
                    code,
                )
            else:
                # 修复（err→ok）——skill_key 攻击者可控，过脱敏。
                logger.info(
                    "skill integrity 变化：%s 已修复", _safe_log_id(skill_key)
                )

    # —— 后台循环（Task 14 lifespan wiring；顶层自愈 + shutdown）——

    async def run_loop(self) -> None:
        """顶层 while-True 自愈循环（spec §3.5/§9，brief 冻结形态）。

        单轮 ``tick_once`` 异常 → warn 后继续（``except Exception`` 不捕
        ``CancelledError``：后者派生自 ``BaseException``，lifespan cancel 时沿本
        task 传播——in-flight tick 的 cleanup 沿同协程帧闭环，F4）。

        sleep **不在 finally**：``tick_once()`` 抛出的 ``CancelledError`` 必须立即
        沿协程帧向上传播退出循环，绝不再进一次 tick 间隔 sleep（否则 cancel 路径最
        长要多等一个完整 ``PROBE_TICK_INTERVAL_SECONDS``=30s）。成功/异常两条正常路
        径的语义不变（tick → sleep）；只有 cancel 路径跳过 sleep 直接退出。
        ``shutdown()`` 置 ``_stopping`` 后循环下一轮自然退出。
        """
        while not self._stopping:
            try:
                await self.tick_once()
            except Exception:  # noqa: BLE001 - 自愈：单轮异常不终止循环
                logger.warning("B9 探测循环单轮异常（自愈，继续下一轮）", exc_info=True)
            # cancel 被下游（如 MCP init 的 except BaseException）吞掉时，tick 会正常
            # 返回——sleep 前复查 _stopping，避免协作停多等一个完整 tick 间隔。
            if self._stopping:
                break
            await asyncio.sleep(PROBE_TICK_INTERVAL_SECONDS)

    async def shutdown(self) -> None:
        self._stopping = True


# ======================================================================
# Task 13：DefaultExtensionProber（真实探测）+ error mapper
# ======================================================================

# —— 探测侧内建超时边界（spec §3.2）——
# A2A：8s 对齐 A2AClientManager 的 A2A_DISCOVERY_TIMEOUT_SECONDS（F3）。
# MCP：临时 manager 复用 mcp.py 的 per-server 5s 连接超时（F5），无需在此重复设置。
A2A_PROBE_TIMEOUT_SECONDS = 8
# R1#4 修（grep-verified）：与 A2AClientManager 实际取卡路径一致（a2a.py:114-117）。
A2A_AGENT_CARD_PATH = "/.well-known/agent-card.json"


def _map_mcp_error(error_msg: str) -> str:
    """MCPClientManager.errors 是字符串——按已知格式前缀归类。

    F5 错误串在 mcp.py 的 ``_connect_mcp_servers`` 生成（超时串含"超时"，其余为
    ``连接MCP服务器[...]出错: {str(e)}`` 携带底层异常文本）。
    """
    if "超时" in error_msg:
        return "timeout"
    lowered = error_msg.lower()
    if any(t in lowered for t in ("filenotfound", "no such file", "spawn", "executable")):
        return "spawn_failed"
    if any(t in lowered for t in ("connect", "connection", "refused", "unreachable")):
        return "connect_failed"
    if any(t in lowered for t in ("401", "403", "unauthorized", "forbidden", "auth")):
        return "auth_failed"
    return "protocol_error"


def _map_mcp_exception(exc: BaseException) -> str:
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return "timeout"
    if isinstance(exc, (FileNotFoundError, PermissionError)):
        # FileNotFoundError/PermissionError 是 OSError 子类——必须先于 OSError 判定
        return "spawn_failed"
    if isinstance(exc, (ConnectionError, OSError)):
        return "connect_failed"
    return "protocol_error"


def _map_a2a_exception(exc: BaseException) -> str:
    if isinstance(exc, httpx.TimeoutException):
        return "timeout"
    if isinstance(exc, httpx.ConnectError):
        return "connect_failed"
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        if code in (401, 403):
            return "auth_failed"
        return "protocol_error"
    return "protocol_error"


class DefaultExtensionProber:
    """P-5 ``ExtensionProber`` 真实实现（Task 14 lifespan 注入）。

    - MCP：临时单项 ``MCPClientManager`` 全握手（init→读 errors/tools→cleanup），
      **init 与 cleanup 在同一协程帧（同一 asyncio Task）闭环（INV-B9-5）**——
      anyio cancel scope 要求 aclose() 与 open 在同一 Task，否则抛 RuntimeError。
    - A2A：direct httpx GET agent-card（R9#3）；display_name 承载 ``agent_card.name``（R1#5）。
    - error_message 返回 raw ``str(exc)``——脱敏由 ``ExtensionProbeService`` 落 record 时做。
    """

    async def probe_mcp(
        self, server_name: str, config: "MCPServerConfig"
    ) -> ProbeOutcome:
        """临时单项 manager 全握手：init→读 errors/tools→cleanup 同协程闭环（INV-B9-5）。"""
        from app.domain.models.app_config import MCPConfig

        manager = MCPClientManager(mcp_config=MCPConfig(mcpServers={server_name: config}))
        started = time.monotonic()
        try:
            await manager.initialize()
            latency = int((time.monotonic() - started) * 1000)
            error_msg = manager.errors.get(server_name)
            if error_msg:
                return ProbeOutcome(
                    ok=False,
                    latency_ms=latency,
                    error_code=_map_mcp_error(error_msg),
                    error_message=error_msg,
                )
            tool_count = len(manager.tools.get(server_name, []))
            return ProbeOutcome(ok=True, latency_ms=latency, tool_count=tool_count)
        except Exception as exc:  # noqa: BLE001 - 映射为 outcome，不外泄
            latency = int((time.monotonic() - started) * 1000)
            return ProbeOutcome(
                ok=False,
                latency_ms=latency,
                error_code=_map_mcp_exception(exc),
                error_message=str(exc),
            )
        finally:
            await manager.cleanup()  # 与 init 同一协程帧（INV-B9-5）

    async def probe_a2a(self, config: "A2AServerConfig") -> ProbeOutcome:
        """direct httpx agent-card GET（R9#3）；display_name 承载 agent_card.name（R1#5）。"""
        started = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=A2A_PROBE_TIMEOUT_SECONDS) as client:
                resp = await client.get(f"{config.base_url}{A2A_AGENT_CARD_PATH}")
                resp.raise_for_status()
                payload = resp.json()
            latency = int((time.monotonic() - started) * 1000)
            name = payload.get("name") if isinstance(payload, dict) else None
            return ProbeOutcome(
                ok=True,
                latency_ms=latency,
                display_name=str(name) if name else None,
            )
        except Exception as exc:  # noqa: BLE001 - 映射为 outcome，不外泄
            # JSON 解析失败（resp.json() 抛 ValueError）落 protocol_error——mapper 兜底覆盖
            latency = int((time.monotonic() - started) * 1000)
            return ProbeOutcome(
                ok=False,
                latency_ms=latency,
                error_code=_map_a2a_exception(exc),
                error_message=str(exc),
            )
