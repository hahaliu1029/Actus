"""D1a §9.2 治理服务——行政动作 + 观测刷新 + 批量 + 审计翻页（治理路由消费面）。

四面 port 组合（T8 read/write/admission + T18 prober）+ app_config_provider 现值 +
skill_repository / plugins_root（skill/plugin 本地 artifact 重算）：

- ``summary()``：read_port.governance_counters + admission.mode（GET /governance on 路）。
- 行政动作（``quarantine`` / ``reapprove`` / ``set_enabled``）：薄委托 WritePort（§2 前置
  表 + §3.6 CAS 由 port 保证；异常经 interfaces exception_handlers 映射 HTTP，T19 已注册）。
- ``refresh_observation``（§5.2）：mcp/a2a=probe→config 观测→(drift 则 reset)→surface 观测；
  skill/plugin=本地 ``compute_content_hash``→artifact 观测。**顺序唯一化**（R32#2/R33#2）：
  config 观测 → reset_pins_after_config_drift → surface 观测；reset CAS 丢失 → 中止（不做
  surface 观测）抛 RevisionConflictError。观测 conflict → RevisionConflictError（单项 409）。
  I/O/hash 异常统一「可重试获取失败」→ probe_failed（绝不 500）。
- ``approve_pins`` / ``refresh_observations_batch``：批量复用单项逻辑，逐项 outcome 回报
  （批量不 404——deleted/unknown→skipped_invalid_state；conflict 原样回报）。
- ``list_audit``：复合游标翻页委托 ReadPort（cursor=(created_at, id)）。

分层：application 层——**禁 import FastAPI/SQLAlchemy**。RefreshResult/ItemOutcome 是
纯 dataclass DTO（路由序列化消费）。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

from app.application.errors.exceptions import NotFoundError
from app.domain.external.extension_admission import (
    AdmissionDecision,
    AuditPage,
    Observation,
)
from app.domain.models.extension_governance import (
    HASH_SCHEMA_VERSION,
    RevisionConflictError,
)
from app.domain.services.extension_hashing import (
    a2a_config_fingerprint,
    mcp_config_fingerprint,
)
from app.domain.services.skills_guard import SkillsGuard

logger = logging.getLogger(__name__)

# observed_surface 描述截断（脱敏摘要，非全文——对齐 ExtensionInstallService._SURFACE_DESC_MAX）
_SURFACE_DESC_MAX = 200


@dataclass(frozen=True)
class RefreshResult:
    """单项 refresh 结果（outcome ∈ refreshed / probe_failed；conflict/skipped 经批量归约）。"""
    outcome: str
    row_revision: int | None
    surface_summary: Any = None


@dataclass(frozen=True)
class ItemOutcome:
    """批量逐项回报（approve_pins / refresh_observations_batch）。"""
    kind: str
    ext_id: str
    outcome: str
    row_revision: int | None = None


def _reduce_refresh_outcome(decisions: list[AdmissionDecision]) -> str:
    """spec §9.2「refreshed=全部必需观测 ∈ {persisted, unchanged}」归约（R3#6）。

    任一 conflict → RevisionConflictError（单项→409；批量捕获→conflict）。
    "none"（本次调用不含观测）对 valid 行不应出现——保守按 conflict 处理 + warn
    （不假报 refreshed）。
    """
    outcomes = [d.observation_outcome for d in decisions]
    if any(o == "conflict" for o in outcomes):
        raise RevisionConflictError("observation conflict during refresh")
    if all(o in ("persisted", "unchanged") for o in outcomes):
        return "refreshed"
    logger.warning("unexpected observation_outcome 'none' in refresh: %s", outcomes)
    raise RevisionConflictError("incomplete observation set")


class ExtensionGovernanceService:
    def __init__(
        self,
        read_port: Any,
        write_port: Any,
        admission_port: Any,
        prober: Any,
        app_config_provider: Any,
        skill_repository: Any,
        plugins_root: Any,
    ) -> None:
        self._read = read_port
        self._write = write_port
        self._admission = admission_port
        self._prober = prober
        self._app_config_provider = app_config_provider
        self._skill_repository = skill_repository
        self._plugins_root = plugins_root

    # ---- 读模型 ----
    async def summary(self) -> dict[str, Any]:
        counters = await self._read.governance_counters()
        return {
            "mode": self._admission.mode,
            "unpinned_count": counters.unpinned_count,
            "missing_observation_count": counters.missing_observation_count,
            "quarantined_count": counters.quarantined_count,
        }

    async def list_audit(
        self,
        *,
        kind: str | None = None,
        ext_id: str | None = None,
        event: str | None = None,
        cursor: str | None = None,
        limit: int = 50,
    ) -> AuditPage:
        return await self._read.list_audit(
            kind=kind, ext_id=ext_id, event=event, cursor=cursor, limit=limit)

    # ---- 行政动作（薄委托；§2 前置 + §3.6 CAS 由 WritePort 保证）----
    async def quarantine(
        self, kind: str, ext_id: str, *,
        expected_row_revision: int, actor_id: str, note: str | None = None,
    ) -> int:
        return await self._write.quarantine(
            kind, ext_id, expected_row_revision=expected_row_revision,
            actor_user_id=actor_id, note=note)

    async def reapprove(
        self, kind: str, ext_id: str, *, expected_row_revision: int, actor_id: str,
    ) -> int:
        return await self._write.reapprove(
            kind, ext_id, expected_row_revision=expected_row_revision,
            actor_user_id=actor_id)

    async def set_enabled(
        self, kind: str, ext_id: str, *,
        enabled: bool, expected_row_revision: int, actor_id: str,
    ) -> int:
        return await self._write.set_governance_enabled(
            kind, ext_id, enabled=enabled,
            expected_row_revision=expected_row_revision, actor_user_id=actor_id)

    # ---- 观测刷新（单项）----
    async def refresh_observation(self, kind: str, ext_id: str) -> RefreshResult:
        row = await self._read.get_row(kind, ext_id)
        if row is None:
            raise NotFoundError(msg=f"extension {kind}/{ext_id} not found")
        if kind in ("mcp", "a2a"):
            entry = self._config_entry(kind, ext_id)   # app_config_provider 现值；缺→probe_failed
            if entry is None:
                return RefreshResult(outcome="probe_failed", row_revision=row.row_revision)
            outcome = await self._probe_safe(kind, ext_id, entry)
            if outcome is None or not outcome.ok or outcome.surface_payload is None:
                return RefreshResult(outcome="probe_failed", row_revision=row.row_revision)
            fp = (mcp_config_fingerprint(entry) if kind == "mcp"
                  else a2a_config_fingerprint(entry.base_url))
            cfg_decision = (await self._admission.check_many(
                kind, [ext_id], config_fingerprints={ext_id: fp}))[ext_id]
            if cfg_decision.observation_outcome == "conflict":
                raise RevisionConflictError("config observation conflict")   # R36#F6
            if cfg_decision.config_drift_detected:
                # 顺序唯一化：reset 在 surface 观测之前；CAS 丢失 → 中止不做 surface（R33#2）
                if cfg_decision.row_revision is None or not await self._write.reset_pins_after_config_drift(
                        kind, ext_id, row_revision=cfg_decision.row_revision):
                    raise RevisionConflictError("pin-reset CAS lost")
            surface_decision = await self._admission.verify_observation(
                kind, ext_id,
                Observation(category="surface", payload=outcome.surface_payload,
                            schema_version=HASH_SCHEMA_VERSION, under_config_fingerprint=fp))
            if surface_decision.observation_outcome == "conflict":
                raise RevisionConflictError("surface observation conflict")
            # config+surface 双观测归约——单跑 config 或 surface 假报 refreshed 的路径由此关闭
            return RefreshResult(
                outcome=_reduce_refresh_outcome([cfg_decision, surface_decision]),
                row_revision=surface_decision.row_revision,
                surface_summary=self._redacted_summary(kind, outcome))
        # skill / plugin：本地重算
        path = self._artifact_dir(kind, ext_id, row)
        try:
            if path is None or not path.exists():
                raise FileNotFoundError(path)
            content_hash = SkillsGuard.compute_content_hash(path)
        except Exception:  # noqa: BLE001 - R3#5：I/O/hash 异常统一「可重试获取失败」
            logger.warning(
                "refresh artifact recompute failed for %s/%s", kind, ext_id, exc_info=True)
            return RefreshResult(outcome="probe_failed", row_revision=row.row_revision)
        decision = await self._admission.verify_observation(
            kind, ext_id,
            Observation(category="artifact", payload=content_hash,
                        schema_version=HASH_SCHEMA_VERSION))
        return RefreshResult(outcome=_reduce_refresh_outcome([decision]),
                             row_revision=decision.row_revision)

    # ---- 观测刷新（批量）----
    async def refresh_observations_batch(
        self, *, all: bool = False, items: Sequence[Any] | None = None,
    ) -> list[ItemOutcome]:
        if all:
            targets = [(r.kind, r.ext_id) for r in await self._read.list_live_rows()]
        else:
            targets = [(it.kind, it.ext_id) for it in (items or [])]
        outcomes: list[ItemOutcome] = []
        for kind, ext_id in targets:
            try:
                result = await self.refresh_observation(kind, ext_id)
                outcomes.append(ItemOutcome(
                    kind=kind, ext_id=ext_id, outcome=result.outcome,
                    row_revision=result.row_revision))
            except NotFoundError:
                outcomes.append(ItemOutcome(
                    kind=kind, ext_id=ext_id, outcome="skipped_invalid_state"))
            except RevisionConflictError:
                outcomes.append(ItemOutcome(kind=kind, ext_id=ext_id, outcome="conflict"))
        return outcomes

    # ---- pin 批量转正 ----
    async def approve_pins(
        self, *, all: bool = False, items: Sequence[Any] | None = None, actor_id: str,
    ) -> list[ItemOutcome]:
        outcomes: list[ItemOutcome] = []
        if all:
            rows = [r for r in await self._read.list_live_rows() if r.status == "active"]
            for r in rows:
                # R7#3：Port 合同四参必填（漏 kind/ext_id/actor 会 TypeError 且 audit actor 错）
                oc = await self._write.approve_pin(
                    r.kind, r.ext_id, expected_row_revision=r.row_revision,
                    actor_user_id=actor_id)
                outcomes.append(ItemOutcome(kind=r.kind, ext_id=r.ext_id, outcome=oc))
            return outcomes
        for it in (items or []):
            row = await self._read.get_row(it.kind, it.ext_id)
            if row is None:                       # None/deleted → skipped_invalid_state（不 404）
                outcomes.append(ItemOutcome(
                    kind=it.kind, ext_id=it.ext_id, outcome="skipped_invalid_state"))
                continue
            # row_revision 可空（R36#7）：显式给则 CAS 用之，否则回退 fetched row.row_revision
            expected = (it.expected_row_revision
                        if getattr(it, "expected_row_revision", None) is not None
                        else row.row_revision)
            oc = await self._write.approve_pin(
                it.kind, it.ext_id, expected_row_revision=expected, actor_user_id=actor_id)
            outcomes.append(ItemOutcome(kind=it.kind, ext_id=it.ext_id, outcome=oc))
        return outcomes

    # ---- helpers ----
    def _config_entry(self, kind: str, ext_id: str) -> Any:
        app_config = self._app_config_provider()
        if kind == "mcp":
            return app_config.mcp_config.mcpServers.get(ext_id)
        return next(
            (s for s in app_config.a2a_config.a2a_servers if s.id == ext_id), None)

    async def _probe_safe(self, kind: str, ext_id: str, config: Any) -> Any:
        """§7.1 对称语义：probe 失败不阻塞——try/except 返 None → 装未 pin/probe_failed。"""
        try:
            if kind == "mcp":
                return await self._prober.probe_mcp(ext_id, config)
            return await self._prober.probe_a2a(config)
        except Exception:  # noqa: BLE001 - probe 失败 → probe_failed，不外泄
            logger.warning(
                "D1a refresh-time probe 失败（非致命，probe_failed）: %s/%s",
                kind, ext_id, exc_info=True)
            return None

    def _artifact_dir(self, kind: str, ext_id: str, row: Any) -> Any:
        if kind == "skill":
            return self._skill_repository.get_skill_dir(ext_id)
        # plugin：plugins_root/{ext_id}/{version}（对齐 reconciler.run_startup_reconcile）
        if self._plugins_root is None:
            return None
        return self._plugins_root / ext_id / (row.version or "")

    def _redacted_summary(self, kind: str, outcome: Any) -> Any:
        """probe 表面脱敏摘要（对齐 ExtensionInstallService._redacted_*surface）。"""
        payload = outcome.surface_payload
        if kind == "mcp":
            if not isinstance(payload, list):
                return None
            return [
                {"name": t.get("name"), "description": _truncate(t.get("description"))}
                for t in payload if isinstance(t, dict)
            ]
        if not isinstance(payload, dict):
            return None
        skills = payload.get("skills")
        skill_names = (
            [s.get("name") for s in skills if isinstance(s, dict)]
            if isinstance(skills, list) else None)
        return {
            "name": payload.get("name"),
            "description": _truncate(payload.get("description")),
            "skills": skill_names,
        }


def _truncate(text: Any) -> str | None:
    if text is None:
        return None
    return str(text)[:_SURFACE_DESC_MAX]
