"""D1a §3.6/§4.1 受限 admission 写 port 实现。

唯一允许的写 = record_observation_and_maybe_quarantine（observed_*/last_* +
条件 quarantine + audit，单 DB 事务）。禁止 pin 写/行政写（INV-D1-3 AST gate T8）。
"""
from __future__ import annotations

import dataclasses
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.domain.external.extension_admission import AdmissionDecision, Observation
from app.domain.models.extension_governance import (
    ADMINISTRATIVE_REASONS,
    DETECTION_REASONS,
    HASH_SCHEMA_VERSION,
)
from app.domain.services.extension_admission_logic import (
    decide_admitted,
    pin_presence,
    select_reason,
    should_persist_observation,
)
from app.domain.services.extension_hashing import a2a_surface_hash, mcp_surface_hash
from app.infrastructure.external.governance.audit import insert_audit
from app.infrastructure.models.extension_governance import (
    ExtensionModel,
    PluginMembershipModel,
)

logger = logging.getLogger(__name__)

_CATEGORY_OBSERVED_COL = {
    "surface": "observed_surface_hash",
    "artifact": "observed_artifact_hash",
    "config_fingerprint": "observed_config_fingerprint",
}
_CATEGORY_PIN_COL = {
    "surface": "surface_hash",
    "artifact": "artifact_hash",
    "config_fingerprint": "config_fingerprint",
}


def _reason_rank(reason: str) -> int:
    """T6 三分严重度（复用 §2 权威 trichotomy 集合）：行政/结构类(2) > 检测类(1) > 中性(0)。
    check_many 双侧 admitted 合并按此排序——检测原因不得被中性 ok 吞掉（F6）。"""
    if reason in ADMINISTRATIVE_REASONS:
        return 2
    if reason in DETECTION_REASONS:
        return 1
    return 0


class DbExtensionAdmissionPort:
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        mode: str,
        sampling_window_seconds: int = 3600,
    ) -> None:
        self._session_factory = session_factory
        self._mode = mode
        self.mode = mode          # Protocol 公开属性（R1#4：gates 兜底分流读取）
        self._window = sampling_window_seconds

    # ---------- Protocol: check_many ----------

    async def check_many(
        self,
        kind: str,
        ext_ids: Sequence[str],
        config_fingerprints: Mapping[str, str] | None = None,
    ) -> dict[str, AdmissionDecision]:
        try:
            return await self._check_many_inner(kind, list(ext_ids), config_fingerprints)
        except Exception:
            logger.warning("admission check_many failed (kind=%s); mode=%s", kind, self._mode,
                           exc_info=True)
            fallback = AdmissionDecision(
                admitted=decide_admitted(self._mode, "registry_unavailable"),
                reason="registry_unavailable", row_revision=None,
            )
            return {e: fallback for e in ext_ids}

    async def _check_many_inner(
        self, kind: str, ext_ids: list[str],
        config_fingerprints: Mapping[str, str] | None,
    ) -> dict[str, AdmissionDecision]:
        out: dict[str, AdmissionDecision] = {}
        # G1 的 config 指纹入参 → 与 verify_observation 完全相同的原子记录路径（R9#3）。
        # R1#1 修：config 观测结果只覆盖 config 类别——最终决策必须**合并其余必需类别**
        # 的 pin 状态（如 mcp 行 config pin 正常但 surface unpinned → enforce 仍不准入，
        # spec §4.1 unpinned/stale 语义 + §5.2 per-kind 写集）。
        fp_ids = [e for e in ext_ids if (config_fingerprints or {}).get(e) is not None]
        obs_decisions: dict[str, AdmissionDecision] = {}
        for ext_id in fp_ids:
            obs_decisions[ext_id] = await self._record_observation(
                kind, ext_id, category="config_fingerprint",
                digest=config_fingerprints[ext_id],
                under_config_fingerprint=None, obs_schema_version=HASH_SCHEMA_VERSION,
            )
        # 单批状态查（membership LEFT JOIN，不 N+1）——fp 项也要读行合并 pin 检测
        async with self._session_factory() as session:
            rows = await self._load_rows(session, kind, ext_ids)
        for ext_id in ext_ids:
            row = rows.get(ext_id)
            status_decision = self._status_decision(kind, row)
            obs = obs_decisions.get(ext_id)
            if obs is None:
                out[ext_id] = status_decision
                continue
            # 合并规则：unknown/行政类以 status_decision 为准（优先序）；两侧均达检测层时，
            # config 观测的检测结果（config_drift/unpinned/pin_stale）与其余类别 pin 检测
            # 任一不准入即不准入；reason 取首个不准入者（config 侧优先——它携带 drift 字段）
            if status_decision.reason in ("unknown", "quarantined", "disabled",
                                          "deleted", "parent_blocked"):
                out[ext_id] = dataclasses.replace(
                    status_decision,
                    observation_outcome=obs.observation_outcome,
                    config_drift_detected=obs.config_drift_detected)
            elif not obs.admitted:
                out[ext_id] = obs
            elif not status_decision.admitted:
                out[ext_id] = dataclasses.replace(
                    status_decision,
                    observation_outcome=obs.observation_outcome,
                    config_drift_detected=obs.config_drift_detected)
            elif _reason_rank(status_decision.reason) > _reason_rank(obs.reason):
                # 双侧 admitted（shadow）：按 reason 严重度合并——status 侧检测原因
                # （surface unpinned/pin_stale）胜过 config 侧中性 ok，不被吞掉（F6 修）；
                # 携带 config 观测 outcome/drift 字段（正交信号）。
                out[ext_id] = dataclasses.replace(
                    status_decision,
                    observation_outcome=obs.observation_outcome,
                    config_drift_detected=obs.config_drift_detected)
            else:
                # 同级或 config 侧更严重：保留 obs（携带 drift 字段，config 侧优先既有约定）。
                out[ext_id] = obs
        return out

    def _status_decision(self, kind: str, row: dict[str, Any] | None) -> AdmissionDecision:
        if row is None:
            return AdmissionDecision(
                admitted=decide_admitted(self._mode, "unknown"),
                reason="unknown", row_revision=None)
        detection = self._pin_detection(kind, row)
        reason = select_reason(
            row_exists=True, status=row["status"],
            parent_blocked=row["parent_blocked"], detection=detection)
        return AdmissionDecision(
            admitted=decide_admitted(self._mode, reason),
            reason=reason, row_revision=row["row_revision"])

    def _pin_detection(self, kind: str, row: dict[str, Any]) -> str | None:
        """必需类别任一 unpinned/pin_stale → 该 reason（check_many 状态面不做 surface 比对）。"""
        from app.domain.services.extension_admission_logic import REQUIRED_OBSERVED_CATEGORIES
        for category in sorted(REQUIRED_OBSERVED_CATEGORIES[kind]):
            state = pin_presence(
                row[_CATEGORY_PIN_COL[category]], row["hash_schema_version"], HASH_SCHEMA_VERSION)
            if state != "pinned":
                return "unpinned" if state == "unpinned" else "pin_stale"
        return None

    def _detection_for(self, category: str, digest: str, row: dict[str, Any]) -> tuple[str | None, bool]:
        """§3.6 ③ digest-aware pin 比对（正常流与 CAS 输家重判共用同一判定路径）：
        返回 (detection_reason, drift_detected)——surface/artifact 失配→pin_mismatch；
        config 失配→config_drift（drift_detected=True）；否则按 pin_presence 给
        unpinned/pin_stale；pinned 且匹配→None。"""
        pin_state = pin_presence(
            row[_CATEGORY_PIN_COL[category]], row["hash_schema_version"], HASH_SCHEMA_VERSION)
        mismatch = pin_state == "pinned" and digest != row[_CATEGORY_PIN_COL[category]]
        drift_detected = category == "config_fingerprint" and mismatch
        if mismatch:
            detection = "config_drift" if category == "config_fingerprint" else "pin_mismatch"
        elif pin_state != "pinned":
            detection = "unpinned" if pin_state == "unpinned" else "pin_stale"
        else:
            detection = None
        return detection, drift_detected

    def _decision_from_row(
        self, kind: str, row: dict[str, Any] | None, category: str, digest: str,
    ) -> AdmissionDecision:
        """§3.6 CAS 输家重判入口：对 fresh 行重算**完整 admission 决策**——含本次 digest 与
        最新 pin 的比对（F1 修：旧实现仅回 pin_presence/status 判定，会把「本次 digest 与
        现行 pin 失配」误判为 ok fail-open 放行一次错配配置）。绝不 quarantine（调用点已返回）。"""
        if row is None:
            return AdmissionDecision(
                admitted=decide_admitted(self._mode, "unknown"),
                reason="unknown", row_revision=None)
        detection, drift_detected = self._detection_for(category, digest, row)
        reason = select_reason(
            row_exists=True, status=row["status"],
            parent_blocked=row["parent_blocked"], detection=detection)
        return AdmissionDecision(
            admitted=decide_admitted(self._mode, reason), reason=reason,
            row_revision=row["row_revision"], config_drift_detected=drift_detected)

    # ---------- Protocol: verify_observation ----------

    async def verify_observation(
        self, kind: str, ext_id: str, obs: Observation,
    ) -> AdmissionDecision:
        try:
            digest = self._digest_for(kind, obs)
            return await self._record_observation(
                kind, ext_id, category=obs.category, digest=digest,
                under_config_fingerprint=obs.under_config_fingerprint,
                obs_schema_version=obs.schema_version)
        except Exception:
            logger.warning("admission verify failed (kind=%s ext=%s); mode=%s",
                           kind, ext_id, self._mode, exc_info=True)
            return AdmissionDecision(
                admitted=decide_admitted(self._mode, "registry_unavailable"),
                reason="registry_unavailable", row_revision=None)

    def _digest_for(self, kind: str, obs: Observation) -> str:
        if obs.category == "surface":
            return mcp_surface_hash(obs.payload) if kind == "mcp" else a2a_surface_hash(obs.payload)
        # artifact / config_fingerprint：调用方已算 hash（R9#3 / G5 hash 已在手）
        return str(obs.payload)

    # ---------- §3.6 原子路径 ----------

    async def _record_observation(
        self, kind: str, ext_id: str, *, category: str, digest: str,
        under_config_fingerprint: str | None, obs_schema_version: int,
    ) -> AdmissionDecision:
        now = datetime.now(timezone.utc)
        async with self._session_factory() as session:
            async with session.begin():
                row = await self._load_row(session, kind, ext_id)
                if row is None:
                    return AdmissionDecision(
                        admitted=decide_admitted(self._mode, "unknown"),
                        reason="unknown", row_revision=None)

                # (ii) surface 持久化门（R34#2/R35#F1）：非持久化早退
                if category == "surface" and row["observed_config_fingerprint"] is not None:
                    if under_config_fingerprint != row["observed_config_fingerprint"]:
                        reason = select_reason(
                            row_exists=True, status=row["status"],
                            parent_blocked=row["parent_blocked"], detection="config_drift")
                        return AdmissionDecision(
                            admitted=decide_admitted(self._mode, reason), reason=reason,
                            row_revision=row["row_revision"], observation_outcome="conflict")

                observed_col = _CATEGORY_OBSERVED_COL[category]
                stored = row[observed_col]
                version_stale = (row["observed_hash_schema_version"] is not None
                                 and row["observed_hash_schema_version"] != HASH_SCHEMA_VERSION)
                persist = should_persist_observation(
                    stored_value=stored, new_value=digest,
                    last_observed_at=row["last_observed_at"], now=now,
                    window_seconds=self._window,
                    stored_schema_version=row["observed_hash_schema_version"],
                    current_schema_version=HASH_SCHEMA_VERSION)

                effective_stored = None if version_stale else stored
                value_write = persist and (effective_stored is None or digest != effective_stored)
                pin_state = pin_presence(
                    row[_CATEGORY_PIN_COL[category]], row["hash_schema_version"], HASH_SCHEMA_VERSION)
                mismatch = pin_state == "pinned" and digest != row[_CATEGORY_PIN_COL[category]]
                drift_detected = category == "config_fingerprint" and mismatch  # R47#7 可比 pin 前置

                latest_revision = row["row_revision"]
                outcome = "unchanged"
                if persist:
                    values: dict[str, Any] = {
                        observed_col: digest,
                        "last_observed_at": now,
                        "observed_hash_schema_version": HASH_SCHEMA_VERSION,
                    }
                    if version_stale:
                        # R43#3 版本门：同一原子事务先清三 observed 列（旧版本观测整体作废）
                        for col in _CATEGORY_OBSERVED_COL.values():
                            values.setdefault(col, None)
                        values[observed_col] = digest
                    if category == "config_fingerprint" and (stored is None or digest != stored):
                        # R33#2：config 值变化同事务失效陈旧 surface——
                        # **含首次 NULL→值（保守触发，R1#2 修）**：防「旧配置 surface +
                        # 新配置 fingerprint」混合快照被 approve（spec §3.6 (i)）
                        values["observed_surface_hash"] = None
                    if category != "config_fingerprint":
                        # §5.2 写矩阵：**仅匹配的 pinned 比对**打 last_verified_at；失配打
                        # last_mismatch_at；unpinned/pin_stale 两者都不打（无 pin 无「verified」
                        # 语义，F8 修——旧 else 分支把 unpinned 首观测误标 verified）。
                        if mismatch:
                            values["last_mismatch_at"] = now
                        elif pin_state == "pinned":
                            values["last_verified_at"] = now
                    if value_write:
                        values["row_revision"] = ExtensionModel.row_revision + 1
                    result = await session.execute(
                        update(ExtensionModel)
                        .where(ExtensionModel.id == row["id"],
                               ExtensionModel.row_revision == row["row_revision"])
                        .values(**values)
                        .returning(ExtensionModel.row_revision))
                    returned = result.scalar_one_or_none()
                    if returned is None:
                        # §3.6 CAS 输家：重读 fresh 行并重算**完整判定**（含 digest 与最新 pin
                        # 比对），不 quarantine——旧观测不得隔离新 pin；但也不得以 pin_presence-only
                        # 重判掩盖本次 digest 与现行 pin 的真实失配（F1 修：fail-open→完整重判）。
                        fresh = await self._load_row(session, kind, ext_id)
                        return dataclasses.replace(
                            self._decision_from_row(kind, fresh, category, digest),
                            observation_outcome="conflict")
                    latest_revision = returned
                    outcome = "persisted"
                    # audit 分派（spec §3.5/§3.6；R1#3a 修）：
                    # - 首次 NULL→值 → 恒 observed_first（**不判 drift/mismatch audit**——
                    #   §3.6 (i) 字面「首次仅记 observed_first」；field/reason 判定不受影响，
                    #   codex R1#3b 按 spec 驳回并在此显式注释）
                    # - 值变化失配（surface/artifact）：若本次将进入 enforce quarantine 分支，
                    #   **跳过**独立 pin_mismatch——双 audit 由 quarantine 赢家一次性写
                    #   （spec「仅赢家写两条、输家零条」；否则三条超基数）
                    will_quarantine = (mismatch and category != "config_fingerprint"
                                       and self._mode == "enforce" and row["status"] == "active")
                    if effective_stored is None:
                        await insert_audit(session, kind=kind, ext_id=ext_id,
                                           extension_id=row["id"], event="observed_first")
                    elif value_write and mismatch and category != "config_fingerprint" and not will_quarantine:
                        await insert_audit(session, kind=kind, ext_id=ext_id,
                                           extension_id=row["id"], event="pin_mismatch")
                    elif value_write and drift_detected:
                        await insert_audit(session, kind=kind, ext_id=ext_id,
                                           extension_id=row["id"], event="config_drift_detected")

                # quarantine 分支：仅 surface/artifact（R11#1）
                if (mismatch and category != "config_fingerprint"
                        and self._mode == "enforce" and row["status"] == "active"):
                    correlation = uuid.uuid4()
                    q = await session.execute(
                        update(ExtensionModel)
                        .where(ExtensionModel.id == row["id"],
                               ExtensionModel.status == "active",
                               ExtensionModel.row_revision == latest_revision)
                        .values(status="quarantined", quarantine_reason="pin_mismatch",
                                last_mismatch_at=now,
                                row_revision=ExtensionModel.row_revision + 1)
                        .returning(ExtensionModel.row_revision))
                    q_rev = q.scalar_one_or_none()
                    if q_rev is not None:
                        # 赢家双 audit（共享 correlation；豁免值变化去重，R43#1）
                        await insert_audit(session, kind=kind, ext_id=ext_id,
                                           extension_id=row["id"], event="pin_mismatch",
                                           correlation_id=correlation)
                        await insert_audit(session, kind=kind, ext_id=ext_id,
                                           extension_id=row["id"], event="quarantined",
                                           correlation_id=correlation)
                        latest_revision = q_rev
                        row = dict(row, status="quarantined")

                # 正常流最终判定复用同一 digest-aware 判定路径（与 CAS 输家重判同源）；
                # 守卫值/pin 在成功 CAS 后与 r0 一致，post-quarantine 的 status 变更不影响
                # pin 列 → detection 恒等于内联计算（drift_detected 沿用 r0 局部值）。
                detection, _ = self._detection_for(category, digest, row)
                reason = select_reason(
                    row_exists=True, status=row["status"],
                    parent_blocked=row["parent_blocked"], detection=detection)
                return AdmissionDecision(
                    admitted=decide_admitted(self._mode, reason), reason=reason,
                    row_revision=latest_revision, observation_outcome=outcome,
                    config_drift_detected=drift_detected)

    # ---------- 行加载（membership join 派生 parent_blocked，§8.5）----------

    async def _load_row(self, session: AsyncSession, kind: str, ext_id: str) -> dict[str, Any] | None:
        rows = await self._load_rows(session, kind, [ext_id])
        return rows.get(ext_id)

    async def _load_rows(
        self, session: AsyncSession, kind: str, ext_ids: list[str],
    ) -> dict[str, dict[str, Any]]:
        parent = ExtensionModel.__table__.alias("parent")
        stmt = (
            select(
                ExtensionModel.id, ExtensionModel.ext_id, ExtensionModel.status,
                ExtensionModel.row_revision, ExtensionModel.hash_schema_version,
                ExtensionModel.artifact_hash, ExtensionModel.surface_hash,
                ExtensionModel.config_fingerprint,
                ExtensionModel.observed_surface_hash, ExtensionModel.observed_artifact_hash,
                ExtensionModel.observed_config_fingerprint,
                ExtensionModel.observed_hash_schema_version, ExtensionModel.last_observed_at,
                parent.c.status.label("parent_status"),
            )
            .select_from(ExtensionModel)
            .outerjoin(PluginMembershipModel,
                       PluginMembershipModel.child_extension_id == ExtensionModel.id)
            .outerjoin(parent, parent.c.id == PluginMembershipModel.plugin_id)
            .where(ExtensionModel.kind == kind,
                   ExtensionModel.ext_id.in_(ext_ids),
                   ExtensionModel.deleted_at.is_(None))
        )
        result = await session.execute(stmt)
        out: dict[str, dict[str, Any]] = {}
        for m in result.mappings():
            row = dict(m)
            row["parent_blocked"] = row.pop("parent_status") in ("quarantined", "disabled", "deleted")
            out[row["ext_id"]] = row
        return out
