"""D1a §6.1 delta reconciler（单一 writer 的批量驱动器——insert/update 权见 spec §6.1）。

实现约束（R2#8）：本模块不得 import AppConfigService（Protocol 反向解耦）。
分支二分（R47#3+#8）：install_context 非 None=独立安装/重装分支（不依赖 diff、
零 drift 观测零 reset）；None=纯 reconcile（added/removed/changed 三路）。
"""
from __future__ import annotations

import logging
from typing import Protocol

from app.domain.external.extension_admission import (
    InstallContext,
    Observation,
    UninstallContext,
)
from app.domain.models.app_config import A2AConfig, MCPConfig
from app.domain.models.extension_governance import HASH_SCHEMA_VERSION, TRUST_ORIGINS
from app.domain.services.extension_hashing import (
    a2a_config_fingerprint,
    mcp_config_fingerprint,
)
from app.domain.services.extension_scan import canonicalize_source_ref

logger = logging.getLogger(__name__)

D1A_STARTUP_RECONCILE_LOCK_KEY = 74520011   # §3.6：pg_try_advisory_lock 固定 key（单跑者）


class ConfigDeltaReconciler(Protocol):
    async def reconcile_mcp_delta(self, old: MCPConfig | None, new: MCPConfig | None, *,
                                  actor_id: str | None = None,
                                  install_context: InstallContext | None = None,
                                  target_ext_id: str | None = None,
                                  uninstall_context: UninstallContext | None = None) -> None: ...
    async def reconcile_a2a_delta(self, old: A2AConfig | None, new: A2AConfig | None, *,
                                  actor_id: str | None = None,
                                  install_context: InstallContext | None = None,
                                  target_ext_id: str | None = None,
                                  uninstall_context: UninstallContext | None = None) -> None: ...


class ExtensionReconciler:
    def __init__(self, write_port, admission_port) -> None:
        self._write = write_port
        self._admission = admission_port

    async def reconcile_mcp_delta(self, old, new, *, actor_id=None, install_context=None,
                                  target_ext_id=None, uninstall_context=None) -> None:
        old_servers = dict(old.mcpServers) if old and old.mcpServers else {}
        new_servers = dict(new.mcpServers) if new and new.mcpServers else {}
        await self._reconcile(
            kind="mcp", old_keys=set(old_servers), new_keys=set(new_servers),
            fingerprint_of=lambda k: mcp_config_fingerprint(new_servers[k]),
            changed=lambda k: mcp_config_fingerprint(old_servers[k]) != mcp_config_fingerprint(new_servers[k]),
            install_context=install_context, target_ext_id=target_ext_id,
            uninstall_context=uninstall_context)

    async def reconcile_a2a_delta(self, old, new, *, actor_id=None, install_context=None,
                                  target_ext_id=None, uninstall_context=None) -> None:
        old_servers = {s.id: s for s in (old.a2a_servers if old else [])}
        new_servers = {s.id: s for s in (new.a2a_servers if new else [])}
        await self._reconcile(
            kind="a2a", old_keys=set(old_servers), new_keys=set(new_servers),
            fingerprint_of=lambda k: a2a_config_fingerprint(new_servers[k].base_url),
            changed=lambda k: old_servers[k].base_url != new_servers[k].base_url,
            install_context=install_context, target_ext_id=target_ext_id,
            uninstall_context=uninstall_context)

    async def _reconcile(self, *, kind, old_keys, new_keys, fingerprint_of, changed,
                         install_context, target_ext_id, uninstall_context) -> None:
        if install_context is not None:
            # (a) 独立安装/重装分支：显式 record_install（§6.4 暂态重试由此收敛），
            #     不做带外 drift 观测/reset——安装即新 TOFU 基线（R47#3+#8）
            if target_ext_id is None:
                raise ValueError("install_context requires target_ext_id")
            await self._write.record_install(kind, target_ext_id, install_context)
            return
        removed = old_keys - new_keys
        if removed and uninstall_context is None:
            # R28#1：注解不阻止显式 None——删除分支先拒绝
            raise ValueError("uninstall_context is required for delete delta")
        for key in sorted(new_keys - old_keys):
            await self._write.record_reconciled_seen(
                kind, key, source_type="config", source_ref=None, version=None,
                trust_origin="user_installed")
        for key in sorted(removed):
            await self._write.record_delete(kind, key, uninstall_context=uninstall_context)
        changed_keys = [k for k in sorted(old_keys & new_keys) if changed(k)]
        if changed_keys:
            fingerprints = {k: fingerprint_of(k) for k in changed_keys}
            # R6#1：delta changed 与 startup 观测逐字同语义——共享 _observe_config_and_reset。
            await self._observe_config_and_reset(kind, fingerprints)

    async def run_startup_reconcile(self, *, app_config, skill_repository,
                                    read_port, plugins_root=None) -> None:
        """§6.1-2 startup 全量对账（四段全序第④段；advisory lock 由 lifespan 层持有）。"""
        rows = await read_port.list_live_rows()
        rows_by_kind: dict[str, dict[str, object]] = {}
        for r in rows:
            rows_by_kind.setdefault(r.kind, {})[r.ext_id] = r

        # --- mcp/a2a：缺行补建（委托唯一 writer，R17#2）+ config 观测（drift→reset）---
        mcp_servers = dict(app_config.mcp_config.mcpServers) if app_config.mcp_config else {}
        a2a_servers = {s.id: s for s in (app_config.a2a_config.a2a_servers
                                         if app_config.a2a_config else [])}
        await self.reconcile_mcp_delta(None, app_config.mcp_config)
        await self.reconcile_a2a_delta(None, app_config.a2a_config)
        if mcp_servers:
            await self._observe_config_and_reset(
                "mcp", {n: mcp_config_fingerprint(c) for n, c in mcp_servers.items()})
        if a2a_servers:
            await self._observe_config_and_reset(
                "a2a", {i: a2a_config_fingerprint(s.base_url) for i, s in a2a_servers.items()})

        # --- 源缺失/恢复（两段确认，§6.3）---
        await self._reconcile_sources("mcp", rows_by_kind.get("mcp", {}), set(mcp_servers))
        await self._reconcile_sources("a2a", rows_by_kind.get("a2a", {}), set(a2a_servers))

        # --- skill：缺行补建 + artifact 观测（R4-03，本地盘零远程 I/O）---
        skills = await skill_repository.list()   # R1#14：权威接口=list()（skill_repository.py:16），无 get_all
        skill_ids = set()
        for skill in skills:
            skill_ids.add(skill.id)
            if skill.id not in rows_by_kind.get("skill", {}):
                # T17 review 硬化：SkillSourceType(str,Enum) 是 str 子类，落库良性（Text 列 asyncpg
                # 绑字符串缓冲=值），但 str(member) 会得成员名而非值——取 .value 归一为纯 str，与本
                # 方法上方 delta 路径传纯字面量（source_type="config"）一致，杜绝未来 str()化 footgun。
                _skill_src = getattr(skill, "source_type", "local") or "local"
                _skill_src = getattr(_skill_src, "value", _skill_src)
                await self._write.record_reconciled_seen(
                    "skill", skill.id, source_type=_skill_src,
                    source_ref=canonicalize_source_ref(getattr(skill, "source_ref", None)),
                    version=(skill.manifest or {}).get("version"),
                    trust_origin=(skill.trust_origin
                                  if skill.trust_origin in TRUST_ORIGINS
                                  else "user_installed"))
                # R5#5 source_ref 不丢；R7#4 trust_origin 归一化（spec §2 不透传自由 str；
                # TRUST_ORIGINS import 自 T1 词表）
            skill_dir = skill_repository.get_skill_dir(skill.id)
            if skill_dir and skill_dir.exists():
                from app.domain.services.skills_guard import SkillsGuard
                content_hash = SkillsGuard.compute_content_hash(skill_dir)
                await self._admission.verify_observation(
                    "skill", skill.id,
                    Observation(category="artifact", payload=content_hash,
                                schema_version=HASH_SCHEMA_VERSION))
        await self._reconcile_sources("skill", rows_by_kind.get("skill", {}), skill_ids)

        # --- plugin：bundle 目录存在性 + artifact 观测（§5.1 plugin 行；行来自 PR-6）---
        plugin_rows = rows_by_kind.get("plugin", {})
        present_plugins = set()
        for ext_id, row in plugin_rows.items():
            bundle_dir = (plugins_root / ext_id / (row.version or "")) if plugins_root else None
            if bundle_dir and bundle_dir.exists():
                present_plugins.add(ext_id)
                from app.domain.services.skills_guard import SkillsGuard
                await self._admission.verify_observation(
                    "plugin", ext_id,
                    Observation(category="artifact",
                                payload=SkillsGuard.compute_content_hash(bundle_dir),
                                schema_version=HASH_SCHEMA_VERSION))
        await self._reconcile_sources("plugin", plugin_rows, present_plugins)

    async def _reconcile_sources(self, kind, rows: dict, present: set[str]) -> None:
        for ext_id, row in rows.items():
            if ext_id in present:
                if row.source_missing_at is not None:
                    await self._write.mark_source_restored(kind, ext_id)   # R1#14
            elif row.source_missing_at is None:
                await self._write.mark_source_missing(kind, ext_id)        # 首次：宽限
            else:
                await self._write.record_reconciled_missing(kind, ext_id)  # 二次确认（R33#5 分流在 T8）

    async def _observe_config_and_reset(self, kind, fingerprints: dict[str, str]) -> None:
        decisions = await self._admission.check_many(
            kind, list(fingerprints), config_fingerprints=fingerprints)
        for key, decision in decisions.items():
            if decision.config_drift_detected and decision.row_revision is not None:
                if not await self._write.reset_pins_after_config_drift(
                        kind, key, row_revision=decision.row_revision):
                    logger.info("startup pin-reset CAS lost for %s/%s", kind, key)
