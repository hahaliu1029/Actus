"""Skill 统一工具层"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shlex
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.domain.external.sandbox import SandboxAccessor
from app.domain.models.skill import Skill, SkillRuntimeType
from app.domain.models.tool_result import (
    AllowError,
    AllowSuccess,
    Asked,
    DecisionReason,
    ToolOutcome,
    ToolResult,
)
from app.domain.services.extension_admission_gates import skill_invoke_gate
from core.config import get_settings

from .a2a import A2ATool
from .base import BaseTool
from .mcp import MCPTool
from .skill_bundle_sync import SkillBundleSyncManager

logger = logging.getLogger(__name__)
TOOL_NAME_MAX_LENGTH = 64


def _skill_tool_function_name(
    skill_slug: str, tool_name: str, name_index: dict[str, int]
) -> str:
    """Single source of truth for the generated ``skill_{slug}_{tool}`` name
    (+ ``_N`` dedup + sha1 cap).

    Extracted verbatim from the original ``SkillTool._build_function_name``
    body, adapting the one ``self.`` reference: ``self._normalize_function_part``
    → ``SkillTool._normalize_function_part`` (a ``@classmethod``, callable on the
    class). Shared by the preserved bound method (monkeypatch contract) and the
    sandbox-free ``SkillTool.generate_tool_names`` (S4 §13/R7-2). [codex-R4-F1]
    """
    slug_part = SkillTool._normalize_function_part(skill_slug)
    tool_part = SkillTool._normalize_function_part(tool_name)
    base = f"skill_{slug_part}_{tool_part}"
    suffix_num = name_index.get(base, 0)
    name_index[base] = suffix_num + 1

    candidate = base if suffix_num == 0 else f"{base}_{suffix_num}"
    if len(candidate) <= TOOL_NAME_MAX_LENGTH:
        return candidate

    digest = hashlib.sha1(candidate.encode("utf-8")).hexdigest()[:8]
    prefix = candidate[: TOOL_NAME_MAX_LENGTH - 9].rstrip("_")
    return f"{prefix}_{digest}"


def _skill_tool_is_model_invocable(skill: "Skill", manifest_tool: dict) -> bool:
    """Module-level mirror of ``SkillTool._is_model_invocable``; shared by the
    preserved bound method and ``generate_tool_names`` (sandbox-free)."""
    policy = SkillTool._get_tool_policy(skill, manifest_tool)  # already @staticmethod
    return bool(policy.get("model_invocable", True))


class SkillTool(BaseTool):
    """统一 Skill 工具层，支持 native/mcp/a2a 三类运行时"""

    name: str = "skill"

    def __init__(
        self,
        sandbox_accessor: SandboxAccessor | None,
        mcp_tool: MCPTool,
        a2a_tool: A2ATool,
        risk_mode: str = "off",
        blocked_command_patterns: list[str] | None = None,
        bundle_sync_manager: SkillBundleSyncManager | None = None,
        skill_sandbox_bundle_root: str | None = None,
        admission_port: Any = None,  # D1a §4.1: ExtensionAdmissionPort | None（off=None → 旧路径零调用）
    ) -> None:
        # PR-1b (SPM Task 9): accessor is Optional — off assembly wires only
        # MCP/A2A paths and never touches the native (sandbox) runtime.
        super().__init__()
        self._sandbox_accessor = sandbox_accessor
        self._mcp_tool = mcp_tool
        self._a2a_tool = a2a_tool
        self._risk_mode = risk_mode
        self._bundle_sync_manager = bundle_sync_manager
        self._admission_port = admission_port  # D1a §4.1: G 逻辑在 T10-T12 消费
        self._skills: list[Skill] = []
        self._tools: list[dict[str, Any]] = []
        self._tool_bindings: dict[str, dict[str, Any]] = {}
        self._tool_name_index: dict[str, int] = {}
        settings = get_settings()
        if blocked_command_patterns is not None:
            self._blocked_command_patterns = blocked_command_patterns
        else:
            self._blocked_command_patterns = [
                item.strip()
                for item in str(settings.skill_blocked_command_patterns).split(",")
                if item.strip()
            ]
        root_dir = str(skill_sandbox_bundle_root or settings.skill_sandbox_bundle_root or "").strip()
        if root_dir:
            self._skill_sandbox_bundle_root = root_dir.rstrip("/")
        else:
            self._skill_sandbox_bundle_root = "/home/ubuntu/workspace/.skills"

    async def initialize(self, skills: list[Skill]) -> None:
        """初始化可用 Skill 列表并生成工具声明

        原子性契约（#27）：构造失败时，self._skills / _tools / _tool_bindings /
        _tool_name_index / _tools_cache 五个字段保持调用前的值不变。实现方式
        是"local 变量累积 + 末尾批量赋值"——循环中途 raise 则 local 变量随
        栈丢弃，self._* 从未被触碰。

        注意：本方法 body 内无 `await`，因此构成同 event loop 上不可被其它
        协程抢占的临界区。末尾五行赋值整体对 reader 原子可见。详见 design doc
        docs/superpowers/specs/2026-04-13-skill-tool-initialize-atomicity-design.md
        """
        new_skills = [skill for skill in skills if skill.enabled]
        new_tools: list[dict[str, Any]] = []
        new_tool_bindings: dict[str, dict[str, Any]] = {}
        new_tool_name_index: dict[str, int] = {}

        for skill in new_skills:
            runtime_type = skill.runtime_type
            manifest_tools = (skill.manifest or {}).get("tools", [])
            if not isinstance(manifest_tools, list):
                continue

            for manifest_tool in manifest_tools:
                if not isinstance(manifest_tool, dict):
                    continue

                raw_tool_name = str(manifest_tool.get("name") or "").strip()
                if not raw_tool_name:
                    continue

                if not self._is_model_invocable(skill, manifest_tool):
                    continue

                function_name = self._build_function_name(
                    skill.slug, raw_tool_name, new_tool_name_index
                )
                parameters = manifest_tool.get("parameters")
                required = manifest_tool.get("required")
                description = self._build_tool_description(skill, manifest_tool)

                if not isinstance(parameters, dict):
                    parameters = {}
                if not isinstance(required, list):
                    required = []

                tool_schema = {
                    "type": "function",
                    "function": {
                        "name": function_name,
                        "description": description,
                        "parameters": {
                            "type": "object",
                            "properties": parameters,
                            "required": required,
                        },
                    },
                }
                new_tools.append(tool_schema)
                new_tool_bindings[function_name] = {
                    "skill": skill,
                    "runtime_type": runtime_type,
                    "manifest_tool": manifest_tool,
                    # R3: risk data for langchain_dynamic_skill_tools metadata injection
                    "final_risk": self._compute_final_risk(skill),
                    "trust_origin": skill.trust_origin,
                    "scan_verdict": (skill.scan_report or {}).get("verdict", "safe"),
                }

        # Batch atomic assignment. No `await` between these five lines, so
        # on a single asyncio event loop no other coroutine can observe an
        # intermediate state. Python's GIL also guarantees each individual
        # attribute assignment is atomic at the bytecode level.
        self._skills = new_skills
        self._tools = new_tools
        self._tool_bindings = new_tool_bindings
        self._tool_name_index = new_tool_name_index
        self._tools_cache = new_tools

        logger.info(
            "SkillTool 初始化完成: enabled_skills=%s, available_tools=%s",
            len(new_skills),
            [tool["function"]["name"] for tool in new_tools],
        )

    def get_tools(self) -> List[Dict[str, Any]]:
        return self._tools

    def has_tool(self, tool_name: str) -> bool:
        return tool_name in self._tool_bindings

    def resolve_skill_dir(self, tool_name: str) -> Path | None:
        """Public port for SkillRiskRefresher (PE-1).

        Encapsulates: binding lookup → bundle_sync_manager._skills_root_dir
        → ``skills_root / skill.id``. Returns ``None`` on any state preventing
        resolution (binding miss / no manager / no root). Path existence is
        the caller's concern.

        PE-1 promotes the operation to a public method so SkillRiskRefresher
        does not reach across two classes' private attrs (spec Round 3 P1#5).
        """
        binding = self._tool_bindings.get(tool_name)
        if not binding:
            return None
        if not self._bundle_sync_manager:
            return None
        skills_root = getattr(self._bundle_sync_manager, "_skills_root_dir", None)
        if not skills_root:
            return None
        skill = binding["skill"]
        return Path(skills_root) / skill.id

    def refresh_risk_if_stale(self, tool_name: str) -> str | None:
        """Check if the skill's content has changed since init and rescan if needed.

        Returns the updated risk_level string (e.g. "high", "medium", "low") if
        the risk changed, or None if still fresh. Call this BEFORE Stage P so
        approval decisions use the latest risk, not a stale snapshot.
        """
        binding = self._tool_bindings.get(tool_name)
        if not binding:
            return None
        skill: Skill = binding["skill"]
        cached_hash = (skill.scan_report or {}).get("content_hash")
        if not cached_hash:
            # No content_hash = never scanned. Return "high" to force Stage P
            # (consistent with _compute_final_risk's "missing = dangerous" rule).
            # The actual scan will happen via startup backfill or on next init.
            return "high"

        # Need filesystem access through bundle sync manager
        if not self._bundle_sync_manager:
            return None
        skills_root = getattr(self._bundle_sync_manager, "_skills_root_dir", None)
        if not skills_root:
            return None

        from app.domain.services.skills_guard import SkillsGuard
        skill_dir = skills_root / skill.id
        if not skill_dir.exists():
            return None

        current_hash = SkillsGuard.compute_content_hash(skill_dir)
        if current_hash == cached_hash:
            return None  # fresh, no change

        # Content changed — rescan and update binding
        from app.domain.services.trust_matrix import scan_skill_source
        report = scan_skill_source(skill.runtime_type, skill_dir)
        skill.scan_report = report.to_dict()
        new_risk = self._compute_final_risk(skill)

        # Update binding in-place so subsequent calls in this step use new risk
        binding["final_risk"] = new_risk
        binding["scan_verdict"] = report.verdict
        logger.warning(
            "R3: skill %s risk refreshed mid-session: %s → %s (hash changed)",
            skill.id, binding.get("final_risk"), new_risk,
        )
        return new_risk

    @staticmethod
    def _compute_final_risk(skill: Skill) -> str:
        from app.domain.services.trust_matrix import compute_base_floor, compute_final_risk
        base = compute_base_floor(skill.runtime_type, skill.trust_origin)
        manifest_risk = None
        if isinstance(skill.manifest, dict):
            policy = skill.manifest.get("policy", {})
            if isinstance(policy, dict):
                manifest_risk = policy.get("risk_level")
        # Missing scan_report = not yet scanned = treat as dangerous (fail-closed).
        # This ensures unscanned legacy skills enter Stage P until backfill completes.
        scan_verdict = (skill.scan_report or {}).get("verdict")
        if skill.scan_report is None:
            scan_verdict = "dangerous"
        return compute_final_risk(base, scan_verdict, manifest_risk).name.lower()

    @staticmethod
    def _tool_result_to_outcome(result: ToolResult) -> ToolOutcome:
        if result.success:
            content = result.message or ""
            if isinstance(result.data, str):
                content = result.data
            elif isinstance(result.data, dict):
                output = result.data.get("output")
                if isinstance(output, str):
                    content = output
                elif content == "":
                    content = json.dumps(result.data, ensure_ascii=False)
            return AllowSuccess(
                content=content,
                data=result.data if isinstance(result.data, dict) else None,
            )
        message = result.message or "skill tool failed"
        return AllowError(
            content=message,
            reason=DecisionReason(
                type="exception",
                code="skill_runtime_error",
                message=message,
            ),
        )

    async def invoke(self, tool_name: str, **kwargs) -> ToolOutcome:
        binding = self._tool_bindings.get(tool_name)
        if not binding:
            return AllowError(
                content=f"Skill工具[{tool_name}]不存在",
                reason=DecisionReason(
                    type="exception",
                    code="skill_tool_not_found",
                    message=f"Skill工具[{tool_name}]不存在",
                ),
            )

        skill: Skill = binding["skill"]
        runtime_type: SkillRuntimeType = binding["runtime_type"]
        manifest_tool: dict[str, Any] = binding["manifest_tool"]

        # D1a §4.2 G4b：invoke 前状态门——独立于 PE，不依赖 R3 refresh（R2#2）。
        # 错误形态复刻 _tool_result_to_outcome 先例（AllowError + DecisionReason，
        # type="exception"），仅 code 改 extension_unavailable。off → None 直通零改动。
        _gate_reason = await skill_invoke_gate(self._admission_port, skill.id)
        if _gate_reason is not None:
            _gate_msg = f"Skill 被治理策略阻断（{_gate_reason}），本次调用不执行"
            return AllowError(
                content=_gate_msg,
                reason=DecisionReason(
                    type="exception",
                    code="extension_unavailable",
                    message=_gate_msg,
                ),
            )

        if runtime_type == SkillRuntimeType.NATIVE:
            result = await self._invoke_native(skill, manifest_tool, kwargs)
            return self._tool_result_to_outcome(result)
        if runtime_type == SkillRuntimeType.MCP:
            result = await self._invoke_mcp(manifest_tool, kwargs)
            return self._tool_result_to_outcome(result)
        if runtime_type == SkillRuntimeType.A2A:
            result = await self._invoke_a2a(manifest_tool, kwargs)
            return self._tool_result_to_outcome(result)

        return AllowError(
            content=f"暂不支持的Skill运行时: {runtime_type}",
            reason=DecisionReason(
                type="exception",
                code="unsupported_skill_runtime",
                message=str(runtime_type),
            ),
        )

    async def cleanup(self) -> None:
        """清空 SkillTool 内部状态（对称于 initialize 的原子性契约 #27）。

        五个字段（_skills / _tools / _tool_bindings / _tool_name_index /
        _tools_cache）在连续赋值中清空。本方法无 await，对同 event loop
        上的 reader 原子可见。
        """
        self._skills = []
        self._tools = []
        self._tool_bindings = {}
        self._tool_name_index = {}
        self._tools_cache = []

    async def _invoke_native(
        self,
        skill: Skill,
        manifest_tool: dict[str, Any],
        kwargs: dict[str, Any],
    ) -> ToolResult:
        entry = manifest_tool.get("entry")
        if not isinstance(entry, dict):
            return ToolResult(success=False, message="native skill 缺少 entry 配置")

        synced_skill_dir = ""
        if self._bundle_sync_manager:
            synced_skill_dir, sync_error = await self._bundle_sync_manager.ensure_ready_for_invoke(
                skill.id,
                skill=skill,
            )
            if sync_error:
                return ToolResult(
                    success=False,
                    message=f"Skill[{skill.id}]同步失败: {sync_error}",
                )

        default_exec_dir = synced_skill_dir or f"{self._skill_sandbox_bundle_root}/{skill.id}"
        exec_dir = str(entry.get("exec_dir") or "").strip() or default_exec_dir
        if not exec_dir:
            return ToolResult(success=False, message="native skill 缺少可用执行目录")

        # PR-1b (SPM Task 9): pull the concrete handle lazily at use time.
        # native skills require a sandbox — the accessor is non-None on this path
        # (off assembly never reaches _invoke_native); assert for fail-fast + narrowing.
        assert self._sandbox_accessor is not None, (
            "native skill invocation requires a provisioned sandbox accessor"
        )
        sandbox = await self._sandbox_accessor.get()

        exists_result = await sandbox.check_file_exists(exec_dir)
        if not exists_result.success:
            return ToolResult(
                success=False,
                message=f"native skill 执行目录检查失败: {exists_result.message or ''}",
            )
        exists_data = exists_result.data if isinstance(exists_result.data, dict) else {}
        if not exists_data.get("exists"):
            return ToolResult(success=False, message=f"native skill 执行目录不存在: {exec_dir}")

        command = str(entry.get("command") or "").strip()
        if not command:
            return ToolResult(success=False, message="native skill 缺少 entry.command")

        # N1: AST validator runs first (per spec §4.5(c))
        from app.domain.services.safety.shell_ast_validator import (
            to_legacy_tool_result,
            validate,
        )
        from app.domain.services.safety.command_policy_evaluator import (
            build_command_policy,
            evaluate_command,
        )
        try:
            ast_result = validate(command, effective_cwd=exec_dir)
        except Exception as _ast_exc:  # defensive — validate() should never raise
            logger.exception(
                "SkillTool._invoke_native AST validator 兜底触发 (should not happen)"
            )
            return ToolResult(
                success=False,
                message=(
                    f"native skill AST validator 内部异常 (fail-closed): "
                    f"{type(_ast_exc).__name__}"
                ),
            )
        _cmd_decision = evaluate_command(
            validation_code=ast_result.code,
            policy=build_command_policy(effective_cwd=ast_result.effective_cwd, is_default_cwd=False),
        )
        if not _cmd_decision.allowed:
            # Interim telemetry for ast_block_rate_skill (spec §8.6);
            # session_id not yet created here (runs before line ~349),
            # so use skill.id + manifest_tool name. INFO level (not WARN) —
            # denial is a normal decision event.
            logger.info(
                "skill_ast_deny skill_id=%s tool_name=%s code=%s category=%s",
                skill.id,
                manifest_tool.get("name", ""),
                ast_result.code,
                ast_result.category_zh,
            )
            return to_legacy_tool_result(ast_result, original_command=command)

        # Belt-and-suspenders: legacy regex remains as secondary layer.
        # DEPRECATED — retained during N1 narrow; remove in R5+CS4 cleanup.
        if self._contains_blocked_command(command):
            return ToolResult(
                success=False,
                message="native skill 命令命中禁止规则 (legacy blocklist)",
            )

        payload = json.dumps(kwargs, ensure_ascii=False) if kwargs else "{}"
        full_command = f"{command} {shlex.quote(payload)}"
        session_id = f"skill-{uuid.uuid4()}"

        result = await sandbox.exec_command(session_id, exec_dir, full_command)
        if not result.success:
            # 即使失败也传递 session_id，便于 UI 终端读取输出
            if result.data is None:
                result.data = {}
            if isinstance(result.data, dict):
                result.data["shell_session_id"] = session_id
                result.data["exec_dir"] = exec_dir
            return result

        output = await sandbox.read_shell_output(session_id)
        final = output if output.success else result
        # 将 session_id 和 exec_dir 注入 data，供 agent_task_runner 使用
        if final.data is None:
            final.data = {}
        if isinstance(final.data, dict):
            final.data["shell_session_id"] = session_id
            final.data["exec_dir"] = exec_dir
        return final

    async def _invoke_mcp(
        self,
        manifest_tool: dict[str, Any],
        kwargs: dict[str, Any],
    ) -> ToolResult:
        entry = manifest_tool.get("entry")
        if not isinstance(entry, dict):
            return ToolResult(success=False, message="mcp skill 缺少 entry 配置")

        actual_tool_name = str(entry.get("tool_name") or manifest_tool.get("name") or "").strip()
        if not actual_tool_name:
            return ToolResult(success=False, message="mcp skill 缺少 entry.tool_name")

        return await self._mcp_tool.invoke(actual_tool_name, **kwargs)

    async def _invoke_a2a(
        self,
        manifest_tool: dict[str, Any],
        kwargs: dict[str, Any],
    ) -> ToolResult:
        entry = manifest_tool.get("entry")
        if not isinstance(entry, dict):
            return ToolResult(success=False, message="a2a skill 缺少 entry 配置")

        agent_id = str(entry.get("agent_id") or "").strip()
        if not agent_id:
            return ToolResult(success=False, message="a2a skill 缺少 entry.agent_id")

        query = kwargs.get("query")
        if not isinstance(query, str) or not query.strip():
            query = json.dumps(kwargs, ensure_ascii=False)

        return await self._a2a_tool.call_remote_agent(id=agent_id, query=query)

    @classmethod
    def _normalize_function_part(cls, raw: str) -> str:
        normalized = re.sub(r"[^a-zA-Z0-9_]+", "_", raw or "").strip("_").lower()
        return normalized or "tool"

    def _build_function_name(
        self, skill_slug: str, tool_name: str, name_index: dict[str, int]
    ) -> str:
        # Thin wrapper over the module-level helper. Signature UNCHANGED so the
        # initialize() call site + the monkeypatch atomicity tests keep working.
        return _skill_tool_function_name(skill_slug, tool_name, name_index)

    @staticmethod
    def generate_tool_names(skills: "list[Skill]") -> list[str]:
        """[S4 §13/R7-2] The generated ``skill_{slug}_{tool}`` names a SkillTool
        would build — same filter (enabled / list / dict / non-empty name /
        model_invocable) + naming (SHARED name_index + sha1 cap) as
        ``initialize()``, WITHOUT constructing a SkillTool (no sandbox). The
        anti-drift test locks it to real ``get_tools()``."""
        name_index: dict[str, int] = {}
        names: list[str] = []
        for skill in skills:
            if not skill.enabled:
                continue
            manifest_tools = (skill.manifest or {}).get("tools", [])
            if not isinstance(manifest_tools, list):
                continue
            for manifest_tool in manifest_tools:
                if not isinstance(manifest_tool, dict):
                    continue
                raw = str(manifest_tool.get("name") or "").strip()
                if not raw:
                    continue
                if not _skill_tool_is_model_invocable(skill, manifest_tool):
                    continue
                names.append(_skill_tool_function_name(skill.slug, raw, name_index))
        return names

    @classmethod
    def _extract_skill_md_summary(cls, skill_md: str) -> str:
        if not skill_md:
            return ""

        lines: list[str] = []
        for raw_line in skill_md.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("#"):
                continue
            line = re.sub(r"`+", "", line)
            lines.append(line)
            if len(" ".join(lines)) >= 180:
                break

        summary = " ".join(lines).strip()
        if len(summary) > 200:
            summary = summary[:197].rstrip() + "..."
        return summary

    def _build_tool_description(self, skill: Skill, manifest_tool: dict[str, Any]) -> str:
        base_desc = str(manifest_tool.get("description") or "").strip()
        if not base_desc:
            base_desc = (skill.description or "").strip()

        skill_md_summary = self._extract_skill_md_summary(
            str((skill.manifest or {}).get("skill_md") or "")
        )

        parts = [f"[{skill.name}] ({skill.runtime_type.value})"]
        if base_desc:
            parts.append(base_desc)
        if skill_md_summary:
            parts.append(f"Skill guide: {skill_md_summary}")

        return " ".join(parts)[:512]

    @staticmethod
    def _get_tool_policy(skill: Skill, manifest_tool: dict[str, Any]) -> dict[str, Any]:
        policy: dict[str, Any] = {}
        manifest_policy = (skill.manifest or {}).get("policy")
        if isinstance(manifest_policy, dict):
            policy.update(manifest_policy)
        tool_policy = manifest_tool.get("policy")
        if isinstance(tool_policy, dict):
            policy.update(tool_policy)
        return policy

    def _is_model_invocable(self, skill: Skill, manifest_tool: dict[str, Any]) -> bool:
        # Thin wrapper over the module-level helper. Signature UNCHANGED so the
        # initialize() call site + the monkeypatch atomicity tests keep working.
        return _skill_tool_is_model_invocable(skill, manifest_tool)

    def _contains_blocked_command(self, command: str) -> bool:
        """DEPRECATED (N1): retained as belt-and-suspenders secondary layer.

        Primary check is now ``shell_ast_validator.validate()``. Will be
        removed in R5+CS4 cleanup once legacy_regex_block_rate reaches zero.
        """
        for pattern in self._blocked_command_patterns:
            if re.search(pattern, command, flags=re.IGNORECASE):
                return True
        return False
