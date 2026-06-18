"""main_graph — outer orchestration as a LangGraph StateGraph.

Replaces PlannerReActFlow.invoke() while-loop.
Nodes: planner_node, executor_node, updater_node, summarizer_node
Edges: See design doc §4.2

Reference: docs/plans/2026-03-10-langchain-langgraph-migration-design.md §4.1-4.2
"""

from __future__ import annotations

import asyncio
import logging
import re
import uuid
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Literal, TYPE_CHECKING

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from .message_utils import truncate_tool_content
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, RetryPolicy, interrupt

from app.domain.models.app_config import AgentConfig

from app.application.errors.exceptions import ServerRequestsError
# PR-9b-A6 — PatchApplier construction at run time inside
# ``_run_parallel_backend``. The existing late import of ``ApplyStatus``
# (below, inside the function body) established the domain → application
# import as an accepted exception; hoisting the class symbol up keeps
# the per-run ctor call readable.
from app.application.services.group_lineage import GroupLineageFields
from app.application.services.patch_applier import PatchApplier
from app.domain.models.file import File
from app.domain.models.llm_responses import PlanResponse, PlanUpdateResponse, StepDef
from app.domain.models.event import (
    DoneEvent,
    MessageEvent,
    PlanEvent,
    PlanEventStatus,
    StepEvent,
    StepEventStatus,
    TitleEvent,
    WaitEvent,
)
from app.domain.models.plan import ExecutionStatus, Plan, Step
from app.domain.repositories.uow import IUnitOfWork
from app.domain.services.flows.base import FlowStatus
from app.domain.services.json_envelope import unwrap_message_envelope
from app.domain.services.planner_guardrails import salvage_empty_memory_recall_plan

from .message_utils import build_multimodal_content, dedup_messages, format_attachments_text
from .state import MainGraphState

if TYPE_CHECKING:
    from .context_assembler import ContextAssembler
    from app.domain.services.prompts.assembler import PromptAssembler
    from app.domain.services.prompts.memory_snapshot import MemorySnapshot

logger = logging.getLogger(__name__)

# Browser tool names whose content can be compacted
_BROWSER_COMPACT_TOOLS = frozenset(["browser_view", "browser_navigate"])


@dataclass(frozen=True)
class ParallelBackendOutcome:
    """Structured result of the C2 coordinator branch (``_run_parallel_backend``).

    ``summary`` is the operator-facing text the orchestrator forwards into
    ``execution_summary`` / ``Step.result``. ``success`` is the authoritative
    pass/fail signal — derived from the reducer's ``GroupOutcome`` (and, on the
    SUCCESS-with-apply path, the ``ApplyStatus``), NOT inferred from the summary
    string. ``executor_node`` reads ``success`` to set ``Step.success`` exactly
    like the react branch reads ``failure_count == 0``.
    """

    success: bool
    summary: str


async def _run_parallel_backend(
    state: Any, config: Any, step: Any
) -> ParallelBackendOutcome:
    """C2 PR-3 §7.2 — invoke parallel_execution_subgraph for a coordinator step.

    The subgraph's ``dispatch_node`` is responsible for ``coordinator_run_id``
    derivation (``peek`` → ``bump`` flow on ``sessions.coordinator_attempts``
    JSONB). main_graph does NOT pre-compute the run id — it only forwards
    ``step.id`` + the planner's ``ParallelWorkUnitGroupRequest.work_units``.
    """
    cfg = (config.get("configurable") or {}) if config else {}
    subgraph = cfg.get("parallel_execution_subgraph")
    if subgraph is None:
        raise RuntimeError(
            "executor_node: parallel dispatch branch entered without "
            "configurable.parallel_execution_subgraph wired. "
            "(PR-3 cold code; PR-9 flips ACTUS_C2_COORDINATOR_ENABLED + wires "
            "the subgraph + collaborators into composition/DI.)"
        )
    # C2 PR-3 §7.2 [r1 P0-3 + r2 P1-3 fix] — id derivation.
    #
    # ``parent_session_id``: MainGraphState carries ``session_id`` (not
    # ``parent_session_id``); planner_react.py:1493-1510 confirms input shape.
    #
    # ``user_id``: planner_react.py:1264-1265 places ``user_id`` in
    # ``configurable`` (not state). Fall back to cfg for the live shape.
    #
    # ``root_session_id``: MainGraphState has no ``root_session_id`` field.
    # Phase 1 ``max_subagent_depth=1`` invariant guarantees the parent IS the
    # root (see ``SessionService.create_session_with_parent`` guard:
    # ``parent.parent_session_id is not None or parent.worker_type != "root"
    # -> SpawnCapExceeded``). When Phase 2 lifts max_subagent_depth, the
    # planner graph state will need an explicit ``root_session_id`` and this
    # fallback must be removed in tandem.
    parent_session_id = state.get("session_id")
    user_id = state.get("user_id") or cfg.get("user_id")
    root_session_id = state.get("root_session_id") or parent_session_id
    if not parent_session_id:
        raise RuntimeError(
            "executor_node parallel branch: state.session_id is required"
        )
    if not user_id:
        raise RuntimeError(
            "executor_node parallel branch: user_id required "
            "(state.user_id or configurable.user_id)"
        )
    final_state = await subgraph.ainvoke(
        {
            "coordinator_run_id": None,
            "step_id": step.id,
            "work_unit_requests": list(step.parallel_work_units.work_units),
            "work_units": [],
            "parent_session_id": parent_session_id,
            "user_id": user_id,
            "root_session_id": root_session_id,
            "child_session_ids": {},
            "orchestrator_task": None,
            "worker_results": [],
            "apply_plan": None,
            "group_outcome": None,
            "step_result_candidate": None,
        },
        config={"configurable": cfg},
    )
    step_result_candidate = final_state.get("step_result_candidate", "") or ""

    # ── C2 PR-7 §12.5: apply re-entry short-circuit ─────────────────────
    #
    # When dispatch_node's rehydrate_service.detect_existing_run returns
    # a non-None ``already_applied`` (success | rollback_partial |
    # crash_mid_apply | in_progress_recent), the subgraph short-circuits
    # to END with ``step_result_candidate = "ALREADY_APPLIED:{status}:{audit_id}"``.
    # main_graph must then SKIP the patch_applier branch below — we are
    # NOT re-running an apply that already committed (success), an
    # operator-locked failure path (rollback_partial / crash_mid_apply),
    # or a concurrent attempt by another pod (in_progress_recent).
    #
    # The HealthEvent for rollback_partial + crash_mid_apply was already
    # emitted by ``CoordinatorRehydrateService._check_already_applied``;
    # this branch only formats the operator-facing summary string.
    if step_result_candidate.startswith("ALREADY_APPLIED:"):
        parts = step_result_candidate.split(":", 2)
        status = parts[1] if len(parts) > 1 else "unknown"
        audit_id = parts[2] if len(parts) > 2 else "unknown"
        if status == "success":
            # The apply already committed in a prior run — the step's goal
            # was achieved, so the coordinator step is a success.
            return ParallelBackendOutcome(
                success=True,
                summary=f"Apply already succeeded (audit {audit_id}).",
            )
        if status == "rollback_partial":
            # Spec §10.4: 不 auto-retry; HealthEvent 已由 rehydrate_service emit。
            # Needs manual recovery → NOT a success.
            return ParallelBackendOutcome(
                success=False,
                summary=(
                    f"Apply rollback_partial detected (audit {audit_id}); "
                    f"not auto-retrying — manual recovery required."
                ),
            )
        if status == "crash_mid_apply":
            return ParallelBackendOutcome(
                success=False,
                summary=(
                    f"⚠️ apply 中途 pod crash（audit {audit_id} "
                    f"in_progress > 5min）。Workspace 可能不一致，请人工检查 "
                    f"+ 清理 audit row 后重试。"
                ),
            )
        if status == "in_progress_recent":
            return ParallelBackendOutcome(
                success=False,
                summary="apply 仍在执行（另一 pod？），等 Redis lock 释放后重试。",
            )
        # Unknown status — surface verbatim so operator can diagnose.
        # Conservative: an undiagnosable apply state is NOT a success.
        return ParallelBackendOutcome(
            success=False, summary=step_result_candidate
        )

    # ── C2 PR-5 §10.6: invoke PatchApplier on SUCCESS ────────────────────
    #
    # Branching contract: the applier fires only when ALL of the
    # following are true:
    #   (a) reducer reported SUCCESS
    #   (b) an apply_plan with at least one file was built
    #   (c) the composition root wired ``patch_applier_deps`` (PR-9b-A6)
    #       + parent_sandbox + artifact_storage into ``configurable``
    #
    # PR-9b-A6 swap: spec §5.1.4 originally specified a singleton
    # ``patch_applier`` cfg key. The production reality (codex R3 audit)
    # requires PER-RUN construction so the ``emit_event`` callable can
    # close over the per-stream ``event_queue`` bound by GraphEventBridge
    # at invocation time (event_bridge.py:74-79). main_graph now builds
    # PatchApplier per-coordinator-run from ``cfg['patch_applier_deps']``
    # (lifespan-scoped PatchApplierDeps holding snapshot_store / audit_repo
    # / redis) + an async ``_emit_event_into_queue`` closure.
    #
    # Legacy ``cfg.get('patch_applier')`` is still consulted as a
    # backward-compat fallback for existing unit tests that pre-construct
    # a fake applier and inject it directly (test_run_parallel_backend_apply_*).
    # Production composition root only wires ``patch_applier_deps``.
    from app.domain.models.patch_apply_plan import GroupOutcome
    group_outcome = final_state.get("group_outcome")
    if group_outcome != GroupOutcome.SUCCESS:
        # Any non-SUCCESS group outcome (FAILED / CANCELLED / TIMED_OUT /
        # NEEDS_AUTHORIZATION / CONFLICT / INCOMPLETE / MIXED) → step failed.
        return ParallelBackendOutcome(
            success=False, summary=step_result_candidate
        )

    apply_plan = final_state.get("apply_plan")
    if apply_plan is None or apply_plan.file_count == 0:
        # SUCCESS with no files (exploration-only step) — nothing to apply.
        return ParallelBackendOutcome(
            success=True, summary=step_result_candidate
        )

    # PR-9b-A6: prefer per-run construction from PatchApplierDeps. Falls back
    # to a singleton ``patch_applier`` only when the new key is absent.
    parent_sandbox = cfg.get("parent_sandbox")
    minio = cfg.get("artifact_storage")
    deps = cfg.get("patch_applier_deps")
    legacy_applier = cfg.get("patch_applier")

    applier = legacy_applier
    if deps is not None:
        # PR-9b-A6 production path — per-run PatchApplier with an emit
        # closure over the per-stream event_queue. Fail loud when the
        # event_queue is absent: GraphEventBridge must always merge it
        # into the coordinator-path configurable (INV-A3 regression
        # guard); a silent drop would lose CoordinatorApplyEvent +
        # HealthEvent emits.
        event_queue = cfg.get("event_queue")
        if event_queue is None:
            raise RuntimeError(
                "coordinator path entered without event_queue in "
                "configurable — GraphEventBridge wiring regression "
                "at event_bridge.py:74-79"
            )

        async def _emit_event_into_queue(event: object) -> None:
            # [INV-A3] put_nowait avoids the cancellation window from
            # awaiting a full queue. The PatchApplier's terminal emits
            # (CoordinatorApplyEvent at _finalize, HealthEvent on
            # rollback_partial) flow through this closure. Emit failures
            # are swallowed by PatchApplier's own try/except so the
            # ApplyOutcome reaches main_graph unchanged (INV-A4).
            event_queue.put_nowait(event)

        applier = PatchApplier(
            snapshot_store=deps.snapshot_store,
            audit_repo=deps.audit_repo,
            redis=deps.redis,
            emit_event=_emit_event_into_queue,
        )

    if applier is None or parent_sandbox is None or minio is None:
        # Cold-code path: ports not wired yet (PR-5 ships before PR-7/8).
        # Return reducer's text without applying. The audit trail will
        # show no apply attempt, which is the correct cold-code signal.
        #
        # [codex R3 P2#7] WARN log so production operators can detect a
        # composition-root misconfig: if the coordinator feature flag is
        # enabled but the DI ports were not bound, we will silently
        # report "reducer succeeded" without ever applying. The log
        # surfaces that the apply was skipped + which ports were missing.
        missing = [
            name for name, port in (
                ("patch_applier_deps_or_applier", applier),
                ("parent_sandbox", parent_sandbox),
                ("artifact_storage", minio),
            )
            if port is None
        ]
        logger.warning(
            "_run_parallel_backend: SUCCESS with %d files but applier "
            "ports not wired (missing: %s); apply step SKIPPED — "
            "expected only during PR-5 cold-code before PR-7/8 wires "
            "the composition root.",
            apply_plan.file_count, missing,
        )
        # The reducer reported SUCCESS; the skipped apply is a composition-root
        # wiring gap (already surfaced as a WARNING above) and never happens on
        # the flag-ON production path where the ports are always bound. Treat
        # the reducer's SUCCESS as the authoritative work signal.
        return ParallelBackendOutcome(
            success=True, summary=step_result_candidate
        )

    from app.application.services.patch_applier import ApplyStatus
    # [codex R2 P1] Thread the orchestrator's cancel_event into the
    # applier so parent-cancel during reducer→apply or mid-apply
    # triggers APPLY_ABORTED + rollback. Without this, the applier's
    # cancel contract only fired in unit tests; production main_graph
    # would continue writing the parent sandbox after the parent was
    # cancelled.
    # [PR-9b-B Task B6 / INV-B4] Thread group-scope lineage into apply so the
    # group-level CoordinatorApplyEvent carries root/parent parity. Reuse the
    # locals already derived above (session_id → parent mapping + root fallback);
    # do NOT re-read state here — MainGraphState has no ``parent_session_id`` key.
    lineage = GroupLineageFields(
        root_session_id=root_session_id,
        parent_session_id=parent_session_id,
    )
    apply_outcome = await applier.apply(
        apply_plan,
        parent_sandbox=parent_sandbox,
        minio_client=minio,
        cancel_event=cfg.get("cancel_event"),
        lineage=lineage,
    )
    if apply_outcome.status == ApplyStatus.SUCCESS:
        return ParallelBackendOutcome(
            success=True,
            summary=(
                f"{step_result_candidate}\n"
                f"应用成功（写入 {apply_plan.file_count} 个文件）。"
            ),
        )
    if apply_outcome.status == ApplyStatus.ROLLBACK_PARTIAL:
        # Surface critical health signal — applier already emitted the
        # HealthEvent via the emit_event port; the additional message
        # here is what the orchestrator passes to the summarizer.
        failed_path = (
            apply_outcome.failed_at.path
            if apply_outcome.failed_at else "未知路径"
        )
        return ParallelBackendOutcome(
            success=False,
            summary=f"应用回滚不完整（卡在 {failed_path}）；需要人工恢复。",
        )
    # Other ApplyStatus (DIGEST_DRIFT / FILE_MISSING / WRITE_IO_ERROR /
    # POST_WRITE_DIGEST_MISMATCH / MINIO_FETCH_FAILED / APPLY_ABORTED).
    # The rollback completed (status != ROLLBACK_PARTIAL), so the sandbox
    # is consistent — orchestrator routes via the result text. Operator
    # text in Chinese (project convention); the machine-readable status
    # value remains the canonical routing surface for PR-6/8.
    return ParallelBackendOutcome(
        success=False,
        summary=f"应用失败（状态 {apply_outcome.status.value}）",
    )


def _assign_fallback_step_id(plan_id: str, index: int) -> str:
    """[C2 PR-1 §4.2 r7 P0-2] Deterministic fallback when LLM omits StepDef.id.

    Used by ``_build_plan_from_response`` and the planner_react flow to
    produce stable, reproducible step ids without injecting random UUIDs.
    Format: ``step_{plan_id}_{index:02d}``.
    """
    return f"step_{plan_id}_{index:02d}"


def _build_plan_from_response(
    response: PlanResponse, *, plan_id: str | None = None
) -> Plan:
    """[C2 PR-1 §4.2 r7 P0-2] Build a Plan from a PlanResponse.

    Propagates ``StepDef.id → Step.id`` directly; when the LLM omitted the
    id, falls back to ``_assign_fallback_step_id(plan_id, index)`` so
    downstream cross-step references stay deterministic. Other Plan-level
    fields (title/goal/language/message) are carried over from the
    response by the caller after this helper produces the inner Step list.

    The returned Plan uses Plan's default factory for ``id`` so the
    caller can either accept the random Plan id or rebuild Plan with an
    explicit id. The ``plan_id`` argument here only governs the fallback
    step id derivation; "default" is used when caller didn't supply one.
    """
    pid = plan_id or "default"
    steps: list[Step] = []
    seen_ids: set[str] = set()
    for i, sd in enumerate(response.steps):
        # When LLM repeats a StepDef.id (e.g. multiple steps share "1"),
        # fall back to the deterministic helper so updater_node's id-based
        # step lookup doesn't collapse unrelated steps onto each other.
        if sd.id and sd.id not in seen_ids:
            step_id = sd.id
        else:
            step_id = _assign_fallback_step_id(pid, i)
        seen_ids.add(step_id)
        # [C2b rollout WS0 §3A.4] Flag-off hard sanitation. The prompt-gate
        # (WS0 teaching section) only REDUCES emission; parallel_work_units is
        # in the with_structured_output schema (StepDef), so a flag-off real
        # provider can emit it straight from the schema (F17). Clear it at the
        # parse->Step boundary so no flag-off session reaches the executor
        # coordinator gate (main_graph.py:751-755). This helper is shared by
        # planner_node AND the detection planner (planner_react.py:1239), so
        # sanitizing here covers BOTH paths. is_coordinator_enabled is domain.
        from app.domain.services.coordinator_feature_flag import (
            is_coordinator_enabled,
        )
        _pwu = sd.parallel_work_units if is_coordinator_enabled() else None
        steps.append(
            Step(
                id=step_id,
                description=sd.description,
                parallel_work_units=_pwu,
            )
        )
    return Plan(steps=steps)


def _compact_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Compact step messages to reduce context size between steps.

    Mirrors the main-branch Memory.compact() behaviour:
    - Replace browser tool results with short summaries
    - Truncate very long tool results
    - Strip multimodal image blocks from HumanMessage to avoid re-sending
      large base64 payloads on every subsequent step
    """
    compacted: list[BaseMessage] = []
    for msg in messages:
        # Strip multimodal image content blocks — images were already "seen" in step 1,
        # subsequent steps only need the text portion to avoid bloating context with base64.
        if isinstance(msg, HumanMessage) and isinstance(msg.content, list):
            text_parts = [
                block.get("text", "")
                for block in msg.content
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            text_only = "\n".join(text_parts) if text_parts else str(msg.content)
            compacted.append(HumanMessage(content=text_only))
            continue
        if isinstance(msg, ToolMessage) and isinstance(msg.content, str) and msg.name in _BROWSER_COMPACT_TOOLS:
            # Extract title and short preview from HTML content
            content = msg.content
            title_match = re.search(
                r"<title[^>]*>(.*?)</title>", content, re.IGNORECASE | re.DOTALL
            )
            title = title_match.group(1).strip() if title_match else ""
            plain = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", content)).strip()
            preview = plain[:200]
            parts = [f"[已执行] {msg.name}:"]
            if title:
                parts.append(f"页面标题: {title},")
            parts.append(f"内容摘要: {preview}")
            compacted.append(ToolMessage(
                content=" ".join(parts),
                tool_call_id=msg.tool_call_id,
                name=msg.name,
            ))
        elif isinstance(msg, ToolMessage) and isinstance(msg.content, str) and len(msg.content) > 2000:
            # Tier 2: 截断超长工具结果，压缩跨 step 存储
            compacted.append(ToolMessage(
                content=truncate_tool_content(msg.content, 2000),
                tool_call_id=msg.tool_call_id,
                name=msg.name,
            ))
        else:
            compacted.append(msg)
    return compacted


_PRIOR_STEP_OUTPUTS_EMPTY_BY_LANG = {"zh": "无", "en": "None"}
_PRIOR_STEP_DESCRIPTION_MAX_CHARS = 80


def _format_prior_step_outputs(plan: Plan | None, language: str = "zh") -> str:
    """Render completed plan steps' attachments as an executor-facing hint.

    The output is injected into ``EXECUTION_PROMPT`` so later steps know
    which files earlier steps produced and can ``file_read`` them instead
    of re-issuing searches. Only ``ExecutionStatus.COMPLETED`` steps with
    non-empty ``attachments`` are rendered; order matches ``plan.steps``.

    Returns a locale-appropriate placeholder (``"无"`` for ZH, ``"None"``
    for EN, falling back to ZH for unknown languages) when there is
    nothing to show — keeps the EN executor prompt free of stray Chinese
    characters.
    """
    empty = _PRIOR_STEP_OUTPUTS_EMPTY_BY_LANG.get(
        language, _PRIOR_STEP_OUTPUTS_EMPTY_BY_LANG["zh"]
    )
    if plan is None or not plan.steps:
        return empty
    lines: list[str] = []
    for s in plan.steps:
        if s.status != ExecutionStatus.COMPLETED or not s.attachments:
            continue
        desc = (s.description or "").strip()
        if len(desc) > _PRIOR_STEP_DESCRIPTION_MAX_CHARS:
            desc = desc[: _PRIOR_STEP_DESCRIPTION_MAX_CHARS - 1] + "…"
        paths = ", ".join(s.attachments)
        lines.append(f"- [{desc}] {paths}" if desc else f"- {paths}")
    return "\n".join(lines) if lines else empty


def build_main_graph(
    planner_llm: BaseChatModel,
    react_graph: CompiledStateGraph,
    summary_llm: BaseChatModel,
    uow_factory: Callable[[], IUnitOfWork],
    session_id: str,
    agent_config: AgentConfig | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    assembler: ContextAssembler | None = None,
    prompt_assembler: "PromptAssembler | None" = None,
    supports_vision: bool = True,
    memory_snapshot_provider: "Callable[[], Awaitable[MemorySnapshot | None]] | None" = None,
    node_decorator: "Callable[[Callable[..., Awaitable[Any]]], Callable[..., Awaitable[Any]]] | None" = None,
    _allow_default_prompt_assembler: bool = False,
) -> CompiledStateGraph:
    """Build and compile the main orchestration graph.

    Parameters
    ----------
    planner_llm : BaseChatModel for plan generation/update (structured output).
    react_graph : Compiled react_graph for step execution.
    summary_llm : BaseChatModel for summary generation (streaming via astream).
    uow_factory : Factory for UoW instances.
    session_id : Current session ID.
    checkpointer : LangGraph checkpointer for interrupt/resume support.
    assembler : ContextAssembler for cross-step message trimming (unchanged).
    prompt_assembler : B5 C5b PromptAssembler for section-based system prompt
        assembly. Required for executor/planner/updater nodes to run. In
        production this MUST be provided (``AgentTaskRunner`` constructs
        a configured instance and passes it in). Missing it raises
        ``RuntimeError`` unless ``_allow_default_prompt_assembler=True``.
    memory_snapshot_provider : M2 PR-4 optional async callable returning a
        ``MemorySnapshot`` per render. ``PlannerReActFlow`` builds a closure
        that opens a DB session, calls ``build_memory_snapshot(repo, user_id)``,
        and swallows exceptions (returns None) so an unavailable DB never
        crashes the agent loop — memory injection degrades to "sections emit
        nothing", which is the same as the no-memory default.

        **Invoked ONLY from ``executor_node``.** The three memory sections
        (``memory_rules`` / ``memory_user_profile`` / ``memory_fact_index``)
        live exclusively in the executor registry per M2 design doc
        §585-599 — planner / updater registries declare a small ``planner_*``
        section set that doesn't read ``ctx.memory_snapshot``. Fetching from
        those two nodes would burn four extra DB queries per ainvoke with
        zero behavioral change; the CI guard
        ``test_planner_and_updater_never_invoke_provider`` locks this
        decision. If a future PR ever adds a memory-aware section to the
        planner or updater registry, re-add the fetch at that call site
        and update the guard.

        None means memory sections stay inert (tests, flows without
        memory wiring).
    _allow_default_prompt_assembler : Test-only escape hatch (leading
        underscore to mark internal). When True AND ``prompt_assembler``
        is None, this constructor builds a minimal-budget default
        assembler and logs a WARNING. Production callers should NEVER
        set this — missing DI should fail loud.
    """
    from app.domain.services.prompts import (
        get_prompt_bundle,
        get_prompt_section_bundle,
    )
    from app.domain.services.prompts.render_context import build_render_context
    from app.domain.services.prompts.section import PromptMode

    # B5 C7.5 + audit MEDIUM #7: by default a missing ``prompt_assembler``
    # is a production DI bug — raise loud. Test fixtures that exercise
    # ``build_main_graph`` directly pass ``_allow_default_prompt_assembler=True``
    # to opt into the legacy test-ergonomics behavior.
    if prompt_assembler is None:
        if not _allow_default_prompt_assembler:
            raise RuntimeError(
                "build_main_graph requires a PromptAssembler instance. "
                "In production, AgentTaskRunner._build_prompt_assembler "
                "constructs one and passes it via PlannerReActFlow. If "
                "this is a test that needs the default, pass "
                "_allow_default_prompt_assembler=True explicitly."
            )
        from app.domain.services.graphs.token_estimator import TokenEstimator
        from app.domain.services.prompts.assembler import (
            PromptAssembler as _PromptAssemblerImpl,
        )
        from app.domain.services.prompts.budget import SystemPromptBudget

        logger.warning(
            "build_main_graph: prompt_assembler is None and "
            "_allow_default_prompt_assembler=True — constructing a minimal "
            "default. This path is intended for tests only; production "
            "should always inject a configured PromptAssembler via "
            "AgentTaskRunner._build_prompt_assembler."
        )
        prompt_assembler = _PromptAssemblerImpl(
            budget=SystemPromptBudget(max_tokens=10000),
            token_estimator=TokenEstimator(strategy="hybrid"),
            telemetry=None,
        )

    async def _resolve_memory_snapshot() -> "MemorySnapshot | None":
        """Invoke ``memory_snapshot_provider`` defensively.

        The provider itself already swallows its own DB / timeout errors
        (see ``PlannerReActFlow._ensure_graphs`` — it wraps the session /
        repo / ``build_memory_snapshot`` chain in try/except and logs).
        We still catch here as a second safety net so any unexpected
        raise from the closure contract never escalates to crashing
        planner / executor / updater — a missing snapshot is equivalent
        to "no memory" and should NOT abort plan generation.
        """
        if memory_snapshot_provider is None:
            return None
        try:
            return await memory_snapshot_provider()
        except Exception as exc:
            logger.warning(
                "memory_snapshot_provider raised; degrading to no-memory prompt: %s",
                exc,
            )
            return None

    # ---- Nodes --------------------------------------------------------- #

    async def planner_node(state: MainGraphState, config: RunnableConfig) -> dict:
        """Call planner LLM to create a plan from user message."""
        bundle = get_prompt_bundle(state.get("language", "zh"))
        attachments = state.get("attachments", [])
        image_blocks = state.get("image_content_blocks", [])
        # Planner 不传图片但需要知道附件包含图片，使用 planner 专用提示
        prompt = bundle.CREATE_PLAN_PROMPT.format(
            message=state["message"],
            attachments=format_attachments_text(
                attachments, has_image_blocks=bool(image_blocks), for_planner=True,
                supports_vision=supports_vision,
            ),
        )

        # B5 C7.5: PromptAssembler is the single code path. planner_node
        # does not receive the LangGraph ``config`` (only ``state``), so
        # we build a dummy config with an empty ``configurable`` —
        # ``bound_tool_names`` will be empty, which is correct: the planner
        # runs BEFORE react_graph_provider and has no per-step tool binding.
        section_bundle = get_prompt_section_bundle(state.get("language", "zh"))
        planner_config = {"configurable": {}}
        # M2 PR-4: planner registry does NOT include memory sections —
        # skip the snapshot fetch. See ``memory_snapshot_provider`` docstring.
        ctx = build_render_context(state, planner_config, agent_config)
        result = prompt_assembler.assemble(
            section_bundle.planner,
            ctx,
            PromptMode.FULL,
            # ``fallback_used`` tracks executor degradation only. Planner
            # never has ``react_graph_provider`` (it runs before execution
            # starts), so the planner call is not a fallback — pass False
            # to keep the telemetry metric semantically clean.
            fallback_used=False,
        )
        system_content = result.text

        # Planner 不传图片：planner 识图不可靠，容易幻觉图片内容并写入 step description，
        # 导致 executor 被错误的描述误导。图片分析留给 executor 通过 MCP 工具完成。
        # 附件文本信息（路径、URL）仍保留，让 planner 知道有附件存在。
        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=prompt),
        ]
        structured_llm = planner_llm.with_structured_output(PlanResponse)

        try:
            parsed: PlanResponse = await structured_llm.ainvoke(messages)
        except ServerRequestsError:
            # Transient transport error — let ``planner_retry``
            # (RetryPolicy on ServerRequestsError at line ~965) handle it.
            # Swallowing here would mask retryable failures as "parse
            # failed" and push a degraded single-step plan downstream.
            raise
        except Exception:
            logger.warning(
                "Planner structured output failed, using fallback plan",
                exc_info=True,
            )
            parsed = PlanResponse(
                title="Task",
                goal=state["message"],
                language=state.get("language", "zh"),
                steps=[StepDef(description=state["message"])],
                message="好的，我来帮你处理。",
            )
        parsed = salvage_empty_memory_recall_plan(
            parsed,
            user_message=state["message"],
            fallback_language=state.get("language", "zh"),
            has_memory_tools=bool(
                (config.get("configurable") or {}).get("has_memory_tools", False)
            ),
            skill_context=state.get("skill_context"),
        )

        # [C2 PR-1 §4.2 r7 P0-2] Build steps via deterministic helper so
        # StepDef.id is propagated directly; when omitted, fall back to
        # _assign_fallback_step_id (no random UUIDs).
        plan = Plan(
            title=parsed.title or "Task",
            goal=parsed.goal or state["message"],
            language=parsed.language or state.get("language", "zh"),
            steps=_build_plan_from_response(parsed).steps,
            message=parsed.message or "",
            status=ExecutionStatus.RUNNING,
        )

        # B5 post-audit LOW #1: notify the runner of the detected session
        # language so subsequent LLM calls' telemetry events record the
        # real lang instead of the default "zh". Injected via
        # configurable["language_callback"] by ``planner_react._build_config``
        # after AgentTaskRunner wires ``self.set_language`` as the callback.
        language_callback = (
            config.get("configurable", {}).get("language_callback")
        )
        if callable(language_callback) and plan.language:
            try:
                language_callback(plan.language)
            except Exception as exc:
                logger.warning(
                    "planner_node: language_callback failed (swallowed): %s",
                    exc,
                )

        events = [
            TitleEvent(title=plan.title),
            MessageEvent(role="assistant", message=plan.message),
            PlanEvent(plan=plan, status=PlanEventStatus.CREATED),
        ]

        return {
            "plan": plan,
            "current_step": plan.get_next_step(),
            "flow_status": FlowStatus.EXECUTING.value,
            "original_request": plan.goal,
            "events": events,
            # B5 C0a fix: write LLM-detected language back to state so executor /
            # updater / summarizer can read it via state["language"]. Without this,
            # state["language"] keeps the initial "zh" default from planner_react,
            # and the bundle dispatch in subsequent nodes is inert.
            "language": plan.language,
        }

    async def executor_node(
        state: MainGraphState, config: RunnableConfig,
    ) -> Command[Literal["updater_node", "interrupt_node", "__end__"]]:
        """Execute current step via react_graph sub-graph.

        Streams react events to the event_queue in real-time so the frontend
        sees tool calls / results as they happen, rather than after the entire
        step completes.
        """
        from app.domain.services.execution_watchdog import _should_terminate

        bundle = get_prompt_bundle(state.get("language", "zh"))

        event_queue: asyncio.Queue | None = (
            config.get("configurable", {}).get("event_queue")
        )

        async def _emit(evt: Any) -> None:
            if event_queue is not None:
                await event_queue.put(evt)

        # D5: Cooperative termination check
        if _should_terminate(config):
            logger.info("executor_node: should_terminate=True, routing to END")
            return Command(
                update={
                    "flow_status": FlowStatus.COMPLETED.value,
                    "events": [],
                    "messages": state.get("messages", []),
                },
                goto=END,
            )

        # D5: Reset tracker blocked set at step boundary
        tracker = config.get("configurable", {}).get("tool_failure_tracker")
        if tracker is not None:
            tracker.reset_blocked()

        step = state["current_step"]
        if not step:
            logger.warning("executor_node: current_step is None, routing to END")
            return Command(
                update={
                    "flow_status": FlowStatus.COMPLETED.value,
                    "events": [],
                    "messages": state.get("messages", []),
                },
                goto=END,
            )

        # ── C2 PR-3 §7.2 — parallel dispatch branch ──────────────────────────
        # When the planner emits ``Step.parallel_work_units``, route execution
        # to the coordinator subgraph instead of react_graph. Hard-gated by
        # ``assert_coordinator_enabled()`` so an env var typo cannot silently
        # enable the cold code path.
        if getattr(step, "parallel_work_units", None) is not None:
            from app.domain.services.coordinator_feature_flag import (
                assert_coordinator_enabled,
            )
            assert_coordinator_enabled()
            # Parity with the react branch (below): emit STARTED, run the
            # coordinator backend, then complete the step.
            await _emit(StepEvent(step=step, status=StepEventStatus.STARTED))
            outcome = await _run_parallel_backend(state, config, step)
            # Mark the step COMPLETED with the authoritative success flag +
            # result text — exactly like the react branch does after a step.
            # Without this the step stays PENDING, so updater_node's
            # Plan.get_next_step() re-returns this same parallel step forever
            # (executor↔updater loop → GraphRecursionError). ``success`` comes
            # from the structured ParallelBackendOutcome, not the summary text.
            step = step.model_copy(update={
                "status": ExecutionStatus.COMPLETED,
                "success": outcome.success,
                "result": outcome.summary,
            })
            await _emit(StepEvent(step=step, status=StepEventStatus.COMPLETED))
            _metrics = config.get("configurable", {}).get("execution_metrics")
            if _metrics is not None:
                if outcome.success:
                    _metrics.steps_completed += 1
                else:
                    _metrics.steps_failed += 1
            return Command(
                update={
                    "current_step": step,
                    # ``execution_summary`` (NOT ``step_result``) is the field
                    # updater_node reads to drive replanning + step sync.
                    "execution_summary": outcome.summary,
                    "resume_value": None,
                    "flow_status": FlowStatus.UPDATING.value,
                    "events": [],  # StepEvents already emitted via queue
                    "messages": state.get("messages", []),
                },
                goto="updater_node",
            )

        # Phase 3: 获取当前 step 的编译后 react_graph（渐进式 Skill 加载）
        # B5 C5a: react_graph_provider now returns (CompiledStateGraph, StepMetadata).
        # StepMetadata carries per-step authoritative bound_tool_names / skill_context
        # / skill_ids. The executor reads these LOCALLY for prompt assembly — see
        # the "Two-Clock Architecture" section in CONTRIBUTING.md. This node must
        # NOT write skill_context back to state (enforced by the AST lint at
        # tests/domain/services/graphs/test_executor_no_skill_context_writeback.py).
        react_graph_provider = (config.get("configurable") or {}).get("react_graph_provider")
        # B5 PR-S2-2 + FOLLOW-10: pin ``step_id`` into a fresh configurable
        # so the downstream subgraph + tool callbacks read the canonical
        # per-step identity. ``traced_node`` reads this slot for the
        # executor span; ``OtelToolSpanCallback`` reads it for tool spans.
        # Always build the fresh dict (even when ``react_graph_provider``
        # is None or raises) so the contract holds for the legacy /
        # fallback paths too.
        fresh_configurable = dict(config.get("configurable") or {})
        fresh_configurable["step_id"] = step.id
        fresh_config = {**config, "configurable": fresh_configurable}
        if react_graph_provider:
            try:
                step_react, step_meta = await react_graph_provider(step.description)
                # Inject per-step bound_tool_names into the fresh configurable for
                # downstream consumers (C5b PromptAssembler will read it).
                fresh_configurable["bound_tool_names"] = step_meta.bound_tool_names
                fresh_skill_context = step_meta.skill_context
                fresh_skill_ids: list[str] = list(step_meta.skill_ids)
            except Exception as _exc:
                # P1 #6 fix — when the runner has a ``tool_filter`` in
                # effect (subagent), ``_react_graph_provider_for_executor``
                # raises ``ToolFilterProviderFailure``. Propagate that
                # marker so we do NOT fall back to the default unfiltered
                # ``react_graph`` (which is built from the unfiltered
                # ``_collect_all_tools()`` at flow construction time and
                # would silently re-arm the subagent with the parent
                # agent's complete tool set). The chat surfaces an error
                # event instead of silently downgrading.
                #
                # Import is local to avoid a domain ↔ runner module cycle.
                from app.domain.services.agent_task_runner import (
                    ToolFilterProviderFailure,
                )
                if isinstance(_exc, ToolFilterProviderFailure):
                    raise
                logger.warning("react_graph_provider 失败，使用默认（无动态Skill工具）")
                step_react = react_graph
                fresh_skill_context = state.get("skill_context", "") or ""
                fresh_skill_ids = list(state.get("skill_names_in_context") or [])
        else:
            step_react = react_graph
            fresh_skill_context = state.get("skill_context", "") or ""
            fresh_skill_ids = list(state.get("skill_names_in_context") or [])

        resume_value = state.get("resume_value")

        # Emit StepEvent(STARTED) — skip on resume to avoid duplicate events
        if resume_value is None:
            await _emit(StepEvent(step=step, status=StepEventStatus.STARTED))

        # Build initial messages with system prompt + execution prompt
        attachments = state.get("attachments", [])
        image_blocks = state.get("image_content_blocks", [])
        language = state.get("language", "zh")

        # B5 C5b: section_assembly_meta carries {version_hash, tokens_used}
        # for the main_graph state writeback. Stays empty on the resume
        # path (system_content is not rebuilt).
        section_assembly_meta: dict[str, Any] = {}

        # Resume path reuses ``saved_messages`` verbatim and appends a
        # HumanMessage(resume_hint) — it never uses ``system_content``. Skip
        # the whole system prompt build when resuming so the PromptAssembler
        # path doesn't run wastefully.
        system_content = ""

        # B5 C7.5: PromptAssembler is the single code path. The whole
        # build is gated on ``resume_value is None`` so resume completely
        # skips prompt assembly. See CONTRIBUTING.md "Prompt Assembly 不变式"
        # section for the two-clock architecture around ``skill_context``.
        if resume_value is None:
            section_bundle = get_prompt_section_bundle(language)
            # Build a LOCAL state view with the fresh per-step skill context.
            # This is the two-clock architecture: we do NOT write skill_context
            # back to state — we only use this view for prompt assembly.
            state_for_render = {
                **state,
                "skill_context": fresh_skill_context,
                "skill_names_in_context": fresh_skill_ids,
            }
            # NOTE: ``agent_config`` here is the ``build_main_graph`` closure
            # parameter (not per-request state). It's stable for the lifetime
            # of the graph and safe to read via ``getattr(..., default)``
            # inside ``build_render_context``.
            memory_snapshot = await _resolve_memory_snapshot()
            ctx = build_render_context(
                state_for_render, fresh_config, agent_config,
                memory_snapshot=memory_snapshot,
            )
            # fallback_used=True when react_graph_provider was unavailable
            # and we're relying on the legacy state.skill_context path.
            fallback_used = react_graph_provider is None
            assembly_result = prompt_assembler.assemble(
                section_bundle.executor,
                ctx,
                PromptMode.FULL,
                fallback_used=fallback_used,
            )
            system_content = assembly_result.text
            section_assembly_meta = {
                "system_prompt_version_hash": assembly_result.version_hash,
                "system_prompt_tokens": assembly_result.tokens_used,
            }

        # 两分支 messages 构建逻辑（使用 LangChain 消息类型）
        # resume_value 由 interrupt_node 在恢复时设置，或由 DB fallback 路径直接注入
        saved_messages: list[BaseMessage] = state.get("messages") or []

        if resume_value is not None:
            # Resume path: messages 已由 checkpointer 保存，追加用户回复
            last_tool = next(
                (m for m in reversed(saved_messages)
                 if isinstance(m, ToolMessage) and m.name == "message_ask_user"),
                None,
            )
            if last_tool is not None:
                resume_hint = (
                    f"用户已回复你之前的提问。请根据用户回复继续执行当前步骤：{step.description}\n"
                    f"用户回复：{resume_value}"
                )
            else:
                resume_hint = (
                    f"用户已完成接管并交还控制。请继续执行当前步骤：{step.description}\n"
                    f"用户消息：{resume_value}"
                )
            initial_messages = saved_messages + [HumanMessage(content=resume_hint)]
        elif saved_messages:
            # 非首步/有历史：更新 system prompt 为最新版本，追加新 execution prompt
            # 图片已在首步消息中（已被 compact 剥离），无需重复添加
            first = saved_messages[0]
            updated_first = SystemMessage(content=system_content) if isinstance(first, SystemMessage) else first
            initial_messages = [
                updated_first,
                *saved_messages[1:],
                HumanMessage(content=bundle.EXECUTION_PROMPT.format(
                    message=state["message"],
                    attachments=format_attachments_text(attachments),
                    prior_step_outputs=_format_prior_step_outputs(state.get("plan"), language),
                    language=language,
                    step=step.description,
                )),
            ]
        else:
            # 首步/无历史：干净的 system + execution prompt（含图片多模态内容）
            attachments_text = format_attachments_text(
                attachments, has_image_blocks=bool(image_blocks),
                supports_vision=supports_vision,
            )
            execution_text = bundle.EXECUTION_PROMPT.format(
                message=state["message"],
                attachments=attachments_text,
                prior_step_outputs=_format_prior_step_outputs(state.get("plan"), language),
                language=language,
                step=step.description,
            )
            logger.debug("[IMG] executor_node: image_blocks=%d, attachments=%s", len(image_blocks), attachments)
            prompt_content = build_multimodal_content(execution_text, image_blocks)
            logger.debug("[IMG] executor_node: prompt_content type=%s, is_list=%s", type(prompt_content).__name__, isinstance(prompt_content, list))
            initial_messages = [
                SystemMessage(content=system_content),
                HumanMessage(content=prompt_content),
            ]

        # Build react_graph input
        react_input = {
            "messages": initial_messages,
            "step_description": step.description,
            "original_request": state.get("original_request", ""),
            "language": language,
            "attachments": attachments,
            "image_content_blocks": image_blocks,
            "events": [],
            "should_interrupt": False,
            "soft_hint_sent": False,
            "attempt_count": 0,
            "failure_count": 0,
        }

        # ── Cross-step context assembly ──
        if assembler is not None and resume_value is None:
            asm_result = assembler.assemble(react_input["messages"])
            react_input["messages"] = asm_result.messages
            if asm_result.actions:
                logger.info("context_assembler(cross-step): %s", asm_result.actions)

        # Stream react_graph — emit events in real-time.
        # IMPORTANT: Must use stream_mode="updates" explicitly.
        # LangGraph 1.0.x defaults to "values" (full state per chunk), but we
        # need "updates" ({node_name: state_update} per chunk) so that we can
        # accumulate only NEW messages from each node and emit events in order.
        react_final: dict[str, Any] = {}
        all_react_messages: list = []
        seen_interrupt = False
        # B5 C5a: pass fresh_config (with injected bound_tool_names) so
        # downstream PromptAssembler consumers (C5b) can read the per-step
        # bound tool set. When react_graph_provider is None, fresh_config is
        # simply the original config.
        async for chunk in step_react.astream(
            react_input, config=fresh_config, stream_mode="updates",
        ):
            for _node_name, node_output in chunk.items():
                if not isinstance(node_output, dict):
                    continue
                # Accumulate messages separately — pop before update to
                # prevent react_final["messages"] from being overwritten
                all_react_messages.extend(node_output.pop("messages", []))
                if node_output.get("should_interrupt"):
                    seen_interrupt = True
                react_final.update(node_output)
                # Push react events to frontend immediately
                for evt in node_output.get("events") or []:
                    await _emit(evt)
        # Reconstruct full message list: initial input + all new messages,
        # deduplicating by ID to replicate add_messages semantics.
        react_final["messages"] = dedup_messages(initial_messages + all_react_messages)
        if seen_interrupt:
            react_final["should_interrupt"] = True

        # Extract execution summary from last AI message, and also parse
        # the JSON envelope (``{"result"|"message", "attachments"}``) so we
        # can populate ``step.attachments`` for the next step's
        # ``prior_step_outputs`` hint. We parse the FULL content (not the
        # 500-char summary) because envelopes with long ``result`` bodies
        # place ``attachments`` after the text and would be cut off.
        #
        # Scan ONLY the messages added in this react run (``all_react_messages``).
        # ``react_final["messages"]`` is initial + new deduped, so scanning
        # that list would fall back to the previous step's AI envelope
        # when the current step produced no new AIMessage(content) — and
        # ``prior_step_outputs`` would then carry the previous step's
        # attachments into every subsequent step.
        summary = ""
        step_attachments: list[str] = []
        for msg in reversed(all_react_messages):
            if isinstance(msg, AIMessage) and msg.content:
                content_str = (
                    msg.content if isinstance(msg.content, str) else str(msg.content)
                )
                summary = content_str[:500]
                _, step_attachments = unwrap_message_envelope(content_str)
                break

        # Detect step success from react_graph's accumulated failure_count
        step_success = react_final.get("failure_count", 0) == 0

        # Compact messages to reduce context size for next step
        # (mirrors main-branch compact_memory behaviour)
        final_messages = _compact_messages(
            react_final.get("messages", state.get("messages", []))
        )

        # Check for interrupt (user takeover request) BEFORE marking step as completed.
        # 中断时步骤尚未完成，保持 RUNNING 状态以便恢复后继续执行。
        should_interrupt = react_final.get("should_interrupt", False)
        if should_interrupt:
            # 中断的步骤不标记为 COMPLETED，保留 RUNNING 状态
            await _emit(WaitEvent())
            return Command(
                update={
                    "messages": final_messages,
                    "current_step": step,
                    "execution_summary": summary,
                    "should_interrupt": True,
                    "resume_value": None,
                    "flow_status": FlowStatus.EXECUTING.value,
                    "original_request": state.get("original_request", ""),
                    "events": [],  # WaitEvent 已通过 _emit 发送，避免重复
                    # B5 C5b: observability — only populated when PromptAssembler path ran.
                    # Never includes skill_context (two-clock architecture).
                    **section_assembly_meta,
                },
                goto="interrupt_node",
            )

        # 通过 model_copy 创建新对象避免直接变异 state 对象
        # （LangGraph 要求节点返回 partial update dict，不可直接修改 state）
        step = step.model_copy(update={
            "status": ExecutionStatus.COMPLETED,
            "success": step_success,
            "result": summary,
            "attachments": step_attachments,
        })
        await _emit(StepEvent(step=step, status=StepEventStatus.COMPLETED))

        # D5: Record step outcome in metrics
        _metrics = config.get("configurable", {}).get("execution_metrics")
        if _metrics is not None:
            if step_success:
                _metrics.steps_completed += 1
            else:
                _metrics.steps_failed += 1

        return Command(
            update={
                "messages": final_messages,
                "current_step": step,
                "execution_summary": summary,
                "resume_value": None,
                "flow_status": FlowStatus.UPDATING.value,
                "events": [],  # already emitted via queue
                # B5 C5b: observability — only populated when PromptAssembler path ran.
                # Never includes skill_context (two-clock architecture).
                **section_assembly_meta,
            },
            goto="updater_node",
        )

    async def updater_node(
        state: MainGraphState, config: RunnableConfig,
    ) -> Command[Literal["executor_node", "__end__"]]:
        """Update the plan after step execution — mark step done, call planner
        to update remaining steps with execution context, then get next step.

        This mirrors the main-branch PlannerReActFlow UPDATING phase:
        1. Sync completed step back into plan
        2. Call planner LLM with UPDATE_PLAN_PROMPT (execution_summary)
        3. Replace pending steps with planner's updated steps
        4. Emit PlanEvent(UPDATED)
        """
        from app.domain.services.execution_watchdog import _should_terminate

        bundle = get_prompt_bundle(state.get("language", "zh"))

        event_queue: asyncio.Queue | None = (
            config.get("configurable", {}).get("event_queue")
        )

        # D5: Cooperative termination check
        if _should_terminate(config):
            logger.info("updater_node: should_terminate=True, routing to END")
            return Command(
                update={
                    "flow_status": FlowStatus.COMPLETED.value,
                    "events": [],
                },
                goto=END,
            )

        plan = state["plan"]
        if not plan:
            return Command(
                update={"flow_status": FlowStatus.COMPLETED.value, "events": []},
                goto=END,
            )

        # 1. Sync completed step back into plan.steps
        completed_step = state.get("current_step")
        if completed_step and completed_step.done:
            updated_steps = [
                completed_step if s.id == completed_step.id else s
                for s in plan.steps
            ]
            plan = plan.model_copy(update={"steps": updated_steps})

        # 2. Call planner LLM to update remaining steps based on execution results.
        # Skip when the plan has no pending steps left — there is nothing to
        # re-plan, and making a redundant LLM round-trip has two real costs
        # we just observed in a production trace:
        #   (a) wasted tokens / latency on an unused PlanUpdateResponse;
        #   (b) if the outer LangGraph task is cancelled while that call is
        #       in flight (e.g. SSE client disconnect), the httpx/openai
        #       read raises asyncio.CancelledError which bypasses our
        #       ``except Exception`` below (CancelledError is BaseException
        #       in py3.8+) and surfaces as a run-failure stack trace even
        #       though the last step already succeeded.
        execution_summary = state.get("execution_summary", "")
        plan_updated = False
        has_pending_step = plan.get_next_step() is not None
        if completed_step and execution_summary and has_pending_step:
            try:
                query = bundle.UPDATE_PLAN_PROMPT.format(
                    plan=plan.model_dump_json(),
                    step=completed_step.model_dump_json(),
                    execution_summary=execution_summary or bundle.EXECUTION_SUMMARY_NONE_FALLBACK,
                )

                # B5 C7.5: PromptAssembler is the single code path.
                # Updater runs without ``react_graph_provider`` by design
                # (it has no per-step tool binding), so ``fallback_used``
                # is always False. The updater registry shares sections
                # with the planner registry — see bundles/zh.py.
                section_bundle = get_prompt_section_bundle(
                    state.get("language", "zh")
                )
                updater_config = {"configurable": {}}
                # M2 PR-4: updater registry does NOT include memory sections —
                # skip the snapshot fetch. See ``memory_snapshot_provider``
                # docstring.
                ctx = build_render_context(state, updater_config, agent_config)
                result = prompt_assembler.assemble(
                    section_bundle.updater,
                    ctx,
                    PromptMode.FULL,
                    fallback_used=False,
                )
                system_content = result.text

                update_messages = [
                    SystemMessage(content=system_content),
                    HumanMessage(content=query),
                ]
                structured_llm = planner_llm.with_structured_output(PlanUpdateResponse)
                parsed_obj: PlanUpdateResponse = await structured_llm.ainvoke(update_messages)

                if parsed_obj and parsed_obj.steps:
                    # [C2 PR-1 §4.2 r7 P0-2] Reuse existing Step.id when the
                    # LLM keeps it; fall back to _assign_fallback_step_id
                    # rooted on the plan's id so the fallback is stable
                    # across replan rounds within the same plan.
                    # [C2b rollout WS0 §3A.4] Flag-off sanitation (updater path).
                    from app.domain.services.coordinator_feature_flag import (
                        is_coordinator_enabled,
                    )
                    _coord_on = is_coordinator_enabled()
                    new_steps = [
                        Step(
                            description=s.description,
                            id=s.id or _assign_fallback_step_id(plan.id, i),
                            parallel_work_units=(
                                s.parallel_work_units if _coord_on else None
                            ),
                        )
                        for i, s in enumerate(parsed_obj.steps)
                        if s.description
                    ]

                    if new_steps:
                        # Find first pending step index
                        first_pending_index = None
                        for idx, s in enumerate(plan.steps):
                            if not s.done:
                                first_pending_index = idx
                                break

                        if first_pending_index is not None:
                            # Preserve completed steps, replace pending with new
                            merged = list(plan.steps[:first_pending_index]) + new_steps
                            plan = plan.model_copy(update={"steps": merged})

                        plan_updated = True
                        logger.info(
                            "Planner 更新计划成功，新步骤数: %d", len(new_steps)
                        )
            except Exception as exc:
                # Plan update failure is non-fatal — continue with original plan
                logger.warning("Planner 更新计划失败，沿用原计划继续: %s", exc)

        events: list = []
        if plan_updated:
            events.append(PlanEvent(plan=plan, status=PlanEventStatus.UPDATED))
            if event_queue is not None:
                for evt in events:
                    await event_queue.put(evt)

        # 3. Find next step
        next_step = plan.get_next_step()

        # Phase 3: 根据下一步描述刷新 skill context
        new_skill_context = state.get("skill_context", "")
        if next_step:
            refresher = (config.get("configurable") or {}).get("skill_context_refresher")
            if refresher:
                try:
                    new_skill_context = await refresher(next_step.description)
                except Exception as exc:
                    logger.warning("Step-level skill refresh 失败: %s", exc)

        if not next_step:
            plan.status = ExecutionStatus.COMPLETED
            events.append(PlanEvent(plan=plan, status=PlanEventStatus.COMPLETED))
            if event_queue is not None:
                for evt in events:
                    await event_queue.put(evt)
            return Command(
                update={
                    "plan": plan,
                    "current_step": None,
                    "flow_status": FlowStatus.COMPLETED.value,
                    "skill_context": new_skill_context,
                    "events": [],  # already emitted via queue
                },
                goto=END,
            )

        return Command(
            update={
                "plan": plan,
                "current_step": next_step,
                "flow_status": FlowStatus.EXECUTING.value,
                "skill_context": new_skill_context,
                "events": events,
            },
            goto="executor_node",
        )

    async def summarizer_node(state: MainGraphState, config: RunnableConfig) -> dict:
        """Generate final summary with streaming and emit completion events.

        Streams partial MessageEvent chunks via event_queue for real-time
        frontend updates. The final MessageEvent carries partial=False,
        the same stream_id, and parsed attachments.
        """
        bundle = get_prompt_bundle(state.get("language", "zh"))

        event_queue: asyncio.Queue | None = (
            config.get("configurable", {}).get("event_queue")
        )

        async def _emit(evt: Any) -> None:
            if event_queue is not None:
                await event_queue.put(evt)

        plan = state["plan"]
        events: list = []

        if plan:
            # 通过 model_copy 避免直接变异 state 对象
            plan = plan.model_copy(update={"status": ExecutionStatus.COMPLETED})
            plan_evt = PlanEvent(plan=plan, status=PlanEventStatus.COMPLETED)
            events.append(plan_evt)
            await _emit(plan_evt)

        # Stream LLM summary to frontend in real-time
        react_messages: list[BaseMessage] = state.get("messages", [])
        stream_id = str(uuid.uuid4())

        if react_messages:
            try:
                # summary_llm is BaseChatModel — pass LangChain messages directly
                messages: list[BaseMessage] = list(react_messages) + [
                    HumanMessage(content=bundle.SUMMARIZE_PROMPT),
                ]

                # Stream with cumulative prefix for frontend
                chunks: list[str] = []
                async for chunk in summary_llm.astream(messages):
                    if chunk.content:
                        chunks.append(chunk.content)
                        cumulative_text = "".join(chunks)
                        await _emit(MessageEvent(
                            role="assistant",
                            message=cumulative_text,
                            stream_id=stream_id,
                            partial=True,
                        ))

                full_text = "".join(chunks)

                # Tolerant {message, attachments} JSON unwrap — four-tier
                # fallback handles pseudo-JSON with unescaped newlines in
                # string values (common when markdown content is long).
                # See app.domain.services.json_envelope.
                summary_text = full_text
                summary_attachments: list[str] = []
                if full_text:
                    summary_text, summary_attachments = unwrap_message_envelope(
                        full_text
                    )

                # Build File attachments
                file_attachments: list[File] = []
                for path in summary_attachments:
                    filename = path.rsplit("/", 1)[-1]
                    ext = filename.rsplit(".", 1)[-1] if "." in filename else ""
                    file_attachments.append(File(
                        filename=filename,
                        filepath=path,
                        extension=ext,
                    ))

                # Final MessageEvent (partial=False, same stream_id, with attachments)
                final_msg_evt = MessageEvent(
                    role="assistant",
                    message=summary_text,
                    attachments=file_attachments,
                    stream_id=stream_id,
                    partial=False,
                )
                events.append(final_msg_evt)
                await _emit(final_msg_evt)
            except Exception as exc:
                logger.warning("Summarizer LLM call failed: %s", exc)

        done_evt = DoneEvent()
        events.append(done_evt)
        await _emit(done_evt)

        result: dict = {
            "flow_status": FlowStatus.COMPLETED.value,
            "events": [],  # events already emitted via queue
        }
        if plan:
            result["plan"] = plan
        return result

    async def interrupt_node(state: MainGraphState) -> dict:
        """Pause graph execution for user input via LangGraph native interrupt().

        This node is only reached when executor_node sets should_interrupt=True.
        On resume, interrupt() returns the user's response, which is stored in
        resume_value for executor_node to consume.
        """
        user_response = interrupt({
            "reason": "user_input_required",
            "step": state["current_step"].description if state.get("current_step") else "",
        })
        return {
            "resume_value": user_response,
            "should_interrupt": False,
        }

    # ---- Routing ------------------------------------------------------- #

    def route_entry(state: MainGraphState) -> Literal[
        "planner_node", "executor_node", "updater_node",
    ]:
        """Route from START based on flow_status."""
        status = state.get("flow_status", FlowStatus.IDLE.value)

        if status in (FlowStatus.IDLE.value, FlowStatus.PLANNING.value):
            return "planner_node"
        if status == FlowStatus.EXECUTING.value:
            return "executor_node"
        return "updater_node"

    # ---- Build Graph --------------------------------------------------- #

    g: StateGraph = StateGraph(MainGraphState)

    # RetryPolicy for transient planner LLM errors
    planner_retry = RetryPolicy(
        max_attempts=3,
        initial_interval=2.0,
        backoff_factor=2.0,
        retry_on=ServerRequestsError,
    )

    # B5 PR-S2-2: optional infrastructure-supplied decorator wraps each
    # registered node coroutine. Composition layer
    # (``app/application/composition/graph_assembly.py``) passes
    # ``traced_node(OtelTracer())`` so each invocation produces a
    # ``graph.node.<name>`` span; ``None`` keeps the legacy behaviour.
    # The decorator MUST preserve ``__name__`` (LangGraph introspects it)
    # — ``functools.wraps`` is the standard tool.
    _wrap = node_decorator if node_decorator is not None else (lambda fn: fn)
    g.add_node("planner_node", _wrap(planner_node), retry_policy=planner_retry)
    g.add_node("executor_node", _wrap(executor_node))
    g.add_node("updater_node", _wrap(updater_node))
    g.add_node("interrupt_node", _wrap(interrupt_node))

    g.add_conditional_edges(START, route_entry)
    g.add_edge("planner_node", "executor_node")
    # executor_node and updater_node use Command(goto=...) for routing —
    # no conditional edges needed. Command handles all outgoing transitions.
    g.add_edge("interrupt_node", "executor_node")

    return g.compile(checkpointer=checkpointer)
