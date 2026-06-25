"""C2 v1 ChildAgentTaskRunnerFactory — full build (spec §5.3 + §8.5.1).

Constructs the inner ``AgentTaskRunner`` that a coordinator child wraps,
plus a ``BuiltChildRunner`` wrapper carrying the runtime concerns
(``cancel_event``, ``child_permission_context``) that the live
``AgentTaskRunner`` ctor does not (and should not) accept.

Why the wrapper instead of extending AgentTaskRunner.__init__:
- ``AgentTaskRunner.__init__`` already takes ~40 kwargs covering provider,
  storage, browser, sandbox, telemetry, permission, memory, etc. Extending
  it with cancel_event + child_permission_context blurs the construction
  surface (runtime vs. construction concerns).
- ``cancel_event`` is consumed by the react_graph nodes at runtime via
  ``config["configurable"]["cancel_event"]`` (PR-4 Task 4.5 checkpoints).
  CoordinatorChildRunner (PR-4 Task 4.7) threads it into config when it
  invokes the runner.
- ``child_permission_context`` is consumed by ChildScopeGate (PE Phase 1)
  and by finalizers attributing scope violations. It travels with the
  child, not the runner construction.

Spec-anchored behavior:
- ``tool_filter_preset == "coordinator_step"`` →
  ``terminal_envelope_publisher_disabled = True``
  (§8.5.1 r6 P0-1 — CoordinatorChildRunner is the sole terminal publisher
   for this preset; runner staying silent prevents duplicate
   RESULT_READY/CANCEL_ACK envelopes on the wire).
- ``tool_filter_preset == "subagent_research"`` → disabled = False
  (default AgentTaskRunner publisher path remains in charge).
- ``tool_filter_preset`` unknown → ``resolve_preset`` raises ``ValueError``;
  the factory propagates it (fail closed: silently defaulting to no filter
  would be an allowlist escape).
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, FrozenSet, Optional, Protocol

from app.domain.models.tool_filter_presets import COORDINATOR_STEP_PRESET
from app.domain.services.coordinator_shell_mode_flag import (
    is_coordinator_shell_mode_enabled,
)
from app.domain.services.tool_filter_presets import resolve_preset

# [S2 §3.5] The 5 raw-shell tools the bind-time widen unions into the
# coordinator_step allowlist when shell-mode. Must match
# child_scope_gate.SHELL_HARD_BLOCKED_NAMES (lockstep gate un-block).
_SHELL_MODE_UNION_TOOLS: frozenset[str] = frozenset({
    "shell_execute", "shell_wait_process", "shell_kill_process",
    "shell_write_input", "shell_read_output",
})


def _apply_member_skill_bind(tool_filter, child_permission_context, tool_filter_preset):
    """[S4 §11] Union the member skill GENERATED tool names into the bind floor,
    gated on the coordinator-step preset (self-defends a future non-coordinator
    reuse, same as the shell widen). Bind floor = preset ∪ member_skill_tools —
    NEVER WorkUnit.allowed_tools (the planner is an LLM; it must not widen its own
    bind surface)."""
    member_tools = frozenset(
        getattr(child_permission_context, "member_skill_tools", frozenset()) or frozenset()
    )
    if tool_filter is not None and member_tools and tool_filter_preset == COORDINATOR_STEP_PRESET:
        return tool_filter | member_tools
    return tool_filter


class ChildRunnerBuilder(Protocol):
    """[C2 PR-4 r7 P1] Protocol the factory expects from its ``runner_class``
    injection point.

    The live ``AgentTaskRunner.__init__`` takes ~40 kwargs (uow_factory, llm,
    agent_config, mcp_config, a2a_config, user_id, file_storage, browser,
    search_engine, sandbox, ...). The factory CANNOT build it directly with
    just the 4 kwargs below.

    PR-5's runner_starter (the integration point at composition root) is
    responsible for ``functools.partial`` -wrapping ``AgentTaskRunner`` with
    the 36 other dependencies pre-bound, producing a callable that conforms
    to this Protocol. That bound callable is what the factory receives as
    ``runner_class``.

    Passing the raw ``AgentTaskRunner`` class would fail with TypeError on
    the missing required args — typing this as a Protocol surfaces the
    wiring contract that runner_starter must satisfy.
    """

    def __call__(
        self,
        *,
        session_id: str,
        tool_filter: Optional[FrozenSet[str]],
        mailbox_publisher: Any,
        terminal_envelope_publisher_disabled: bool,
        sandbox: Any,
        browser: Any,
        user_id: str,
        cost_callback_handler: Any,
    ) -> Any: ...


@dataclass(frozen=True)
class BuiltChildRunner:
    """Wrapper bundling the adapter-wrapped child runner with runtime
    concerns that travel with the child but are not AgentTaskRunner ctor
    inputs.

    [C2b budget §3-8 — docstring re-anchored to shipped reality]
    ``.runner`` is the ``AgentTaskRunnerInvokeAdapter`` ALREADY wrapping the
    raw AgentTaskRunner (build() constructs the adapter below — the
    PR-4-era "starter wraps it later" wording predated that). The live
    consumer is ``DefaultCoordinatorChildRunnerStarter.start`` which:
    1. Passes ``.runner`` (the adapter) as CoordinatorChildRunner's
       inner_runner (the adapter satisfies CoordinatorChildInnerRunner).
    2. Calls ``.runner.set_budget_callback(...)`` for the C2b late-injected
       BudgetEnforcementCallback (adapter → raw runner → flow chain).
    3. The adapter ctor itself already wired ``cancel_event`` +
       ``child_permission_context`` into the raw runner at build() time.
    4. Reads ``.terminal_envelope_publisher_disabled`` as a sanity-check
       against the runner's own ``_terminal_envelope_publisher_disabled``.

    ``runner``                              — the constructed AgentTaskRunner
    ``cancel_event``                        — asyncio.Event the parent sets
                                              to abort the child via react_graph
                                              cancel checkpoints (Task 4.5)
    ``child_permission_context``            — ChildPermissionContext consumed
                                              by ChildScopeGate at tool dispatch
    ``terminal_envelope_publisher_disabled`` — mirror of the kwarg passed to
                                              the runner, kept on the wrapper
                                              for downstream assertion + audit
    """

    runner: Any
    cancel_event: asyncio.Event
    child_permission_context: Any
    terminal_envelope_publisher_disabled: bool


class ChildAgentTaskRunnerFactory:
    """Builds the inner ``AgentTaskRunner`` for a coordinator-step child.

    ``runner_class`` is injected (defaults to ``AgentTaskRunner`` at the
    composition root) so tests can swap a MagicMock without monkey-patching.
    ``mailbox_publisher`` is the wire publisher shared across all children
    spawned in a coordinator run.
    """

    def __init__(
        self, *, runner_class: ChildRunnerBuilder, mailbox_publisher: Any, task_cls: Any,
    ) -> None:
        # ``runner_class`` is typed as ChildRunnerBuilder — see Protocol above
        # for why the live AgentTaskRunner class itself can't be passed
        # directly. PR-5's runner_starter wraps AgentTaskRunner via
        # functools.partial to satisfy this contract.
        self._runner_class = runner_class
        self._mailbox_publisher = mailbox_publisher
        # ``task_cls`` is the RedisStreamTask class the invoke-adapter drives
        # (Task.create(task_runner=...)); injected so tests can substitute a
        # MagicMock without monkey-patching the adapter's import.
        self._task_cls = task_cls

    async def build(
        self,
        *,
        child_session_id: str,
        child_permission_context: Any,
        tool_filter_preset: str,
        cancel_event: asyncio.Event,
        sandbox: Any,
        browser: Any,
        user_id: str,
        cost_callback_handler: Any,
        coordinator_metrics_recorder: Any = None,
    ) -> BuiltChildRunner:
        """Build the child AgentTaskRunner via the shared runner builder and
        wrap it in the invoke-adapter (§5.1 Shape-1-variant). The adapter ctor
        performs the cancel-event wiring (set_coordinator_cancel_event).

        Raises ``ValueError`` if ``tool_filter_preset`` is not registered.
        """
        from app.application.services.agent_task_runner_invoke_adapter import (
            AgentTaskRunnerInvokeAdapter,
        )
        tool_filter = resolve_preset(tool_filter_preset)
        # [S2 §3.5] Bind-time dual-loosening: when this child runs shell-mode
        # (master flag ON AND its spawn_manifest.shell_mode True), union the 5
        # raw-shell tools into the resolved allowlist so llm.bind_tools exposes
        # them. Keep the PERSISTED preset string (coordinator_step) — runtime
        # union only, NO DB migration (§10-B). Lockstep with the runtime gate
        # un-block (child_scope_gate.py): preset-widen WITHOUT gate-unblock
        # would bind a tool the gate then HARD_BLOCKs, and vice-versa.
        # [codex PR-5 R1 P1] Scope the widen to the coordinator-step preset.
        # shell-mode is a coordinator feature; the live caller always passes
        # COORDINATOR_STEP_PRESET, and a non-coordinator child never carries
        # shell_mode=True. Gating on the preset self-defends this reusable
        # factory boundary so a future non-coordinator reuse with a hand-built
        # shell_mode context can NEVER bind raw exec shell.
        _shell_mode = bool(
            getattr(child_permission_context, "shell_mode", False)
        )
        _shell_widen = (
            tool_filter is not None
            and _shell_mode
            and tool_filter_preset == COORDINATOR_STEP_PRESET
            and is_coordinator_shell_mode_enabled()
        )
        if _shell_widen:
            tool_filter = tool_filter | _SHELL_MODE_UNION_TOOLS
        # [S4 §11] Union the member's GENERATED skill tool names into the bind
        # floor (preset ∪ member_skill_tools), gated on the coordinator-step
        # preset — same self-defense as the shell widen above. Empty member
        # tools / non-coordinator preset / tool_filter=None ⇒ identity, so
        # flag-OFF / no-team children keep byte-identical bind behavior.
        tool_filter = _apply_member_skill_bind(
            tool_filter, child_permission_context, tool_filter_preset
        )
        terminal_disabled = tool_filter_preset == COORDINATOR_STEP_PRESET
        raw_runner = self._runner_class(
            session_id=child_session_id,
            tool_filter=tool_filter,
            mailbox_publisher=self._mailbox_publisher,
            terminal_envelope_publisher_disabled=terminal_disabled,
            sandbox=sandbox,
            browser=browser,
            user_id=user_id,
            cost_callback_handler=cost_callback_handler,
        )
        adapter = AgentTaskRunnerInvokeAdapter(
            runner=raw_runner, cancel_event=cancel_event, task_cls=self._task_cls,
            child_permission_context=child_permission_context,
            coordinator_metrics_recorder=coordinator_metrics_recorder,
        )
        return BuiltChildRunner(
            runner=adapter,
            cancel_event=cancel_event,
            child_permission_context=child_permission_context,
            terminal_envelope_publisher_disabled=terminal_disabled,
        )
