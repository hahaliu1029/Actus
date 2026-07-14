"""Bridge between LangGraph execution and Actus Event stream.

Uses an asyncio.Queue so that events from both the main graph nodes and
the nested react_graph are yielded to the frontend in real-time.

Nodes that receive an ``event_queue`` via LangGraph config can push events
directly; other nodes' events are picked up from astream output.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncGenerator

from app.domain.models.event import (
    BaseEvent,
    ControlAction,
    ControlEvent,
    HealthEvent,
    HealthStatus,
    ToolConfirmationEvent,
    WaitEvent,
)
from app.domain.services.execution_watchdog import (
    ExecutionControl,
    ExecutionWatchdog,
    WatchdogVerdict,
    _is_progress_event,
)

logger = logging.getLogger(__name__)


def _is_hitl_control_event(event: BaseEvent) -> bool:
    """Whether consumers stop immediately so graph interrupt state must drain."""
    if isinstance(event, (WaitEvent, ToolConfirmationEvent)):
        return True
    return (
        isinstance(event, ControlEvent)
        and event.action == ControlAction.REQUESTED
    )


class GraphEventBridge:
    """Runs a LangGraph and streams events in real-time via an async queue."""

    def __init__(self) -> None:
        self._final_state: dict[str, Any] = {}
        self._was_interrupted: bool = False

    @property
    def final_state(self) -> dict[str, Any]:
        """The full graph output state after execution."""
        return self._final_state

    @property
    def was_interrupted(self) -> bool:
        """Whether the graph was interrupted (has pending interrupt_node)."""
        return self._was_interrupted

    async def run(
        self,
        graph: Any,
        input_state: dict[str, Any] | Any,
        config: dict[str, Any] | None = None,
    ) -> AsyncGenerator[BaseEvent, None]:
        """Stream the graph, yielding events as each node produces them.

        Events reach the caller via two paths:

        1. **Queue path** — nodes that receive ``event_queue`` in config push
           events directly (used by ``executor_node`` for react sub-graph).
        2. **State path** — nodes that only return ``{"events": [...]}`` have
           their events picked up here from the ``astream`` output.

        Parameters
        ----------
        graph : Compiled LangGraph StateGraph.
        input_state : Input state dict, or a Command for resuming.
        config : Optional config dict with ``configurable`` keys (e.g. thread_id).
        """
        queue: asyncio.Queue[BaseEvent | None] = asyncio.Queue()
        # 用 input_state 初始化 _final_state，确保节点未返回的字段保留输入值
        # （例如 executor_node 跳过时不会丢失 messages）
        # 仅当 input_state 为 dict 时初始化（Command 不可展开为 dict）
        if isinstance(input_state, dict):
            self._final_state = dict(input_state)
        else:
            self._final_state = {}

        # Merge event_queue into config
        merged_config: dict[str, Any] = {"configurable": {"event_queue": queue}}
        if config:
            for key, value in config.items():
                if key == "configurable":
                    merged_config["configurable"].update(value)
                else:
                    merged_config[key] = value

        # D5: Extract watchdog + control from config (created by PlannerReActFlow)
        configurable = merged_config.get("configurable", {})
        watchdog: ExecutionWatchdog | None = configurable.get("execution_watchdog")
        control: ExecutionControl | None = configurable.get("execution_control")
        wait_guard_factory = configurable.get("coordinator_wait_guard_factory")
        if (
            watchdog is not None
            and configurable.get("coordinator_wait_guard") is None
            and callable(wait_guard_factory)
        ):
            # The composition root supplies an opaque application-layer
            # factory. Keeping construction here makes the guard share this
            # invoke's exact watchdog without importing application code into
            # the domain graph layer.
            configurable["coordinator_wait_guard"] = wait_guard_factory(
                watchdog=watchdog
            )

        async def _drive_graph() -> None:
            """Run the graph and forward state-path events to the queue."""
            try:
                async for chunk in graph.astream(
                    input_state,
                    config=merged_config,
                    stream_mode="updates",
                ):
                    for _node_name, node_output in chunk.items():
                        if not isinstance(node_output, dict):
                            continue
                        self._final_state.update(node_output)
                        # D5: Record progress from astream path
                        if watchdog is not None:
                            watchdog.record_progress(node_name=_node_name)
                        # Emit events that were NOT already pushed via queue
                        # (nodes using queue return events=[])
                        for evt in node_output.get("events") or []:
                            if isinstance(evt, BaseEvent):
                                await queue.put(evt)
            except Exception:
                logger.exception("GraphEventBridge: graph execution error")
                raise
            finally:
                # Detect if graph was interrupted via checkpointer state
                try:
                    graph_state = await graph.aget_state(merged_config)
                    if graph_state and graph_state.next:
                        self._was_interrupted = True
                except Exception:
                    # No checkpointer or aget_state unavailable (e.g. mock graph) —
                    # fall back to checking should_interrupt in final state
                    self._was_interrupted = self._final_state.get(
                        "should_interrupt", False
                    )
                await queue.put(None)  # sentinel

        task = asyncio.create_task(_drive_graph())

        # Track exit path for the finally block:
        #   _sentinel_exit = True  → graph finished naturally (sentinel received)
        #   _watchdog_terminated = True → HARD_TERMINATE cancelled the task
        #   neither → GeneratorExit from caller cleanup (e.g. WaitEvent)
        _sentinel_exit = False
        _watchdog_terminated = False
        _last_yield_requires_hitl_drain = False

        def _emit_terminating() -> HealthEvent:
            """Build the HealthEvent(TERMINATING) payload."""
            return HealthEvent(
                status=HealthStatus.TERMINATING,
                reason="执行即将超时终止",
                last_node=watchdog.last_node if watchdog else None,
                idle_seconds=round(watchdog.idle_seconds, 1) if watchdog else None,
                action="hard_terminate",
            )

        try:
            while True:
                # D5: Use wait_for with idle timeout when watchdog is active.
                # When total_timeout is finite and smaller than idle_timeout,
                # shrink the wait so the total cap is not starved waiting for
                # the idle tick. Minimum wait of 0.1s to avoid a tight loop.
                if watchdog is not None:
                    _wait_timeout = watchdog.idle_timeout_seconds
                    if watchdog.total_timeout_seconds > 0:
                        _remaining_total = max(
                            0.1,
                            watchdog.total_timeout_seconds - watchdog.elapsed_seconds,
                        )
                        _wait_timeout = min(_wait_timeout, _remaining_total)
                    try:
                        event = await asyncio.wait_for(
                            queue.get(),
                            timeout=_wait_timeout,
                        )
                    except asyncio.TimeoutError:
                        # Always check total-only first (it's the hard cap).
                        # If the wait was shortened for total, evaluate() may
                        # still classify as HEALTHY for idle but HARD for total.
                        verdict = watchdog.evaluate()
                        if verdict == WatchdogVerdict.SOFT_RECOVER:
                            logger.warning(
                                "watchdog SOFT_RECOVER: idle=%.1fs last_node=%s session=%s",
                                watchdog.idle_seconds,
                                watchdog.last_node,
                                configurable.get("session_id", "?"),
                            )
                            _last_yield_requires_hitl_drain = False
                            yield HealthEvent(
                                status=HealthStatus.DEGRADED,
                                reason="Agent 似乎遇到了困难，正在尝试恢复...",
                                last_node=watchdog.last_node,
                                idle_seconds=round(watchdog.idle_seconds, 1),
                                action="soft_recovery",
                            )
                            # Inject recovery hint for llm_node
                            if control is not None:
                                control.idle_recovery_hint = (
                                    "[SYSTEM] You appear stuck with no progress. "
                                    "Try a different approach or summarize current progress."
                                )
                            continue
                        elif verdict == WatchdogVerdict.HARD_TERMINATE:
                            logger.warning(
                                "watchdog HARD_TERMINATE: elapsed=%.1fs idle=%.1fs last_node=%s session=%s",
                                watchdog.elapsed_seconds,
                                watchdog.idle_seconds,
                                watchdog.last_node,
                                configurable.get("session_id", "?"),
                            )
                            # Set internal state FIRST so callers that react
                            # synchronously to TERMINATING (and then stop
                            # consuming) still see should_terminate=True.
                            if control is not None:
                                control.should_terminate = True
                            task.cancel()
                            _watchdog_terminated = True
                            _last_yield_requires_hitl_drain = False
                            yield _emit_terminating()
                            break
                        else:
                            # HEALTHY after re-evaluation (e.g. total not yet reached)
                            continue
                else:
                    event = await queue.get()

                if event is None:
                    _sentinel_exit = True
                    break

                # D5: Record progress from queue-path events
                if watchdog is not None and _is_progress_event(event):
                    watchdog.record_progress()

                # D5: Check total_timeout on every event (not just idle).
                # Otherwise a graph that keeps producing output bypasses
                # the total cap entirely.
                if watchdog is not None and watchdog.check_total_only():
                    logger.warning(
                        "watchdog HARD_TERMINATE(total): elapsed=%.1fs last_node=%s session=%s",
                        watchdog.elapsed_seconds,
                        watchdog.last_node,
                        configurable.get("session_id", "?"),
                    )
                    # Set state first so sync break-on-TERMINATING still sees it.
                    if control is not None:
                        control.should_terminate = True
                    task.cancel()
                    _watchdog_terminated = True
                    # Yield the triggering event first so the frontend sees
                    # it before the termination notice.
                    _last_yield_requires_hitl_drain = _is_hitl_control_event(event)
                    yield event
                    _last_yield_requires_hitl_drain = False
                    yield _emit_terminating()
                    break

                _last_yield_requires_hitl_drain = _is_hitl_control_event(event)
                yield event
                _last_yield_requires_hitl_drain = False
        finally:
            if _sentinel_exit and not _watchdog_terminated:
                # Normal finish: task should complete cleanly.
                await task
            elif _last_yield_requires_hitl_drain and not _watchdog_terminated:
                # The live consumer returns immediately on WaitEvent,
                # ToolConfirmationEvent, or takeover REQUESTED. In all three
                # paths the real LangGraph then naturally completes its
                # interrupt node and persists should_interrupt/checkpoint
                # state. Drain that bounded graph transition instead of
                # cancelling it. No task-wide timeout belongs here.
                try:
                    await task
                except Exception:
                    logger.warning(
                        "GraphEventBridge: suppressed _drive_graph error during "
                        "HITL state drain (error already logged above)"
                    )
            else:
                # Ordinary early consumer shutdown must not wait indefinitely
                # for a graph still blocked in a node. HARD_TERMINATE already
                # cancels the task; task.cancel() is idempotent in that path.
                if not task.done():
                    task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    # Expected for explicit bridge cleanup / HARD_TERMINATE.
                    pass
                except Exception:
                    # Suppress _drive_graph exceptions during cleanup.
                    # Three scenarios:
                    # 1. Ordinary GeneratorExit from caller cleanup.
                    # 2. HARD_TERMINATE: we cancelled the task, CancelledError expected.
                    # 3. Other non-normal exits: already logged by _drive_graph's except.
                    # Without this suppression, errors propagate to agent_task_runner's
                    # `except Exception` handler, which overwrites WAITING status to COMPLETED.
                    logger.warning(
                        "GraphEventBridge: suppressed _drive_graph error during "
                        "generator cleanup (error already logged above)"
                    )
