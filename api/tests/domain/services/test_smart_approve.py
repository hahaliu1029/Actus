import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from app.domain.services.smart_approve import SmartApprove


def _run(coro):
    return asyncio.run(coro)


class TestSmartApprove:
    def setup_method(self):
        self.llm = AsyncMock()
        self.smart = SmartApprove(llm=self.llm)

    def test_approve(self):
        self.llm.ainvoke.return_value = MagicMock(content="APPROVE")
        assert _run(self.smart.evaluate("shell_execute", {"command": "ls"}, "high", [], "listing files")) == "approve"

    def test_deny(self):
        self.llm.ainvoke.return_value = MagicMock(content="DENY")
        assert _run(self.smart.evaluate("shell_execute", {"command": "rm -rf /"}, "high", ["recursive_delete"], "cleanup")) == "deny"

    def test_escalate(self):
        self.llm.ainvoke.return_value = MagicMock(content="ESCALATE")
        assert _run(self.smart.evaluate("shell_execute", {"command": "pip install foo"}, "high", [], "deps")) == "escalate"

    def test_fallback_on_error(self):
        self.llm.ainvoke.side_effect = Exception("timeout")
        assert _run(self.smart.evaluate("shell_execute", {"command": "ls"}, "high", [], "test")) == "escalate"

    def test_unexpected_response_escalates(self):
        self.llm.ainvoke.return_value = MagicMock(content="MAYBE")
        assert _run(self.smart.evaluate("shell_execute", {"command": "ls"}, "high", [], "test")) == "escalate"


# ---------------------------------------------------------------------------
# B5 PR-S3-2: decision_trace integration locks.
# ---------------------------------------------------------------------------


class TestSmartApproveDecisionEvents:
    """SmartApprove writes a ``decision.smart_approve`` span event on
    the currently-active span for each outcome path (approve / deny /
    escalate-on-unexpected / escalate-on-error). Locks the wiring so a
    future refactor that drops the ``record_decision`` calls fails
    loudly rather than silently breaking dashboards.

    Round-2 P3: the recorder is now injected via the constructor
    instead of imported by the domain module. Pass the OTel-backed
    ``record_decision`` explicitly so this suite still exercises the
    real production wiring; downstream tests that don't care about
    observability can leave ``decision_recorder=None``.
    """

    def setup_method(self):
        from app.infrastructure.observability.decision_trace import (
            record_decision,
        )

        self.llm = AsyncMock()
        self.smart = SmartApprove(
            llm=self.llm, decision_recorder=record_decision
        )
        self.provider = TracerProvider()
        self.exporter = InMemorySpanExporter()
        self.provider.add_span_processor(
            SimpleSpanProcessor(self.exporter)
        )
        self.tracer = self.provider.get_tracer("actus-test")

    def _events_on_one_span(self) -> list[Any]:
        spans = self.exporter.get_finished_spans()
        assert len(spans) == 1, f"expected single span; got {len(spans)}"
        return list(spans[0].events)

    def test_approve_emits_decision_event_with_outcome(self):
        self.llm.ainvoke.return_value = MagicMock(content="APPROVE")

        async def _go():
            with self.tracer.start_as_current_span("tool.shell_execute"):
                return await self.smart.evaluate(
                    "shell_execute", {"command": "ls"}, "high", [], "listing"
                )

        assert _run(_go()) == "approve"
        events = self._events_on_one_span()
        assert len(events) == 1
        ev = events[0]
        assert ev.name == "decision.smart_approve"
        assert ev.attributes.get("decision_outcome") == "approve"
        assert ev.attributes.get("tool_name") == "shell_execute"
        # Reason intentionally absent on the happy path.
        assert "decision_reason" not in ev.attributes

    def test_deny_emits_decision_event_with_outcome(self):
        self.llm.ainvoke.return_value = MagicMock(content="DENY")

        async def _go():
            with self.tracer.start_as_current_span("tool.shell_execute"):
                return await self.smart.evaluate(
                    "shell_execute", {"command": "rm -rf /"}, "high",
                    ["recursive_delete"], "cleanup",
                )

        assert _run(_go()) == "deny"
        ev = self._events_on_one_span()[0]
        assert ev.attributes.get("decision_outcome") == "deny"

    def test_unexpected_response_emits_event_with_reason(self):
        self.llm.ainvoke.return_value = MagicMock(content="MAYBE")

        async def _go():
            with self.tracer.start_as_current_span("tool.shell_execute"):
                return await self.smart.evaluate(
                    "shell_execute", {"command": "ls"}, "high", [], "test"
                )

        assert _run(_go()) == "escalate"
        ev = self._events_on_one_span()[0]
        assert ev.attributes.get("decision_outcome") == "escalate"
        assert ev.attributes.get("decision_reason") == "unexpected_response"

    def test_llm_error_emits_event_with_reason(self):
        self.llm.ainvoke.side_effect = Exception("timeout")

        async def _go():
            with self.tracer.start_as_current_span("tool.shell_execute"):
                return await self.smart.evaluate(
                    "shell_execute", {"command": "ls"}, "high", [], "test"
                )

        assert _run(_go()) == "escalate"
        ev = self._events_on_one_span()[0]
        assert ev.attributes.get("decision_outcome") == "escalate"
        assert ev.attributes.get("decision_reason") == "llm_error"

    def test_no_raw_tool_args_in_event_attributes(self):
        """Privacy lock: tool args MUST NOT leak into the decision
        event. Only ``tool_name`` is recorded — args go through
        ``OtelToolSpanCallback`` as a hash on the parent span.
        """
        canary_command = "rm -rf /CANARY-PATH-aXf9zQ"
        self.llm.ainvoke.return_value = MagicMock(content="DENY")

        async def _go():
            with self.tracer.start_as_current_span("tool.shell_execute"):
                return await self.smart.evaluate(
                    "shell_execute",
                    {"command": canary_command},
                    "high",
                    [],
                    "cleanup",
                )

        _run(_go())
        ev = self._events_on_one_span()[0]
        for k, v in ev.attributes.items():
            assert "CANARY-PATH-aXf9zQ" not in str(v), (
                f"raw tool args leaked into decision event attr {k}={v!r}"
            )


class TestSmartApproveNoRecorderIsSilent:
    """Round-2 P3 lock: when ``decision_recorder=None`` (default)
    SmartApprove emits NO span events — the domain layer pays zero
    observability cost when composition hasn't wired the recorder.

    This is the contract that lets tests instantiate
    ``SmartApprove(llm=mock)`` without touching OTel globals, and
    that lets downstream consumers (CLI / scripts / non-graph
    flows) opt out of decision telemetry.
    """

    def setup_method(self):
        self.llm = AsyncMock()
        # No decision_recorder → silent.
        self.smart = SmartApprove(llm=self.llm)
        self.provider = TracerProvider()
        self.exporter = InMemorySpanExporter()
        self.provider.add_span_processor(
            SimpleSpanProcessor(self.exporter)
        )
        self.tracer = self.provider.get_tracer("actus-test")

    def test_default_recorder_emits_no_event(self):
        self.llm.ainvoke.return_value = MagicMock(content="APPROVE")

        async def _go():
            with self.tracer.start_as_current_span("tool.shell_execute"):
                return await self.smart.evaluate(
                    "shell_execute", {"command": "ls"}, "high", [], "listing"
                )

        assert _run(_go()) == "approve"
        spans = self.exporter.get_finished_spans()
        assert len(spans) == 1
        assert len(spans[0].events) == 0


class TestSmartApproveBrokenRecorder:
    """Round-3 P3 port-contract lock: a raising recorder MUST NOT
    change or break SmartApprove outcomes.

    Without per-call wrap inside ``evaluate``, a recorder that raises
    on the APPROVE path would: (a) escape the happy path, (b) get
    caught by the outer ``except``, (c) cause the except branch to
    call the **same failing recorder** a second time, (d) propagate
    the second exception out of ``evaluate()`` — silently turning
    legitimate ``approve`` outcomes into uncaught exceptions.

    The OTel-backed production recorder is best-effort by design so
    production isn't affected, but the port contract MUST hold for
    test fakes and future custom recorder impls too.
    """

    def _make_smart_with_broken_recorder(self, llm_content: str | None = None,
                                          llm_raises: bool = False):
        def _broken(*a, **kw):
            raise RuntimeError("recorder broke")

        llm = AsyncMock()
        if llm_raises:
            llm.ainvoke.side_effect = Exception("upstream timeout")
        else:
            llm.ainvoke.return_value = MagicMock(content=llm_content)
        return SmartApprove(llm=llm, decision_recorder=_broken)

    def test_broken_recorder_in_approve_path_returns_approve(self):
        smart = self._make_smart_with_broken_recorder("APPROVE")
        result = _run(
            smart.evaluate("shell_execute", {"cmd": "ls"}, "high", [], "test")
        )
        assert result == "approve"

    def test_broken_recorder_in_deny_path_returns_deny(self):
        smart = self._make_smart_with_broken_recorder("DENY")
        result = _run(
            smart.evaluate(
                "shell_execute", {"cmd": "rm -rf /"}, "high",
                ["recursive_delete"], "cleanup",
            )
        )
        assert result == "deny"

    def test_broken_recorder_in_escalate_path_returns_escalate(self):
        smart = self._make_smart_with_broken_recorder("ESCALATE")
        result = _run(
            smart.evaluate("shell_execute", {"cmd": "x"}, "high", [], "test")
        )
        assert result == "escalate"

    def test_broken_recorder_in_unexpected_response_path_returns_escalate(
        self,
    ):
        smart = self._make_smart_with_broken_recorder("MAYBE")
        result = _run(
            smart.evaluate("shell_execute", {"cmd": "ls"}, "high", [], "test")
        )
        assert result == "escalate"

    def test_broken_recorder_in_llm_error_path_returns_escalate(self):
        """Both LLM and recorder raise — SmartApprove still returns
        ``escalate`` cleanly. This is the worst-case path: without
        per-call wrap, the LLM exception lands the except, the except
        re-raises through the recorder, exception leaks. With wrap,
        both raise sites are absorbed.
        """
        smart = self._make_smart_with_broken_recorder(llm_raises=True)
        result = _run(
            smart.evaluate("shell_execute", {"cmd": "ls"}, "high", [], "test")
        )
        assert result == "escalate"
