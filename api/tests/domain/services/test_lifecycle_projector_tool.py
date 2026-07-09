"""C7 PR3 — 投影器 tool 分支映射测试（spec §4.3）。"""
from typing import Any, Dict, Optional

import pytest

from app.domain.models.event import ToolEvent, ToolEventStatus
from app.domain.models.lifecycle import (
    LifecycleEventKind as K,
    LifecycleState as S,
    LifecycleType as T,
)
from app.domain.services.lifecycle_projector import LifecycleProjector


@pytest.fixture
def projector() -> LifecycleProjector:
    return LifecycleProjector(parent_session_id="sess-1")


def _tool_event(
    status: ToolEventStatus,
    *,
    variant: Optional[str] = None,
    reason_type: Optional[str] = None,
    function_result: Any = None,
) -> ToolEvent:
    artifact: Optional[Dict[str, Any]] = None
    if variant is not None:
        outcome: Dict[str, Any] = {"variant": variant}
        if variant == "allow_error":
            # message 用 "boom"：test_no_raw_error_text_leaks_into_lifecycle 的
            # `"boom" not in dumped` 断言依赖源 artifact 真带该文本（R5 review P3-2）
            outcome["reason"] = {"type": reason_type or "exception", "code": "x", "message": "boom"}
        artifact = {
            "tool_call_id": "tc-1", "tool_name": "shell",
            "tool_source": "native", "outcome": outcome,
        }
    ev = ToolEvent(
        tool_call_id="tc-1", tool_name="shell",
        function_name="shell_execute", function_args={},
        status=status, artifact=artifact, function_result=function_result,
    )
    ev.seq = 30
    ev.id = "src-30"
    return ev


class TestPhaseEvents:
    @pytest.mark.asyncio
    async def test_calling_maps_started_pending(self, projector):
        out = await projector.project(_tool_event(ToolEventStatus.CALLING))
        assert [(o.lifecycle_type, o.event, o.state) for o in out] == [(T.TOOL, K.STARTED, S.PENDING)]
        assert out[0].unit_id == "tc-1"

    @pytest.mark.asyncio
    async def test_running_maps_progress_running(self, projector):
        out = await projector.project(_tool_event(ToolEventStatus.RUNNING))
        assert [(o.event, o.state) for o in out] == [(K.PROGRESS, S.RUNNING)]


class TestCalledOutcomes:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "variant,reason_type,expected_kind,expected_state,expected_reason",
        [
            ("allow_success", None, K.COMPLETED, S.COMPLETED, None),
            ("allow_error", "exception", K.FAILED, S.FAILED, "tool_error"),
            ("allow_error", "timeout", K.FAILED, S.FAILED, "tool_timeout"),
            ("denied", None, K.CANCELLED, S.CANCELLED, "denied"),
            ("passthrough", None, K.COMPLETED, S.COMPLETED, "passthrough"),
        ],
    )
    async def test_called_variant_mapping(
        self, projector, variant, reason_type, expected_kind, expected_state, expected_reason
    ):
        out = await projector.project(
            _tool_event(ToolEventStatus.CALLED, variant=variant, reason_type=reason_type)
        )
        assert [(o.event, o.state, o.reason) for o in out] == [
            (expected_kind, expected_state, expected_reason)
        ]

    @pytest.mark.asyncio
    async def test_asked_is_not_projected(self, projector):
        # Asked 走 ToolConfirmationEvent 独立通道（§4.3 注）
        out = await projector.project(_tool_event(ToolEventStatus.CALLED, variant="asked"))
        assert list(out) == []

    @pytest.mark.asyncio
    async def test_denied_never_maps_failed(self, projector):
        # 取消非失败——denied 的 state 必须是 cancelled（§4.3）
        out = await projector.project(_tool_event(ToolEventStatus.CALLED, variant="denied"))
        assert out[0].state is S.CANCELLED and out[0].state is not S.FAILED

    @pytest.mark.asyncio
    async def test_no_raw_error_text_leaks_into_lifecycle(self, projector):
        # 敏感信息禁入：allow_error 的 message/code 只留在 source，lifecycle 仅受控 code
        out = await projector.project(
            _tool_event(ToolEventStatus.CALLED, variant="allow_error", reason_type="exception")
        )
        dumped = out[0].model_dump_json()
        assert "boom" not in dumped and '"message"' not in dumped


class TestLegacyFallback:
    class _LegacyResult:
        def __init__(self, success: bool) -> None:
            self.success = success

    @pytest.mark.asyncio
    async def test_called_no_artifact_success_true(self, projector):
        ev = _tool_event(ToolEventStatus.CALLED)
        ev.function_result = None
        object.__setattr__(ev, "function_result", None)  # 保底：pydantic 可直接赋值则删本行
        ev2 = _tool_event(ToolEventStatus.CALLED)
        ev2.function_result = self._LegacyResult(True)  # ToolResult 鸭子型（仅用 .success，镜像 legacy 判别）
        out = await projector.project(ev2)
        assert [(o.event, o.reason) for o in out] == [(K.COMPLETED, None)]

    @pytest.mark.asyncio
    async def test_called_no_artifact_success_false(self, projector):
        ev = _tool_event(ToolEventStatus.CALLED)
        ev.function_result = self._LegacyResult(False)
        out = await projector.project(ev)
        assert [(o.event, o.state, o.reason) for o in out] == [(K.FAILED, S.FAILED, "tool_error")]

    @pytest.mark.asyncio
    async def test_called_nothing_resolvable_returns_empty(self, projector):
        out = await projector.project(_tool_event(ToolEventStatus.CALLED))
        assert list(out) == []  # 不硬造终态

    @pytest.mark.asyncio
    async def test_malformed_artifact_falls_back_gracefully(self, projector):
        ev = _tool_event(ToolEventStatus.CALLED)
        ev.artifact = {"outcome": "not-a-dict"}
        out = await projector.project(ev)
        assert list(out) == []  # try/except 兜底，不 raise 不硬造
