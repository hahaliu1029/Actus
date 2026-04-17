"""R4 P1 executable guard 3 (Round 2f): _translate_outcome 在 runtime 拒绝
未启用 variant. 证明 enabled_outcome_variants 不是死配置.
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.anyio


@pytest.fixture()
def anyio_backend() -> str:
    return "asyncio"


class TestEnabledVariantsRuntimeGuard:
    """config 里去掉某个 variant 后, _translate_outcome 遇到它必须 fail-fast."""

    async def test_disabled_variant_raises_value_error(self) -> None:
        """enabled_outcome_variants 不含 'passthrough' 时, 产出 Passthrough
        outcome 必须抛 ValueError, 不让 wrapper 绕过运维灰度."""
        from app.domain.models.tool_result import (
            MultimodalPayload, Passthrough, TextBlock,
        )
        from app.domain.services.graphs.react_graph import _translate_outcome
        from app.domain.services.tools.tool_source_resolver import ToolSource

        ts = ToolSource(source="native", category="shell", canonical_name="shell_execute")
        outcome = Passthrough(
            content="hi",
            data=MultimodalPayload(blocks=[TextBlock(text="hi")]),
        )
        tool_call = {"id": "c1", "name": "shell_execute", "args": {}}

        with pytest.raises(ValueError, match="variant .+ not enabled"):
            await _translate_outcome(
                outcome=outcome, tool_call=tool_call, tool_source=ts,
                session_ctx=None, tool_result_max_chars=8000,
                guide_injector=None,
                enabled_outcome_variants=[
                    "allow_success", "allow_error", "denied", "asked",
                ],  # passthrough 被禁
            )

    async def test_enabled_variant_passes(self) -> None:
        """config 允许的 variant 正常产出 envelope."""
        from app.domain.models.tool_result import AllowSuccess
        from app.domain.services.graphs.react_graph import _translate_outcome
        from app.domain.services.tools.tool_source_resolver import ToolSource

        ts = ToolSource(source="native", category="shell", canonical_name="shell_execute")
        outcome = AllowSuccess(content="ok")
        tool_call = {"id": "c1", "name": "shell_execute", "args": {}}

        _, _, events = await _translate_outcome(
            outcome=outcome, tool_call=tool_call, tool_source=ts,
            session_ctx=None, tool_result_max_chars=8000,
            guide_injector=None,
            enabled_outcome_variants=[
                "allow_success", "allow_error", "denied", "asked", "passthrough",
            ],
        )
        assert len(events) == 1

    async def test_no_config_no_enforcement_backward_compat(self) -> None:
        """enabled_outcome_variants=None 时跳过校验（backward-compat 兜底）."""
        from app.domain.models.tool_result import AllowSuccess
        from app.domain.services.graphs.react_graph import _translate_outcome
        from app.domain.services.tools.tool_source_resolver import ToolSource

        ts = ToolSource(source="native", category="shell", canonical_name="shell_execute")
        outcome = AllowSuccess(content="ok")
        tool_call = {"id": "c1", "name": "shell_execute", "args": {}}

        _, _, events = await _translate_outcome(
            outcome=outcome, tool_call=tool_call, tool_source=ts,
            session_ctx=None, tool_result_max_chars=8000,
            guide_injector=None,
            enabled_outcome_variants=None,
        )
        assert len(events) == 1
