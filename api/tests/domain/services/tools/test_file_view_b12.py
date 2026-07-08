"""B12 P2: file_view media_type producer gate."""
import asyncio
from unittest.mock import AsyncMock, MagicMock

from app.domain.external.file_processor import FileProcessResult
from app.domain.models.tool_result import Passthrough


class _FakeImageProcessor:
    async def process(self, sandbox_path, filename, mime_type, supports_vision, supports_pdf_input=False):
        return FileProcessResult(
            text=f"[Image: {filename}, 10x10]",
            image_blocks=(
                {"type": "image_url", "image_url": {"url": "https://minio/a.png", "detail": "auto"}},
            ),
            media_type="image/png",
        )


class _FakeLookup:
    def get_processor(self, mime_type):
        if mime_type.startswith("image/"):
            return _FakeImageProcessor()
        return None


def _make_sandbox_mock(mime_output: str = "image/png"):
    sandbox = AsyncMock()
    mock_result = MagicMock()
    mock_result.success = True
    mock_result.data = {"returncode": 0, "output": mime_output}
    mock_result.__str__ = lambda self: mime_output
    sandbox.exec_command = AsyncMock(return_value=mock_result)
    return sandbox


def _invoke_file_view(**flags):
    from app.domain.services.tools.langchain_tools import _make_file_view_tools

    tools = _make_file_view_tools(
        _make_sandbox_mock(), _FakeLookup(), supports_vision=True, **flags
    )
    file_view = tools[0]
    return asyncio.run(file_view.ainvoke({
        "id": "call_fv", "name": "file_view",
        "args": {"filepath": "/home/ubuntu/a.png"}, "type": "tool_call",
    }))


def test_file_view_media_type_flag_off_omits_media_type() -> None:
    result = _invoke_file_view(file_view_media_type_enabled=False)
    assert isinstance(result.artifact, Passthrough)
    assert result.artifact.data.media_type is None


def test_file_view_media_type_flag_on_sets_media_type() -> None:
    result = _invoke_file_view(file_view_media_type_enabled=True)
    assert isinstance(result.artifact, Passthrough)
    assert result.artifact.data.media_type == "image/png"


def test_planner_collect_native_tools_forwards_media_type_flag() -> None:
    """R2#P2-1: planner_react._collect_native_tools 也须透传 producer flag（主路径）。
    create_native_tools 在 planner_react 模块级 import → 可 patch 该符号。"""
    from unittest.mock import MagicMock, patch

    from app.domain.models.app_config import ToolRuntimeConfig
    from app.domain.services.flows.planner_react import PlannerReActFlow

    flow = PlannerReActFlow.__new__(PlannerReActFlow)
    flow._sandbox = MagicMock()
    flow._browser = MagicMock()
    flow._search_engine = MagicMock()
    flow._file_processor_lookup = MagicMock()
    flow._supports_vision = True
    flow._supports_pdf_input = False
    flow._execution_supervisor = None
    flow._user_id = ""
    flow._tool_runtime = ToolRuntimeConfig(file_view_media_type_enabled=True)

    with patch(
        "app.domain.services.flows.planner_react.create_native_tools", return_value=[]
    ) as m:
        flow._collect_native_tools()
    assert m.call_args.kwargs["file_view_media_type_enabled"] is True
