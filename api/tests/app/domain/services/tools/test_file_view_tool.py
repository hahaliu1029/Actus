import asyncio
from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import ToolMessage

from app.domain.external.file_processor import FileProcessResult


class FakeProcessor:
    async def process(self, sandbox_path, filename, mime_type, supports_vision, supports_pdf_input=False):
        return FileProcessResult(
            text=f"[Image: {filename}, 100x200]",
            image_blocks=({"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},),
        )


class FakeLookup:
    def get_processor(self, mime_type):
        if mime_type.startswith("image/"):
            return FakeProcessor()
        return None


def _make_sandbox_mock(mime_output: str = "image/png", returncode: int = 0, success: bool = True):
    sandbox = AsyncMock()
    mock_result = MagicMock()
    mock_result.success = success
    mock_result.data = {
        "returncode": returncode,
        "output": mime_output,
    }
    mock_result.__str__ = lambda self: mime_output
    sandbox.exec_command = AsyncMock(return_value=mock_result)
    return sandbox


class TestFileViewTool:
    def test_file_view_plain_ainvoke_returns_summary_text(self):
        from app.domain.services.tools.langchain_tools import _make_file_view_tools

        tools = _make_file_view_tools(_make_sandbox_mock(), FakeLookup(), supports_vision=True)
        file_view = tools[0]

        result = asyncio.run(
            file_view.ainvoke({"filepath": "/home/ubuntu/test.png"})
        )
        assert isinstance(result, str)
        assert "100x200" in result

    def test_file_view_tool_call_returns_tool_message_with_passthrough_artifact(self):
        from app.domain.services.tools.langchain_tools import _make_file_view_tools

        tools = _make_file_view_tools(_make_sandbox_mock(), FakeLookup(), supports_vision=True)
        file_view = tools[0]

        result = asyncio.run(
            file_view.ainvoke(
                {
                    "id": "call_file_view",
                    "name": "file_view",
                    "args": {"filepath": "/home/ubuntu/test.png"},
                    "type": "tool_call",
                }
            )
        )

        assert isinstance(result, ToolMessage)
        assert result.content == "[Image: test.png, 100x200]"
        assert result.artifact.variant == "passthrough"

    def test_file_view_unsupported_type_returns_string(self):
        from app.domain.services.tools.langchain_tools import _make_file_view_tools

        tools = _make_file_view_tools(_make_sandbox_mock("text/plain"), FakeLookup(), supports_vision=True)
        file_view = tools[0]

        result = asyncio.run(
            file_view.ainvoke({"filepath": "/home/ubuntu/readme.txt"})
        )
        assert isinstance(result, str)
        assert "Unsupported" in result

    def test_file_view_extension_fallback(self):
        """When `file --mime-type` returns octet-stream, fall back to extension."""
        from app.domain.services.tools.langchain_tools import _make_file_view_tools

        tools = _make_file_view_tools(
            _make_sandbox_mock("application/octet-stream"), FakeLookup(), supports_vision=True,
        )
        file_view = tools[0]

        result = asyncio.run(
            file_view.ainvoke({"filepath": "/home/ubuntu/photo.jpg"})
        )
        # Extension .jpg maps to image/jpeg → FakeLookup matches image/ prefix
        assert isinstance(result, str)
        assert "100x200" in result

    def test_file_view_returncode_127_falls_back_to_extension(self):
        """When `file` command is not installed (returncode 127), fall back to extension."""
        import pytest
        from app.domain.services.tools.langchain_tools import _make_file_view_tools

        sandbox = _make_sandbox_mock(
            mime_output="/bin/bash: file: 未找到命令\n",
            returncode=127,
        )
        tools = _make_file_view_tools(sandbox, FakeLookup(), supports_vision=True)
        file_view = tools[0]

        result = asyncio.run(
            file_view.ainvoke({"filepath": "/home/ubuntu/upload/photo.png"})
        )
        # .png → image/png → FakeLookup matches image/ prefix
        assert isinstance(result, str)
        assert "100x200" in result

    def test_file_view_nonzero_returncode_returns_error_text(self):
        """When `file` fails for a real reason, wrapper returns typed error content."""
        from app.domain.services.tools.langchain_tools import _make_file_view_tools

        sandbox = _make_sandbox_mock(
            mime_output="cannot open `/no/such/file` (No such file or directory)\n",
            returncode=1,
        )
        tools = _make_file_view_tools(sandbox, FakeLookup(), supports_vision=True)
        file_view = tools[0]

        result = asyncio.run(file_view.ainvoke({"filepath": "/no/such/file.png"}))
        assert result.startswith("Cannot detect file type")

    def test_create_native_tools_includes_file_view(self):
        from app.domain.services.tools.langchain_tools import create_native_tools

        tools = create_native_tools(
            sandbox=AsyncMock(), browser=AsyncMock(), search_engine=AsyncMock(),
            processor_lookup=FakeLookup(), supports_vision=True,
        )
        tool_names = [t.name for t in tools]
        assert "file_view" in tool_names

    def test_create_native_tools_without_lookup_has_no_file_view(self):
        from app.domain.services.tools.langchain_tools import create_native_tools

        tools = create_native_tools(
            sandbox=AsyncMock(), browser=AsyncMock(), search_engine=AsyncMock(),
        )
        tool_names = [t.name for t in tools]
        assert "file_view" not in tool_names
