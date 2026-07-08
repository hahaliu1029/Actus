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


# --------------------------------------------------------------------------- #
# B12 P3: session-scoped file_view image cache (Task 3.2)
# --------------------------------------------------------------------------- #


class _CachingLookup:
    def __init__(self, cached, fresh):
        self._cached = cached
        self._proc = AsyncMock()
        self._proc.process = AsyncMock(return_value=fresh)
        self.get_calls: list = []
        self.put_calls: list = []

    def get_processor(self, mime):
        return self._proc

    def cache_get(self, key):
        self.get_calls.append(key)
        return self._cached

    def cache_put(self, key, result, ts):
        self.put_calls.append(key)


def _sandbox_with_stat(mime: str = "image/png"):
    sandbox = AsyncMock()

    async def _exec(session, exec_dir, command):
        r = MagicMock()
        r.success = True
        if "python3" in command:  # stat 命令
            r.data = {"returncode": 0,
                      "output": '{"realpath":"/w/a.png","size":10,"mtime_ns":1,"dev":2,"ino":3}'}
        else:  # file --mime-type
            r.data = {"returncode": 0, "output": mime}
        r.__str__ = lambda self: mime
        return r

    sandbox.exec_command = AsyncMock(side_effect=_exec)
    sandbox.generation = 1
    return sandbox


def _img_result(text: str):
    return FileProcessResult(
        text=text,
        image_blocks=({"type": "image_url", "image_url": {"url": "https://minio/a.png", "detail": "auto"}},),
        media_type="image/png",
    )


def _invoke_cache(lookup, **flags):
    from app.domain.services.tools.langchain_tools import _make_file_view_tools

    tools = _make_file_view_tools(_sandbox_with_stat(), lookup, supports_vision=True, **flags)
    return asyncio.run(tools[0].ainvoke({
        "id": "c", "name": "file_view", "args": {"filepath": "/w/a.png"}, "type": "tool_call",
    }))


def test_file_view_cache_hit_skips_process() -> None:
    lookup = _CachingLookup(cached=_img_result("[cached]"), fresh=_img_result("[fresh]"))
    result = _invoke_cache(lookup, file_view_image_cache_enabled=True)
    assert result.artifact.content == "[cached]"
    lookup._proc.process.assert_not_called()
    assert len(lookup.get_calls) == 1


def test_file_view_cache_miss_processes_and_puts() -> None:
    lookup = _CachingLookup(cached=None, fresh=_img_result("[fresh]"))
    result = _invoke_cache(lookup, file_view_image_cache_enabled=True)
    assert result.artifact.content == "[fresh]"
    lookup._proc.process.assert_awaited_once()
    assert len(lookup.put_calls) == 1


def test_file_view_cache_disabled_no_lookup() -> None:
    lookup = _CachingLookup(cached=_img_result("[cached]"), fresh=_img_result("[fresh]"))
    result = _invoke_cache(lookup, file_view_image_cache_enabled=False)
    assert result.artifact.content == "[fresh]"       # processed, cache ignored
    lookup._proc.process.assert_awaited_once()
    assert len(lookup.get_calls) == 0                  # cache never consulted


def test_cache_hit_result_still_materializes_for_kimi() -> None:
    """spec §12 / R2#P2-3: cache 命中的 result（含 presigned URL）→ file_view Passthrough
    → _translate_outcome(materialize) 仍为 accepts_image_url=False provider 拉 bytes 转 base64
    （证 P1 materialize × P3 cache 组合无死角）。"""
    from app.domain.services.graphs.react_graph import _translate_outcome
    from app.domain.services.tools.tool_source_resolver import ToolSource

    cached = FileProcessResult(
        text="[cached]",
        image_blocks=(
            {"type": "image_url",
             "image_url": {"url": "https://minio:9000/manus/a.png", "detail": "auto"}},
        ),
        media_type="image/png",
    )
    lookup = _CachingLookup(cached=cached, fresh=cached)
    result = _invoke_cache(lookup, file_view_image_cache_enabled=True)
    passthrough = result.artifact  # Passthrough carrying the cached presigned URL

    class _Kimi:
        provider_id = "kimi"
        accepts_image_url = False
        accepts_image_base64 = True
        image_max_bytes = 5 * 1024 * 1024

    class _Resolver:
        async def load_image_bytes(self, display_url, *, max_bytes):
            return b"PNGBYTES"

    ts = ToolSource(source="native", category="file", canonical_name="file_view")
    _msg, deferred, _ev = asyncio.run(_translate_outcome(
        passthrough, {"id": "c", "name": "file_view", "args": {}, "type": "tool_call"},
        ts, None, tool_result_max_chars=8000, guide_injector=None,
        materialize_enabled=True, image_transport_profile=_Kimi(), image_bytes_resolver=_Resolver(),
    ))
    urls = [b["image_url"]["url"] for b in deferred[0].content if b.get("type") == "image_url"]
    assert urls and urls[0].startswith("data:image/png;base64,")


def test_file_view_cache_miss_on_ctime_change() -> None:
    """PR-3 codex R1#P2: 原地改文件（同 size + os.utime 恢复 mtime_ns）但 ctime_ns 变 → key 变 → 必 miss。

    mtime 可被 os.utime 恢复，**ctime 不可**（utime 反把 ctime 顶到当前时刻）——故 ctime_ns 是
    防同 session stale-hit 的正确判别维度。用真 FileProcessorRegistry（真 key-based cache_get/put）+
    两次 stat 仅 ctime_ns 变，证第二次 file_view 重新 process（非返 stale [first]）。"""
    from app.infrastructure.external.file_processors.registry import FileProcessorRegistry
    from app.domain.services.tools.langchain_tools import _make_file_view_tools

    reg = FileProcessorRegistry(sandbox=AsyncMock(), file_uploader=AsyncMock())
    proc = AsyncMock()
    proc.process = AsyncMock(side_effect=[_img_result("[first]"), _img_result("[second]")])
    reg.get_processor = lambda mime: proc  # type: ignore[assignment]

    ctimes = [100, 999]  # 同 size+mtime_ns，仅 ctime_ns 变
    n = {"i": 0}

    async def _exec(session, exec_dir, command):
        r = MagicMock()
        r.success = True
        if "python3" in command:  # stat
            ct = ctimes[min(n["i"], 1)]
            n["i"] += 1
            r.data = {"returncode": 0,
                      "output": ('{"realpath":"/w/a.png","size":10,"mtime_ns":1,"ctime_ns":'
                                 + str(ct) + ',"dev":2,"ino":3}')}
        else:  # file --mime-type
            r.data = {"returncode": 0, "output": "image/png"}
        r.__str__ = lambda self: "image/png"
        return r

    sandbox = AsyncMock()
    sandbox.exec_command = AsyncMock(side_effect=_exec)
    sandbox.generation = 1

    tools = _make_file_view_tools(sandbox, reg, supports_vision=True,
                                  file_view_image_cache_enabled=True)

    def _call():
        return asyncio.run(tools[0].ainvoke({
            "id": "c", "name": "file_view", "args": {"filepath": "/w/a.png"}, "type": "tool_call",
        }))

    first = _call()
    second = _call()
    assert first.artifact.content == "[first]"
    assert second.artifact.content == "[second]"   # ctime 变 → miss → 重新 process（非 stale [first]）
    assert proc.process.await_count == 2


# --------------------------------------------------------------------------- #
# B12 P5: file_view document_preview producer gate (Task 5.2)
# --------------------------------------------------------------------------- #


class _PdfLookup:
    def __init__(self):
        from app.domain.models.tool_result import DocumentPreview
        self._result = FileProcessResult(
            text="[PDF: r.pdf, 3 pages]",
            document_blocks=(
                {"type": "file", "file": {"filename": "r.pdf",
                                          "file_data": "data:application/pdf;base64,abc"}},
            ),
            media_type="application/pdf",
            document_preview=DocumentPreview(
                filename="r.pdf", media_type="application/pdf", page_count=3),
        )

    def get_processor(self, mime):
        proc = AsyncMock()
        proc.process = AsyncMock(return_value=self._result)
        return proc


def _invoke_pdf(**flags):
    from app.domain.services.tools.langchain_tools import _make_file_view_tools

    tools = _make_file_view_tools(
        _make_sandbox_mock("application/pdf"), _PdfLookup(),
        supports_vision=True, supports_pdf_input=True, **flags,
    )
    return asyncio.run(tools[0].ainvoke({
        "id": "c", "name": "file_view", "args": {"filepath": "/w/r.pdf"}, "type": "tool_call",
    }))


def test_file_view_document_preview_flag_off() -> None:
    result = _invoke_pdf(document_preview_enabled=False)
    assert result.artifact.data.document_preview is None


def test_file_view_document_preview_flag_on() -> None:
    result = _invoke_pdf(document_preview_enabled=True)
    assert result.artifact.data.document_preview is not None
    assert result.artifact.data.document_preview.filename == "r.pdf"
