"""PDF file processor — dual-path: native PDF block or pymupdf4llm extraction."""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import shlex
import uuid
from typing import Awaitable, Callable

from app.domain.external.file_processor import FileProcessResult
from app.domain.external.sandbox import SandboxHandle

logger = logging.getLogger(__name__)

FileUploader = Callable[[bytes, str], Awaitable[str | None]]

from app.infrastructure.external.llm.message_sanitizer import MAX_PDF_BYTES

_MAX_NATIVE_SIZE = MAX_PDF_BYTES  # Aligned with sanitizer limit
_MAX_EXTRACT_SIZE = 100 * 1024 * 1024
_MAX_PAGES_PER_CALL = 100
_MAX_TEXT_CHARS = 8000
_MAX_PAGE_IMAGES = 10

# Markdown patterns for complex page detection
_MD_IMAGE_PATTERN = re.compile(r'!\[.*?\]\(.*?\)')
_MD_TABLE_SEPARATOR = re.compile(r'^\|[\s:]*-{3,}[\s:]*\|', re.MULTILINE)
_FENCED_CODE_BLOCK = re.compile(r'```[\s\S]*?```', re.MULTILINE)

_PDF_EXTRACT_SCRIPT = '''
import pymupdf4llm, json, sys, os
work_dir = sys.argv[2]
os.makedirs(work_dir, exist_ok=True)
image_path = os.path.join(work_dir, "images")
os.makedirs(image_path, exist_ok=True)
result = pymupdf4llm.to_markdown(sys.argv[1], page_chunks=True, write_images=True, image_path=image_path)
print(json.dumps(result, ensure_ascii=False))
'''.strip()

_PDF_RENDER_SCRIPT = '''
import fitz, sys, json, os
work_dir = sys.argv[3]
os.makedirs(work_dir, exist_ok=True)
doc = fitz.open(sys.argv[1])
pages = json.loads(sys.argv[2])
for i in pages:
    page = doc[i]
    pix = page.get_pixmap(dpi=150)
    pix.save(os.path.join(work_dir, f"page_{i}.jpg"))
'''.strip()


def _is_complex_page(page_markdown: str) -> bool:
    """Determine whether a page's markdown content is complex (image/table-heavy).

    A page is complex if:
    - Very little text (<200 chars) AND has images or tables → visually dominated
    - Many image references (>=3) at any text length → chart/figure heavy

    Note: tables in long text pages are NOT flagged as complex — pymupdf4llm's
    Markdown table output is usually readable enough, and rendering long text+table
    pages as images wastes upload bandwidth and context tokens.
    """
    text = page_markdown.strip()
    text_no_code = _FENCED_CODE_BLOCK.sub("", text)
    image_refs = len(_MD_IMAGE_PATTERN.findall(text_no_code))
    has_table = bool(_MD_TABLE_SEPARATOR.search(text_no_code))

    # Short page with any visual element → visually dominated
    if len(text) < 200 and (image_refs > 0 or has_table):
        return True
    # Many images at any length → chart/figure dense
    if image_refs >= 3:
        return True
    return False


class PdfFileProcessor:
    def __init__(self, sandbox: SandboxHandle, file_uploader: FileUploader) -> None:
        self._sandbox = sandbox
        self._file_uploader = file_uploader

    async def process(
        self,
        sandbox_path: str,
        filename: str,
        mime_type: str,
        supports_vision: bool,
        supports_pdf_input: bool = False,
    ) -> FileProcessResult:
        file_io = await self._sandbox.download_file(sandbox_path)
        file_bytes = file_io.read() if hasattr(file_io, "read") else file_io

        if not file_bytes[:5] == b"%PDF-":
            return FileProcessResult(
                text=f"[PDF: {filename} — not a valid PDF file (invalid header)]"
            )

        if supports_pdf_input:
            result = await self._native_path(sandbox_path, file_bytes, filename)
            if result.document_blocks:
                return result
            # Native path couldn't produce document_blocks (oversized, too many pages,
            # page count failed) → fall back to extraction path instead of hard failure
            logger.info("Native PDF path unavailable for %s, falling back to extraction", filename)
        return await self._extraction_path(sandbox_path, filename, file_bytes, supports_vision)

    async def _count_pages_via_sandbox(self, sandbox_path: str) -> int | None:
        try:
            # Path passed as shell argument (shlex.quote), read via sys.argv in Python.
            # Using shlex.quote inside a Python string literal would produce invalid
            # Python for simple paths (e.g. fitz.open(/tmp/test.pdf) — missing quotes).
            result = await asyncio.wait_for(
                self._sandbox.exec_command(
                    "default",
                    "",
                    f"python3 -c 'import fitz,sys; print(len(fitz.open(sys.argv[1])))' {shlex.quote(sandbox_path)}",
                ),
                timeout=10.0,
            )
            if hasattr(result, "data") and isinstance(result.data, dict):
                if result.data.get("returncode", -1) == 0:
                    return int((result.data.get("output") or "").strip())
        except Exception:
            pass
        return None

    async def _native_path(
        self, sandbox_path: str, file_bytes: bytes, filename: str
    ) -> FileProcessResult:
        if len(file_bytes) > _MAX_NATIVE_SIZE:
            return FileProcessResult(
                text=f"[PDF: {filename}, {len(file_bytes)} bytes — too large for native processing]"
            )

        page_count = await self._count_pages_via_sandbox(sandbox_path)
        if page_count is not None and page_count > _MAX_PAGES_PER_CALL:
            return FileProcessResult(
                text=f"[PDF: {filename}, {page_count} pages — exceeds {_MAX_PAGES_PER_CALL} page limit]"
            )
        if page_count is None:
            return FileProcessResult(
                text=f"[PDF: {filename} — unable to determine page count, rejecting native path]"
            )

        b64 = base64.b64encode(file_bytes).decode()
        return FileProcessResult(
            text=f"[PDF: {filename}, {page_count} pages]",
            document_blocks=(
                {
                    "type": "file",
                    "file": {
                        "filename": filename,
                        "file_data": f"data:application/pdf;base64,{b64}",
                    },
                },
            ),
        )

    async def _extraction_path(
        self,
        sandbox_path: str,
        filename: str,
        file_bytes: bytes,
        supports_vision: bool,
    ) -> FileProcessResult:
        if len(file_bytes) > _MAX_EXTRACT_SIZE:
            return FileProcessResult(
                text=f"[PDF: {filename}, {len(file_bytes)} bytes — too large to extract]"
            )

        work_dir = f"/tmp/_fv_{uuid.uuid4().hex[:8]}"
        script_path = "/tmp/_pdf_extract.py"

        try:
            from app.infrastructure.external.file_processors._sandbox_exec import exec_and_wait

            await self._sandbox.write_file(script_path, _PDF_EXTRACT_SCRIPT)
            safe_args = f"{shlex.quote(sandbox_path)} {shlex.quote(work_dir)}"
            cmd = f"python3 {shlex.quote(script_path)} {safe_args}"
            exec_result = await exec_and_wait(self._sandbox, cmd, timeout=120.0)

            if exec_result["returncode"] != 0:
                error = exec_result["output"][:500]
                return FileProcessResult(
                    text=f"[PDF: {filename} — extraction failed (rc={exec_result['returncode']}): {error}]"
                )
            output = exec_result["output"]
            if not output.strip():
                return FileProcessResult(
                    text=f"[PDF: {filename} — extraction returned empty output "
                    f"(status={exec_result.get('status', '?')}, rc={exec_result['returncode']})]"
                )

            # Sandbox stdout may contain warnings before the JSON array.
            # e.g. "onnxruntime cpuid_info warning: ...\n[{...}]"
            json_start = output.find("[")
            if json_start > 0:
                output = output[json_start:]
            pages = json.loads(output)
            if not pages:
                return FileProcessResult(text=f"[PDF: {filename} — empty document]")

            total_pages = len(pages)
            truncated = total_pages > _MAX_PAGES_PER_CALL
            pages = pages[:_MAX_PAGES_PER_CALL]
            header = f"[PDF: {filename}, {total_pages} pages"
            if truncated:
                header += f" — showing first {_MAX_PAGES_PER_CALL}, remaining {total_pages - _MAX_PAGES_PER_CALL} pages omitted"
            header += "]"
            text_parts: list[str] = [header]
            image_blocks: list[dict] = []
            complex_page_indices: list[int] = []

            # Classify pages; defer text assignment for complex pages
            # until we know which ones actually get rendered.
            page_markdowns: dict[int, str] = {}  # complex page index → original markdown
            for i, page in enumerate(pages):
                md = page.get("text", "")
                if _is_complex_page(md) and supports_vision:
                    complex_page_indices.append(i)
                    page_markdowns[i] = md  # Keep markdown for fallback
                    text_parts.append(None)  # placeholder, filled below
                else:
                    text_parts.append(f"\n--- Page {i+1} ---\n{md}")

            # Render complex pages — only the first _MAX_PAGE_IMAGES
            render_pages = complex_page_indices[:_MAX_PAGE_IMAGES]
            rendered_ok: set[int] = set()

            if render_pages:
                render_script_path = "/tmp/_pdf_render.py"
                await self._sandbox.write_file(render_script_path, _PDF_RENDER_SCRIPT)
                pages_json = json.dumps(render_pages)
                render_args = (
                    f"{shlex.quote(sandbox_path)} {shlex.quote(pages_json)} {shlex.quote(work_dir)}"
                )
                render_result = await asyncio.wait_for(
                    self._sandbox.exec_command(
                        "default", "", f"python3 {shlex.quote(render_script_path)} {render_args}"
                    ),
                    timeout=60.0,
                )
                # Check render returncode
                render_ok = (
                    hasattr(render_result, "data")
                    and isinstance(render_result.data, dict)
                    and render_result.data.get("returncode", -1) == 0
                )

                if render_ok:
                    for idx in render_pages:
                        try:
                            img_io = await self._sandbox.download_file(
                                f"{work_dir}/page_{idx}.jpg"
                            )
                            img_bytes = img_io.read() if hasattr(img_io, "read") else img_io
                            url = await self._file_uploader(img_bytes, f"{filename}_p{idx}.jpg")
                            if url:
                                image_blocks.append(
                                    {"type": "image_url", "image_url": {"url": url, "detail": "auto"}}
                                )
                                rendered_ok.add(idx)
                        except Exception as e:
                            logger.warning("Failed to upload page %d image: %s", idx, e)
                else:
                    logger.warning("PDF page rendering failed")

            # Fill in placeholders for complex pages
            for i in complex_page_indices:
                # text_parts index = i + 1 (offset by header at [0])
                tp_idx = i + 1
                if i in rendered_ok:
                    text_parts[tp_idx] = f"[Page {i+1}: rendered as image]"
                elif i in set(render_pages) - rendered_ok:
                    # Was attempted but failed → fall back to extracted text
                    text_parts[tp_idx] = f"\n--- Page {i+1} (render failed, text fallback) ---\n{page_markdowns[i]}"
                else:
                    # Beyond _MAX_PAGE_IMAGES limit → keep extracted text
                    text_parts[tp_idx] = f"\n--- Page {i+1} (complex, text fallback) ---\n{page_markdowns[i]}"

            full_text = "\n".join(text_parts)
            if len(full_text) > _MAX_TEXT_CHARS:
                half = _MAX_TEXT_CHARS // 2
                full_text = (
                    full_text[:half]
                    + f"\n...(已截断，共 {len(full_text)} 字符)\n"
                    + full_text[-half:]
                )

            return FileProcessResult(text=full_text, image_blocks=tuple(image_blocks))

        except asyncio.TimeoutError:
            return FileProcessResult(text=f"[PDF: {filename} — extraction timed out]")
        except json.JSONDecodeError as e:
            return FileProcessResult(
                text=f"[PDF: {filename} — extraction output invalid: {e}]"
            )
        except Exception as e:
            return FileProcessResult(text=f"[PDF: {filename} — extraction error: {e}]")
        finally:
            try:
                await self._sandbox.exec_command(
                    "default", "", f"rm -rf {shlex.quote(work_dir)}"
                )
            except Exception:
                pass
