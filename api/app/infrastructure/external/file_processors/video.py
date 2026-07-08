"""Video file processor — scene detection + audio transcription."""
from __future__ import annotations

import asyncio
import json
import logging
import shlex
import uuid
from typing import Any, Awaitable, Callable

from app.domain.external.file_processor import FileProcessResult
from app.domain.external.sandbox import SandboxHandle

logger = logging.getLogger(__name__)

FileUploader = Callable[[bytes, str], Awaitable[str | None]]

from app.domain.external.file_processor import MAX_FILE_VIEW_IMAGES as _MAX_FRAME_IMAGES

_MAX_VIDEO_SIZE = 500 * 1024 * 1024  # 500MB
_MAX_TRANSCRIPT_CHARS = 8000
# Max vision model calls per video for frame description (fallback path).
# Failed downloads/oversized frames don't consume this quota.
MAX_VISION_DESCRIBE = 5


def _compute_target_frames(duration_seconds: float, max_keyframes: int) -> int:
    if duration_seconds < 30:
        target = 5
    elif duration_seconds < 300:
        target = 10
    elif duration_seconds < 1800:
        target = 20
    else:
        target = 30
    return min(target, max_keyframes)


def _estimate_frame_timestamp(
    kf_name: str, duration: float, uniform_count: int,
    format_fn: callable,
) -> str:
    """Estimate a frame's timestamp label.

    - Uniform frames (uniform_NNN.jpg): position = N * interval where
      interval = duration / (uniform_count + 1), N is 1-indexed from filename.
    - Scene-detected frames (keyframe_*.jpg): position is unknown, show '?'.
    - If duration is 0: show '?'.
    """
    if duration <= 0:
        return "?"
    if "uniform_" in kf_name and uniform_count > 0:
        # Extract frame number from filename: uniform_001.jpg → 1 (1-indexed)
        import re
        m = re.search(r'uniform_(\d+)', kf_name)
        if m:
            frame_num = int(m.group(1))  # 1-indexed
            interval = duration / (uniform_count + 1)
            return f"~{format_fn(interval * frame_num)}"
    return "?"


class VideoFileProcessor:
    def __init__(
        self,
        sandbox: SandboxHandle,
        file_uploader: FileUploader,
        audio_processor: Any | None,  # AudioFileProcessor or None
        video_config: Any,  # VideoProcessorConfig
        vision_model: Any | None = None,
    ) -> None:
        self._sandbox = sandbox
        self._file_uploader = file_uploader
        self._audio_processor = audio_processor
        self._video_config = video_config
        self._vision_model = vision_model

    async def process(
        self,
        sandbox_path: str,
        filename: str,
        mime_type: str,
        supports_vision: bool,
        supports_pdf_input: bool = False,
    ) -> FileProcessResult:
        # Get video metadata via ffprobe (includes size check)
        meta = await self._get_metadata(sandbox_path)
        if meta is None:
            return FileProcessResult(text=f"[Video: {filename} — unable to read metadata]")

        file_size = meta.get("size", 0)
        if file_size > _MAX_VIDEO_SIZE:
            return FileProcessResult(
                text=f"[Video: {filename}, {file_size} bytes — too large (limit {_MAX_VIDEO_SIZE // (1024*1024)}MB)]"
            )

        duration = meta.get("duration", 0)
        width = meta.get("width", 0)
        height = meta.get("height", 0)
        has_audio = meta.get("has_audio", False)

        header = f"[Video: {filename}, {self._format_duration(duration)}"
        if width and height:
            header += f", {width}x{height}"
        header += "]"

        work_dir = f"/tmp/_fv_{uuid.uuid4().hex[:8]}"

        try:
            # Run audio transcription and keyframe extraction in parallel
            tasks = []

            # Audio transcription (if available and enabled)
            transcript_text = ""
            do_audio = has_audio and self._audio_processor and self._video_config.extract_audio

            if do_audio:
                tasks.append(self._extract_and_transcribe_audio(sandbox_path, work_dir))

            # Keyframe extraction
            target_frames = _compute_target_frames(duration, self._video_config.max_keyframes)
            tasks.append(self._extract_keyframes(sandbox_path, work_dir, target_frames, duration))

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Parse results based on whether audio task was included
            if do_audio:
                audio_result = results[0]
                keyframe_result = results[1] if len(results) > 1 else []
                if isinstance(audio_result, str):
                    transcript_text = audio_result
                elif isinstance(audio_result, Exception):
                    logger.warning("Audio extraction failed: %s", audio_result)
            else:
                keyframe_result = results[0]

            # Process keyframes
            keyframe_files: list[str] = []
            if isinstance(keyframe_result, list):
                keyframe_files = keyframe_result
            elif isinstance(keyframe_result, Exception):
                logger.warning("Keyframe extraction failed: %s", keyframe_result)

            # Build image blocks based on vision capability
            image_blocks: list[dict] = []
            frame_text_parts: list[str] = []

            if keyframe_files:
                if supports_vision:
                    from app.infrastructure.external.llm.message_sanitizer import MAX_IMAGE_BYTES
                    # Aggregate budget for inline data URLs (upload-failure fallback).
                    # Cap at 3 inline frames to prevent payload explosion when storage is down.
                    _MAX_INLINE_FALLBACK = 3
                    inline_count = 0
                    _uc = sum(1 for f in keyframe_files if "uniform_" in f)
                    for i, kf_path in enumerate(keyframe_files):
                        kf_name = kf_path.rsplit("/", 1)[-1] if "/" in kf_path else kf_path
                        approx_ts = _estimate_frame_timestamp(
                            kf_name, duration, _uc, self._format_duration,
                        )

                        # Already at image cap → all remaining frames are text-only
                        if len(image_blocks) >= _MAX_FRAME_IMAGES:
                            frame_text_parts.append(f"[Keyframe {i+1}: {approx_ts}, {kf_name} — image limit reached]")
                            continue

                        try:
                            img_io = await self._sandbox.download_file(kf_path)
                            img_bytes = img_io.read() if hasattr(img_io, "read") else img_io
                            url = None
                            try:
                                url = await self._file_uploader(
                                    img_bytes, f"{filename}_{kf_name}"
                                )
                            except Exception as e:
                                logger.warning("Keyframe upload failed for %s: %s", kf_path, e)
                            if url:
                                image_blocks.append(
                                    {"type": "image_url", "image_url": {"url": url, "detail": "auto"}}
                                )
                                frame_text_parts.append(f"[Keyframe {i+1}: {approx_ts}, {kf_name}]")
                            elif len(img_bytes) <= MAX_IMAGE_BYTES and inline_count < _MAX_INLINE_FALLBACK:
                                import base64 as _b64
                                data_url = f"data:image/jpeg;base64,{_b64.b64encode(img_bytes).decode()}"
                                image_blocks.append(
                                    {"type": "image_url", "image_url": {"url": data_url, "detail": "auto"}}
                                )
                                inline_count += 1
                                frame_text_parts.append(f"[Keyframe {i+1}: {approx_ts}, {kf_name}]")
                            else:
                                frame_text_parts.append(f"[Keyframe {i+1}: {approx_ts}, {kf_name} — image unavailable]")
                        except Exception as e:
                            logger.warning("Failed to process keyframe %s: %s", kf_path, e)
                            frame_text_parts.append(f"[Keyframe {i+1}: {approx_ts}, {kf_name} — image unavailable]")
                elif self._vision_model:
                    # Describe frames with vision model (fallback).
                    # Limit actual vision API calls to 5 (not frame index — failed
                    # downloads/oversized frames don't consume the quota).
                    vision_call_count = 0
                    _uc = sum(1 for f in keyframe_files if "uniform_" in f)
                    for i, kf_path in enumerate(keyframe_files):
                        kf_name = kf_path.rsplit("/", 1)[-1] if "/" in kf_path else kf_path
                        approx_ts = _estimate_frame_timestamp(
                            kf_name, duration, _uc, self._format_duration,
                        )
                        if vision_call_count >= MAX_VISION_DESCRIBE:
                            frame_text_parts.append(f"[Keyframe {i+1}: {approx_ts}, {kf_name} — not described]")
                            continue
                        used_vision, desc = await self._describe_frame(kf_path)
                        frame_text_parts.append(f"[Keyframe {i+1}: {approx_ts}, {kf_name}] {desc}")
                        if used_vision:
                            vision_call_count += 1
                else:
                    # No vision: list keyframes with filename-derived position info.
                    # Scene-detected frames are named keyframe_001.jpg etc.;
                    # we can approximate position from the frame index relative to duration.
                    _uc = sum(1 for f in keyframe_files if "uniform_" in f)
                    for i, kf_path in enumerate(keyframe_files):
                        kf_name = kf_path.rsplit("/", 1)[-1] if "/" in kf_path else kf_path
                        ts = _estimate_frame_timestamp(
                            kf_name, duration, _uc, self._format_duration,
                        )
                        if ts != "?":
                            frame_text_parts.append(f"[Keyframe {i + 1}: {ts}, {kf_name}]")
                        else:
                            frame_text_parts.append(f"[Keyframe {i + 1}: {kf_name}]")

            # Assemble output
            parts = [header]
            if transcript_text:
                parts.append(f"\nTranscript:\n{transcript_text}")
            if frame_text_parts:
                parts.append("\nKeyframes:\n" + "\n".join(frame_text_parts))
            elif keyframe_files:
                parts.append(f"\nKeyframes: {len(keyframe_files)} frames extracted")

            full_text = "\n".join(parts)
            if len(full_text) > _MAX_TRANSCRIPT_CHARS:
                half = _MAX_TRANSCRIPT_CHARS // 2
                full_text = full_text[:half] + "\n...(已截断)\n" + full_text[-half:]

            return FileProcessResult(
                text=full_text,
                image_blocks=tuple(image_blocks),
                # B12 PR-1 producer-contract: keyframe blocks carry the SOURCE video
                # mime (mirrors pdf.py page-image blocks → media_type='application/pdf').
                # No keyframe blocks (no-vision text-only path) → None, like image.py/
                # pdf.py text-only returns. Other 3 returns are text-only/error → None.
                media_type=mime_type if image_blocks else None,
            )

        except Exception as e:
            return FileProcessResult(text=f"[Video: {filename} — processing error: {e}]")
        finally:
            try:
                await self._sandbox.exec_command("default", "", f"rm -rf {shlex.quote(work_dir)}")
            except Exception:
                pass

    async def _get_metadata(self, sandbox_path: str) -> dict | None:
        try:
            cmd = (
                f"ffprobe -v quiet -print_format json "
                f"-show_format -show_streams {shlex.quote(sandbox_path)}"
            )
            result = await asyncio.wait_for(
                self._sandbox.exec_command("default", "", cmd),
                timeout=15.0,
            )
            if hasattr(result, "data") and isinstance(result.data, dict):
                if result.data.get("returncode", -1) != 0:
                    return None
                data = json.loads(result.data.get("output", "{}"))
            else:
                data = json.loads(str(result))

            duration = float(data.get("format", {}).get("duration", 0))
            streams = data.get("streams", [])
            video_stream = next(
                (s for s in streams if s.get("codec_type") == "video"), None
            )
            has_audio = any(s.get("codec_type") == "audio" for s in streams)

            file_size = int(data.get("format", {}).get("size", 0))

            return {
                "duration": duration,
                "width": int(video_stream.get("width", 0)) if video_stream else 0,
                "height": int(video_stream.get("height", 0)) if video_stream else 0,
                "has_audio": has_audio,
                "size": file_size,
            }
        except Exception as e:
            logger.warning("ffprobe metadata extraction failed: %s", e)
            return None

    async def _extract_keyframes(
        self, sandbox_path: str, work_dir: str, target: int, duration: float = 0
    ) -> list[str]:
        await self._sandbox.exec_command("default", "", f"mkdir -p {shlex.quote(work_dir)}")

        strategy = getattr(self._video_config, "frame_strategy", "scene")
        threshold = getattr(self._video_config, "scene_threshold", 0.3)

        files: list[str] = []

        if strategy == "scene":
            # Phase 1: scene detection
            scene_cmd = (
                f"ffmpeg -i {shlex.quote(sandbox_path)} "
                f"-vf \"select='gt(scene,{threshold})',scale='min(1280,iw):-2'\" "
                f"-vsync vfr -frames:v {target} "
                f"{shlex.quote(work_dir)}/keyframe_%03d.jpg 2>/dev/null"
            )
            scene_result = await asyncio.wait_for(
                self._sandbox.exec_command("default", "", scene_cmd),
                timeout=60.0,
            )
            if self._check_rc(scene_result, "ffmpeg scene detection"):
                files = await self._list_keyframes(work_dir, "keyframe_")
            else:
                logger.warning("Scene detection failed, falling back to uniform sampling")

            # Phase 2: if scene detection produced too few frames, supplement
            # with uniform sampling (scene frames first, then uniform to fill remaining).
            # Ceil half: (target+1)//2 so odd targets (e.g. 3→2) don't under-trigger.
            if len(files) < max(1, (target + 1) // 2):
                remaining = target - len(files)
                interval = max(1, int(duration / (remaining + 1))) if duration > 0 else 5
                uniform_cmd = (
                    f"ffmpeg -i {shlex.quote(sandbox_path)} "
                    f"-vf \"fps=1/{interval},scale='min(1280,iw):-2'\" "
                    f"-frames:v {remaining} "
                    f"{shlex.quote(work_dir)}/uniform_%03d.jpg 2>/dev/null"
                )
                uniform_result = await asyncio.wait_for(
                    self._sandbox.exec_command("default", "", uniform_cmd),
                    timeout=60.0,
                )
                if self._check_rc(uniform_result, "ffmpeg uniform sampling"):
                    uniform_files = await self._list_keyframes(work_dir, "uniform_")
                    # Supplement: scene frames first (content-aware), then uniform
                    files.extend(uniform_files)
                else:
                    logger.warning("Uniform keyframe extraction also failed")
        else:
            # Uniform only: adaptive interval based on duration and target.
            # Use uniform_ prefix so _estimate_frame_timestamp recognizes them.
            interval = max(1, int(duration / (target + 1))) if duration > 0 else 5
            uniform_cmd = (
                f"ffmpeg -i {shlex.quote(sandbox_path)} "
                f"-vf \"fps=1/{interval},scale='min(1280,iw):-2'\" "
                f"-frames:v {target} "
                f"{shlex.quote(work_dir)}/uniform_%03d.jpg 2>/dev/null"
            )
            uniform_result = await asyncio.wait_for(
                self._sandbox.exec_command("default", "", uniform_cmd),
                timeout=60.0,
            )
            if self._check_rc(uniform_result, "ffmpeg uniform extraction"):
                files = await self._list_keyframes(work_dir, "uniform_")
            else:
                logger.warning("Keyframe extraction failed")

        return files[:target]

    @staticmethod
    def _check_rc(result: object, label: str) -> bool:
        """Return True if sandbox exec_command succeeded (returncode 0)."""
        if hasattr(result, "data") and isinstance(result.data, dict):
            rc = result.data.get("returncode", -1)
            if rc != 0:
                output = (result.data.get("output") or "")[:200]
                logger.warning("%s failed (rc=%d): %s", label, rc, output)
                return False
            return True
        return False  # Unexpected format → treat as failure

    async def _list_keyframes(self, work_dir: str, prefix: str) -> list[str]:
        ls_result = await self._sandbox.exec_command(
            "default", "", f"ls -1 {shlex.quote(work_dir)}/{prefix}*.jpg 2>/dev/null"
        )
        files: list[str] = []
        if hasattr(ls_result, "data") and isinstance(ls_result.data, dict):
            output = ls_result.data.get("output") or ""
            files = [f.strip() for f in output.strip().split("\n") if f.strip()]
        return sorted(files)

    async def _extract_and_transcribe_audio(
        self, sandbox_path: str, work_dir: str
    ) -> str:
        await self._sandbox.exec_command("default", "", f"mkdir -p {shlex.quote(work_dir)}")
        audio_path = f"{work_dir}/audio.wav"

        # Extract audio track
        cmd = (
            f"ffmpeg -i {shlex.quote(sandbox_path)} "
            f"-vn -acodec pcm_s16le -ar 16000 -ac 1 "
            f"{shlex.quote(audio_path)} 2>/dev/null"
        )
        extract_result = await asyncio.wait_for(
            self._sandbox.exec_command("default", "", cmd),
            timeout=60.0,
        )

        if not self._check_rc(extract_result, "ffmpeg audio extraction"):
            return "[Audio extraction failed]"

        # Use audio processor for transcription
        result = await self._audio_processor.process(
            audio_path, "audio.wav", "audio/wav",
            supports_vision=False,
        )
        return result.text

    async def _describe_frame(self, frame_path: str) -> tuple[bool, str]:
        """Describe a video frame using the vision model.

        Returns (used_vision, text):
          - (False, error_text) if frame couldn't be loaded or was too large
          - (True, description) if vision model was called (even if it failed)
        """
        import base64

        from langchain_core.messages import HumanMessage
        from app.infrastructure.external.llm.message_sanitizer import MAX_IMAGE_BYTES

        try:
            img_io = await self._sandbox.download_file(frame_path)
            img_bytes = img_io.read() if hasattr(img_io, "read") else img_io
        except Exception as e:
            return False, f"[Frame unavailable: {e}]"

        if len(img_bytes) > MAX_IMAGE_BYTES:
            return False, "[Frame too large for vision description]"

        b64 = base64.b64encode(img_bytes).decode()
        data_url = f"data:image/jpeg;base64,{b64}"

        msg = HumanMessage(content=[
            {
                "type": "text",
                "text": "Describe this video frame briefly. Focus on key visual elements.",
            },
            {"type": "image_url", "image_url": {"url": data_url, "detail": "auto"}},
        ])
        try:
            response = await asyncio.wait_for(
                self._vision_model.ainvoke([msg]), timeout=30.0
            )
            text = (
                response.content
                if isinstance(response.content, str)
                else str(response.content)
            )
            return True, text
        except Exception as e:
            return True, f"[Frame description failed: {e}]"

    @staticmethod
    def _format_duration(seconds: float) -> str:
        s = int(seconds)
        mins, secs = divmod(s, 60)
        hrs, mins = divmod(mins, 60)
        if hrs > 0:
            return f"{hrs}:{mins:02d}:{secs:02d}"
        return f"{mins:02d}:{secs:02d}"
