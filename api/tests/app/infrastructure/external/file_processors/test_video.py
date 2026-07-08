"""Tests for VideoFileProcessor — scene detection, audio transcription, three-tier keyframe."""
from __future__ import annotations

import asyncio
import io
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.infrastructure.external.file_processors.video import MAX_VISION_DESCRIBE

from app.domain.external.file_processor import FileProcessResult
from app.domain.models.app_config import VideoProcessorConfig
from app.infrastructure.external.file_processors.video import (
    VideoFileProcessor,
    _compute_target_frames,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_video_config(**kwargs) -> VideoProcessorConfig:
    defaults = {
        "max_keyframes": 5,
        "extract_audio": True,
        "frame_strategy": "scene",
        "scene_threshold": 0.3,
    }
    defaults.update(kwargs)
    return VideoProcessorConfig(**defaults)


def _make_exec_result(returncode: int = 0, output: str = "") -> MagicMock:
    result = MagicMock()
    result.data = {"returncode": returncode, "output": output}
    return result


def _make_ffprobe_output(
    duration: float = 60.0,
    width: int = 1920,
    height: int = 1080,
    has_audio: bool = True,
    size: int = 1000,
) -> str:
    streams = [
        {"codec_type": "video", "width": width, "height": height},
    ]
    if has_audio:
        streams.append({"codec_type": "audio"})
    return json.dumps({
        "format": {"duration": str(duration), "size": str(size)},
        "streams": streams,
    })


def _make_ls_output(files: list[str]) -> str:
    return "\n".join(files) + "\n" if files else ""


def _make_sandbox(
    ffprobe_output: str | None = None,
    ls_output: str = "",
    returncode: int = 0,
    file_content: bytes = b"fake_image_data",
) -> AsyncMock:
    """Create a command-aware mock sandbox.

    Inspects the command string to decide which result to return,
    so test doesn't break when call order changes (e.g. scene + uniform fallback).
    """
    sandbox = AsyncMock()
    _ffprobe_out = ffprobe_output or _make_ffprobe_output()

    async def _exec_command(session_id: str, cmd_id: str, command: str):
        if "ffprobe" in command:
            return _make_exec_result(returncode, _ffprobe_out)
        if command.startswith("mkdir"):
            return _make_exec_result(0, "")
        if "ffmpeg" in command:
            return _make_exec_result(0, "")
        if command.startswith("ls"):
            # Only return ls_output for the prefix that matches the file names
            if "uniform_" in command and ls_output and "uniform_" not in ls_output:
                return _make_exec_result(0, "")
            return _make_exec_result(0, ls_output)
        if "rm -rf" in command:
            return _make_exec_result(0, "")
        return _make_exec_result(0, "")

    sandbox.exec_command = AsyncMock(side_effect=_exec_command)
    sandbox.download_file = AsyncMock(return_value=io.BytesIO(file_content))
    return sandbox


def _make_processor(
    sandbox: AsyncMock | None = None,
    file_uploader: AsyncMock | None = None,
    audio_processor: AsyncMock | None = None,
    video_config: VideoProcessorConfig | None = None,
    vision_model: AsyncMock | None = None,
) -> VideoFileProcessor:
    return VideoFileProcessor(
        sandbox=sandbox or _make_sandbox(),
        file_uploader=file_uploader or AsyncMock(return_value="https://minio.example.com/frame.jpg"),
        audio_processor=audio_processor,
        video_config=video_config or _make_video_config(),
        vision_model=vision_model,
    )


# ---------------------------------------------------------------------------
# Tests: _compute_target_frames
# ---------------------------------------------------------------------------


class TestComputeTargetFrames:
    def test_short_video_under_30s_returns_5(self):
        assert _compute_target_frames(20.0, 10) == 5

    def test_short_video_at_boundary_29s(self):
        assert _compute_target_frames(29.9, 10) == 5

    def test_medium_video_30_to_300s_returns_10(self):
        assert _compute_target_frames(30.0, 20) == 10
        assert _compute_target_frames(299.9, 20) == 10

    def test_long_video_300_to_1800s_returns_20(self):
        assert _compute_target_frames(300.0, 30) == 20
        assert _compute_target_frames(1799.9, 30) == 20

    def test_very_long_video_over_1800s_returns_30(self):
        assert _compute_target_frames(1800.0, 50) == 30

    def test_max_keyframes_caps_result(self):
        # max_keyframes=3 caps the target for a 30s video (target=10, but capped to 3)
        assert _compute_target_frames(30.0, 3) == 3

    def test_max_keyframes_not_exceeded_when_target_is_lower(self):
        # target=5 for <30s, max=10 → returns 5
        assert _compute_target_frames(10.0, 10) == 5


# ---------------------------------------------------------------------------
# Tests: VideoProcessorConfig validation
# ---------------------------------------------------------------------------


class TestVideoProcessorConfigValidation:
    def test_max_keyframes_zero_rejected(self):
        with pytest.raises(Exception):  # Pydantic ValidationError
            _make_video_config(max_keyframes=0)

    def test_max_keyframes_negative_rejected(self):
        with pytest.raises(Exception):
            _make_video_config(max_keyframes=-1)

    def test_frame_strategy_invalid_rejected(self):
        with pytest.raises(Exception):
            _make_video_config(frame_strategy="sceen")

    def test_scene_threshold_below_zero_rejected(self):
        with pytest.raises(Exception):
            _make_video_config(scene_threshold=-0.1)

    def test_scene_threshold_above_one_rejected(self):
        with pytest.raises(Exception):
            _make_video_config(scene_threshold=1.1)

    def test_valid_config_accepted(self):
        cfg = _make_video_config(max_keyframes=1, frame_strategy="uniform", scene_threshold=0.0)
        assert cfg.max_keyframes == 1


# ---------------------------------------------------------------------------
# Tests: frame_strategy="uniform" explicit path
# ---------------------------------------------------------------------------


class TestUniformStrategyExplicit:
    def test_uniform_strategy_skips_scene_detection_and_has_timestamps(self):
        """frame_strategy='uniform' should use uniform_ prefix and produce timestamps."""
        call_log = []
        # Use uniform_ prefix (matches what video.py now produces for uniform-only)
        uf_files = [f"/tmp/_fv_test/uniform_{i:03d}.jpg" for i in range(1, 4)]

        async def _exec(session_id, cmd_id, command):
            call_log.append(command)
            if "ffprobe" in command:
                return _make_exec_result(0, _make_ffprobe_output(duration=60.0))
            if command.startswith("mkdir"):
                return _make_exec_result(0, "")
            if "ffmpeg" in command:
                return _make_exec_result(0, "")
            if command.startswith("ls"):
                return _make_exec_result(0, "\n".join(uf_files) + "\n")
            if "rm -rf" in command:
                return _make_exec_result(0, "")
            return _make_exec_result(0, "")

        sandbox = AsyncMock()
        sandbox.exec_command = AsyncMock(side_effect=_exec)
        sandbox.download_file = AsyncMock(return_value=io.BytesIO(b"fake"))

        proc = _make_processor(
            sandbox=sandbox,
            video_config=_make_video_config(frame_strategy="uniform"),
        )
        result = asyncio.run(
            proc.process("/tmp/v.mp4", "v.mp4", "video/mp4", supports_vision=True)
        )

        assert len(result.image_blocks) == 3
        assert not any("select=" in cmd for cmd in call_log)
        assert any("fps=" in cmd for cmd in call_log)
        # Uniform frames should have ~timestamps (not "?")
        assert "~" in result.text
        assert "uniform_001.jpg" in result.text

    def test_uniform_strategy_duration_zero_uses_default_interval(self):
        """frame_strategy='uniform' + duration=0 → uses fps=1/5 default, no timestamps."""
        call_log = []
        kf_file = "/tmp/_fv_test/uniform_001.jpg"

        async def _exec(session_id, cmd_id, command):
            call_log.append(command)
            if "ffprobe" in command:
                return _make_exec_result(0, _make_ffprobe_output(duration=0.0))
            if command.startswith("mkdir"):
                return _make_exec_result(0, "")
            if "ffmpeg" in command:
                return _make_exec_result(0, "")
            if command.startswith("ls"):
                return _make_exec_result(0, kf_file + "\n")
            if "rm -rf" in command:
                return _make_exec_result(0, "")
            return _make_exec_result(0, "")

        sandbox = AsyncMock()
        sandbox.exec_command = AsyncMock(side_effect=_exec)
        sandbox.download_file = AsyncMock(return_value=io.BytesIO(b"fake"))

        proc = _make_processor(
            sandbox=sandbox,
            video_config=_make_video_config(frame_strategy="uniform"),
        )
        result = asyncio.run(
            proc.process("/tmp/v.mp4", "v.mp4", "video/mp4", supports_vision=True)
        )

        assert len(result.image_blocks) == 1
        assert any("fps=1/5" in cmd for cmd in call_log)


# ---------------------------------------------------------------------------
# Tests: _extract_keyframes — scene fallback edge cases
# ---------------------------------------------------------------------------


class TestSceneFallbackEdgeCases:
    def test_max_keyframes_1_scene_empty_falls_back_to_uniform(self):
        """target=1, scene detection returns 0 frames → should fall back to uniform."""
        # Scene ls returns empty, uniform ls returns 1 file
        call_log = []
        uniform_file = "/tmp/_fv_test/uniform_001.jpg"

        async def _exec(session_id, cmd_id, command):
            call_log.append(command)
            if "ffprobe" in command:
                return _make_exec_result(0, _make_ffprobe_output(duration=10.0))
            if command.startswith("mkdir"):
                return _make_exec_result(0, "")
            if "ffmpeg" in command and "select=" in command:
                return _make_exec_result(0, "")  # scene detection ok but 0 frames
            if "ffmpeg" in command and "fps=" in command:
                return _make_exec_result(0, "")  # uniform ok
            if command.startswith("ls") and "uniform_" in command:
                return _make_exec_result(0, uniform_file + "\n")
            if command.startswith("ls"):
                return _make_exec_result(0, "")  # scene: empty
            if "rm -rf" in command:
                return _make_exec_result(0, "")
            return _make_exec_result(0, "")

        sandbox = AsyncMock()
        sandbox.exec_command = AsyncMock(side_effect=_exec)
        sandbox.download_file = AsyncMock(return_value=io.BytesIO(b"fake"))

        proc = _make_processor(
            sandbox=sandbox,
            video_config=_make_video_config(max_keyframes=1),
        )
        result = asyncio.run(
            proc.process("/tmp/v.mp4", "v.mp4", "video/mp4", supports_vision=True)
        )

        # Should have 1 image from uniform fallback
        assert len(result.image_blocks) == 1
        # Verify uniform ffmpeg was actually called
        assert any("fps=" in cmd for cmd in call_log)

    def test_duration_zero_scene_empty_falls_back_to_uniform(self):
        """duration=0 (unknown), scene returns 0 frames → uniform with default interval."""
        call_log = []
        uniform_file = "/tmp/_fv_test/uniform_001.jpg"

        async def _exec(session_id, cmd_id, command):
            call_log.append(command)
            if "ffprobe" in command:
                return _make_exec_result(0, _make_ffprobe_output(duration=0.0))
            if command.startswith("mkdir"):
                return _make_exec_result(0, "")
            if "ffmpeg" in command and "select=" in command:
                return _make_exec_result(0, "")
            if "ffmpeg" in command and "fps=" in command:
                return _make_exec_result(0, "")
            if command.startswith("ls") and "uniform_" in command:
                return _make_exec_result(0, uniform_file + "\n")
            if command.startswith("ls"):
                return _make_exec_result(0, "")
            if "rm -rf" in command:
                return _make_exec_result(0, "")
            return _make_exec_result(0, "")

        sandbox = AsyncMock()
        sandbox.exec_command = AsyncMock(side_effect=_exec)
        sandbox.download_file = AsyncMock(return_value=io.BytesIO(b"fake"))

        proc = _make_processor(sandbox=sandbox)
        result = asyncio.run(
            proc.process("/tmp/v.mp4", "v.mp4", "video/mp4", supports_vision=True)
        )

        assert len(result.image_blocks) == 1
        # Verify uniform was called with default interval (fps=1/5)
        assert any("fps=1/5" in cmd for cmd in call_log)

    def test_odd_target_scene_1_frame_falls_back_to_uniform(self):
        """target=3 (ceil half=2), scene returns 1 frame → should fall back."""
        call_log = []
        scene_file = "/tmp/_fv_test/keyframe_001.jpg"
        uniform_files = [f"/tmp/_fv_test/uniform_{i:03d}.jpg" for i in range(1, 4)]

        async def _exec(session_id, cmd_id, command):
            call_log.append(command)
            if "ffprobe" in command:
                return _make_exec_result(0, _make_ffprobe_output(duration=20.0))
            if command.startswith("mkdir"):
                return _make_exec_result(0, "")
            if "ffmpeg" in command and "select=" in command:
                return _make_exec_result(0, "")
            if "ffmpeg" in command and "fps=" in command:
                return _make_exec_result(0, "")
            if command.startswith("ls") and "uniform_" in command:
                return _make_exec_result(0, "\n".join(uniform_files) + "\n")
            if command.startswith("ls") and "keyframe_" in command:
                return _make_exec_result(0, scene_file + "\n")  # only 1 scene frame
            if "rm -rf" in command:
                return _make_exec_result(0, "")
            return _make_exec_result(0, "")

        sandbox = AsyncMock()
        sandbox.exec_command = AsyncMock(side_effect=_exec)
        sandbox.download_file = AsyncMock(return_value=io.BytesIO(b"fake"))

        proc = _make_processor(
            sandbox=sandbox,
            video_config=_make_video_config(max_keyframes=3),
        )
        result = asyncio.run(
            proc.process("/tmp/v.mp4", "v.mp4", "video/mp4", supports_vision=True)
        )

        # 1 scene frame < ceil(3/2)=2 → supplement with 2 uniform frames (total 3)
        # Scene frame kept (content-aware), uniform frames appended
        assert len(result.image_blocks) == 3
        assert any("fps=" in cmd for cmd in call_log)
        # Scene keyframe preserved + uniform supplement added
        assert "keyframe_001.jpg" in result.text
        assert "uniform_001.jpg" in result.text

    def test_uniform_success_but_empty_keeps_scene_frames(self):
        """Uniform ffmpeg succeeds but produces 0 files → keep existing scene frames."""
        scene_file = "/tmp/_fv_test/keyframe_001.jpg"

        async def _exec(session_id, cmd_id, command):
            if "ffprobe" in command:
                return _make_exec_result(0, _make_ffprobe_output(duration=20.0))
            if command.startswith("mkdir"):
                return _make_exec_result(0, "")
            if "ffmpeg" in command:
                return _make_exec_result(0, "")  # both scene and uniform succeed
            if command.startswith("ls") and "uniform_" in command:
                return _make_exec_result(0, "")  # uniform produced nothing
            if command.startswith("ls") and "keyframe_" in command:
                return _make_exec_result(0, scene_file + "\n")  # scene got 1
            if "rm -rf" in command:
                return _make_exec_result(0, "")
            return _make_exec_result(0, "")

        sandbox = AsyncMock()
        sandbox.exec_command = AsyncMock(side_effect=_exec)
        sandbox.download_file = AsyncMock(return_value=io.BytesIO(b"fake"))

        proc = _make_processor(
            sandbox=sandbox,
            video_config=_make_video_config(max_keyframes=3),
        )
        result = asyncio.run(
            proc.process("/tmp/v.mp4", "v.mp4", "video/mp4", supports_vision=True)
        )

        # Uniform was empty → should keep the 1 scene frame, not replace with []
        assert len(result.image_blocks) == 1
        assert "keyframe_001.jpg" in result.text

    def test_uniform_supplements_scene_frames(self):
        """Scene has 2 frames (insufficient), uniform adds 1 → total 3 (scene + uniform)."""
        scene_files = ["/tmp/_fv_test/keyframe_001.jpg", "/tmp/_fv_test/keyframe_002.jpg"]
        uniform_file = "/tmp/_fv_test/uniform_001.jpg"

        async def _exec(session_id, cmd_id, command):
            if "ffprobe" in command:
                return _make_exec_result(0, _make_ffprobe_output(duration=20.0))
            if command.startswith("mkdir"):
                return _make_exec_result(0, "")
            if "ffmpeg" in command:
                return _make_exec_result(0, "")
            if command.startswith("ls") and "uniform_" in command:
                return _make_exec_result(0, uniform_file + "\n")
            if command.startswith("ls") and "keyframe_" in command:
                return _make_exec_result(0, "\n".join(scene_files) + "\n")
            if "rm -rf" in command:
                return _make_exec_result(0, "")
            return _make_exec_result(0, "")

        sandbox = AsyncMock()
        sandbox.exec_command = AsyncMock(side_effect=_exec)
        sandbox.download_file = AsyncMock(return_value=io.BytesIO(b"fake"))

        proc = _make_processor(
            sandbox=sandbox,
            video_config=_make_video_config(max_keyframes=5),
        )
        result = asyncio.run(
            proc.process("/tmp/v.mp4", "v.mp4", "video/mp4", supports_vision=True)
        )

        # Scene frames kept + uniform supplement = 2 + 1 = 3
        assert len(result.image_blocks) == 3
        assert "keyframe_001.jpg" in result.text
        assert "keyframe_002.jpg" in result.text
        assert "uniform_001.jpg" in result.text


# ---------------------------------------------------------------------------
# Tests: process — with vision
# ---------------------------------------------------------------------------


class TestVideoFileProcessorWithVision:
    def test_process_vision_true_returns_image_blocks(self):
        # Provide >= target//2 scene keyframes to avoid uniform fallback.
        # For 60s video, target=10, threshold=5.
        kf_files = [f"/tmp/_fv_test/keyframe_{i:03d}.jpg" for i in range(1, 6)]
        sandbox = _make_sandbox(ls_output=_make_ls_output(kf_files))
        uploader = AsyncMock(return_value="https://minio.example.com/frame.jpg")
        proc = _make_processor(sandbox=sandbox, file_uploader=uploader)

        result = asyncio.run(
            proc.process("/tmp/video.mp4", "video.mp4", "video/mp4", supports_vision=True)
        )

        assert isinstance(result, FileProcessResult)
        assert "video.mp4" in result.text
        assert len(result.image_blocks) == 5
        assert result.image_blocks[0]["type"] == "image_url"

    def test_process_vision_keyframes_set_media_type(self):
        """B12 PR-1: vision keyframe passthrough carries the SOURCE video media_type.

        Mirrors test_process_vision_true_returns_image_blocks; adds the media_type
        producer-contract assertion. Consistent with image.py (media_type=mime_type)
        and pdf.py (media_type='application/pdf') — the SOURCE file mime, not the
        block image mime. RED before the fix: the :242 success return omits
        media_type, so it defaults to None.
        """
        kf_files = [f"/tmp/_fv_test/keyframe_{i:03d}.jpg" for i in range(1, 6)]
        sandbox = _make_sandbox(ls_output=_make_ls_output(kf_files))
        uploader = AsyncMock(return_value="https://minio.example.com/frame.jpg")
        proc = _make_processor(sandbox=sandbox, file_uploader=uploader)

        result = asyncio.run(
            proc.process("/tmp/video.mp4", "video.mp4", "video/mp4", supports_vision=True)
        )

        # Keyframe blocks produced → media_type carries the source video mime.
        assert len(result.image_blocks) > 0
        assert result.media_type == "video/mp4"

    def test_process_vision_true_includes_duration_and_resolution(self):
        sandbox = _make_sandbox(
            ffprobe_output=_make_ffprobe_output(duration=125.0, width=1280, height=720),
            ls_output="",
        )
        proc = _make_processor(sandbox=sandbox)

        result = asyncio.run(
            proc.process("/tmp/video.mp4", "video.mp4", "video/mp4", supports_vision=True)
        )

        assert "02:05" in result.text
        assert "1280x720" in result.text

    def test_process_vision_multiple_keyframes_uploaded(self):
        files = [f"/tmp/_fv_test/keyframe_{i:03d}.jpg" for i in range(1, 4)]
        sandbox = _make_sandbox(ls_output=_make_ls_output(files))
        uploader = AsyncMock(return_value="https://minio.example.com/frame.jpg")
        proc = _make_processor(sandbox=sandbox, file_uploader=uploader)

        result = asyncio.run(
            proc.process("/tmp/video.mp4", "video.mp4", "video/mp4", supports_vision=True)
        )

        # 3 keyframes → 3 image blocks
        assert len(result.image_blocks) == 3

    def test_process_vision_uploader_failure_falls_back_to_data_url(self):
        """Upload fails → small keyframes should fall back to inline data URL."""
        kf_files = [f"/tmp/_fv_test/keyframe_{i:03d}.jpg" for i in range(1, 6)]
        sandbox = _make_sandbox(ls_output=_make_ls_output(kf_files))
        uploader = AsyncMock(return_value=None)  # uploader returns None
        proc = _make_processor(sandbox=sandbox, file_uploader=uploader)

        result = asyncio.run(
            proc.process("/tmp/video.mp4", "video.mp4", "video/mp4", supports_vision=True)
        )

        # Small test frames → first 3 get data URL (inline budget cap), rest text fallback
        assert len(result.image_blocks) == 3
        assert all(
            b["image_url"]["url"].startswith("data:image/jpeg;base64,")
            for b in result.image_blocks
        )
        # Remaining 2 frames should appear as text fallback
        assert "image unavailable" in result.text

    def test_process_vision_download_failure_shows_unavailable(self):
        """download_file exception → frame shows 'image unavailable', not silently dropped."""
        kf_files = [f"/tmp/_fv_test/keyframe_{i:03d}.jpg" for i in range(1, 6)]
        sandbox = _make_sandbox(ls_output=_make_ls_output(kf_files))
        # Make download_file raise for all frames
        sandbox.download_file = AsyncMock(side_effect=ConnectionError("sandbox down"))
        proc = _make_processor(sandbox=sandbox)

        result = asyncio.run(
            proc.process("/tmp/video.mp4", "video.mp4", "video/mp4", supports_vision=True)
        )

        assert result.image_blocks == ()
        # All 5 frames should appear as "image unavailable" in text
        assert result.text.count("image unavailable") == 5

    def test_process_vision_image_cap_limits_blocks(self):
        """When keyframes exceed _MAX_FRAME_IMAGES, surplus frames get text-only."""
        # Create 12 keyframe files (exceeds _MAX_FRAME_IMAGES=10)
        kf_files = [f"/tmp/_fv_test/keyframe_{i:03d}.jpg" for i in range(1, 13)]
        # Long video (2000s) so _compute_target_frames(2000, 15)=15 → all 12 files used
        sandbox = _make_sandbox(
            ffprobe_output=_make_ffprobe_output(duration=2000.0),
            ls_output=_make_ls_output(kf_files),
        )
        uploader = AsyncMock(return_value="https://minio.example.com/frame.jpg")
        proc = _make_processor(
            sandbox=sandbox,
            file_uploader=uploader,
            video_config=_make_video_config(max_keyframes=15),
        )

        result = asyncio.run(
            proc.process("/tmp/video.mp4", "video.mp4", "video/mp4", supports_vision=True)
        )

        from app.domain.external.file_processor import MAX_FILE_VIEW_IMAGES
        # Should cap at shared limit
        assert len(result.image_blocks) == MAX_FILE_VIEW_IMAGES
        # Remaining frames should show "image limit reached"
        assert result.text.count("image limit reached") == 12 - MAX_FILE_VIEW_IMAGES
        # All 12 frames should have text entries
        assert "Keyframe 1:" in result.text
        assert "Keyframe 12:" in result.text


# ---------------------------------------------------------------------------
# Tests: process — no vision, no fallback
# ---------------------------------------------------------------------------


class TestVideoFileProcessorNoVisionNoFallback:
    def test_process_no_vision_no_model_returns_keyframe_count_text(self):
        files = [f"/tmp/_fv_test/keyframe_{i:03d}.jpg" for i in range(1, 4)]
        sandbox = _make_sandbox(ls_output=_make_ls_output(files))
        proc = _make_processor(sandbox=sandbox, vision_model=None)

        result = asyncio.run(
            proc.process("/tmp/video.mp4", "video.mp4", "video/mp4", supports_vision=False)
        )

        assert result.image_blocks == ()
        assert "Keyframe 1:" in result.text
        assert "keyframe_001.jpg" in result.text
        assert "Keyframe 2:" in result.text
        assert "Keyframe 3:" in result.text

    def test_process_no_vision_keyframes_media_type_is_none(self):
        """B12 PR-1: no image blocks produced (text-only keyframe listing) →
        media_type stays None, consistent with image.py/pdf.py text-only returns.

        Proves the `if image_blocks else None` conditional is non-vacuous: keyframes
        WERE extracted, but no multimodal block was emitted (no-vision path), so
        media_type must remain None rather than asserting the source video mime.
        """
        files = [f"/tmp/_fv_test/keyframe_{i:03d}.jpg" for i in range(1, 4)]
        sandbox = _make_sandbox(ls_output=_make_ls_output(files))
        proc = _make_processor(sandbox=sandbox, vision_model=None)

        result = asyncio.run(
            proc.process("/tmp/video.mp4", "video.mp4", "video/mp4", supports_vision=False)
        )

        assert result.image_blocks == ()
        assert result.media_type is None

    def test_process_no_keyframes_no_frame_section(self):
        sandbox = _make_sandbox(ls_output="")
        proc = _make_processor(sandbox=sandbox, vision_model=None)

        result = asyncio.run(
            proc.process("/tmp/video.mp4", "video.mp4", "video/mp4", supports_vision=False)
        )

        assert result.image_blocks == ()
        assert "Keyframe" not in result.text


# ---------------------------------------------------------------------------
# Tests: process — no vision, with vision model fallback
# ---------------------------------------------------------------------------


class TestVideoFileProcessorVisionModelFallback:
    def test_process_no_vision_with_model_calls_describe_frame(self):
        # Provide enough scene keyframes to avoid uniform fallback
        kf_files = [f"/tmp/_fv_test/keyframe_{i:03d}.jpg" for i in range(1, 6)]
        sandbox = _make_sandbox(ls_output=_make_ls_output(kf_files))

        vision_model = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = "A person walking in a park"
        vision_model.ainvoke = AsyncMock(return_value=mock_response)

        proc = _make_processor(sandbox=sandbox, vision_model=vision_model)

        result = asyncio.run(
            proc.process("/tmp/video.mp4", "video.mp4", "video/mp4", supports_vision=False)
        )

        assert result.image_blocks == ()
        assert "A person walking in a park" in result.text
        assert vision_model.ainvoke.await_count == MAX_VISION_DESCRIBE  # 5 keyframes, all described
        # All 5 should have labels with timestamps
        assert "Keyframe 1:" in result.text
        assert "Keyframe 5:" in result.text

    def test_process_no_vision_with_model_over_5_frames_shows_not_described(self):
        """Vision model fallback describes first 5 frames; rest get text-only labels."""
        kf_files = [f"/tmp/_fv_test/keyframe_{i:03d}.jpg" for i in range(1, 9)]
        # Long video so target allows 8 frames
        sandbox = _make_sandbox(
            ffprobe_output=_make_ffprobe_output(duration=2000.0),
            ls_output=_make_ls_output(kf_files),
        )

        vision_model = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = "Frame description"
        vision_model.ainvoke = AsyncMock(return_value=mock_response)

        proc = _make_processor(
            sandbox=sandbox, vision_model=vision_model,
            video_config=_make_video_config(max_keyframes=10),
        )

        result = asyncio.run(
            proc.process("/tmp/video.mp4", "video.mp4", "video/mp4", supports_vision=False)
        )

        assert result.image_blocks == ()
        # Only first 5 described via vision model
        assert vision_model.ainvoke.await_count == MAX_VISION_DESCRIBE
        # Frames 6-8 should show "not described"
        assert result.text.count("not described") == 3
        # All 8 frames have text entries
        assert "Keyframe 1:" in result.text
        assert "Keyframe 8:" in result.text

    def test_process_no_vision_describe_frame_download_failure(self):
        """download_file raises → 'Frame unavailable' (not 'description failed')."""
        kf_files = [f"/tmp/_fv_test/keyframe_{i:03d}.jpg" for i in range(1, 6)]
        sandbox = _make_sandbox(ls_output=_make_ls_output(kf_files))
        sandbox.download_file = AsyncMock(side_effect=ConnectionError("sandbox down"))

        vision_model = AsyncMock()
        proc = _make_processor(sandbox=sandbox, vision_model=vision_model)

        result = asyncio.run(
            proc.process("/tmp/video.mp4", "video.mp4", "video/mp4", supports_vision=False)
        )

        assert "Frame unavailable" in result.text
        # Vision model should never be called since download fails first
        vision_model.ainvoke.assert_not_awaited()

    def test_process_no_vision_describe_frame_model_failure(self):
        """Vision model raises → 'Frame description failed'."""
        kf_files = [f"/tmp/_fv_test/keyframe_{i:03d}.jpg" for i in range(1, 6)]
        sandbox = _make_sandbox(ls_output=_make_ls_output(kf_files))

        vision_model = AsyncMock()
        vision_model.ainvoke = AsyncMock(side_effect=RuntimeError("model timeout"))
        proc = _make_processor(sandbox=sandbox, vision_model=vision_model)

        result = asyncio.run(
            proc.process("/tmp/video.mp4", "video.mp4", "video/mp4", supports_vision=False)
        )

        assert "Frame description failed" in result.text

    def test_process_no_vision_describe_frame_too_large(self):
        """Frame exceeds MAX_IMAGE_BYTES → 'Frame too large'."""
        kf_files = [f"/tmp/_fv_test/keyframe_{i:03d}.jpg" for i in range(1, 6)]
        big_content = b"\xff" * (6 * 1024 * 1024)  # 6MB > 5MB limit
        sandbox = _make_sandbox(ls_output=_make_ls_output(kf_files))
        # Override download_file to return fresh BytesIO with large content each time
        sandbox.download_file = AsyncMock(side_effect=lambda p: io.BytesIO(big_content))

        vision_model = AsyncMock()
        proc = _make_processor(sandbox=sandbox, vision_model=vision_model)

        result = asyncio.run(
            proc.process("/tmp/video.mp4", "video.mp4", "video/mp4", supports_vision=False)
        )

        assert "Frame too large" in result.text
        vision_model.ainvoke.assert_not_awaited()

    def test_process_no_vision_early_failures_dont_consume_quota(self):
        """First 2 frames fail download; vision model should still describe frames 3-7."""
        # 8 frames total, first 2 fail download, frames 3-7 succeed, frame 8 = not described
        kf_files = [f"/tmp/_fv_test/keyframe_{i:03d}.jpg" for i in range(1, 9)]
        sandbox = _make_sandbox(
            ffprobe_output=_make_ffprobe_output(duration=2000.0),
            ls_output=_make_ls_output(kf_files),
        )

        call_count = [0]
        def _download(path):
            call_count[0] += 1
            if call_count[0] <= 2:
                # First 2 downloads fail
                raise ConnectionError("sandbox flaky")
            return io.BytesIO(b"fake_frame_data")

        sandbox.download_file = AsyncMock(side_effect=_download)

        vision_model = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = "Frame described"
        vision_model.ainvoke = AsyncMock(return_value=mock_response)

        proc = _make_processor(
            sandbox=sandbox, vision_model=vision_model,
            video_config=_make_video_config(max_keyframes=10),
        )

        result = asyncio.run(
            proc.process("/tmp/video.mp4", "video.mp4", "video/mp4", supports_vision=False)
        )

        # Frames 1-2: "Frame unavailable" (download failed, doesn't consume quota)
        assert result.text.count("Frame unavailable") == 2
        # Frames 3-7: described (5 actual vision calls)
        assert vision_model.ainvoke.await_count == MAX_VISION_DESCRIBE
        # Frame 8: "not described" (quota exhausted)
        assert result.text.count("not described") == 1
        # All 8 frames have text entries
        assert "Keyframe 1:" in result.text
        assert "Keyframe 8:" in result.text

    def test_process_no_vision_early_oversized_dont_consume_quota(self):
        """First 2 frames oversized; vision model should still describe frames 3-7."""
        kf_files = [f"/tmp/_fv_test/keyframe_{i:03d}.jpg" for i in range(1, 9)]
        sandbox = _make_sandbox(
            ffprobe_output=_make_ffprobe_output(duration=2000.0),
            ls_output=_make_ls_output(kf_files),
        )

        call_count = [0]
        big_content = b"\xff" * (6 * 1024 * 1024)  # 6MB > 5MB limit
        small_content = b"fake_frame_data"

        def _download(path):
            call_count[0] += 1
            if call_count[0] <= 2:
                return io.BytesIO(big_content)
            return io.BytesIO(small_content)

        sandbox.download_file = AsyncMock(side_effect=_download)

        vision_model = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = "Frame described"
        vision_model.ainvoke = AsyncMock(return_value=mock_response)

        proc = _make_processor(
            sandbox=sandbox, vision_model=vision_model,
            video_config=_make_video_config(max_keyframes=10),
        )

        result = asyncio.run(
            proc.process("/tmp/video.mp4", "video.mp4", "video/mp4", supports_vision=False)
        )

        # Frames 1-2: "Frame too large" (oversized, doesn't consume quota)
        assert result.text.count("Frame too large") == 2
        # Frames 3-7: described (5 actual vision calls)
        assert vision_model.ainvoke.await_count == MAX_VISION_DESCRIBE
        # Frame 8: "not described" (quota exhausted)
        assert result.text.count("not described") == 1

    def test_process_no_vision_model_failure_consumes_quota(self):
        """Model failures count as vision calls (used_vision=True).
        First 2 frames: model raises → 'description failed' (consumes quota).
        Frames 3-5: model succeeds (consumes quota, total = 5).
        Frame 6+: 'not described' (quota exhausted)."""
        kf_files = [f"/tmp/_fv_test/keyframe_{i:03d}.jpg" for i in range(1, 9)]
        sandbox = _make_sandbox(
            ffprobe_output=_make_ffprobe_output(duration=2000.0),
            ls_output=_make_ls_output(kf_files),
        )

        call_count = [0]
        mock_success = MagicMock()
        mock_success.content = "Frame described"

        async def _ainvoke(msgs):
            call_count[0] += 1
            if call_count[0] <= 2:
                raise RuntimeError("model timeout")
            return mock_success

        vision_model = AsyncMock()
        vision_model.ainvoke = AsyncMock(side_effect=_ainvoke)

        proc = _make_processor(
            sandbox=sandbox, vision_model=vision_model,
            video_config=_make_video_config(max_keyframes=10),
        )

        result = asyncio.run(
            proc.process("/tmp/video.mp4", "video.mp4", "video/mp4", supports_vision=False)
        )

        # Frames 1-2: model failure (used_vision=True, consumes quota)
        assert result.text.count("description failed") == 2
        # Frames 3-5: model success (consumes quota, total = 5)
        assert vision_model.ainvoke.await_count == MAX_VISION_DESCRIBE
        # Frames 6-8: "not described" (quota exhausted)
        assert result.text.count("not described") == 3


# ---------------------------------------------------------------------------
# Tests: process — audio handling
# ---------------------------------------------------------------------------


class TestVideoFileProcessorAudioHandling:
    def test_has_audio_false_skips_transcription(self):
        sandbox = _make_sandbox(
            ffprobe_output=_make_ffprobe_output(has_audio=False),
            ls_output="",
        )
        audio_processor = AsyncMock()

        proc = _make_processor(sandbox=sandbox, audio_processor=audio_processor)

        result = asyncio.run(
            proc.process("/tmp/video.mp4", "video.mp4", "video/mp4", supports_vision=False)
        )

        # audio processor should NOT have been called
        audio_processor.process.assert_not_awaited()
        assert isinstance(result, FileProcessResult)

    def test_extract_audio_false_skips_transcription(self):
        sandbox = _make_sandbox(
            ffprobe_output=_make_ffprobe_output(has_audio=True),
            ls_output="",
        )
        audio_processor = AsyncMock()
        video_config = _make_video_config(extract_audio=False)

        proc = _make_processor(
            sandbox=sandbox,
            audio_processor=audio_processor,
            video_config=video_config,
        )

        result = asyncio.run(
            proc.process("/tmp/video.mp4", "video.mp4", "video/mp4", supports_vision=False)
        )

        audio_processor.process.assert_not_awaited()
        assert isinstance(result, FileProcessResult)

    def test_audio_processor_none_skips_transcription(self):
        sandbox = _make_sandbox(
            ffprobe_output=_make_ffprobe_output(has_audio=True),
            ls_output="",
        )

        proc = _make_processor(sandbox=sandbox, audio_processor=None)

        result = asyncio.run(
            proc.process("/tmp/video.mp4", "video.mp4", "video/mp4", supports_vision=False)
        )

        assert isinstance(result, FileProcessResult)
        assert "Transcript" not in result.text

    def test_audio_transcription_included_in_output(self):
        # We need more exec_command calls because audio extraction adds mkdir + ffmpeg
        sandbox = AsyncMock()
        ffprobe_result = _make_exec_result(0, _make_ffprobe_output(has_audio=True))
        mkdir_audio = _make_exec_result(0, "")
        ffmpeg_audio = _make_exec_result(0, "")
        mkdir_kf = _make_exec_result(0, "")
        ffmpeg_kf = _make_exec_result(0, "")
        ls_result = _make_exec_result(0, "")
        rm_result = _make_exec_result(0, "")
        sandbox.exec_command = AsyncMock(
            side_effect=[
                ffprobe_result,
                mkdir_audio,
                ffmpeg_audio,
                mkdir_kf,
                ffmpeg_kf,
                ls_result,
                rm_result,
            ]
        )
        sandbox.download_file = AsyncMock(return_value=io.BytesIO(b"fake"))

        audio_processor = AsyncMock()
        audio_processor.process = AsyncMock(
            return_value=FileProcessResult(text="Hello from transcript")
        )

        proc = _make_processor(
            sandbox=sandbox,
            audio_processor=audio_processor,
        )

        result = asyncio.run(
            proc.process("/tmp/video.mp4", "video.mp4", "video/mp4", supports_vision=False)
        )

        assert "Transcript" in result.text
        assert "Hello from transcript" in result.text


# ---------------------------------------------------------------------------
# Tests: metadata extraction failure
# ---------------------------------------------------------------------------


class TestVideoFileProcessorMetadataFailure:
    def test_ffprobe_nonzero_returncode_returns_error_message(self):
        sandbox = AsyncMock()
        sandbox.exec_command = AsyncMock(
            return_value=_make_exec_result(returncode=1, output="error")
        )

        proc = _make_processor(sandbox=sandbox)

        result = asyncio.run(
            proc.process("/tmp/broken.mp4", "broken.mp4", "video/mp4", supports_vision=False)
        )

        assert "unable to read metadata" in result.text
        assert "broken.mp4" in result.text
        assert result.image_blocks == ()

    def test_ffprobe_exception_returns_error_message(self):
        sandbox = AsyncMock()
        sandbox.exec_command = AsyncMock(side_effect=Exception("ffprobe crashed"))

        proc = _make_processor(sandbox=sandbox)

        result = asyncio.run(
            proc.process("/tmp/broken.mp4", "broken.mp4", "video/mp4", supports_vision=False)
        )

        assert "unable to read metadata" in result.text


# ---------------------------------------------------------------------------
# Tests: duration formatting
# ---------------------------------------------------------------------------


class TestFormatDuration:
    def test_under_one_hour_shows_mm_ss(self):
        assert VideoFileProcessor._format_duration(125.0) == "02:05"

    def test_exactly_one_minute(self):
        assert VideoFileProcessor._format_duration(60.0) == "01:00"

    def test_over_one_hour_shows_h_mm_ss(self):
        assert VideoFileProcessor._format_duration(3723.0) == "1:02:03"

    def test_zero_seconds(self):
        assert VideoFileProcessor._format_duration(0.0) == "00:00"

    def test_fractional_seconds_truncated(self):
        assert VideoFileProcessor._format_duration(61.9) == "01:01"


# ---------------------------------------------------------------------------
# Tests: registry integration
# ---------------------------------------------------------------------------


class TestRegistryVideoIntegration:
    def test_video_mp4_returns_video_processor(self):
        from app.infrastructure.external.file_processors.registry import FileProcessorRegistry

        video_config = _make_video_config()
        registry = FileProcessorRegistry(
            sandbox=AsyncMock(),
            file_uploader=AsyncMock(),
            video_config=video_config,
        )
        proc = registry.get_processor("video/mp4")
        assert proc is not None
        assert type(proc).__name__ == "VideoFileProcessor"

    def test_video_webm_returns_video_processor(self):
        from app.infrastructure.external.file_processors.registry import FileProcessorRegistry

        video_config = _make_video_config()
        registry = FileProcessorRegistry(
            sandbox=AsyncMock(),
            file_uploader=AsyncMock(),
            video_config=video_config,
        )
        proc = registry.get_processor("video/webm")
        assert proc is not None
        assert type(proc).__name__ == "VideoFileProcessor"

    def test_video_disabled_by_default(self):
        from app.infrastructure.external.file_processors.registry import FileProcessorRegistry

        registry = FileProcessorRegistry(
            sandbox=AsyncMock(),
            file_uploader=AsyncMock(),
        )
        assert registry.get_processor("video/mp4") is None

    def test_video_processor_receives_audio_processor_from_registry(self):
        from app.domain.models.app_config import AudioProcessorConfig
        from app.infrastructure.external.file_processors.registry import FileProcessorRegistry

        audio_config = AudioProcessorConfig(provider="sandbox_whisper")
        video_config = _make_video_config()
        registry = FileProcessorRegistry(
            sandbox=AsyncMock(),
            file_uploader=AsyncMock(),
            audio_config=audio_config,
            video_config=video_config,
        )
        proc = registry.get_processor("video/mp4")
        assert proc is not None
        # Video processor should have the audio processor injected
        assert proc._audio_processor is not None
        assert type(proc._audio_processor).__name__ == "AudioFileProcessor"

    def test_video_processor_audio_processor_none_when_audio_disabled(self):
        from app.infrastructure.external.file_processors.registry import FileProcessorRegistry

        video_config = _make_video_config()
        registry = FileProcessorRegistry(
            sandbox=AsyncMock(),
            file_uploader=AsyncMock(),
            video_config=video_config,
            # no audio_config
        )
        proc = registry.get_processor("video/mp4")
        assert proc is not None
        assert proc._audio_processor is None
