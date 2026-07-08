# api/tests/domain/services/test_agent_task_runner_profile.py
import base64
import io
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.domain.services.provider_profiles import get_profile

# Project uses anyio (not pytest-asyncio); pin backend to asyncio.
pytestmark = pytest.mark.anyio


@pytest.fixture()
def anyio_backend() -> str:
    return "asyncio"


def _make_runner_with_profile(profile, storage):
    """Helper: minimal runner instance exposing only the attachment builder."""
    from app.domain.services.agent_task_runner import AgentTaskRunner
    runner = AgentTaskRunner.__new__(AgentTaskRunner)
    runner.profile = profile
    runner._file_storage = storage
    runner._supports_vision = True
    runner._image_url_map = {}
    runner._IMAGE_MIME_PREFIXES = ("image/",)
    return runner


async def test_kimi_attachment_builder_produces_base64_block() -> None:
    """T3: profile.accepts_image_url=False → base64 data URL"""
    kimi = get_profile("kimi_k2")
    storage = MagicMock()
    fake_bytes = b"\x89PNG\r\n\x1a\nfake"
    storage.download_file = AsyncMock(return_value=(io.BytesIO(fake_bytes), None))
    runner = _make_runner_with_profile(kimi, storage)

    attachment = MagicMock()
    attachment.id = "att1"
    attachment.mime_type = "image/png"
    attachment.filepath = None
    attachment.multimodal_eligible = True
    attachment.width = attachment.height = None
    attachment.filename = "pic.png"

    blocks = await runner._build_image_blocks([attachment])
    assert len(blocks) == 1
    assert blocks[0]["type"] == "image_url"
    assert blocks[0]["image_url"]["url"].startswith("data:image/png;base64,")


async def test_kimi_attachment_builder_degrades_to_text_placeholder_on_io_failure(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """T3c: base64 encode failed (I/O / oversize / decode) → text placeholder"""
    kimi = get_profile("kimi_k2")
    storage = MagicMock()
    storage.download_file = AsyncMock(side_effect=OSError("network error"))
    runner = _make_runner_with_profile(kimi, storage)

    attachment = MagicMock()
    attachment.id = "att1"
    attachment.mime_type = "image/png"
    attachment.filename = "pic.png"
    attachment.filepath = None
    attachment.multimodal_eligible = True
    attachment.width = attachment.height = None

    import logging
    with caplog.at_level(logging.WARNING):
        blocks = await runner._build_image_blocks([attachment])

    assert all(b.get("type") != "image_url"
               or b.get("image_url", {}).get("url", "").startswith("data:")
               for b in blocks)
    text_blocks = [b for b in blocks if b.get("type") == "text"]
    assert any("[image unavailable:" in b.get("text", "") for b in text_blocks)
    assert any("att1" in r.message or "network error" in r.message for r in caplog.records)


async def test_openai_profile_still_uses_presigned_url(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """profile.accepts_image_url=True → presigned URL path.

    Explicitly pins every attribute the downstream metadata branch reads so
    MagicMock auto-attribute creation cannot silently route into the outer
    ``except Exception`` and mask real bugs. An ERROR log in this path must
    fail the test, not pass it.
    """
    openai_p = get_profile("openai_official")
    storage = MagicMock()
    runner = _make_runner_with_profile(openai_p, storage)
    runner._get_image_presigned_url = AsyncMock(return_value="https://s3.example/x")

    attachment = MagicMock()
    attachment.id = "att1"
    attachment.mime_type = "image/png"
    attachment.filepath = "f"
    attachment.filename = "pic.png"
    attachment.multimodal_eligible = True
    attachment.width = 500
    attachment.height = 500
    attachment.original_width = 500
    attachment.original_height = 500

    import logging
    with caplog.at_level(logging.ERROR):
        blocks = await runner._build_image_blocks([attachment])

    assert blocks[0]["image_url"]["url"].startswith("https://")
    # Regression guard: the OpenAI path must not silently trip the outer
    # ``except Exception`` and mask a bug in a later branch.
    error_records = [
        r for r in caplog.records
        if r.levelno >= logging.ERROR
        and "_build_image_blocks unexpected error" in r.getMessage()
    ]
    assert not error_records, (
        f"Unexpected outer-except ERROR fired: "
        f"{[r.getMessage() for r in error_records]}"
    )


async def test_openai_profile_base64_caps_at_legacy_5mb_base64_ceiling(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """P1 regression guard: without a presigned URL, the OpenAI/generic
    base64 fallback must enforce the legacy 3.75 MB raw ≈ 5 MB base64 cap.

    A 4.5 MB raw image used to be rejected (~6 MB data URL violates OpenAI's
    5 MB payload limit). Task 1.8 initially widened this to 5 MB raw via the
    base profile default; `openai_official` / `generic_openai` now override
    ``image_max_bytes=3_932_160`` so the OpenAI payload ceiling is preserved.
    """
    openai_p = get_profile("openai_official")
    assert openai_p.image_max_bytes == 3_932_160

    oversized = b"\x89PNG\r\n\x1a\n" + b"x" * (4_500_000 - 8)  # 4.5 MB raw
    storage = MagicMock()
    storage.download_file = AsyncMock(return_value=(io.BytesIO(oversized), None))
    runner = _make_runner_with_profile(openai_p, storage)
    # No presigned URL → forces the base64 fallback even for OpenAI.
    runner._get_image_presigned_url = AsyncMock(return_value=None)

    attachment = MagicMock()
    attachment.id = "att1"
    attachment.mime_type = "image/png"
    attachment.filename = "big.png"
    attachment.filepath = None
    attachment.multimodal_eligible = True
    attachment.width = attachment.height = None
    attachment.original_width = None
    attachment.original_height = None

    import logging
    with caplog.at_level(logging.WARNING):
        blocks = await runner._build_image_blocks([attachment])

    # No oversized data URL block may be emitted.
    for b in blocks:
        if b.get("type") == "image_url":
            url = b["image_url"]["url"]
            assert not url.startswith("data:"), (
                f"oversized image was base64-encoded despite profile cap "
                f"(url len={len(url)})"
            )

    # Degrade contract: text placeholder emitted, WARN logged with reason.
    text_blocks = [b for b in blocks if b.get("type") == "text"]
    assert any("[image unavailable:" in b.get("text", "") for b in text_blocks)
    assert any(
        "image_max_bytes" in r.getMessage() or "exceeds" in r.getMessage()
        for r in caplog.records
        if r.levelno >= logging.WARNING
    )


async def test_deepseek_profile_skips_image_blocks_when_supports_vision_false() -> None:
    """P1: profile.supports_vision=False must skip image embedding entirely.

    Even if the runner was constructed with supports_vision=True, the profile
    ceiling must kick in: no image_url block produced, no base64 attempt.
    """
    deepseek = get_profile("deepseek_reasoner")
    storage = MagicMock()
    storage.download_file = AsyncMock(
        side_effect=AssertionError("storage must not be touched when profile disables vision"),
    )
    runner = _make_runner_with_profile(deepseek, storage)
    # Even if _supports_vision is True, the profile ceiling must override.
    runner._supports_vision = True
    runner._get_image_presigned_url = AsyncMock(return_value="https://s3.example/x")

    attachment = MagicMock()
    attachment.id = "att1"
    attachment.mime_type = "image/png"
    attachment.filepath = "f"
    attachment.filename = "pic.png"
    attachment.multimodal_eligible = True
    attachment.width = attachment.height = None
    attachment.original_width = attachment.original_height = None

    blocks = await runner._build_image_blocks([attachment])
    assert blocks == [] or all(b.get("type") != "image_url" for b in blocks)
    storage.download_file.assert_not_called()


async def test_profile_forbids_both_url_and_base64_emits_text_placeholder(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """P1: accepts_image_url=False AND accepts_image_base64=False -> text placeholder,
    no download attempt.

    DeepSeek Reasoner has this combination. Runner should NOT call storage.download_file.
    """
    import logging
    deepseek = get_profile("deepseek_reasoner")
    storage = MagicMock()
    storage.download_file = AsyncMock(
        side_effect=AssertionError("must not attempt download when base64 forbidden"),
    )
    runner = _make_runner_with_profile(deepseek, storage)
    # Force vision on (bypass the separate supports_vision short-circuit),
    # we want to test the accepts_image_base64=False specific path.
    # The supports_vision profile ceiling check fires first, so temporarily
    # patch profile.supports_vision to True via replace() to isolate this path.
    from dataclasses import replace
    runner.profile = replace(deepseek, supports_vision=True)
    runner._supports_vision = True
    runner._get_image_presigned_url = AsyncMock(return_value=None)  # no URL path

    attachment = MagicMock()
    attachment.id = "att1"
    attachment.mime_type = "image/png"
    attachment.filename = "pic.png"
    attachment.filepath = None
    attachment.multimodal_eligible = True
    attachment.width = attachment.height = None
    attachment.original_width = attachment.original_height = None

    with caplog.at_level(logging.WARNING):
        blocks = await runner._build_image_blocks([attachment])

    text_blocks = [b for b in blocks if b.get("type") == "text"]
    assert any("[image unavailable:" in b.get("text", "") for b in text_blocks)
    storage.download_file.assert_not_called()
    assert any(
        "forbids both image URL and base64" in r.getMessage()
        for r in caplog.records if r.levelno >= logging.WARNING
    )


async def test_profile_over_5mb_capped_by_sanitizer_limit(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """R2#P2-2: profile.image_max_bytes > 5MiB 时，base64 fallback 仍按 5MiB(sanitizer)
    cap——6MiB raw 图应被拒（min(6MiB,5MiB)=5MiB），出 text placeholder。"""
    from dataclasses import replace
    import logging
    base = get_profile("kimi_k2")  # accepts_image_url=False → 强制 base64 路
    prof = replace(base, image_max_bytes=6 * 1024 * 1024, accepts_image_url=False,
                   accepts_image_base64=True, supports_vision=True)
    oversized = b"\x89PNG\r\n\x1a\n" + b"x" * (5_500_000)  # 5.5MiB raw
    storage = MagicMock()
    storage.download_file = AsyncMock(return_value=(io.BytesIO(oversized), None))
    runner = _make_runner_with_profile(prof, storage)
    runner._get_image_presigned_url = AsyncMock(return_value=None)  # 无 URL → base64 路

    attachment = MagicMock()
    attachment.id = "att1"
    attachment.mime_type = "image/png"
    attachment.filename = "big.png"
    attachment.filepath = None
    attachment.multimodal_eligible = True
    attachment.width = attachment.height = None
    attachment.original_width = attachment.original_height = None

    with caplog.at_level(logging.WARNING):
        blocks = await runner._build_image_blocks([attachment])

    for b in blocks:
        if b.get("type") == "image_url":
            assert not b["image_url"]["url"].startswith("data:"), "oversized image base64 despite 5MiB cap"
    assert any("[image unavailable:" in b.get("text", "") for b in blocks)
