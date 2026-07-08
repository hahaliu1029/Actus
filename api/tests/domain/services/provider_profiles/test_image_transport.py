"""B12 P1: select_image_transport 5 分支（+ no-URL 边界）。"""
import asyncio
import base64

from app.domain.services.provider_profiles._image_transport import select_image_transport


class _Profile:
    def __init__(self, accepts_image_url: bool, accepts_image_base64: bool):
        self.accepts_image_url = accepts_image_url
        self.accepts_image_base64 = accepts_image_base64
        self.provider_id = "test"
        self.image_max_bytes = 5 * 1024 * 1024


async def _bytes_ok():
    return b"\x89PNG rawbytes"


async def _bytes_fail():
    return None


def _run(coro):
    return asyncio.run(coro)


def test_data_url_base64_ok_kept() -> None:
    block = _run(select_image_transport(
        "data:image/png;base64,abc", "image/png", "a.png",
        _Profile(accepts_image_url=False, accepts_image_base64=True), _bytes_fail,
    ))
    assert block == {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc", "detail": "auto"}}


def test_data_url_base64_forbidden_placeholder() -> None:
    block = _run(select_image_transport(
        "data:image/png;base64,abc", "image/png", "a.png",
        _Profile(accepts_image_url=False, accepts_image_base64=False), _bytes_fail,
    ))
    assert block == {"type": "text", "text": "[image unavailable: a.png]"}


def test_http_url_ok_kept() -> None:
    block = _run(select_image_transport(
        "https://minio/a.png", "image/png", "a.png",
        _Profile(accepts_image_url=True, accepts_image_base64=True), _bytes_fail,
    ))
    assert block["image_url"]["url"] == "https://minio/a.png"


def test_http_url_forbidden_base64_ok() -> None:
    block = _run(select_image_transport(
        "https://minio/a.png", "image/png", "a.png",
        _Profile(accepts_image_url=False, accepts_image_base64=True), _bytes_ok,
    ))
    expected_b64 = base64.b64encode(b"\x89PNG rawbytes").decode("ascii")
    assert block["image_url"]["url"] == f"data:image/png;base64,{expected_b64}"


def test_http_url_forbidden_base64_fetch_fails_placeholder() -> None:
    block = _run(select_image_transport(
        "https://minio/a.png", "image/png", "a.png",
        _Profile(accepts_image_url=False, accepts_image_base64=True), _bytes_fail,
    ))
    assert block == {"type": "text", "text": "[image unavailable: a.png]"}


def test_profile_none_keeps_url() -> None:
    block = _run(select_image_transport(
        "https://minio/a.png", "image/png", "a.png", None, _bytes_fail,
    ))
    assert block["image_url"]["url"] == "https://minio/a.png"


def test_no_url_falls_back_to_base64() -> None:
    """attachment 路 presigned_url=None 时：无 URL → base64。"""
    block = _run(select_image_transport(
        "", "image/png", "a.png",
        _Profile(accepts_image_url=True, accepts_image_base64=True), _bytes_ok,
    ))
    assert block["image_url"]["url"].startswith("data:image/png;base64,")
