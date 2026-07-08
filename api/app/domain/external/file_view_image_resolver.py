"""B12 P1 domain Protocol: 按 display URL 取图片原始 bytes（provider 不接受 URL 时
的 base64 materialize 用）。infra 实现负责 SSRF 硬门。"""
from __future__ import annotations

from typing import Protocol


class FileViewImageBytesResolver(Protocol):
    async def load_image_bytes(self, display_url: str, *, max_bytes: int) -> bytes | None:
        """返回图片 bytes 或 None（拉取失败/超限/非法/被 SSRF 门拒 → None，
        调用方降级 text placeholder）。max_bytes = min(profile.image_max_bytes,
        sanitizer MAX_IMAGE_BYTES)；实现须 streaming 读到 max_bytes+1 早停。"""
        ...
