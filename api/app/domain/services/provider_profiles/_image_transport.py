"""B12 P1: provider-aware image transport selection (pure策略, domain 内无 infra import).

file_view 图片 block 有 3 个消费者（LLM message / FE result_blocks / artifact 回放），
本函数只决定 LLM-facing transport（Kimi 类 accepts_image_url=False → base64 / placeholder）。
bytes 拉取通过注入的 load_bytes thunk 完成（domain 不碰 storage/HTTP）。
"""
from __future__ import annotations

import base64
from typing import Any, Awaitable, Callable
from urllib.parse import urlsplit


def _image_url_block(url: str, detail: str = "auto") -> dict:
    return {"type": "image_url", "image_url": {"url": url, "detail": detail}}


def _text_placeholder(filename: str) -> dict:
    return {"type": "text", "text": f"[image unavailable: {filename}]"}


async def select_image_transport(
    display_url: str,
    media_type: str | None,
    filename: str,
    profile: Any | None,
    load_bytes: Callable[[], Awaitable[bytes | None]],
    detail: str = "auto",
) -> dict:
    """按 provider profile 决定图片 block 的 LLM-facing transport.

    分支（spec §3）:
    - data: URL → profile None/accepts_image_base64 → 原样保留; 否则 text placeholder
    - 非 loopback http(s) + accepts_image_url(或 profile None) → 保留 URL
    - loopback http(s) → 仅 base64 或 placeholder，绝不把 URL 直接交给 LLM
    - 无可用 URL / accepts_image_url=False → accepts_image_base64 时拉 bytes 转 base64;
      拉取失败或禁 base64 → text placeholder（不回退 http URL，那会再触发 rewrite 崩）
    """
    has_url = bool(display_url)
    is_data_url = has_url and display_url.startswith("data:")

    if is_data_url:
        if profile is None or profile.accepts_image_base64:
            return _image_url_block(display_url, detail)
        return _text_placeholder(filename)

    try:
        parsed_url = urlsplit(display_url)
        is_loopback_http = (
            parsed_url.scheme.lower() in {"http", "https"}
            and parsed_url.hostname in {"localhost", "127.0.0.1", "::1"}
        )
    except ValueError:
        is_loopback_http = False

    if has_url and not is_loopback_http and (
        profile is None or profile.accepts_image_url
    ):
        return _image_url_block(display_url, detail)

    # 无 URL / provider 禁 URL → base64 或 placeholder
    if profile is None or profile.accepts_image_base64:
        raw = await load_bytes()
        if raw is not None:
            mime = media_type or "image/png"
            b64 = base64.b64encode(raw).decode("ascii")
            return _image_url_block(f"data:{mime};base64,{b64}", detail)
    return _text_placeholder(filename)
