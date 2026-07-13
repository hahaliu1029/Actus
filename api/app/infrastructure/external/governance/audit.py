"""D1a §3.5 audit 写入 helper + allowlist sanitize（INV-D1-7：禁 secrets/探测原文/scan match 原文）。"""
from __future__ import annotations

import re
import uuid
from typing import Any, Mapping

from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.services.extension_scan import canonicalize_source_ref
from app.infrastructure.models.extension_governance import ExtensionAuditLogModel

# before/after 仅允许治理字段（R6#8）
ALLOWED_BEFORE_AFTER_KEYS = frozenset({
    "status", "quarantine_reason", "trust_origin",
    "artifact_hash", "surface_hash", "config_fingerprint",
    "observed_surface_hash", "observed_artifact_hash", "observed_config_fingerprint",
})

_REJECTED_STAGES = frozenset({"publish_reverify", "content_write_collision", "recovery_collision"})

# details 按 event 允许键；值=校验函数（None=原样收，字符串裁 500）
DETAILS_ALLOWLIST: dict[str, frozenset[str]] = {
    "installed": frozenset({"probe_failed", "forced"}),
    "quarantined": frozenset({"note"}),
    "install_rejected": frozenset({"stage", "member", "category", "collided_targets", "source_ref"}),
    "reconciled_missing": frozenset({"disposition"}),
    "plugin_expand_compensated": frozenset({"failed_step", "compensated_targets"}),
}

_MAX_STR = 500

# F2（INV-D1-7 值级纵深）：key allowlist 不足——值本身可携带 secrets（note 原文 /
# collided_targets 内的 headers dict / before-after 伪装 dict）。凭据模式脱敏族：
# 覆盖 bearer token、sk-* 密钥、以及 api_key/access_token/secret/password/signature/
# credential/authorization 的 `key: value` / `key=value` 形态。
# G1（INV-D1-7）：key-based 形态**必须脱敏到行尾**（`[^\r\n]*` 而非 `\S+`）——否则
# 分隔符后仅第一 token 被吃，多段值（`Basic <b64>` / `Bearer <tok> <extra>`）的后段幸存。
# standalone `bearer \S+` / `sk-...` 保留单 token 语义，覆盖内联出现。
_CREDENTIAL_PATTERNS = (
    re.compile(r"(?i)bearer\s+\S+"),
    re.compile(r"sk-[A-Za-z0-9_\-]{8,}"),
    re.compile(
        r"(?i)(?:api[_-]?key|access[_-]?token|refresh[_-]?token|secret|password|"
        r"signature|credential|authorization)\s*[:=]\s*[^\r\n]*"
    ),
)

# 值级白名单丢弃哨兵（dict/复杂形态无法结构化白名单 → 整值丢弃）
_DROP = object()


def _redact(text: str) -> str:
    for pattern in _CREDENTIAL_PATTERNS:
        text = pattern.sub("<redacted>", text)
    return text


def _sanitize_value(value: Any) -> Any:
    """值级白名单 + 凭据脱敏（INV-D1-7 纵深）：
    - str → 凭据模式脱敏后裁 `_MAX_STR`（**脱敏在裁剪之前**）；
    - bool / int / float / None → 原样（bool 是 int 子类，需先判）；
    - list → 逐元素递归；dict/复杂元素 → 丢弃（结构不可白名单），标量元素照上处理，整表裁 50；
    - dict 或其它复杂类型 → 返回 `_DROP`（调用方据此整键/整元素丢弃）。"""
    if isinstance(value, str):
        return _redact(value)[:_MAX_STR]
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, list):
        out: list[Any] = []
        for item in value[:50]:
            sanitized = _sanitize_value(item)
            if sanitized is _DROP:
                continue
            out.append(sanitized)
        return out
    return _DROP


def sanitize_details(event: str, details: Mapping[str, Any] | None) -> dict[str, Any] | None:
    allowed = DETAILS_ALLOWLIST.get(event)
    if not details or allowed is None:
        return None
    out: dict[str, Any] = {}
    for key, value in details.items():
        if key not in allowed:
            continue
        if event == "install_rejected" and key == "stage" and value not in _REJECTED_STAGES:
            continue
        if key == "source_ref":
            # R5#4 + R6#A2：INV-D1-7 防御纵深——source_ref 在 sanitize 层强制 canonicalize
            # （不依赖调用者自觉）；**非字符串形态直接 drop**（dict/list 载荷可携带任意
            # secrets 绕过 canonicalize——details 是 JSONB，原样入库不可接受）
            if not isinstance(value, str):
                continue
            value = canonicalize_source_ref(value)
        sanitized = _sanitize_value(value)
        if sanitized is _DROP:
            continue
        out[key] = sanitized
    return out


def sanitize_before_after(snapshot: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if snapshot is None:
        return None
    out: dict[str, Any] = {}
    for key, value in snapshot.items():
        if key not in ALLOWED_BEFORE_AFTER_KEYS:
            continue
        sanitized = _sanitize_value(value)
        if sanitized is _DROP:
            continue
        out[key] = sanitized
    return out


async def insert_audit(
    session: AsyncSession,
    *,
    kind: str,
    ext_id: str | None,
    extension_id: uuid.UUID | None,
    event: str,
    actor_user_id: str | None = None,
    before: Mapping[str, Any] | None = None,
    after: Mapping[str, Any] | None = None,
    details: Mapping[str, Any] | None = None,
    correlation_id: uuid.UUID | None = None,
) -> None:
    # ext_id 可 NULL 的唯一场景=install_rejected 身份解析前（R13#5）——代码层断言
    assert ext_id is not None or event == "install_rejected", event
    session.add(ExtensionAuditLogModel(
        id=uuid.uuid4(),
        extension_id=extension_id,
        kind=kind,
        ext_id=ext_id,
        actor_user_id=actor_user_id,
        event=event,
        before=sanitize_before_after(before),
        after=sanitize_before_after(after),
        details=sanitize_details(event, details),
        correlation_id=correlation_id,
    ))
