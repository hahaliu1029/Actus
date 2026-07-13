"""D1-7：audit details allowlist sanitize（禁 secrets/原文；quarantined.note ≤500）。"""
from app.infrastructure.external.governance.audit import (
    ALLOWED_BEFORE_AFTER_KEYS,
    DETAILS_ALLOWLIST,
    sanitize_before_after,
    sanitize_details,
)


def test_details_unknown_keys_dropped():
    out = sanitize_details("installed", {"probe_failed": True, "api_key": "sk-XXX"})
    assert out == {"probe_failed": True}


def test_quarantined_note_truncated_500():
    out = sanitize_details("quarantined", {"note": "x" * 900})
    assert len(out["note"]) == 500


def test_install_rejected_stage_allowlist():
    out = sanitize_details("install_rejected", {
        "stage": "recovery_collision", "collided_targets": ["mcp_config:s1"],
        "raw_match": "password=123",
    })
    assert out == {"stage": "recovery_collision", "collided_targets": ["mcp_config:s1"]}
    # stage 允许值集合封闭（§3.5）
    assert sanitize_details("install_rejected", {"stage": "bogus"}) == {}


def test_event_without_allowlist_yields_none():
    assert sanitize_details("enabled", None) is None
    assert sanitize_details("enabled", {"whatever": 1}) is None


def test_before_after_only_governance_fields():
    out = sanitize_before_after({
        "status": "active", "quarantine_reason": None,
        "surface_hash": "h1", "config": {"url": "http://x", "headers": {"k": "v"}},
    })
    assert "config" not in out and out["status"] == "active"
    assert set(out) <= ALLOWED_BEFORE_AFTER_KEYS


def test_details_allowlist_covers_spec_events():
    # §3.5 声明过 details 的事件必须在 allowlist 有定义
    for event in ("installed", "quarantined", "install_rejected", "reconciled_missing"):
        assert event in DETAILS_ALLOWLIST, event


def test_note_credential_pattern_redacted():
    # F2 counterexample #1（INV-D1-7）：note 内凭据必须脱敏，不逐字入库
    out = sanitize_details("quarantined", {"note": "Authorization: Bearer sk-demo-secret-token"})
    assert "sk-demo-secret-token" not in out["note"]
    assert "Bearer" not in out["note"]
    assert "<redacted>" in out["note"]


def test_nested_dict_in_list_dropped():
    # F2 counterexample #2：collided_targets 内的 dict 项（可携带 headers/match 原文）→ 丢弃
    out = sanitize_details("install_rejected", {
        "stage": "recovery_collision",
        "collided_targets": [
            {"headers": {"Authorization": "Bearer sk-demo"}, "match": "password=hunter2"},
            "mcp_config:s1",
        ],
    })
    assert out["stage"] == "recovery_collision"
    # dict 项被丢弃，标量项保留
    assert out["collided_targets"] == ["mcp_config:s1"]
    assert "sk-demo" not in str(out) and "hunter2" not in str(out)


def test_dict_masquerading_as_governance_field_dropped():
    # F2 counterexample #3：before/after 的 surface_hash 若是 dict（可藏 secrets）→ 整键丢弃
    out = sanitize_before_after({
        "status": "active",
        "surface_hash": {"malicious": "sk-demo-secret"},
    })
    assert "surface_hash" not in out
    assert out["status"] == "active"
    assert "sk-demo-secret" not in str(out)


def test_bare_secret_patterns_redacted_in_details():
    # sk-/api_key/access_token/secret/password/signature/credential 家族均脱敏
    out = sanitize_details("quarantined", {"note": "key sk-ABCDEFGH12345 and api_key=topsecret"})
    assert "sk-ABCDEFGH12345" not in out["note"]
    assert "topsecret" not in out["note"]


def test_authorization_basic_multipart_fully_redacted():
    # G1 repro (INV-D1-7): `Authorization: Basic <b64>` — old \S+ ate only "Basic",
    # the base64 credential survived. Key-based pattern must redact to end-of-line.
    out = sanitize_details("quarantined", {"note": "Authorization: Basic dXNlcjpwYXNz"})
    assert "dXNlcjpwYXNz" not in out["note"]
    assert "Basic" not in out["note"]
    assert out["note"] == "<redacted>"


def test_authorization_bearer_two_parts_no_survivor():
    # G1: standalone `bearer \S+` grabs only the 1st token; the key-based pattern with
    # [^\r\n]* must eat the whole value so a trailing 2nd part cannot survive.
    out = sanitize_details("quarantined", {"note": "authorization=Bearer tok-part1 tok-part2"})
    assert "tok-part1" not in out["note"]
    assert "tok-part2" not in out["note"]      # old \S+ left this survivor


def test_prose_without_key_separator_untouched():
    # G1 guard: prose mentioning 'password'/'secret' words but NO `key: value` separator
    # must stay intact (redaction anchors on the key separator, no over-redaction).
    note = "The user changed their password and the secret handshake worked"
    out = sanitize_details("quarantined", {"note": note})
    assert out["note"] == note


def test_source_ref_forcibly_canonicalized_in_sanitize_layer():
    # R5#4/INV-D1-7 防御纵深：调用方传原始 ref 也在 sanitize 层被 canonicalize
    out = sanitize_details("install_rejected", {
        "stage": "publish_reverify",
        "source_ref": "https://user:pass@evil.example/repo?token=sk-XXX",
    })
    assert "user:pass" not in out["source_ref"] and "token" not in out["source_ref"]
    out2 = sanitize_details("install_rejected", {
        "stage": "publish_reverify", "source_ref": "file:///home/x/.ssh/id_rsa",
    })
    assert out2["source_ref"] == "<redacted-local-path>"
    # R6#A2：非字符串 source_ref（dict 载荷可携带 secrets）→ 整键 drop
    out3 = sanitize_details("install_rejected", {
        "stage": "publish_reverify", "source_ref": {"token": "sk-XXX"},
    })
    assert "source_ref" not in out3
