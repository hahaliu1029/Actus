"""D1a §4.1 Port 合同：dataclass frozen/字段集/默认值 + Protocol 方法面。"""
import dataclasses
import inspect
import re
from typing import Protocol, get_args

import pytest

from app.domain.external.extension_admission import (
    AdmissionDecision,
    ExtensionAdmissionPort,
    ExtensionRegistryReadPort,
    ExtensionRegistryWritePort,
    GovernanceCounters,
    GovernanceRowSnapshot,
    InstallContext,
    Observation,
    UninstallContext,
)


class TestAdmissionDecision:
    def test_frozen_and_fields(self):
        d = AdmissionDecision(admitted=True, reason="ok", row_revision=3)
        assert d.observation_outcome == "none"           # 默认
        assert d.config_drift_detected is False          # R46#5 默认
        assert dataclasses.fields(AdmissionDecision)
        try:
            d.admitted = False
            raise AssertionError("should be frozen")
        except dataclasses.FrozenInstanceError:
            pass

    def test_field_names_closed(self):
        assert {f.name for f in dataclasses.fields(AdmissionDecision)} == {
            "admitted", "reason", "row_revision",
            "observation_outcome", "config_drift_detected",
        }


class TestObservation:
    def test_categories(self):
        cat_type = next(f for f in dataclasses.fields(Observation) if f.name == "category").type
        # dataclass 保存注解字符串或类型均可——直接构造三类验证
        for c in ("surface", "artifact", "config_fingerprint"):
            Observation(category=c, payload={}, schema_version=1)

    def test_under_config_fingerprint_default_none(self):
        assert Observation(category="artifact", payload={}, schema_version=1).under_config_fingerprint is None


class TestContexts:
    def test_install_context_fields_closed(self):
        assert {f.name for f in dataclasses.fields(InstallContext)} == {
            "actor_user_id", "correlation_id", "source_type", "source_ref",
            "version", "trust_origin", "artifact_hash", "surface_hash",
            "config_fingerprint", "hash_schema_version", "scan",
            "probe_failed", "acknowledged", "forced",
        }

    def test_uninstall_context_fields(self):
        assert {f.name for f in dataclasses.fields(UninstallContext)} == {
            "correlation_id", "actor_user_id",
        }
        ctx = UninstallContext(correlation_id=None, actor_user_id="admin-1")
        assert ctx.actor_user_id == "admin-1"


class TestPortSurfaces:
    def test_admission_port_methods(self):
        assert hasattr(ExtensionAdmissionPort, "check_many")
        assert hasattr(ExtensionAdmissionPort, "verify_observation")
        # 受限写 port：行政/pin 方法不得出现在 admission 面（INV-D1-3 结构侧）
        for forbidden in ("record_install", "record_delete", "quarantine",
                          "reapprove", "approve_pin", "reset_pins_after_config_drift"):
            assert not hasattr(ExtensionAdmissionPort, forbidden), forbidden

    def test_write_port_methods(self):
        for m in ("record_install", "record_delete", "record_reconciled_seen",
                  "mark_source_missing", "mark_source_restored", "record_reconciled_missing",
                  "reset_pins_after_config_drift", "quarantine", "reapprove",
                  "set_governance_enabled", "approve_pin"):
            assert hasattr(ExtensionRegistryWritePort, m), m

    def test_read_port_methods(self):
        for m in ("get_row", "list_live_rows", "governance_counters"):
            assert hasattr(ExtensionRegistryReadPort, m), m


class TestReadModels:
    def test_counters_fields(self):
        c = GovernanceCounters(unpinned_count=0, missing_observation_count=0, quarantined_count=0)
        assert dataclasses.asdict(c) == {
            "unpinned_count": 0, "missing_observation_count": 0, "quarantined_count": 0,
        }

    def test_row_snapshot_has_projection_fields(self):
        names = {f.name for f in dataclasses.fields(GovernanceRowSnapshot)}
        # §9.1 governance block 全字段的读模型来源（provenance 无孤儿列）
        for required in ("status", "trust_origin", "quarantine_reason", "scan_verdict",
                         "last_mismatch_at", "last_verified_at", "row_revision",
                         "observed_surface_hash", "observed_artifact_hash",
                         "observed_config_fingerprint", "observed_hash_schema_version",
                         "pinned_at", "pinned_by", "installed_by", "source_type",
                         "source_ref", "version", "source_missing_at",
                         "parent_plugin_ext_id", "artifact_hash", "surface_hash",
                         "config_fingerprint", "hash_schema_version"):
            assert required in names, required


# ===========================================================================
# Finding 2+3 (P2) — FULL Protocol signature pinning + INV-D1-3 closed set.
# The hasattr checks above only prove a method EXISTS; they don't catch a
# param add/remove, a positional↔keyword-only flip, a return-type change, an
# async→sync flip, or a new method leaking onto the restricted admission面.
# These tables pin the whole surface so future drift fails loudly.
# ===========================================================================

_EMPTY = "<empty>"
_ND = object()  # "no default" sentinel — shared so tuple equality is by identity
_K_POS = inspect.Parameter.POSITIONAL_OR_KEYWORD
_K_KW = inspect.Parameter.KEYWORD_ONLY


def _norm_ann(ann: object) -> str:
    """Whitespace- and forward-ref-quote-insensitive annotation string.

    The module uses ``from __future__ import annotations`` so annotations are
    the stringified source. Collapse all whitespace and treat a PEP 563 string
    forward-ref (``'InstallContext'``) identically to a bare one
    (``InstallContext``) so cosmetic de-quoting doesn't trip the guard. Type
    expressions here contain no whitespace-significant tokens, so this is safe.
    """
    if ann is _EMPTY:
        return _EMPTY
    s = ann if isinstance(ann, str) else getattr(ann, "__name__", str(ann))
    s = re.sub(r"\s+", "", s)
    if len(s) >= 2 and s[0] in "\"'" and s[-1] in "\"'":
        s = s[1:-1]
    return s


def _actual_params(func):
    out = []
    for p in inspect.signature(func).parameters.values():
        ann = _EMPTY if p.annotation is inspect.Parameter.empty else _norm_ann(p.annotation)
        default = _ND if p.default is inspect.Parameter.empty else p.default
        out.append((p.name, p.kind, ann, default))
    return out


def _actual_return(func) -> str:
    ra = inspect.signature(func).return_annotation
    return _EMPTY if ra is inspect.Signature.empty else _norm_ann(ra)


def _norm_expected_params(params):
    return [(n, k, _norm_ann(a), d) for (n, k, a, d) in params]


def _public_callables(port) -> set[str]:
    """Public callable members of ``port``, minus bare-Protocol machinery."""

    class _Bare(Protocol):
        pass

    bare = {n for n in dir(_Bare) if not n.startswith("_")}
    return {
        n
        for n in dir(port)
        if not n.startswith("_") and callable(getattr(port, n, None))
    } - bare


# port method -> (is_coroutine, return_annotation, [(param, kind, annotation, default)])
_ADMISSION_SIGS = {
    "check_many": (True, "dict[str, AdmissionDecision]", [
        ("self", _K_POS, _EMPTY, _ND),
        ("kind", _K_POS, "GovernedExtensionKind", _ND),
        ("ext_ids", _K_POS, "Sequence[str]", _ND),
        ("config_fingerprints", _K_POS, "Mapping[str, str] | None", None),
    ]),
    "verify_observation": (True, "AdmissionDecision", [
        ("self", _K_POS, _EMPTY, _ND),
        ("kind", _K_POS, "GovernedExtensionKind", _ND),
        ("ext_id", _K_POS, "str", _ND),
        ("obs", _K_POS, "Observation", _ND),
    ]),
}

_WRITE_SIGS = {
    "record_install": (True, "None", [
        ("self", _K_POS, _EMPTY, _ND),
        ("kind", _K_POS, "GovernedExtensionKind", _ND),
        ("ext_id", _K_POS, "str", _ND),
        ("install_context", _K_POS, "InstallContext", _ND),
    ]),
    "record_delete": (True, "None", [
        ("self", _K_POS, _EMPTY, _ND),
        ("kind", _K_POS, "GovernedExtensionKind", _ND),
        ("ext_id", _K_POS, "str", _ND),
        ("uninstall_context", _K_KW, "UninstallContext", _ND),
    ]),
    "record_reconciled_seen": (True, "None", [
        ("self", _K_POS, _EMPTY, _ND),
        ("kind", _K_POS, "GovernedExtensionKind", _ND),
        ("ext_id", _K_POS, "str", _ND),
        ("source_type", _K_KW, "str", _ND),
        ("source_ref", _K_KW, "str | None", _ND),
        ("version", _K_KW, "str | None", _ND),
        ("trust_origin", _K_KW, "str", _ND),
    ]),
    "mark_source_missing": (True, "None", [
        ("self", _K_POS, _EMPTY, _ND),
        ("kind", _K_POS, "GovernedExtensionKind", _ND),
        ("ext_id", _K_POS, "str", _ND),
    ]),
    "mark_source_restored": (True, "None", [
        ("self", _K_POS, _EMPTY, _ND),
        ("kind", _K_POS, "GovernedExtensionKind", _ND),
        ("ext_id", _K_POS, "str", _ND),
    ]),
    "record_reconciled_missing": (True, "None", [
        ("self", _K_POS, _EMPTY, _ND),
        ("kind", _K_POS, "GovernedExtensionKind", _ND),
        ("ext_id", _K_POS, "str", _ND),
    ]),
    "reset_pins_after_config_drift": (True, "bool", [
        ("self", _K_POS, _EMPTY, _ND),
        ("kind", _K_POS, "GovernedExtensionKind", _ND),
        ("ext_id", _K_POS, "str", _ND),
        ("row_revision", _K_KW, "int", _ND),
    ]),
    "quarantine": (True, "int", [
        ("self", _K_POS, _EMPTY, _ND),
        ("kind", _K_POS, "GovernedExtensionKind", _ND),
        ("ext_id", _K_POS, "str", _ND),
        ("expected_row_revision", _K_KW, "int", _ND),
        ("actor_user_id", _K_KW, "str", _ND),
        ("note", _K_KW, "str | None", None),
    ]),
    "reapprove": (True, "int", [
        ("self", _K_POS, _EMPTY, _ND),
        ("kind", _K_POS, "GovernedExtensionKind", _ND),
        ("ext_id", _K_POS, "str", _ND),
        ("expected_row_revision", _K_KW, "int", _ND),
        ("actor_user_id", _K_KW, "str", _ND),
    ]),
    "set_governance_enabled": (True, "int", [
        ("self", _K_POS, _EMPTY, _ND),
        ("kind", _K_POS, "GovernedExtensionKind", _ND),
        ("ext_id", _K_POS, "str", _ND),
        ("enabled", _K_KW, "bool", _ND),
        ("expected_row_revision", _K_KW, "int", _ND),
        ("actor_user_id", _K_KW, "str", _ND),
    ]),
    "approve_pin": (True, "PinApprovalOutcome", [
        ("self", _K_POS, _EMPTY, _ND),
        ("kind", _K_POS, "GovernedExtensionKind", _ND),
        ("ext_id", _K_POS, "str", _ND),
        ("expected_row_revision", _K_KW, "int", _ND),
        ("actor_user_id", _K_KW, "str", _ND),
    ]),
}

_READ_SIGS = {
    "get_row": (True, "GovernanceRowSnapshot | None", [
        ("self", _K_POS, _EMPTY, _ND),
        ("kind", _K_POS, "GovernedExtensionKind", _ND),
        ("ext_id", _K_POS, "str", _ND),
    ]),
    "list_live_rows": (True, "list[GovernanceRowSnapshot]", [
        ("self", _K_POS, _EMPTY, _ND),
    ]),
    "governance_counters": (True, "GovernanceCounters", [
        ("self", _K_POS, _EMPTY, _ND),
    ]),
    # T20：§9.2 GET /audit 复合游标翻页读面（ReadPort Protocol 同步加方法）
    "list_audit": (True, "AuditPage", [
        ("self", _K_POS, _EMPTY, _ND),
        ("kind", _K_KW, "str | None", None),
        ("ext_id", _K_KW, "str | None", None),
        ("event", _K_KW, "str | None", None),
        ("cursor", _K_KW, "str | None", None),
        ("limit", _K_KW, "int", 50),
    ]),
}


class TestPortFullSignatures:
    @pytest.mark.parametrize(
        "port, table",
        [
            (ExtensionAdmissionPort, _ADMISSION_SIGS),
            (ExtensionRegistryWritePort, _WRITE_SIGS),
            (ExtensionRegistryReadPort, _READ_SIGS),
        ],
    )
    def test_signatures_pinned(self, port, table):
        # completeness: every public method is pinned, and none has leaked in
        assert _public_callables(port) == set(table), port.__name__
        for name, (coro, ret, params) in table.items():
            func = getattr(port, name)
            assert inspect.iscoroutinefunction(func) is coro, f"{port.__name__}.{name} async"
            assert _actual_return(func) == _norm_ann(ret), f"{port.__name__}.{name} return"
            assert _actual_params(func) == _norm_expected_params(params), (
                f"{port.__name__}.{name} params")

    def test_method_counts(self):
        # spec §4.1: admission=2 / write=11 / read=4（T20 加 list_audit 读面）
        assert len(_ADMISSION_SIGS) == 2
        assert len(_WRITE_SIGS) == 11
        assert len(_READ_SIGS) == 4


class TestAdmissionClosedSet:
    """INV-D1-3 (structural half): the admission面 is exactly the two read-only
    checks — no pin/administrative write may leak onto it. This is the positive
    closed-set companion to the existing negative-list test (both kept)."""

    def test_closed_public_method_set(self):
        assert _public_callables(ExtensionAdmissionPort) == {"check_many", "verify_observation"}

    def test_mode_declared(self):
        # 生效 mode 必须在 port 上声明（R1#4 双保险 / INV-D1-6 fail-open/closed 分流依据）
        assert "mode" in ExtensionAdmissionPort.__annotations__


# ===========================================================================
# Finding 4 (P2) — DTO snapshot: ordered (name, annotation) + frozen + defaults.
# The field-name SET tests above don't lock ORDER, TYPES, frozen-ness, or the
# exact declared defaults; wire projection + persistence depend on all four, so
# a reorder / retype / default-flip must fail here.
# ===========================================================================

_DTO_FIELDS = {
    AdmissionDecision: [
        ("admitted", "bool"),
        ("reason", "AdmissionDecisionReason"),
        ("row_revision", "int | None"),
        ("observation_outcome", "Literal['persisted', 'unchanged', 'conflict', 'none']"),
        ("config_drift_detected", "bool"),
    ],
    UninstallContext: [
        ("correlation_id", "UUID | None"),
        ("actor_user_id", "str"),
    ],
    InstallContext: [
        ("actor_user_id", "str"),
        ("correlation_id", "UUID | None"),
        ("source_type", "str"),
        ("source_ref", "str | None"),
        ("version", "str | None"),
        ("trust_origin", "str"),
        ("artifact_hash", "str | None"),
        ("surface_hash", "str | None"),
        ("config_fingerprint", "str | None"),
        ("hash_schema_version", "int"),
        ("scan", "GovernanceScanSummary | None"),
        ("probe_failed", "bool"),
        ("acknowledged", "bool"),
        ("forced", "bool"),
    ],
    Observation: [
        ("category", "Literal['surface', 'artifact', 'config_fingerprint']"),
        ("payload", "Any"),
        ("schema_version", "int"),
        ("under_config_fingerprint", "str | None"),
    ],
    GovernanceCounters: [
        ("unpinned_count", "int"),
        ("missing_observation_count", "int"),
        ("quarantined_count", "int"),
    ],
    GovernanceRowSnapshot: [
        ("id", "UUID"),
        ("kind", "str"),
        ("ext_id", "str"),
        ("status", "str"),
        ("quarantine_reason", "str | None"),
        ("trust_origin", "str"),
        ("source_type", "str"),
        ("source_ref", "str | None"),
        ("version", "str | None"),
        ("artifact_hash", "str | None"),
        ("surface_hash", "str | None"),
        ("config_fingerprint", "str | None"),
        ("hash_schema_version", "int"),
        ("observed_surface_hash", "str | None"),
        ("observed_artifact_hash", "str | None"),
        ("observed_config_fingerprint", "str | None"),
        ("observed_hash_schema_version", "int | None"),
        ("last_observed_at", "datetime | None"),
        ("last_verified_at", "datetime | None"),
        ("last_mismatch_at", "datetime | None"),
        ("pinned_at", "datetime | None"),
        ("pinned_by", "str | None"),
        ("scan_verdict", "str | None"),
        ("scan_report", "dict[str, Any] | None"),
        ("source_missing_at", "datetime | None"),
        ("installed_by", "str | None"),
        ("row_revision", "int"),
        ("deleted_at", "datetime | None"),
        ("parent_plugin_ext_id", "str | None"),
    ],
}

# Only the fields whose authoritative source (extension_admission.py) declares a
# default. Every OTHER field must stay required (default is dataclasses.MISSING).
_DTO_DEFAULTS = {
    AdmissionDecision: {"observation_outcome": "none", "config_drift_detected": False},
    UninstallContext: {},
    InstallContext: {"probe_failed": False, "acknowledged": False, "forced": False},
    Observation: {"under_config_fingerprint": None},
    GovernanceCounters: {},
    GovernanceRowSnapshot: {"parent_plugin_ext_id": None},
}


class TestDTOSnapshot:
    @pytest.mark.parametrize("cls", list(_DTO_FIELDS))
    def test_field_order_and_types(self, cls):
        fields = dataclasses.fields(cls)
        actual = [(f.name, _norm_ann(cls.__annotations__[f.name])) for f in fields]
        expected = [(n, _norm_ann(a)) for (n, a) in _DTO_FIELDS[cls]]
        assert actual == expected, cls.__name__
        # __annotations__ order must equal fields order (no ClassVar/reorder drift)
        assert list(cls.__annotations__.keys()) == [f.name for f in fields], cls.__name__

    @pytest.mark.parametrize("cls", list(_DTO_FIELDS))
    def test_frozen(self, cls):
        assert cls.__dataclass_params__.frozen is True, cls.__name__

    @pytest.mark.parametrize("cls", list(_DTO_FIELDS))
    def test_defaults(self, cls):
        declared = _DTO_DEFAULTS[cls]
        for f in dataclasses.fields(cls):
            if f.name in declared:
                exp = declared[f.name]
                if exp is None:
                    assert f.default is None, (cls.__name__, f.name)
                elif isinstance(exp, bool):
                    assert f.default is exp, (cls.__name__, f.name)
                else:
                    assert f.default == exp and isinstance(f.default, type(exp)), (
                        cls.__name__, f.name)
            else:
                assert f.default is dataclasses.MISSING, (cls.__name__, f.name)
            # A default_factory would bypass the plain-default lock above and
            # silently make a required field optional (e.g. actor_user_id via
            # field(default_factory=str)) — forbid it on every DTO field.
            assert f.default_factory is dataclasses.MISSING, (cls.__name__, f.name)
