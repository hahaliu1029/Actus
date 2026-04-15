"""R2 CS2 golden matrix validation — 22 JSON fixtures round-trip cleanly.

The companion generator ``r2_tool_outcome_matrix/_generate_fixtures.py``
writes 22 canonical ``ToolArtifact`` JSON files covering the full
(``source`` × ``variant``) matrix. This test suite loads each one and
asserts:

1. **Fixture count** — exactly 22 files (not 21, not 23). Adding a new
   fixture requires deliberately bumping the counter + updating the
   generator.
2. **Every fixture loads via ``TOOL_ARTIFACT_ADAPTER``** — typed
   round-trip through the R2 CS2 discriminated union.
3. **Byte-for-byte round-trip stability** — load → dump → compare
   structure. If the wire format ever drifts (e.g. a key renamed, a
   nested block alias dropped, by_alias=True forgotten), this test
   catches it before the drift reaches golden consumers (front-end,
   audit log, replay).
4. **Structural coverage** — at least one fixture per source and
   per variant so no taxonomy row is silently dropped from the matrix.
5. **Fixture name ↔ content agreement** — e.g. a file named
   ``*_allow_error_*`` must contain an ``allow_error`` variant.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.domain.models.tool_result import (
    TOOL_ARTIFACT_ADAPTER,
    ToolArtifact,
)


FIXTURE_DIR = Path(__file__).parent / "r2_tool_outcome_matrix"


def _all_fixture_files() -> list[Path]:
    """Enumerate every ``*.json`` under the fixture directory.

    Excludes anything starting with ``_`` so the generator script (if
    it's ever renamed to ``.json``) is ignored.
    """
    return sorted(
        f for f in FIXTURE_DIR.glob("*.json") if not f.name.startswith("_")
    )


# ============================================================
# Fixture count + presence
# ============================================================


def test_22_fixtures_present():
    files = _all_fixture_files()
    assert len(files) == 22, (
        f"Expected 22 golden fixtures in {FIXTURE_DIR}, found {len(files)}: "
        f"{sorted(f.name for f in files)}"
    )


def test_fixture_dir_exists():
    assert FIXTURE_DIR.is_dir(), (
        f"Golden fixture directory not found: {FIXTURE_DIR}. "
        f"Run `uv run python {FIXTURE_DIR}/_generate_fixtures.py` to "
        f"regenerate."
    )


# ============================================================
# Typed load (one parametrized case per fixture)
# ============================================================


@pytest.mark.parametrize(
    "fixture_path",
    _all_fixture_files(),
    ids=lambda p: p.name,
)
def test_fixture_loads_as_tool_artifact(fixture_path: Path):
    """Every fixture must validate cleanly through ``TOOL_ARTIFACT_ADAPTER``."""
    data = json.loads(fixture_path.read_text())
    artifact = TOOL_ARTIFACT_ADAPTER.validate_python(data)
    assert isinstance(artifact, ToolArtifact), (
        f"{fixture_path.name}: TOOL_ARTIFACT_ADAPTER.validate_python "
        f"did not yield a ToolArtifact"
    )
    # The canonical fields must be non-empty
    assert artifact.tool_call_id
    assert artifact.tool_name
    assert artifact.tool_source is not None
    assert artifact.outcome is not None


# ============================================================
# Byte-for-byte round-trip stability
# ============================================================


@pytest.mark.parametrize(
    "fixture_path",
    _all_fixture_files(),
    ids=lambda p: p.name,
)
def test_fixture_round_trip_preserves_structure(fixture_path: Path):
    """Load → dump → compare. Any drift in wire format surfaces here.

    We compare the Python dict (not raw bytes) so whitespace-only
    differences don't break the test; structure must be identical.
    """
    original = json.loads(fixture_path.read_text())
    artifact = TOOL_ARTIFACT_ADAPTER.validate_python(original)
    re_dumped = artifact.model_dump(mode="json", by_alias=True)
    assert re_dumped == original, (
        f"{fixture_path.name} round-trip mismatch.\n"
        f"original keys: {sorted(original.keys())}\n"
        f"re-dumped keys: {sorted(re_dumped.keys())}"
    )


# ============================================================
# Structural coverage: all 4 sources, all 5 variants
# ============================================================


def test_every_source_has_at_least_one_fixture():
    """R2 merge-gate requirement: 4 tool sources × meaningful outcome
    combinations. Every source must be represented.
    """
    seen = set()
    for f in _all_fixture_files():
        data = json.loads(f.read_text())
        seen.add(data["tool_source"]["source"])
    assert seen >= {"native", "mcp", "a2a", "skill"}, (
        f"Missing fixtures for sources: "
        f"{ {'native', 'mcp', 'a2a', 'skill'} - seen}"
    )


def test_every_variant_has_at_least_one_fixture():
    """All 5 ``ToolOutcome`` variants must appear at least once so the
    golden matrix reflects the full CS2 taxonomy."""
    seen = set()
    for f in _all_fixture_files():
        data = json.loads(f.read_text())
        seen.add(data["outcome"]["variant"])
    assert seen >= {
        "allow_success",
        "allow_error",
        "denied",
        "asked",
        "passthrough",
    }, (
        f"Missing fixtures for variants: "
        f"{ {'allow_success', 'allow_error', 'denied', 'asked', 'passthrough'} - seen}"
    )


# ============================================================
# Fixture naming ↔ content consistency
# ============================================================


@pytest.mark.parametrize(
    "fixture_path",
    _all_fixture_files(),
    ids=lambda p: p.name,
)
def test_fixture_name_matches_outcome_variant(fixture_path: Path):
    """Fixture filename must reflect the contained outcome variant.

    Convention: ``{source}_{variant_marker}_{extra}.json`` where
    ``variant_marker`` is one of ``allow_success`` / ``allow_error`` /
    ``denied`` / ``asked`` / ``passthrough``. This keeps the directory
    self-documenting — future readers can filter by source+variant
    without opening each file.
    """
    data = json.loads(fixture_path.read_text())
    variant = data["outcome"]["variant"]
    name = fixture_path.name.lower()

    variant_markers = {
        "allow_success": "allow_success",
        "allow_error": "allow_error",
        "denied": "denied",
        "asked": "asked",
        "passthrough": "passthrough",
    }
    marker = variant_markers[variant]
    assert marker in name, (
        f"{fixture_path.name}: outcome variant is {variant!r} but "
        f"filename doesn't contain {marker!r}. Rename the file to match "
        f"the convention ``{{source}}_{{variant}}_{{extra}}.json``."
    )


@pytest.mark.parametrize(
    "fixture_path",
    _all_fixture_files(),
    ids=lambda p: p.name,
)
def test_fixture_name_matches_tool_source(fixture_path: Path):
    """Fixture filename prefix must match its ``tool_source.source``."""
    data = json.loads(fixture_path.read_text())
    source = data["tool_source"]["source"]
    name_prefix = fixture_path.name.split("_", 1)[0]
    assert source == name_prefix, (
        f"{fixture_path.name}: tool_source is {source!r} but filename "
        f"prefix is {name_prefix!r}. Convention: filename must start with "
        f"the source string (native_*, mcp_*, a2a_*, skill_*)."
    )
