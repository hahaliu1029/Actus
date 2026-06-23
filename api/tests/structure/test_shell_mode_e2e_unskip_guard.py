"""[S2 PR-6 — reverse unskip guard, mirror C2 finish-core F5.3]

A fast, DB-free unit test that fails if the live shell-mode E2E
(``test_coordinator_e2e_shell_mode.py``) is ever silently neutered — by a
module-level ``@pytest.mark.skip``, a ``pytest.skip(...)`` call inside a test
body, or removal of the ``coordinator_recovery`` marker that the
``coordinator-e2e`` CI job selects on. Without this guard a future refactor
could green the suite by skipping the only test that proves shell-mode applies
real diffs, and nobody would notice (the E2E itself never runs in the default
suite — it carries ``pytest.mark.sandbox``, which ``api/pytest.ini`` addopts
deselect by default). This test reads the E2E source as TEXT (it does NOT
import or collect it) so it stays green in the default `cd api && uv run
pytest` run that has no pg/redis/minio/sandbox.

Colocated with ``test_coordinator_e2e_unskipped.py`` under ``api/tests/
structure/`` so it can reuse that file's ``parents[3]`` repo-root idiom: from
``api/tests/structure/<this file>`` → ``parents[3]`` is the repo root, then we
append ``api/tests/integration/...`` (verified in Step 2/Step 4). A guard placed
under a deeper dir like ``api/tests/domain/services/graphs/`` would mis-resolve
the path (``parents[3]`` would be ``api/tests`` → ``api/tests/tests/...``).

Scope (deliberate — what this guard does and does NOT catch):
The guard's contract is to catch ACCIDENTAL, SAME-FILE, statically-detectable
neutering — the kinds a normal maintainer edit could introduce. It exhaustively
covers: removing the file; any conditional/nested/second/mutated ``pytestmark``
binding; removing/commenting/misspelling the ``coordinator_recovery``/``sandbox``
markers; ``skip``/``skipif``/``xfail`` as a module marker, on ANY decoratable
node (function/async/class), or as a ``pytest.skip``/``xfail``/``importorskip``
call; deleting all ``test_*`` functions; and removing the real
``{"tool": "shell_execute"}`` driven call. It does NOT (by design) defend against
DELIBERATE circumvention or out-of-file neutering — an aliased import
(``import pytest as pt``), ``pytest.param(marks=...)``, marks built via a
variable/``*starred`` list, ``request.applymarker``, a skip hidden in
``conftest.py``/a plugin, ``raise pytest.skip.Exception``, or gutting a test body
to ``pass`` while keeping a shell dict. Those require intent to dodge the guard
(or live in another file) and are expected to be caught by code review, not this
static text guard. (Audit trail: codex PR-6 R1–R7 drove these to convergence.)
"""
import ast
from pathlib import Path

import pytest

pytestmark = [pytest.mark.structure]

# Repo root, matching api/tests/structure/test_coordinator_e2e_unskipped.py:
#   parents[3] == repo root (api/tests/structure/<file> -> .../structure ->
#   .../tests -> api -> repo root); then api/tests/integration/<e2e>.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_E2E_PATH = (
    _REPO_ROOT
    / "api" / "tests" / "integration" / "test_coordinator_e2e_shell_mode.py"
)


def _source() -> str:
    assert _E2E_PATH.exists(), (
        f"shell-mode E2E file missing at {_E2E_PATH}; PR-6 Task 6.2 must create "
        "it. The unskip guard exists precisely so this file can never quietly "
        "disappear."
    )
    return _E2E_PATH.read_text(encoding="utf-8")


def _pytest_mark_name(node: ast.expr) -> str | None:
    """Return ``<name>`` for a ``pytest.mark.<name>`` attribute or
    ``pytest.mark.<name>(...)`` call AST node; else ``None``."""
    if isinstance(node, ast.Call):
        node = node.func
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "mark"
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == "pytest"
    ):
        return node.attr
    return None


def test_shell_mode_e2e_is_not_silently_skipped() -> None:
    src = _source()
    # Parse the E2E as an AST so the marker/skip checks read the REAL
    # ``pytest.mark.<name>`` nodes — NOT raw text. A substring/regex scan was
    # spoofable two ways (codex PR-6 R1+R2 P0): (1) a marker mentioned only in a
    # docstring made the check vacuous; (2) a marker COMMENTED OUT inside the
    # ``pytestmark = [ ... ]`` list (e.g. ``# pytest.mark.coordinator_recovery,``)
    # still satisfied the substring while pytest never registered it, so
    # ``-m coordinator_recovery`` would silently stop selecting the E2E. AST
    # ignores comments/prose entirely.
    tree = ast.parse(src)

    # Neutering markers = skip / skipif / xfail (an xfail'd FAILING test also
    # reports green — codex PR-6 R3 P0). The guard must catch EVERY realistic
    # silent-neuter edit, so it (1) requires EXACTLY ONE module-level
    # `pytestmark = <literal list/tuple>`, (2) REJECTS any mutation of it
    # (`+=` / `.append`/`.extend`/`.insert` / a second assignment) — otherwise a
    # skip/xfail added AFTER the literal would be invisible to a static scan, and
    # (3) forbids skip/skipif/xfail as a list marker, a decorator, or a
    # `pytest.skip()/pytest.xfail()` body call.
    _NEUTER = {"skip", "skipif", "xfail"}

    # (2a) Reject pytestmark mutation (AugAssign or .append/.extend/.insert).
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.AugAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "pytestmark"
        ):
            raise AssertionError(
                "shell-mode E2E must not mutate `pytestmark` via `+=` — a skip/"
                "xfail added that way silently neuters the live proof; keep a "
                "single literal `pytestmark = [...]`."
            )
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in ("append", "extend", "insert")
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "pytestmark"
        ):
            raise AssertionError(
                "shell-mode E2E must not mutate `pytestmark` via "
                f"`.{node.func.attr}()` — keep a single literal `pytestmark = "
                "[...]` so the markers cannot be silently changed."
            )

    # (2b) Collect EVERY `pytestmark` binding ANYWHERE in the tree (not just the
    # module top level) and require EXACTLY ONE, itself a TOP-LEVEL literal
    # list/tuple. This closes a conditional/nested rebind — e.g.
    # `if os.getenv("X"): pytestmark = [..., pytest.mark.skip(...)]` — that a
    # `tree.body`-only scan would miss (codex PR-6 R6 P1); a second/shadowing
    # assignment; and it accepts both `pytestmark = [...]` (Assign) and an
    # annotated `pytestmark: list = [...]` (AnnAssign) (codex R4 P2).
    def _binds_pytestmark(node: ast.AST) -> bool:
        if isinstance(node, ast.Assign):
            return any(
                isinstance(t, ast.Name) and t.id == "pytestmark" for t in node.targets
            )
        if isinstance(node, ast.AnnAssign):
            return isinstance(node.target, ast.Name) and node.target.id == "pytestmark"
        return False

    pytestmark_assigns = [node for node in ast.walk(tree) if _binds_pytestmark(node)]
    assert len(pytestmark_assigns) == 1, (
        "shell-mode E2E must bind `pytestmark` EXACTLY ONCE anywhere in the file "
        f"(found {len(pytestmark_assigns)}); a second / conditional / nested "
        "binding could swap in a skip the static check would otherwise miss."
    )
    binding = pytestmark_assigns[0]
    assert any(binding is stmt for stmt in tree.body), (
        "shell-mode E2E `pytestmark` must be a SINGLE TOP-LEVEL module assignment "
        "— not nested inside an if/for/try/with/function — so it cannot be "
        "conditionally rebound to a skipping marker set."
    )
    pm_value = binding.value
    assert isinstance(pm_value, (ast.List, ast.Tuple)), (
        "shell-mode E2E `pytestmark` must be a literal list/tuple of "
        "`pytest.mark.*` entries (not a variable / comprehension) so the markers "
        "are statically verifiable."
    )
    pytestmark_names = {
        nm for elt in pm_value.elts if (nm := _pytest_mark_name(elt)) is not None
    }
    # The coordinator-e2e CI job selects `-m coordinator_recovery`; losing this
    # marker means the E2E silently stops running there.
    assert "coordinator_recovery" in pytestmark_names, (
        "shell-mode E2E must keep a REAL `pytest.mark.coordinator_recovery` in "
        "its pytestmark list so the coordinator-e2e CI job (-m coordinator_recovery) "
        "actually selects it (a commented-out marker does NOT count)."
    )
    # The `sandbox` marker keeps it out of the default local suite (api/pytest.ini
    # addopts: `-m "not slow and not sandbox and not browser_eval"`) — without it
    # `cd api && uv run pytest` would try to collect+run a DB/Redis/MinIO/Docker
    # E2E that cannot run locally.
    assert "sandbox" in pytestmark_names, (
        "shell-mode E2E must keep a REAL `pytest.mark.sandbox` in its pytestmark "
        "list so the default local suite deselects it; the coordinator-e2e CI job "
        "does NOT exclude sandbox so it still selects via -m coordinator_recovery."
    )
    # (3) No neutering marker in the pytestmark list ...
    assert not (pytestmark_names & _NEUTER), (
        "shell-mode E2E must not carry a module-level pytest.mark.skip/skipif/"
        "xfail — the coordinator-e2e CI job is the ONLY place it runs."
    )
    # ... nor as a test-function decorator, nor as a `pytest.skip()/pytest.xfail()`
    # body call. AST-based so legit prose mentioning "skip" never false-trips it.
    for node in ast.walk(tree):
        # A neutering decorator on ANY decoratable node — a function, an async
        # function, OR a test CLASS — skips the test(s). `decorator_list` exists
        # ONLY on FunctionDef / AsyncFunctionDef / ClassDef, so iterating it via
        # `getattr` covers all three completely and future-proof (codex PR-6 R5
        # P1: a class-level `@pytest.mark.skip` was missed by a function-only
        # check).
        for dec in getattr(node, "decorator_list", []):
            assert _pytest_mark_name(dec) not in _NEUTER, (
                f"shell-mode E2E {getattr(node, 'name', '<node>')!r} must not "
                "carry @pytest.mark.skip/skipif/xfail (function OR class level) — "
                "that would silently neuter the live proof."
            )
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in ("skip", "xfail", "importorskip")
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "pytest"
        ):
            raise AssertionError(
                f"shell-mode E2E must not call pytest.{node.func.attr}() — if a "
                "precondition (or import) is genuinely missing in CI the job "
                "should fail loudly, not skip (codex PR-6 R4 P1: importorskip "
                "would module-skip the only live shell proof)."
            )
    # (4) The file must still contain ≥1 real test, and must DRIVE a real
    #     shell_execute tool call — not be reduced to a stub that keeps the
    #     markers + a docstring while the actual proof is deleted/refactored
    #     away (codex PR-6 R4 P1). Both are AST checks so a comment/docstring
    #     mention can't satisfy them.
    test_fns = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    ]
    assert test_fns, (
        "shell-mode E2E must define at least one `test_*` function — a file with "
        "the markers but no test would pass collection (0 tests) while the "
        "coordinator-e2e job stays green on OTHER coordinator_recovery tests, "
        "silently removing all live shell-mode proof."
    )
    # A real `{"tool": "shell_execute", ...}` child-response dict must appear —
    # this is what makes the child run RAW shell (the whole point of S2; typed
    # extraction is blind to it). A docstring/comment string would NOT be inside
    # a `{"tool": ...}` dict, so this can't be spoofed by prose.
    drives_shell_execute = any(
        isinstance(node, ast.Dict)
        and any(
            isinstance(k, ast.Constant)
            and k.value == "tool"
            and isinstance(v, ast.Constant)
            and v.value == "shell_execute"
            for k, v in zip(node.keys, node.values)
            if k is not None
        )
        for node in ast.walk(tree)
    )
    assert drives_shell_execute, (
        "shell-mode E2E must drive a real `{\"tool\": \"shell_execute\", ...}` "
        "child response — that raw-shell write is the whole point of S2 "
        "(typed-event extraction is blind to it). A substring in a docstring/"
        "comment does NOT count."
    )
    # Weak smoke (documents the dependency; the AUTHORITATIVE flag enforcement is
    # the CI YAML guard `test_coordinator_e2e_shell_flag_in_ci.py` + the
    # dark-launch test, since the E2E reads the flag from the CI job env).
    assert "ACTUS_C2_COORDINATOR_SHELL_MODE_ENABLED" in src, (
        "shell-mode E2E must reference the master flag env var it depends on."
    )
