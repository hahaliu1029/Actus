"""INV-SPM-1/9 — sandbox create / resume surface confined to a method whitelist.

AST-level structure gate (mirrors ``tests/invariants/test_inv_c7_2_*``). A plain
text grep would false-positive on unrelated ``.resume(`` (task / flow /
``redis_stream_task``) and MISS the ``main.py`` lifespan proxy alias, forcing the
gate to be either red or file-level loose. So we resolve the receiver via AST:

- **bind_new / resume** call = ``ast.Call`` whose ``func`` is
  ``Attribute(attr in {"bind_new","resume"})`` AND the receiver resolves to the
  lifecycle service: its source text contains ``"lifecycle"``, OR it is a
  ``Name`` bound — inside the SAME function — to an expression whose source
  contains ``"lifecycle"`` (``main.py``: ``svc = _pending_lifecycle_ref.get(...)``
  then ``svc.bind_new(...)`` — receiver ``"svc"`` alone lacks the token; the
  alias rule recovers it). ``task.resume()`` / ``flow.resume()`` never resolve.
- **container create** = ``Attribute(attr=="create")`` whose receiver source
  contains ``"sandbox_cls"`` or ``"DockerSandbox"``; PLUS ``Attribute(attr=="run")``
  whose receiver source contains ``"containers"`` (``docker_sandbox``'s
  ``containers.run``).

Whitelists are module-relative to ``api/app``. PR-1b asserts
``{agent_service, starter} ⊆ bind_new callers``; the provisioner membership
assertion is added by PR-1c Task 19 (which also appends the provisioner file to
``BIND_NEW_ALLOWED``). Constants + the single test body are the only edit points.
"""
import ast
import functools
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[2] / "app"


@functools.lru_cache(maxsize=1)
def _parsed_app_files() -> tuple[tuple[str, ast.AST, str], ...]:
    """Parse every ``api/app`` module ONCE (rel-path, tree, source)."""
    out = []
    for path in sorted(APP_ROOT.rglob("*.py")):
        src = path.read_text(encoding="utf-8")
        out.append((path.relative_to(APP_ROOT).as_posix(), ast.parse(src), src))
    return tuple(out)

# ── Whitelists (module-relative to api/app) ─────────────────────────────────
BIND_NEW_ALLOWED = {
    "application/services/agent_service.py",              # always run_start
    "application/services/session_service.py",            # vnc / takeover
    "application/services/coordinator_child_runner_starter.py",  # per-child lease
    "application/services/sandbox_lifecycle_service.py",  # definition site
    "main.py",                                            # lifespan forward-ref proxy
    # PR-1c (Task 19) appends: "application/services/sandbox_provisioner.py"
}
RESUME_ALLOWED = {
    "application/services/agent_service.py",              # reopen / retry_from_suspend
    "application/services/session_service.py",            # vnc / takeover / _acquire_sandbox
    "application/services/sandbox_lifecycle_service.py",  # definition site
    "main.py",                                            # lifespan proxy symmetry
}
CONTAINER_CREATE_ALLOWED = {
    "application/services/sandbox_lifecycle_service.py",
    "application/services/agent_service.py",              # no-lifecycle test fallback create
    "application/services/skill_creator_service.py",      # DD-15 gated exception
    "infrastructure/external/sandbox/docker_sandbox.py",  # definition site
}

# Staged anti-inertness membership (PR-1b). These MUST be observed callers — if
# the predicate is written too narrow they silently disappear and the gate rots.
# Encodes the brief's Step 4b known sites (main.py bind_new + agent_service create).
BIND_NEW_REQUIRED = {
    "application/services/agent_service.py",
    "application/services/coordinator_child_runner_starter.py",
    "main.py",
}
RESUME_REQUIRED = {
    "application/services/agent_service.py",
    "application/services/session_service.py",
}
CONTAINER_CREATE_REQUIRED = {
    "application/services/agent_service.py",
    "infrastructure/external/sandbox/docker_sandbox.py",
}

_LIFECYCLE_TOKENS = ("lifecycle",)
_CONTAINER_CREATE_TOKENS = ("sandbox_cls", "DockerSandbox")


def _src_of(node: ast.AST, src: str) -> str:
    return ast.get_source_segment(src, node) or ""


def _iter_functions(tree: ast.AST):
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield n


def _func_lifecycle_aliases(func: ast.AST, src: str) -> set[str]:
    """Local names in ``func`` bound to an expression whose source mentions lifecycle."""
    names: set[str] = set()
    for n in ast.walk(func):
        if isinstance(n, ast.Assign):
            rhs = _src_of(n.value, src).lower()
            if any(tok in rhs for tok in _LIFECYCLE_TOKENS):
                for tgt in n.targets:
                    if isinstance(tgt, ast.Name):
                        names.add(tgt.id)
    return names


def _receiver_is_lifecycle(recv: ast.expr, aliases: set[str], src: str) -> bool:
    if any(tok in _src_of(recv, src).lower() for tok in _LIFECYCLE_TOKENS):
        return True
    return isinstance(recv, ast.Name) and recv.id in aliases


def _file_calls_lifecycle_method(tree: ast.AST, src: str, attr: str) -> bool:
    for func in _iter_functions(tree):
        aliases = _func_lifecycle_aliases(func, src)
        for n in ast.walk(func):
            if (
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr == attr
                and _receiver_is_lifecycle(n.func.value, aliases, src)
            ):
                return True
    return False


def _file_creates_container(tree: ast.AST, src: str) -> bool:
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
            recv = _src_of(n.func.value, src)
            if n.func.attr == "create" and any(t in recv for t in _CONTAINER_CREATE_TOKENS):
                return True
            if n.func.attr == "run" and "containers" in recv:
                return True
    return False


def _collect(predicate) -> set[str]:
    return {rel for rel, tree, src in _parsed_app_files() if predicate(tree, src)}


def _bind_new_callers() -> set[str]:
    return _collect(lambda t, s: _file_calls_lifecycle_method(t, s, "bind_new"))


def _resume_callers() -> set[str]:
    return _collect(lambda t, s: _file_calls_lifecycle_method(t, s, "resume"))


def _container_create_callers() -> set[str]:
    return _collect(lambda t, s: _file_creates_container(t, s))


def test_bind_new_callers_whitelisted() -> None:
    callers = _bind_new_callers()
    extra = callers - BIND_NEW_ALLOWED
    assert not extra, f"INV-SPM-1 unexpected lifecycle.bind_new callers: {sorted(extra)}"
    missing = BIND_NEW_REQUIRED - callers
    assert not missing, (
        f"predicate too narrow — expected bind_new callers missing: {sorted(missing)}"
    )


def test_resume_callers_whitelisted() -> None:
    callers = _resume_callers()
    extra = callers - RESUME_ALLOWED
    assert not extra, f"INV-SPM-1 unexpected lifecycle.resume callers: {sorted(extra)}"
    missing = RESUME_REQUIRED - callers
    assert not missing, (
        f"predicate too narrow — expected resume callers missing: {sorted(missing)}"
    )


def test_container_create_callers_whitelisted() -> None:
    callers = _container_create_callers()
    extra = callers - CONTAINER_CREATE_ALLOWED
    assert not extra, f"INV-SPM-1 unexpected container-create callers: {sorted(extra)}"
    missing = CONTAINER_CREATE_REQUIRED - callers
    assert not missing, (
        f"predicate too narrow — expected container-create callers missing: {sorted(missing)}"
    )


def test_gate_detects_violation_negative_control(tmp_path) -> None:
    """反向自证：the AST predicates catch an out-of-whitelist lifecycle call and a
    rogue container create (defends against an inert gate)."""
    sample = tmp_path / "app" / "rogue.py"
    sample.parent.mkdir(parents=True)
    sample.write_text(
        "async def evil(self):\n"
        "    svc = build_sandbox_lifecycle_service()\n"
        "    await svc.bind_new('s')\n"
        "    await self._lifecycle.resume('s')\n"
        "    return await self._sandbox_cls.create(user_id='u')\n",
        encoding="utf-8",
    )
    src = sample.read_text(encoding="utf-8")
    tree = ast.parse(src)
    assert _file_calls_lifecycle_method(tree, src, "bind_new")
    assert _file_calls_lifecycle_method(tree, src, "resume")
    assert _file_creates_container(tree, src)
