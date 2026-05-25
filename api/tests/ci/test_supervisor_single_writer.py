"""C3 spec §13.3 — MailboxSupervisor is the sole writer of mailbox-child
terminal sandbox transitions (M1 single-writer invariant, spec §11.4 + §3.2).

AST scan verifies that no application/infrastructure module **other than** the
supervisor handler family invokes ``destroy()`` / ``terminate()`` with a
mailbox-only ``DestroyReason`` value.

The 4 mailbox-only DestroyReason values (spec §7.2) are:

  - ``SUBAGENT_TERMINAL_RESULT``  — RESULT_READY observed
  - ``CANCEL_ACK_OBSERVED``       — child confirmed cooperative cancel
  - ``ORPHAN_TIMEOUT``            — supervisor stale-child detection
  - ``FORCE_TERMINATE``           — cascade cancel policy=TERMINATE

These reasons are emitted exclusively by ``MailboxSupervisor`` handlers
(spec §6.3 dispatch table). Any other module that threads one of them into a
``destroy(reason=…)`` / ``terminate(destroy_reason=…)`` call would race the
supervisor on the M1 single-writer invariant.

This is a *syntactic* gate, not a control-flow proof. It catches the common
violation pattern ``destroy(reason=DestroyReason.X)`` where ``X`` is a
mailbox-only value. Runtime safety is provided by the AST gate +
``mailbox_skip_helper`` + supervisor handler discipline together.

**Whitelisted files** (own the mailbox supervisor surface and may legitimately
emit mailbox DestroyReasons):

- ``mailbox_supervisor.py`` — owns mailbox-plane destroy (M1).
- ``supervisor_registry.py`` — supervisor lifecycle; transitively threads
  destroy reasons through the supervisor.
"""

from __future__ import annotations

import ast
import pathlib


_MAILBOX_ONLY_DESTROY_REASONS: frozenset[str] = frozenset(
    {
        "SUBAGENT_TERMINAL_RESULT",
        "CANCEL_ACK_OBSERVED",
        "ORPHAN_TIMEOUT",
        "FORCE_TERMINATE",
    }
)

# The single-writer invariant applies to lifecycle write methods only.
# Filtering call targets to this set avoids false positives on telemetry /
# logging / audit call sites that legitimately accept a DestroyReason value
# as a label or tag. The kwarg whitelist further narrows to the canonical
# parameter names used by ``SandboxLifecycleService.destroy()`` and the
# supervisor's ``_emit_cascade_terminate(destroy_reason=...)`` style helpers.
_LIFECYCLE_WRITER_METHOD_NAMES: frozenset[str] = frozenset(
    {"destroy", "terminate"}
)
_LIFECYCLE_REASON_KWARG_NAMES: frozenset[str] = frozenset(
    {"reason", "destroy_reason"}
)

_WHITELIST_FILES: frozenset[str] = frozenset(
    {
        "api/app/application/services/mailbox_supervisor.py",
        "api/app/application/services/supervisor_registry.py",
    }
)


def _call_target_method_name(call: ast.Call) -> str | None:
    """Return ``X.Y`` → ``"Y"`` for an attribute-style call target, else None.
    Bare-name calls (``f(...)``) and indexed targets are intentionally
    ignored — the lifecycle writer surface is always method-style
    (``sandbox.destroy(...)`` / ``supervisor.terminate(...)``).
    """
    func = call.func
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _iter_reason_arg_attributes(call: ast.Call) -> list[ast.Attribute]:
    """Return ``ast.Attribute`` nodes used in any positional arg OR in
    the ``reason=`` / ``destroy_reason=`` keyword arg of the call.

    The call-target filter (``_call_target_method_name`` checked
    upstream) already narrows the scan to ``.destroy()`` / ``.terminate()``
    methods — those are short, lifecycle-only signatures so scanning all
    positional args is safe AND necessary. Concrete writers in this
    codebase put reason in different slots:

      * ``SandboxLifecycleService.destroy(self, session_id, reason)``
        → ``svc.destroy(sid, DestroyReason.X)`` — reason is SECOND positional
        (codex r3 P1 fix — the original first-positional-only narrowing
        silently missed this real-world pattern at session_service.py:295).
      * ``MailboxSupervisor._emit_cascade_terminate(..., destroy_reason=...)``
        → keyword-only.
      * ``some.terminate(reason=DestroyReason.X)`` / ``terminate(DestroyReason.X)``
        → either form.

    Scanning all positional + matching kwargs covers every shape; the
    call-target filter prevents the false positives the R2 narrowing
    was meant to address (telemetry / logging calls that happen to
    reference DestroyReason).
    """
    out: list[ast.Attribute] = []
    for arg in call.args:
        if isinstance(arg, ast.Attribute):
            out.append(arg)
    for kw in call.keywords:
        if kw.arg in _LIFECYCLE_REASON_KWARG_NAMES and isinstance(
            kw.value, ast.Attribute
        ):
            out.append(kw.value)
    return out


def test_only_supervisor_family_uses_mailbox_destroy_reasons() -> None:
    """No non-whitelisted module under ``api/app/`` may pass a mailbox-only
    ``DestroyReason`` value into a ``.destroy()`` / ``.terminate()`` call.
    The supervisor handler family (whitelisted files) is the single writer
    for these transitions.

    Scoping notes (codex r2 [P2] follow-up + codex r3 [P1] re-broaden):
      * Call target name must be ``destroy`` or ``terminate`` (the two
        lifecycle write methods on ``SandboxLifecycleService`` + the
        supervisor's ``_emit_cascade_terminate(destroy_reason=...)`` etc).
        This filter is what keeps the gate immune to false positives on
        telemetry / logging call sites that legitimately mention a
        ``DestroyReason`` value.
      * Reason attribute may appear in ANY positional slot OR under
        ``reason=`` / ``destroy_reason=`` kwargs — r3 [P1] showed the
        real ``SandboxLifecycleService.destroy(self, session_id, reason)``
        signature carries reason as the SECOND positional, which the
        r2 first-positional-only narrowing silently missed.
    """
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    scan_root = repo_root / "api" / "app"
    assert scan_root.is_dir(), f"expected {scan_root} to exist"

    violations: list[str] = []
    for py_file in scan_root.rglob("*.py"):
        rel = py_file.relative_to(repo_root).as_posix()
        if rel in _WHITELIST_FILES:
            continue
        source = py_file.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source, filename=str(py_file))
        except SyntaxError as exc:  # pragma: no cover — defensive
            violations.append(f"{rel}: failed to parse ({exc!s})")
            continue

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            target = _call_target_method_name(node)
            if target not in _LIFECYCLE_WRITER_METHOD_NAMES:
                continue
            for attr in _iter_reason_arg_attributes(node):
                if attr.attr in _MAILBOX_ONLY_DESTROY_REASONS:
                    violations.append(
                        f"{rel}:{node.lineno} — non-supervisor module passes "
                        f"mailbox-only `DestroyReason.{attr.attr}` to "
                        f"`.{target}(...)` (M1 single-writer invariant). "
                        f"Whitelist the file or route the destroy through "
                        f"MailboxSupervisor."
                    )

    assert not violations, (
        "M1 single-writer violations (C3 spec §11.4 + §13.3 — mailbox-only "
        "DestroyReason values may only be emitted by MailboxSupervisor / "
        "SupervisorRegistry):\n" + "\n".join(violations)
    )


def test_synthetic_violation_is_detected(tmp_path: pathlib.Path) -> None:
    """Regression: synthetic violator patterns covering the 3 call shapes
    of the real ``destroy``/``terminate`` writers, plus 3 negative cases
    that prove the call-target filter prevents false positives.

    Codex r3 P1 added the second-positional case
    (``destroy(session_id, DestroyReason.X)``) — the real signature of
    ``SandboxLifecycleService.destroy(self, session_id, reason)`` —
    which the R2 first-positional-only narrowing silently missed.
    """
    bad = tmp_path / "fake_module.py"
    bad.write_text(
        "from app.domain.models.session import DestroyReason\n"
        "\n"
        "async def violate_kw(sandbox):\n"
        "    await sandbox.destroy(reason=DestroyReason.FORCE_TERMINATE)\n"
        "\n"
        "async def violate_pos_first(sandbox):\n"
        "    await sandbox.terminate(DestroyReason.ORPHAN_TIMEOUT)\n"
        "\n"
        "async def violate_pos_second(sandbox):\n"
        "    # Real-world signature: destroy(session_id, reason)\n"
        "    await sandbox.destroy('sess-x', DestroyReason.CANCEL_ACK_OBSERVED)\n"
        "\n"
        "async def ok_string_reason_kw(sandbox):\n"
        "    await sandbox.destroy(reason='legacy_reason')\n"
        "\n"
        "async def ok_string_reason_pos(sandbox):\n"
        "    await sandbox.destroy('sess-x', 'legacy_reason')\n"
        "\n"
        "async def ok_non_lifecycle_call(logger_):\n"
        "    # Telemetry call that happens to mention DestroyReason — must NOT be flagged.\n"
        "    logger_.info('reason=%s', DestroyReason.FORCE_TERMINATE)\n",
        encoding="utf-8",
    )

    tree = ast.parse(bad.read_text(encoding="utf-8"), filename=str(bad))
    flagged: list[tuple[int, str, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        target = _call_target_method_name(node)
        if target not in _LIFECYCLE_WRITER_METHOD_NAMES:
            continue
        for attr in _iter_reason_arg_attributes(node):
            if attr.attr in _MAILBOX_ONLY_DESTROY_REASONS:
                flagged.append((node.lineno, target, attr.attr))

    assert flagged == [
        (4, "destroy", "FORCE_TERMINATE"),
        (7, "terminate", "ORPHAN_TIMEOUT"),
        (11, "destroy", "CANCEL_ACK_OBSERVED"),
    ], (
        f"expected exactly the `violate_kw` (line 4), `violate_pos_first` "
        f"(line 7), and `violate_pos_second` (line 11) sites to be "
        f"flagged; got {flagged}"
    )
