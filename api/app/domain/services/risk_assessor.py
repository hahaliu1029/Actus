"""Risk assessment domain model for dangerous tool confirmation.

Provides RiskLevel ordering enum and RiskAssessment frozen dataclass
used throughout the tool approval workflow.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

from app.domain.services.tools.tool_source_resolver import (
    ToolSourceUnknownError,
    resolve_tool_source,
)


class RiskLevel(IntEnum):
    """Ordered risk levels for tool operations.

    Supports standard comparison operators and max() via IntEnum ordering.
    """

    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3


@dataclass(frozen=True)
class RiskAssessment:
    """Immutable risk assessment result for a single tool invocation.

    Attributes:
        tool_name: Name of the tool being assessed.
        tool_args: Arguments passed to the tool.
        static_level: Risk level determined by static pattern matching.
        dynamic_level: Risk level determined by dynamic/contextual analysis.
        final_level: Effective risk level (typically max of static and dynamic).
        risk_reason: Human-readable explanation of the assessed risk.
        matched_patterns: List of pattern identifiers that triggered the assessment.
        suggested_alternative: Optional safer alternative action, if applicable.
        primary_arg: The primary argument value extracted from tool_args (e.g. path or command).
        dir_arg: The directory argument, if relevant to the tool operation.
        arg_digest: A short digest (hash) of the tool arguments for deduplication.
    """

    tool_name: str
    tool_args: dict[str, Any]
    static_level: RiskLevel
    dynamic_level: RiskLevel
    final_level: RiskLevel
    risk_reason: str
    matched_patterns: list[str]
    suggested_alternative: str | None
    primary_arg: str
    dir_arg: str | None
    arg_digest: str


import re
import unicodedata

_ANSI_ESCAPE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


def normalize_command(command: str) -> str:
    """Normalize a shell command string for pattern matching.

    Applies ANSI escape strip, null byte removal, Unicode NFKC normalization,
    lowercase conversion, and leading/trailing whitespace strip.

    Args:
        command: Raw command string, possibly with escape codes or Unicode variants.

    Returns:
        Normalized command string ready for pattern matching.
    """
    command = _ANSI_ESCAPE.sub("", command)
    command = command.replace("\x00", "")
    command = unicodedata.normalize("NFKC", command)
    command = command.lower()
    command = command.strip()
    return command


# Each tuple: (regex_pattern, pattern_name, description)
DANGEROUS_PATTERNS: list[tuple[str, str, str]] = [
    (
        r"rm\s+(-[a-z]*r[a-z]*f[a-z]*|-[a-z]*f[a-z]*r[a-z]*)\s+",
        "recursive_delete",
        "Recursive force deletion (rm -rf)",
    ),
    (
        r"\brm\s+(-[^\s]*r|--(recursive))\b",
        "recursive_delete_no_f",
        "Recursive deletion without -f (rm -r or rm --recursive)",
    ),
    (
        r"mkfs(\.\w+)?\s+",
        "format_filesystem",
        "Filesystem format command (mkfs)",
    ),
    (
        r"\bdd\b.*\bof=/dev/[a-z]",
        "disk_copy",
        "Direct disk write via dd",
    ),
    (
        r"chmod\s+[0-7]*7\s+",
        "world_writable",
        "World-writable permission grant (chmod *7)",
    ),
    (
        r"\bdrop\s+table\b",
        "sql_drop",
        "SQL DROP TABLE statement",
    ),
    (
        r"\bdelete\s+from\s+\S+\s*(?:;|$)",
        "sql_delete_no_where",
        "SQL DELETE without WHERE clause",
    ),
    (
        r"(curl|wget)\s+\S+\s*\|\s*(ba?sh|sh|zsh|fish|dash)",
        "pipe_remote_to_shell",
        "Piping remote script directly to shell",
    ),
    (
        r">\s*/etc/",
        "overwrite_system_config",
        "Overwriting system configuration file in /etc/",
    ),
    (
        r"\bkill\s+-9\s+-1\b",
        "kill_all_processes",
        "Kill all processes (kill -9 -1)",
    ),
    (
        r":\s*\(\s*\)\s*\{.*:\s*\|.*:.*&.*\}",
        "fork_bomb",
        "Fork bomb pattern :(){ :|:& };:",
    ),
    (
        r"\bshutdown\b",
        "shutdown",
        "System shutdown command",
    ),
    (
        r"\breboot\b",
        "reboot",
        "System reboot command",
    ),
    (
        r"\bfind\b.*-delete\b",
        "find_delete",
        "find command with -delete flag",
    ),
    (
        r"\bpython\s+(-\w\s+)*-c\b",
        "python_exec",
        "inline Python execution",
    ),
    (
        r"\bsed\s+-i\b.*\s/etc/",
        "edit_system_config",
        "edit system config in-place",
    ),
    (
        r"\bpkill\s+-9\b",
        "kill_processes",
        "kill processes by name",
    ),
    (
        r"\b>\s*/dev/[sh]d[a-z]",
        "overwrite_disk",
        "overwrite disk device",
    ),
]

_COMPILED_PATTERNS: list[tuple[re.Pattern[str], str, str]] = [
    (re.compile(pattern), name, description)
    for pattern, name, description in DANGEROUS_PATTERNS
]


def match_dangerous_patterns(command: str) -> list[str]:
    """Check a command string against all dangerous patterns.

    Normalizes the command first, then returns a list of pattern names
    that matched.

    Args:
        command: Raw shell or SQL command string to evaluate.

    Returns:
        List of matched pattern name strings (may be empty if command is safe).
    """
    normalized = normalize_command(command)
    matched: list[str] = []
    for compiled, name, _ in _COMPILED_PATTERNS:
        if compiled.search(normalized):
            matched.append(name)
    return matched


# ---------------------------------------------------------------------------
# Static risk levels per tool name
# ---------------------------------------------------------------------------

_STATIC_RISK: dict[str, RiskLevel] = {
    # HIGH — can cause irreversible system-level damage
    "shell_execute": RiskLevel.HIGH,
    "browser_console_exec": RiskLevel.HIGH,
    # MEDIUM — modifies files, but scoped
    "file_write": RiskLevel.MEDIUM,
    "file_str_replace": RiskLevel.MEDIUM,
    # LOW — read/navigation only, no side effects
    "browser_click": RiskLevel.LOW,
    "browser_navigate": RiskLevel.LOW,
    # NONE — purely read / informational
    "file_read": RiskLevel.NONE,
    "browser_view": RiskLevel.NONE,
    "shell_read_output": RiskLevel.NONE,
    "message_ask_user": RiskLevel.NONE,
    "browser_scroll_up": RiskLevel.NONE,
    "browser_scroll_down": RiskLevel.NONE,
    "browser_console_view": RiskLevel.NONE,
}

# ---------------------------------------------------------------------------
# Per-tool argument extractors
# Returns (primary_arg, dir_arg) — dir_arg may be None
# ---------------------------------------------------------------------------

_EFFECT_ARGS_EXTRACTORS: dict[str, Any] = {
    "shell_execute": lambda args: (
        args.get("command", ""),
        args.get("exec_dir") or None,
    ),
    "browser_console_exec": lambda args: (
        args.get("javascript", ""),
        None,
    ),
    "file_write": lambda args: (
        args.get("filepath", ""),
        None,
    ),
    "file_str_replace": lambda args: (
        args.get("filepath", ""),
        None,
    ),
}


def _compute_arg_digest(primary_arg: str, dir_arg: str | None) -> str:
    """Compute a short, stable digest from the primary and directory arguments.

    Args:
        primary_arg: The main argument value (command, filepath, etc.).
        dir_arg: Optional directory argument.

    Returns:
        First 16 hex characters of the SHA-256 digest of the combined args.
    """
    raw = f"{dir_arg}::{primary_arg}" if dir_arg is not None else primary_arg
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Alternative suggestions keyed by dangerous pattern name
# ---------------------------------------------------------------------------

_ALTERNATIVES: dict[str, str] = {
    "recursive_delete": "Consider using 'trash' or 'rm' without -rf, or move files to a temporary directory first.",
    "recursive_delete_no_f": "Consider using 'trash' or moving files to a temporary directory before deleting recursively.",
    "format_filesystem": "Verify the target device before formatting. Consider using a backup tool instead.",
    "disk_copy": "Verify the output device (of=) carefully before running dd. Use dd with 'status=progress' for visibility.",
    "world_writable": "Use the minimum required permissions (e.g., chmod 755) instead of world-writable settings.",
    "sql_drop": "Consider renaming or backing up the table instead of dropping it permanently.",
    "sql_delete_no_where": "Add a WHERE clause to limit the rows affected, or use a transaction to allow rollback.",
    "pipe_remote_to_shell": "Download the script first, review it, then execute it explicitly.",
    "overwrite_system_config": "Back up the original file before overwriting system configuration.",
    "kill_all_processes": "Target specific process IDs instead of using kill -9 -1.",
    "fork_bomb": "This pattern is a fork bomb. Do not execute it.",
    "shutdown": "Ensure all work is saved before issuing a shutdown command.",
    "reboot": "Ensure all work is saved before rebooting.",
    "find_delete": "Preview with 'find ... -print' first, then add -delete only after confirming the results.",
}


# ---------------------------------------------------------------------------
# RiskAssessor
# ---------------------------------------------------------------------------


class RiskAssessor:
    """Stateless risk assessor for tool invocations.

    Combines static (per-tool-name) and dynamic (pattern-based) risk levels
    to produce a final RiskAssessment for a single tool call.
    """

    def assess(self, tool_name: str, tool_args: dict[str, Any]) -> RiskAssessment:
        """Assess the risk of invoking a tool with the given arguments.

        Args:
            tool_name: The registered name of the tool being invoked.
            tool_args: The arguments that will be passed to the tool.

        Returns:
            A frozen RiskAssessment capturing all risk-related metadata.
        """
        # --- Static level --------------------------------------------------
        # R1 CS1: use resolve_tool_source(...).category == "mcp" instead of
        # the historical tool_name.startswith("mcp__") check (double underscore
        # never matched Actus' single-underscore mcp_{server}_{tool} naming).
        # Use category (NOT source) so identity-only discovery meta-tools
        # (list_mcp_tools / get_mcp_tool, category="mcp discovery") stay at
        # NONE risk instead of escalating to MEDIUM.
        try:
            is_mcp_wrapper = resolve_tool_source(tool_name).category == "mcp"
        except ToolSourceUnknownError:
            # Defensive: assess() runs on every tool invocation. If a rogue
            # caller passes an unknown name, fall through to _STATIC_RISK
            # instead of crashing the step.
            is_mcp_wrapper = False
        if is_mcp_wrapper:
            static_level = RiskLevel.MEDIUM
        else:
            static_level = _STATIC_RISK.get(tool_name, RiskLevel.NONE)

        # --- Extract effect args -------------------------------------------
        extractor = _EFFECT_ARGS_EXTRACTORS.get(tool_name)
        if extractor is not None:
            raw_primary, raw_dir = extractor(tool_args)
        else:
            raw_primary = json.dumps(tool_args, ensure_ascii=False, sort_keys=True)
            raw_dir = None

        primary_arg = normalize_command(str(raw_primary))
        dir_arg = normalize_command(str(raw_dir)) if raw_dir is not None else None

        # --- Dynamic level (pattern matching) --------------------------------
        matched_patterns: list[str] = []
        dynamic_level = RiskLevel.NONE
        if static_level >= RiskLevel.MEDIUM:
            matched_patterns = match_dangerous_patterns(primary_arg)
            if matched_patterns:
                dynamic_level = RiskLevel.HIGH

        # --- Final level -----------------------------------------------------
        final_level = RiskLevel(max(static_level, dynamic_level))

        # --- Risk reason -----------------------------------------------------
        if matched_patterns:
            pattern_descriptions = [
                desc
                for _, name, desc in _COMPILED_PATTERNS
                if name in matched_patterns
            ]
            risk_reason = (
                f"Tool '{tool_name}' matched dangerous pattern(s): "
                + "; ".join(pattern_descriptions)
            )
        elif static_level > RiskLevel.NONE:
            risk_reason = (
                f"Tool '{tool_name}' has a static risk level of {static_level.name}."
            )
        else:
            risk_reason = f"Tool '{tool_name}' is considered safe (NONE risk)."

        # --- Suggested alternative -------------------------------------------
        suggested_alternative: str | None = None
        for pattern_name in matched_patterns:
            if pattern_name in _ALTERNATIVES:
                suggested_alternative = _ALTERNATIVES[pattern_name]
                break

        # --- Digest ----------------------------------------------------------
        arg_digest = _compute_arg_digest(primary_arg, dir_arg)

        return RiskAssessment(
            tool_name=tool_name,
            tool_args=tool_args,
            static_level=static_level,
            dynamic_level=dynamic_level,
            final_level=final_level,
            risk_reason=risk_reason,
            matched_patterns=matched_patterns,
            suggested_alternative=suggested_alternative,
            primary_arg=primary_arg,
            dir_arg=dir_arg,
            arg_digest=arg_digest,
        )
