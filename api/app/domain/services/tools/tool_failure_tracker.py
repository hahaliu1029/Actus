"""Per-signature tool failure tracking for ReAct loop protection.

Blocks repeated execution of the same tool with the same arguments after
``max_same_failures`` consecutive failures. Uses SHA-256:16 (64-bit) hash
of serialized args for signature matching.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field


@dataclass
class ToolFailureTracker:
    """Precise per-signature tool failure tracker.

    Injected via ``configurable``. Lives on ``PlannerReActFlow`` to survive
    across ``invoke()`` / ``resume()`` calls within a session.
    """

    max_same_failures: int = 3
    _failure_map: dict[str, int] = field(default_factory=dict)
    _blocked_signatures: set[str] = field(default_factory=set)

    @staticmethod
    def _signature(tool_name: str, args: dict) -> str:
        """Tool name + SHA-256:16 hash of serialized args."""
        try:
            digest = hashlib.sha256(
                json.dumps(args, sort_keys=True, default=str).encode()
            ).hexdigest()[:16]
            return f"{tool_name}:{digest}"
        except Exception:
            return tool_name

    def record_failure(self, tool_name: str, args: dict) -> bool:
        """Record a failure. Returns True if the signature is now blocked."""
        sig = self._signature(tool_name, args)
        self._failure_map[sig] = self._failure_map.get(sig, 0) + 1
        if self._failure_map[sig] >= self.max_same_failures:
            self._blocked_signatures.add(sig)
            return True
        return False

    def record_success(self, tool_name: str, args: dict) -> None:
        """Reset failure count for this signature on success."""
        sig = self._signature(tool_name, args)
        self._failure_map.pop(sig, None)
        self._blocked_signatures.discard(sig)

    def is_blocked(self, tool_name: str, args: dict) -> bool:
        return self._signature(tool_name, args) in self._blocked_signatures

    def get_blocked_summary(self) -> str:
        """Summary of blocked signatures for LLM context injection."""
        if not self._blocked_signatures:
            return ""
        sigs = ", ".join(sorted(self._blocked_signatures))
        return f"以下工具调用模式因连续失败已被暂停：{sigs}"

    def reset_blocked(self) -> None:
        """Clear blocked set at step boundary.

        Called when a new step begins (executor_node). Retains failure_map
        so that if the same signature fails again, it re-blocks quickly.
        """
        self._blocked_signatures.clear()
