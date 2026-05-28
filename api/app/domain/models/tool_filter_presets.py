"""Tool-filter preset string constants.

Single source of truth for the ``"coordinator_step"`` contract string that
spans:

* ``Session.tool_filter_preset`` (domain model ``Literal`` — value side)
* ``parallel_execution_subgraph._dispatch_node`` (writer — creates child
  sessions with this preset)
* ``MailboxSupervisor`` terminal gates (cost rollup / persist terminal /
  cancel-ack persist) — readers
* ``CoordinatorProgressUpdateHandler`` (PROGRESS_UPDATE gate) — reader

A typo at any consumer site silently fails the gate (handler falls
through to stub-forward; supervisor gates skip their best-effort
prologues), with no compile-time or test-time signal. Centralising the
literal here means a typo becomes ``ImportError``/``AttributeError`` at
module load.

The runtime string value MUST remain ``"coordinator_step"`` — persisted
Session rows + the DB CHECK constraint
(``ck_sessions_tool_filter_preset_valid``) and the migration history
(C2 PR-1 Task 1.7) all encode this literal.

See also ``app.domain.services.tool_filter_presets`` for the runtime
allowlist mapping keyed by the same string.
"""
from __future__ import annotations

from typing import Final


COORDINATOR_STEP_PRESET: Final[str] = "coordinator_step"
