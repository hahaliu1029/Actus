"""B9 run-scoped liveness 注册表（spec §4）——纯内存零 IO，asyncio 单线程 dict 无锁。"""
from __future__ import annotations

import logging

from app.domain.models.runtime_extension import LivenessSnapshot

logger = logging.getLogger(__name__)


class RuntimeLivenessRegistry:
    def __init__(self) -> None:
        self._active: dict[tuple[str, str], set[str]] = {}
        self._degraded_run_ids: set[str] = set()
        self._known_runs: set[str] = set()

    def begin_run(self, run_id: str) -> None:
        self._known_runs.add(run_id)

    def end_run(self, run_id: str) -> None:
        """R14#1：无条件清 degraded + 所有 active set 中该 run + prune 空 key。"""
        self._known_runs.discard(run_id)
        self._degraded_run_ids.discard(run_id)
        for key in [k for k, runs in self._active.items() if run_id in runs]:
            self._active[key].discard(run_id)
            if not self._active[key]:
                del self._active[key]

    def acquire(self, kind: str, ext_id: str, run_id: str) -> None:
        self._active.setdefault((kind, ext_id), set()).add(run_id)

    def release(self, kind: str, ext_id: str, run_id: str) -> None:
        runs = self._active.get((kind, ext_id))
        if not runs:
            return
        runs.discard(run_id)
        if not runs:
            del self._active[(kind, ext_id)]

    def mark_degraded(self, run_id: str) -> None:
        self._degraded_run_ids.add(run_id)

    def snapshot(self) -> LivenessSnapshot:
        return LivenessSnapshot(
            active={k: frozenset(v) for k, v in self._active.items()},
            degraded=bool(self._degraded_run_ids),
        )


runtime_liveness_registry = RuntimeLivenessRegistry()
