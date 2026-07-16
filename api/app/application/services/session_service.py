import asyncio
import logging
from typing import TYPE_CHECKING, Callable, List, Literal, Optional, Type

from app.application.errors.exceptions import (
    ForbiddenError,
    NotFoundError,
    ServerRequestsError,
)
from app.domain.errors.sandbox_lifecycle import (
    SandboxAlreadyDestroyed,
    SandboxBindingMissing,
    SandboxLifecycleError,
    SessionFinalizedError,
    SessionSuspendedError,
    SessionUnboundError,
)
from app.domain.external.sandbox import SandboxHandle
from app.domain.external.task import Task
from app.domain.models.file import File
from app.domain.models.session import DestroyReason, Session

# from app.domain.repositories.session_repository import SessionRepository
from app.domain.repositories.uow import IUnitOfWork
from app.interfaces.schemas.session import FileReadResponse, ShellReadResponse
from core.config import get_settings

if TYPE_CHECKING:
    from app.application.services.sandbox_lifecycle_service import SandboxLifecycleService
    from app.domain.services.execution_supervisor import ExecutionSupervisor
    from app.infrastructure.external.memory.fs_reconciler import FsReconciler
    from core.config import Settings, SubagentLimitsConfig

logger = logging.getLogger(__name__)


class SessionService:
    """会话服务"""

    def __init__(
        self,
        uow_factory: Callable[[], IUnitOfWork],
        task_cls: Optional[Type[Task]] = None,
        sandbox_lifecycle_service: Optional["SandboxLifecycleService"] = None,
        fs_reconciler: Optional["FsReconciler"] = None,
        execution_supervisor: Optional["ExecutionSupervisor"] = None,
        *,
        subagent_limits: Optional["SubagentLimitsConfig"] = None,
        settings: Optional["Settings"] = None,
        mailbox_flag_reader: Optional[Callable[[], bool]] = None,
        # SPM PR-1c Task 17: provision-flow metrics singleton
        # (app.state.sandbox_provision_metrics). None → the vnc/takeover
        # trigger three-classification records are no-op'd (tests / legacy).
        sandbox_provision_metrics: object | None = None,
    ) -> None:
        """构造函数，完成会话服务初始化

        ``subagent_limits`` is the C1a spawn-cap config; when ``None`` we
        lazy-construct ``SubagentLimitsConfig()`` inside
        ``create_session_with_parent`` so env overrides
        (``ACTUS_MAX_SUBAGENT_DEPTH`` / ``ACTUS_MAX_DESCENDANTS_PER_ROOT``)
        still take effect for callers that don't go through the DI factory
        (test code, ad-hoc constructors).

        ``settings`` and ``mailbox_flag_reader`` are inert PR-6 retired
        constructor parameters. They were the C3 PR-4.5 hooks for the
        ``mailbox_supervisor_enabled`` feature flag (spec §11.2) that
        controlled the runtime choice between the legacy and mailbox
        subagent control planes. PR-6 (spec §11.7) retires the legacy
        plane and the §11.6 rollback runbook; every new subagent now
        gets ``subagent_control_plane='mailbox'`` unconditionally. The
        parameters are still accepted so existing DI wiring and test
        fixtures don't break, but their values are NOT consulted.
        """
        self._uow_factory = uow_factory
        self._uow = uow_factory()
        self._task_cls = task_cls
        self._lifecycle = sandbox_lifecycle_service
        self._fs_reconciler = fs_reconciler
        self._supervisor = execution_supervisor
        self._subagent_limits = subagent_limits
        # PR-6: kept for back-compat with callers that still pass these
        # kwargs; values are intentionally ignored by
        # ``create_session_with_parent``. Marked private + unused so a
        # future cleanup PR can drop them once all callers stop passing
        # them. Do NOT add new readers — the rollback path is gone.
        self._settings = settings
        self._mailbox_flag_reader = mailbox_flag_reader
        # SPM PR-1c Task 17: provision metrics for vnc/takeover trigger surfaces.
        self._provision_metrics = sandbox_provision_metrics

    def _record_provision(self, *, trigger: str, outcome: str) -> None:
        """SPM Task 17 — non-provisioner trigger three-classification emit
        (spec §5.2d). No-op when no metrics sink is injected. Side-effect-only:
        a metrics hiccup must never mask the real vnc/takeover error."""
        metrics = getattr(self, "_provision_metrics", None)
        if metrics is None:
            return
        try:
            from core.config import get_settings

            metrics.record_provision(
                mode=get_settings().sandbox_provision_mode,
                trigger=trigger,
                outcome=outcome,
            )
        except Exception:  # noqa: BLE001
            logger.warning("session_service provision metric emit failed", exc_info=True)

    async def create_session(self, user_id: str) -> Session:
        """创建一个空白的新任务会话"""
        logger.info("创建一个空白新任务会话")
        session = Session(title="新对话", user_id=user_id)
        async with self._uow:
            await self._uow.session.save(session)
        logger.info(f"成功创建一个新任务会话: {session.id}")
        # M1 PR-5B：fire-and-forget 触发 FsReconciler per-user walk。
        # 第二次起 reconciler 内部 _walked_users 缓存会短路，但首次会扫一次
        # user 目录清孤儿文件 / 重建缺失文件。walk 出错不能影响 session 创建——
        # session 入库已完成，reconciler 只是补数据视图一致性。
        if self._fs_reconciler is not None:
            self._spawn_fs_reconciler_walk(user_id)
        return session

    async def create_session_with_parent(
        self,
        user_id: str,
        *,
        parent_session_id: str,
        tool_filter_preset: Optional[str] = None,
        title: str | None = None,
        # C2 PR-3 §7.5 P0-3 — coordinator-step children carry the run/work-unit
        # ids in the same INSERT so the partial unique index
        # ``ux_sessions_coordinator_wu`` enforces idempotent dispatch.
        # Both kwargs must be supplied together or both omitted.
        coordinator_run_id: Optional[str] = None,
        work_unit_id: Optional[str] = None,
    ) -> Session:
        """C1a (PR-2): owner-checked + FOR UPDATE locked + spawn cap enforced.

        Phase 1 max_depth=1: the parent must itself be a root (``worker_type='root'``,
        ``parent_session_id IS NULL``). If callers want deeper trees later, expand
        ``MAX_SUBAGENT_DEPTH`` and replace the ``parent.parent_session_id is not None``
        guard with a walk-up-the-chain.

        Child sessions carry a non-null ``parent_session_id``; the frontend session
        selector filters them out of the main list. Parent FK is ondelete=RESTRICT —
        deleting the parent while children exist raises IntegrityError, which the API
        layer translates to 409.

        T12 / Phase 1 PR-X: ``tool_filter_preset`` is **required** for every
        child created via this method — closes the codex R1 P1 bypass where
        a child row could be persisted with ``tool_filter_preset = NULL``
        and then reconstructed on resume with no restriction. The value
        must match a key in
        ``app.domain.services.tool_filter_presets.TOOL_FILTER_PRESETS``; an
        unknown name surfaces at runtime via ``resolve_preset(...)``'s
        ``ValueError`` and the DB CHECK constraint
        ``ck_sessions_tool_filter_preset`` is the last-line defense.

        The DB also enforces ``ck_sessions_child_must_have_preset``
        (``parent_session_id IS NULL OR tool_filter_preset IS NOT NULL``);
        this app-level ValueError produces a cleaner error than the
        IntegrityError path.

        Does NOT trigger ``fs_reconciler`` walk: the parent ``create_session``
        already walked the user's memory directory, so the child can skip the
        redundant scan.
        """
        parent_id = parent_session_id
        if parent_id is None:
            raise ValueError(
                "create_session_with_parent: parent_session_id is required"
            )
        if tool_filter_preset is None:
            raise ValueError(
                "create_session_with_parent: tool_filter_preset is required "
                "for child sessions (T12 / Phase 1 PR-X)."
            )
        # C2 PR-3 §7.5 P0-3 — both coordinator ids must be supplied together
        # or both omitted. Forbids mid-state where only one is set.
        if (coordinator_run_id is None) != (work_unit_id is None):
            raise ValueError(
                "create_session_with_parent: coordinator_run_id and work_unit_id "
                "must be supplied together (got coordinator_run_id="
                f"{coordinator_run_id!r}, work_unit_id={work_unit_id!r})."
            )
        from app.domain.services.tool_filter_presets import TOOL_FILTER_PRESETS

        if tool_filter_preset not in TOOL_FILTER_PRESETS:
            raise ValueError(
                f"create_session_with_parent: unknown tool_filter_preset "
                f"{tool_filter_preset!r}. Known: {sorted(TOOL_FILTER_PRESETS)}."
            )

        from app.domain.services.subagent_limits import SpawnCapExceeded

        # Resolve runtime config: prefer DI-injected instance, fall back to
        # env-loaded default. ``SubagentLimitsConfig()`` reads
        # ``ACTUS_MAX_SUBAGENT_DEPTH`` / ``ACTUS_MAX_DESCENDANTS_PER_ROOT`` at
        # construction so non-DI callers (tests, scripts) still honor env
        # overrides without going through the FastAPI Depends graph.
        limits = self._subagent_limits
        if limits is None:
            from core.config import SubagentLimitsConfig

            limits = SubagentLimitsConfig()

        async with self._uow_factory() as uow:
            # SQL pushes (id, user_id) into WHERE so a cross-tenant parent_id
            # never acquires a row lock. Foreign-user collapses to None,
            # identical to a missing id — defeats ID enumeration via lock-timing.
            parent = await uow.session.lock_session_for_spawn(
                parent_id, user_id=user_id
            )
            if parent is None:
                raise NotFoundError(f"parent session {parent_id} not found")

            # C2-full S3 (PR-2) — depth-aware spawn gate (design §4.2). The
            # config validator (SubagentLimitsConfig le=2) caps the ceiling at
            # load time, so the old `!= 1 → NotImplementedError` guard is gone.
            #
            # Corrupt-lineage guard: the persisted `depth` is now the authority,
            # but root-ness historically had two signals (parent_session_id +
            # worker_type). Fail closed if the parent row's lineage signals
            # disagree, rather than trusting a corrupt depth (unreachable
            # post-backfill / INV-A1, but defends against a bad row).
            if (parent.parent_session_id is not None) != (parent.depth > 0):
                raise SpawnCapExceeded(
                    "depth", parent.depth + 1, limits.max_subagent_depth
                )

            child_depth = parent.depth + 1
            if child_depth > limits.max_subagent_depth:
                raise SpawnCapExceeded(
                    "depth", child_depth, limits.max_subagent_depth
                )

            root_id = (
                parent.root_session_id
                if parent.root_session_id is not None
                else parent.id
            )
            descendant_count = await uow.session.count_descendants(
                root_id, user_id=user_id, cap=limits.max_descendants_per_root,
            )
            if descendant_count >= limits.max_descendants_per_root:
                raise SpawnCapExceeded(
                    "descendants",
                    descendant_count,
                    limits.max_descendants_per_root,
                )

            # C3 PR-6 (spec §11.7) — legacy retired. Every new subagent
            # gets ``subagent_control_plane='mailbox'`` unconditionally.
            # The ``mailbox_supervisor_enabled`` feature flag and the
            # §11.6 rollback runbook are decommissioned; the PR-6
            # alembic migration ``c3pr6_retire_legacy_ctrl_plane``
            # rewrites any pre-existing ``legacy`` rows to ``mailbox``.
            control_plane: Literal["legacy", "mailbox"] = "mailbox"
            child = Session(
                user_id=user_id,
                parent_session_id=parent_id,
                worker_type="subagent",
                subagent_control_plane=control_plane,
                tool_filter_preset=tool_filter_preset,
                title=title or "新对话",
                # C2 PR-3 §7.5 P0-3 — atomic same-row write of coordinator lineage.
                coordinator_run_id=coordinator_run_id,
                work_unit_id=work_unit_id,
                # C2-full S3 (PR-1) — persisted lineage, set once from the
                # FOR-UPDATE-locked parent row (INV-A1). depth chains off the
                # parent; root_session_id resolves to the true tree root
                # (parent's root, or the parent itself when the parent is a root).
                depth=parent.depth + 1,
                root_session_id=(
                    parent.root_session_id
                    if parent.root_session_id is not None
                    else parent.id
                ),
            )
            await uow.session.save(child)
            logger.info(
                "成功创建子会话: %s (parent=%s, control_plane=%s)",
                child.id, parent_id, control_plane,
            )
            return child

    # ── C2 PR-3 §7.5 P0-3 — coordinator attempt bump (UoW-bounded) ─────────

    async def peek_coordinator_attempt(
        self, *, session_id: str, step_id: str,
    ) -> Optional[int]:
        """C2 PR-3 §7.5 P0-3 — READ ``coordinator_attempts[step_id]`` inside a UoW.

        UoW commit-on-exit is harmless here (SELECT only) but keeps the call
        symmetric with ``bump_coordinator_attempt``. Returns ``None`` when the
        row is missing OR the JSONB key has never been written.

        Called by ``dispatch_node`` BEFORE deciding whether to bump or
        rehydrate — see [r1 P0-2 fix] for the no-UoW-boundary regression.
        """
        async with self._uow_factory() as uow:
            return await uow.session.peek_coordinator_attempt(
                session_id=session_id, step_id=step_id,
            )

    async def bump_coordinator_attempt(
        self, *, session_id: str, step_id: str,
    ) -> int:
        """C2 PR-3 §7.5 P0-3 — atomic JSONB increment, committed via UoW.

        Returns the new attempt_ix (``>= 1``). Critical: the UoW commit fires
        on ``__aexit__`` so the new attempt counter is durable BEFORE
        ``dispatch_node`` creates any child session rows (which use this
        ``SessionService``'s own UoW for their INSERT). Without this commit
        ordering, a crash between bump and child-create would leave a stale
        counter that violates the PEEK-before-BUMP crash recovery contract.
        """
        async with self._uow_factory() as uow:
            return await uow.session.bump_coordinator_attempt(
                session_id=session_id, step_id=step_id,
            )

    def _spawn_fs_reconciler_walk(self, user_id: str) -> None:
        reconciler = self._fs_reconciler
        if reconciler is None:
            return

        async def _walk_swallowing_errors() -> None:
            try:
                await reconciler.walk_user_directory(user_id)
            except Exception:
                # 不能让 walk 的异常冒泡到 event loop 未处理 handler；reconciler
                # 只是对账，失败 log 了就接着干，不影响 session 生命周期。
                logger.warning(
                    "FsReconciler walk 失败（已吞） user=%s", user_id, exc_info=True
                )

        try:
            asyncio.create_task(_walk_swallowing_errors())
        except RuntimeError:
            # no running loop（理论上 FastAPI 内不会发生；单测环境才可能）
            logger.debug(
                "FsReconciler walk 未能调度（无 event loop），忽略 user=%s",
                user_id,
            )

    async def get_all_sessions(
        self, user_id: str, is_admin: bool = False
    ) -> List[Session]:
        """获取用户所有任务会话列表"""
        async with self._uow:
            if is_admin:
                return await self._uow.session.get_all()
            return await self._uow.session.get_all_by_user(user_id)

    async def clear_unread_message_count(
        self, session_id: str, user_id: str, is_admin: bool = False
    ) -> None:
        """清空指定会话未读消息数"""
        logger.info(f"清除会话[{session_id}]未读消息数")
        async with self._uow:
            await self._get_accessible_session(session_id, user_id, is_admin=is_admin)
            await self._uow.session.update_unread_message_count(session_id, 0)

    async def delete_session(
        self, session_id: str, user_id: str, is_admin: bool = False
    ) -> None:
        """根据传递的会话id删除任务会话

        Eng review #10: destroy first (quiesce), then hard delete row.
        """
        # 1.检查会话是否存在
        logger.info(f"正在删除会话, 会话id: {session_id}")
        async with self._uow:
            session = await self._get_accessible_session(
                session_id, user_id, is_admin=is_admin
            )
            descendants = await self._uow.session.find_descendants(
                session_id,
                user_id=session.user_id or user_id,
                max_depth=32,
                limit=1000,
            )

        delete_targets = list(reversed(descendants)) + [session]

        # 2.清理会话关联的运行态资源（任务/容器）
        for target in delete_targets:
            await self._cleanup_task(target.task_id)

        # 3. Lifecycle-managed sandbox destroy (I6: quiesce barrier)
        destroy_error: SandboxLifecycleError | None = None
        for target in delete_targets:
            if self._lifecycle:
                try:
                    await self._lifecycle.destroy(target.id, DestroyReason.SESSION_DELETE)
                except (SandboxAlreadyDestroyed, SandboxBindingMissing) as e:
                    # C3 PR-1 (spec §3.2 M2 + §7.3): terminal-success signals
                    # — sandbox already gone, treat as no-op for delete path.
                    logger.debug(
                        "Sandbox already terminal for session %s during delete: %s",
                        target.id,
                        e,
                    )
                except SandboxLifecycleError as e:
                    # C3 PR-1 (codex P1 round 1 + round 13 P2): non-terminal sandbox
                    # teardown failure (docker daemon down, network blip, generation
                    # mismatch, etc.) must NOT proceed to DB delete —
                    # reconcile_orphans needs the session row to find the still-bound
                    # sandbox on retry. Capture the error and re-raise AFTER the
                    # background slot cleanup runs: the slot cleanup is idempotent
                    # and a sandbox teardown failure should not strand the Redis
                    # quota slot until TTL, blocking future background-task quota.
                    logger.warning(
                        "Sandbox lifecycle destroy failed for session %s — aborting delete",
                        target.id,
                        exc_info=True,
                    )
                    destroy_error = destroy_error or e

            # Background slot cleanup is idempotent — always run so a sandbox
            # destroy failure doesn't strand the Redis quota slot.
            await self._cleanup_background_slot_if_needed(
                target,
                reason="session_delete",
            )

        if destroy_error is not None:
            # Re-raise the original SandboxLifecycleError so the HTTP DELETE
            # handler returns 5xx and the session row stays for the next
            # reconcile pass.
            raise destroy_error

        # 4.根据传递的会话id删除会话
        async with self._uow:
            for target in delete_targets:
                await self._uow.session.delete_by_id(target.id)
        logger.info(f"删除会话[{session_id}]成功")

    async def _cleanup_background_slot_if_needed(
        self,
        session: Session,
        *,
        reason: str,
    ) -> None:
        if (
            session.execution_mode != "background"
            and not getattr(session, "was_background", False)
        ):
            return
        if not getattr(session, "user_id", None):
            return
        if self._supervisor is None:
            return
        try:
            await self._supervisor.cleanup_background_slot(
                session_id=session.id,
                user_id=str(session.user_id),
                reason=reason,
            )
        except Exception:
            logger.warning(
                "background slot cleanup failed for deleted session %s",
                session.id,
                exc_info=True,
            )

    async def _cleanup_task(self, task_id: Optional[str]) -> None:
        """清理会话关联任务，避免删除会话后后台任务继续运行。"""
        if not task_id or self._task_cls is None:
            return

        try:
            task = self._task_cls.get(task_id)
            if not task:
                logger.info(f"会话任务[{task_id}]不存在或已结束，无需清理")
                return
            task.cancel(reason="session_delete")
            logger.info(f"会话任务[{task_id}]已取消")
        except Exception as e:
            logger.warning(f"清理会话任务[{task_id}]失败: {e}")

    async def _acquire_sandbox(
        self,
        session_id: str,
        *,
        resume_if_suspended: bool = False,
    ) -> SandboxHandle:
        """Acquire a sandbox handle for a session via lifecycle service.

        Raises NotFoundError/ServerRequestsError with user-friendly message
        for all lifecycle error states.
        """
        if not self._lifecycle:
            raise ServerRequestsError("Sandbox lifecycle service not available")
        try:
            return await self._lifecycle.acquire(session_id)
        except SessionUnboundError:
            raise NotFoundError("当前会话无沙箱环境")
        except SessionSuspendedError:
            if resume_if_suspended:
                return await self._lifecycle.resume(session_id)
            raise NotFoundError("当前会话沙箱不存在或已销毁")
        except SessionFinalizedError:
            raise NotFoundError("当前会话沙箱不存在或已销毁")
        except SandboxLifecycleError:
            # Catch-all for SessionCreatingError, SessionDestroyingError, etc.
            raise ServerRequestsError("沙箱正在初始化或销毁中，请稍后重试")

    async def download_file(
        self,
        session_id: str,
        filepath: str,
        user_id: str,
        is_admin: bool = False,
    ) -> "BinaryIO":
        """通过 session + filepath 直接从沙箱下载文件"""
        from typing import BinaryIO

        async with self._uow:
            await self._get_accessible_session(session_id, user_id, is_admin)

        handle = await self._acquire_sandbox(session_id, resume_if_suspended=True)
        return await handle.download_file(filepath)

    async def get_session(
        self, session_id: str, user_id: str, is_admin: bool = False
    ) -> Session:
        """获取指定会话详情信息"""
        async with self._uow:
            return await self._get_accessible_session(session_id, user_id, is_admin)

    async def get_session_files(
        self, session_id: str, user_id: str, is_admin: bool = False
    ) -> List[File]:
        """根据传递的会话id获取指定会话的文件列表信息"""
        logger.info(f"获取指定会话[{session_id}]下的文件列表信息")
        async with self._uow:
            session = await self._get_accessible_session(session_id, user_id, is_admin)
        return session.files

    async def _get_accessible_session(
        self, session_id: str, user_id: str, is_admin: bool = False
    ) -> Session:
        """获取指定用户可访问的会话（需在 uow 上下文中调用）"""
        session = await self._uow.session.get_by_id(session_id)
        if not session:
            logger.error(f"会话[{session_id}]不存在")
            raise NotFoundError("会话不存在")
        if not is_admin and session.user_id != user_id:
            logger.error(f"会话[{session_id}]无权访问")
            raise ForbiddenError("无权访问此会话")
        return session

    async def read_file(
        self, session_id: str, filepath: str, user_id: str, is_admin: bool = False
    ) -> FileReadResponse:
        """根据传递的信息查看会话中指定文件的内容"""
        # 1.检查会话是否存在
        logger.info(f"获取会话[{session_id}]中的文件内容, 文件路径: {filepath}")
        async with self._uow:
            await self._get_accessible_session(session_id, user_id, is_admin)

        # 2.通过 lifecycle service 获取沙箱 handle
        handle = await self._acquire_sandbox(session_id, resume_if_suspended=True)

        # 3.调用沙箱读取文件内容
        result = await handle.read_file(filepath)
        if result.success:
            return FileReadResponse(**result.data)

        raise ServerRequestsError(result.message)

    async def read_shell_output(
        self,
        session_id: str,
        shell_session_id: str,
        user_id: str,
        is_admin: bool = False,
    ) -> ShellReadResponse:
        """根据传递的任务会话id+Shell会话id获取Shell执行结果"""
        # 1.检查会话是否存在
        logger.info(
            f"获取会话[{session_id}]中的Shell内容输出, Shell标识符: {shell_session_id}"
        )
        async with self._uow:
            await self._get_accessible_session(session_id, user_id, is_admin)

        # 2.通过 lifecycle service 获取沙箱 handle
        handle = await self._acquire_sandbox(session_id)

        # 3.调用沙箱查看shell内容
        result = await handle.read_shell_output(
            session_id=shell_session_id, console=True
        )
        if result.success:
            return ShellReadResponse(**result.data)

        raise ServerRequestsError(result.message)

    async def get_vnc_url(
        self, session_id: str, user_id: str, is_admin: bool = False
    ) -> str:
        """获取指定会话的vnc链接"""
        # 1.检查会话是否存在
        logger.info(f"获取会话[{session_id}]的VNC链接")
        async with self._uow:
            session = await self._get_accessible_session(session_id, user_id, is_admin)

        if not self._lifecycle:
            raise ServerRequestsError("Sandbox lifecycle service not available")

        # 2. 获取或创建沙箱（I5: UNBOUND → bind_new, ACTIVE → acquire, SUSPENDED → resume）
        # 注意：_get_accessible_session 允许管理员访问他人 session，所以**不能**
        # 把 `user_id`（请求者身份）透传给 bind_new——否则管理员打开他人 VNC
        # 会把自己的 memory 挂进 session owner 的 sandbox。让 bind_new 走内部
        # 的 session.user_id 回退拿到真正的 session owner。
        # SPM PR-1c Task 17 (§5.2d): non-provisioner trigger three-classification.
        # The acquire→bind_new/resume branching is NORMAL routing (not failure);
        # only exceptions that ESCAPE it count as failed/cancelled.
        try:
            try:
                handle = await self._lifecycle.acquire(session_id)
            except SessionUnboundError:
                handle = await self._lifecycle.bind_new(session_id)
            except SessionSuspendedError:
                handle = await self._lifecycle.resume(session_id)
        except asyncio.CancelledError:
            self._record_provision(trigger="vnc", outcome="cancelled")
            raise
        except Exception:
            self._record_provision(trigger="vnc", outcome="failed")
            raise
        self._record_provision(trigger="vnc", outcome="ok")

        return handle.vnc_url

    async def ensure_takeover_shell_session(
        self,
        session_id: str,
        takeover_id: str,
        user_id: str,
        is_admin: bool = False,
    ) -> tuple[SandboxHandle, str]:
        """确保接管终端会话存在并返回沙箱 handle 与 shell 会话ID。"""
        logger.info("确保会话[%s]接管终端可用，takeover_id=%s", session_id, takeover_id)
        async with self._uow:
            await self._get_accessible_session(session_id, user_id, is_admin)

        if not self._lifecycle:
            raise ServerRequestsError("Sandbox lifecycle service not available")

        # 获取或创建沙箱
        # 同 get_vnc_url：admin takeover 不能用 requester user_id，
        # 交给 bind_new 内部的 session.user_id 回退。
        # SPM PR-1c Task 17 (§5.2d): trigger="takeover" three-classification.
        try:
            try:
                handle = await self._lifecycle.acquire(session_id)
            except SessionUnboundError:
                handle = await self._lifecycle.bind_new(session_id)
            except SessionSuspendedError:
                handle = await self._lifecycle.resume(session_id)
        except asyncio.CancelledError:
            self._record_provision(trigger="takeover", outcome="cancelled")
            raise
        except Exception:
            self._record_provision(trigger="takeover", outcome="failed")
            raise
        self._record_provision(trigger="takeover", outcome="ok")

        shell_session_id = f"takeover_{session_id}_{takeover_id}"
        probe_result = await handle.read_shell_output(
            session_id=shell_session_id,
            console=False,
        )
        if not probe_result.success:
            sandbox_home = get_settings().sandbox_default_cwd or "/root"
            start_result = await handle.exec_command(
                session_id=shell_session_id,
                exec_dir=sandbox_home,
                command="bash -i",
            )
            if not start_result.success:
                raise ServerRequestsError(start_result.message)

        return handle, shell_session_id
