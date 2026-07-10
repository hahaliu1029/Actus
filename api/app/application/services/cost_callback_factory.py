"""B4 M0: DB-backed CostCallbackHandler factory.

Lives in the application layer so both ``agent_service`` (which already holds
a ``uow_factory``) and the HTTP layer (``service_dependencies``) can build a
handler without a reverse-layer import from ``interfaces``.

Each persist opens its own short-lived UoW so a failing write can't take down
the LLM call path and session leaks are bounded per call.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from app.domain.models.cost_record import CostRecord
    from app.domain.repositories.uow import IUnitOfWork
    from app.domain.services.cost_callback_handler import Persister

# The two cost_records parent-row FKs (naming_convention
# ``fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s``,
# infrastructure/models/base.py:9). A 23503 against either means a parent
# row this handler references was deleted — the handler's session_id /
# user_id are fixed for its lifetime, so the FK can never be satisfied
# again (sessions.user_id is SET NULL on user delete, not CASCADE; the
# session row may survive a user delete, but the cost INSERT still can't
# land). Same "row can never land again" class either way.
_PARENT_GONE_CONSTRAINTS: frozenset[str] = frozenset(
    {
        "fk_cost_records_session_id_sessions",
        "fk_cost_records_user_id_users",
    }
)


def _is_parent_row_gone_violation(exc: BaseException) -> bool:
    """True iff ``exc`` is a Postgres FK violation (23503) against one of
    cost_records' parent-row FKs.

    Walks the ``orig``/``__cause__`` chain (SQLAlchemy wraps the asyncpg
    error behind a dialect adapter) checking ``sqlstate``/``pgcode`` — the
    same idiom as memory_management_service's 23505 mapping. The constraint
    is matched via the ``constraint_name`` diag attribute when the driver
    exposes it, falling back to the (naming_convention-stable) constraint
    name embedded in the error text.
    """
    node: BaseException | None = getattr(exc, "orig", None)
    fk_code = False
    named = False
    depth = 0
    while node is not None and depth < 10:
        pgcode = getattr(node, "sqlstate", None) or getattr(node, "pgcode", None)
        if pgcode == "23503":
            fk_code = True
        if getattr(node, "constraint_name", None) in _PARENT_GONE_CONSTRAINTS:
            named = True
        node = node.__cause__
        depth += 1
    if not fk_code:
        return False
    if named:
        return True
    text = str(exc)
    return any(name in text for name in _PARENT_GONE_CONSTRAINTS)


def _make_db_persister(
    uow_factory: Callable[[], "IUnitOfWork"],
) -> "Persister":
    """Shared DB persister for both handler builders.

    Semantics (kept identical across builders on purpose):

    - Each persist opens a fresh short-lived UoW.
    - Commit is explicit and failures propagate: ``DBUnitOfWork.__aexit__``
      catches commit/rollback failures and only logs (SSE-disconnect
      scenarios rely on that swallow behavior) — but for the cost ledger we
      NEED the failure to reach ``CostCallbackHandler._persist_safely`` so
      it can count it and write a degraded marker. If we relied on
      __aexit__'s commit, a silent commit failure would leave the ledger
      incomplete while the aggregate still reported ``actual``.
    - A parent-row FK violation (session deleted while the task was still
      finishing) is translated to ``SessionRowGoneError`` so the handler
      skips instead of chasing the same dead FK with a degraded marker.
    """
    from app.domain.services.cost_callback_handler import SessionRowGoneError
    from app.infrastructure.repositories.db_cost_record_repository import (
        DbCostRecordRepository,
    )

    async def _persist(record: CostRecord) -> None:
        try:
            async with uow_factory() as uow:
                repo = DbCostRecordRepository(uow.db_session)
                await repo.insert(record)
                try:
                    await uow.db_session.commit()
                except Exception:
                    # Roll back any pending state so the UoW's __aexit__
                    # cleanup is clean, then re-raise.
                    try:
                        await uow.db_session.rollback()
                    except Exception:
                        pass
                    raise
        except Exception as exc:
            if _is_parent_row_gone_violation(exc):
                raise SessionRowGoneError(
                    f"cost_records parent row gone for "
                    f"session_id={record.session_id}"
                ) from exc
            raise

    return _persist


def build_cost_callback_handler(
    session_id: str,
    user_id: str,
    uow_factory: Callable[[], "IUnitOfWork"],
):
    """Build a session-scoped CostCallbackHandler with a DB-backed persister.

    Consumers attach the returned handler to LangGraph/LLM invoke config::

        handler = build_cost_callback_handler(session_id, user_id, uow_factory)
        await graph.ainvoke(state, config={"callbacks": [handler]})
        await handler.flush_pending()  # on FINISHING drain
    """
    from app.domain.services.cost_callback_handler import CostCallbackHandler

    return CostCallbackHandler(
        session_id=session_id,
        user_id=user_id,
        persister=_make_db_persister(uow_factory),
    )


def build_supervisor_aware_callback_handler(
    supervisor,
    session_id: str,
    user_id: str,
    uow_factory: Callable[[], "IUnitOfWork"],
):
    """Build a cost handler that also mirrors LLM inflight state.

    The DB persister semantics intentionally match
    ``build_cost_callback_handler`` — both builders share
    ``_make_db_persister``.
    """
    from app.domain.services.cost_callback_handler import (
        SupervisorAwareCallbackHandler,
    )

    return SupervisorAwareCallbackHandler(
        supervisor=supervisor,
        session_id=session_id,
        user_id=user_id,
        persister=_make_db_persister(uow_factory),
    )
