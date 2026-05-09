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
    from app.domain.repositories.uow import IUnitOfWork


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
    from app.domain.models.cost_record import CostRecord
    from app.domain.services.cost_callback_handler import CostCallbackHandler
    from app.infrastructure.repositories.db_cost_record_repository import (
        DbCostRecordRepository,
    )

    async def _persist(record: CostRecord) -> None:
        async with uow_factory() as uow:
            repo = DbCostRecordRepository(uow.db_session)
            await repo.insert(record)
            # Commit explicitly here. ``DBUnitOfWork.__aexit__`` catches
            # commit/rollback failures and only logs (SSE-disconnect
            # scenarios rely on that swallow behavior) — but for the cost
            # ledger we NEED the failure to propagate so
            # ``CostCallbackHandler._persist_safely`` can count it as a
            # persist failure and write a degraded marker. If we relied on
            # __aexit__'s commit, a silent commit failure would leave the
            # ledger incomplete while the aggregate still reported ``actual``.
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

    return CostCallbackHandler(
        session_id=session_id, user_id=user_id, persister=_persist
    )


def build_supervisor_aware_callback_handler(
    supervisor,
    session_id: str,
    user_id: str,
    uow_factory: Callable[[], "IUnitOfWork"],
):
    """Build a cost handler that also mirrors LLM inflight state.

    The DB persister semantics intentionally match
    ``build_cost_callback_handler``: each write gets a short-lived UoW and
    commit failures propagate into ``CostCallbackHandler._persist_safely``.
    """
    from app.domain.models.cost_record import CostRecord
    from app.domain.services.cost_callback_handler import (
        SupervisorAwareCallbackHandler,
    )
    from app.infrastructure.repositories.db_cost_record_repository import (
        DbCostRecordRepository,
    )

    async def _persist(record: CostRecord) -> None:
        async with uow_factory() as uow:
            repo = DbCostRecordRepository(uow.db_session)
            await repo.insert(record)
            try:
                await uow.db_session.commit()
            except Exception:
                try:
                    await uow.db_session.rollback()
                except Exception:
                    pass
                raise

    return SupervisorAwareCallbackHandler(
        supervisor=supervisor,
        session_id=session_id,
        user_id=user_id,
        persister=_persist,
    )
