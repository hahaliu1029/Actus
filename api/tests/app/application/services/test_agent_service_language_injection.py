"""#29: verify ``AgentService._create_task`` derives the right
``initial_language`` from each session input and passes it to
``AgentTaskRunner``.

Drives real ``_create_task`` execution with minimal mocks for the
ambient dependencies and patches ``AgentTaskRunner`` to capture
constructor kwargs per parametrized session input branch.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.domain.models.event import MessageEvent, PlanEvent, PlanEventStatus
from app.domain.models.plan import Plan, Step
from app.domain.models.session import Session

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _make_session(events: list) -> Session:
    """Build a minimal Session with the given events list.

    Session's Pydantic model provides defaults for id (uuid factory),
    user_id (None), etc., so events is the only field we need to set.
    """
    return Session(events=events)


def _make_plan(language: str) -> Plan:
    return Plan(
        title="Test",
        goal="do the thing",
        language=language,
        steps=[Step(description="one step")],
        message="",
    )


@pytest.fixture
def skeletal_service_with_captured_runner(monkeypatch):
    """Build a skeletal ``AgentService`` that can run ``_create_task``
    up to the ``AgentTaskRunner(...)`` call, and patch the runner to
    capture its constructor kwargs.

    Returns ``(service, captured_kwargs)`` where ``captured_kwargs`` is
    populated after ``service._create_task(session)`` is awaited.
    """
    from app.application.services.agent_service import AgentService

    svc = AgentService.__new__(AgentService)

    # --- sandbox chain (called when session.sandbox_id is None) ---
    fake_sandbox = MagicMock()
    fake_sandbox.id = "sbx-1"
    fake_sandbox.get_browser = AsyncMock(return_value=MagicMock())
    svc._sandbox_cls = MagicMock()
    svc._sandbox_cls.get = AsyncMock(return_value=None)
    svc._sandbox_cls.create = AsyncMock(return_value=fake_sandbox)

    # --- uow_factory: session.save(...) after sandbox creation ---
    fake_uow = AsyncMock()
    fake_uow.__aenter__ = AsyncMock(return_value=fake_uow)
    fake_uow.__aexit__ = AsyncMock(return_value=False)
    fake_uow.session = AsyncMock()
    fake_uow.session.save = AsyncMock()
    svc._uow_factory = MagicMock(return_value=fake_uow)

    # --- other deps the runner constructor reads ---
    svc._file_storage = MagicMock()
    svc._search_engine = MagicMock()
    svc._checkpointer_pool = MagicMock()
    svc._memory_flusher = MagicMock()
    svc._memory_embedding_provider = MagicMock()
    svc._memory_session_factory = MagicMock()
    svc._memory_repo_factory = MagicMock()
    svc._memory_write_service = None
    svc._memory_session_save_cap = 20
    # PR-4+8: _create_task now passes gate deps to AgentTaskRunner.
    # Stub all four to None so the path doesn't hit AttributeError; the
    # captured kwargs verify the contract, but this test only asserts on
    # ``initial_language`` so the actual gate values don't matter.
    svc._memory_gate_breaker = None
    svc._memory_gate_daily_cap = None
    svc._memory_notification_emitter = None

    # Skip file_processor_lookup / approval_cache / confirmation_manager
    # side paths — all are guarded on config being populated.
    svc._redis_client = None
    svc._confirmation_manager = None
    svc._sandbox_lifecycle_service = None  # PR1: lifecycle service not needed for this test
    svc._supervisor = MagicMock()

    # --- config snapshot (file_understanding_config=None skips the whole
    # file_processor_lookup branch) ---
    snap = MagicMock()
    snap.file_understanding_config = None
    snap.supports_vision = False
    snap.supports_pdf_input = False
    snap.vision_fallback_model = None
    svc._config_snapshot = snap

    # --- task_cls.create(...) after the runner is built ---
    fake_task = MagicMock()
    fake_task.id = "task-1"
    svc._task_cls = MagicMock()
    svc._task_cls.create = MagicMock(return_value=fake_task)

    # --- the capture itself ---
    captured: dict = {}

    def _capture(*args, **kwargs):
        del args  # positional args unused
        captured.update(kwargs)
        return MagicMock()

    monkeypatch.setattr(
        "app.application.services.agent_service.AgentTaskRunner",
        _capture,
    )

    svc._captured_runner_kwargs = captured  # type: ignore[attr-defined]
    return svc


@pytest.mark.parametrize(
    "events_factory, expected_language",
    [
        # #29 test 3: brand-new session, no events
        (lambda: [], "zh"),
        # #29 test 4: single PlanEvent with language='en'
        (
            lambda: [
                PlanEvent(
                    plan=_make_plan("en"),
                    status=PlanEventStatus.CREATED,
                ),
            ],
            "en",
        ),
        # #29 test 5: two PlanEvents, latest wins (zh → en)
        (
            lambda: [
                PlanEvent(
                    plan=_make_plan("zh"),
                    status=PlanEventStatus.CREATED,
                ),
                PlanEvent(
                    plan=_make_plan("en"),
                    status=PlanEventStatus.UPDATED,
                ),
            ],
            "en",
        ),
        # #29 test 6: latest PlanEvent has empty-string language → "zh"
        (
            lambda: [
                PlanEvent(
                    plan=_make_plan(""),
                    status=PlanEventStatus.CREATED,
                ),
            ],
            "zh",
        ),
        # #29 test 7: session has events but none are PlanEvent → "zh"
        (
            lambda: [
                MessageEvent(role="user", message="hello"),
                MessageEvent(role="assistant", message="hi"),
            ],
            "zh",
        ),
    ],
    ids=[
        "brand_new_session",
        "latest_plan_en",
        "latest_plan_wins_over_older",
        "empty_plan_language_falls_back",
        "no_plan_events_only_messages",
    ],
)
async def test_create_task_passes_correct_initial_language(
    skeletal_service_with_captured_runner,
    events_factory,
    expected_language,
) -> None:
    """#29 tests 3-7: for each session branch, ``_create_task`` must
    compute the right ``initial_language`` and pass it to the runner.

    This is the ONLY test that actually exercises the computation inside
    ``_create_task`` (not a test-file replay of the expression). It
    catches the "implementer hardcodes 'zh' literally" class of bugs
    that the pure-logic-plus-AST design from earlier plan drafts missed.
    """
    service = skeletal_service_with_captured_runner
    captured = service._captured_runner_kwargs
    session = _make_session(events=events_factory())

    await service._create_task(session)

    assert captured.get("initial_language") == expected_language, (
        f"_create_task should have passed initial_language="
        f"{expected_language!r} for this session input; "
        f"got {captured.get('initial_language')!r}"
    )
