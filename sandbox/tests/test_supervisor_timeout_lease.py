"""Sandbox supervisor exact-reset cleanup lease tests."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.interfaces.errors.exceptions import BadRequestException
from app.services import supervisor as supervisor_module
from app.services.supervisor import SupervisorService


class _ClockDateTime(datetime):
    current = datetime(2026, 7, 15, 10, 0, 0)

    @classmethod
    def now(cls, tz=None):  # noqa: ANN001
        del tz
        return cls.current


@pytest.fixture
def service(monkeypatch) -> SupervisorService:
    monkeypatch.setattr(supervisor_module, "datetime", _ClockDateTime)
    monkeypatch.setattr(
        supervisor_module,
        "get_settings",
        lambda: SimpleNamespace(server_timeout_minutes=None),
    )
    monkeypatch.setattr(SupervisorService, "_connect_rpc", lambda self: None)
    instance = SupervisorService()
    instance._setup_timer = MagicMock()
    return instance


async def test_reset_timeout_is_exact_and_increments_generation(
    service: SupervisorService,
) -> None:
    first = await service.reset_timeout(10)

    assert service.shutdown_time == _ClockDateTime.current + timedelta(minutes=10)
    assert first.timeout_minutes == 10
    assert service._timeout_generation == 1

    _ClockDateTime.current += timedelta(minutes=4)
    second = await service.reset_timeout(10)

    assert service.shutdown_time == _ClockDateTime.current + timedelta(minutes=10)
    assert second.timeout_minutes == 10
    assert service._timeout_generation == 2


async def test_stale_generation_never_shutdowns_and_current_fires_once(
    service: SupervisorService,
) -> None:
    await service.reset_timeout(10)
    stale_generation = service._timeout_generation
    await service.reset_timeout(10)
    current_generation = service._timeout_generation
    service.shutdown = AsyncMock()
    _ClockDateTime.current = service.shutdown_time

    await service._handle_timeout_expiry(stale_generation)
    await service._handle_timeout_expiry(current_generation)
    await service._handle_timeout_expiry(current_generation)

    service.shutdown.assert_awaited_once()


async def test_cancelled_old_timer_callback_cannot_beat_rapid_reset(
    monkeypatch,
    service: SupervisorService,
) -> None:
    real_sleep = asyncio.sleep
    sleep_gates = []

    async def cancellation_ignoring_sleep(delay_seconds):
        gate = asyncio.Event()
        sleep_gates.append((delay_seconds, gate))
        try:
            await gate.wait()
        except asyncio.CancelledError:
            # Model a timer whose cancellation raced with wakeup and whose
            # callback still executes. Generation must be the final guard.
            return

    monkeypatch.setattr(supervisor_module.asyncio, "sleep", cancellation_ignoring_sleep)
    service._setup_timer = SupervisorService._setup_timer.__get__(service)
    service.shutdown = AsyncMock()

    await service.reset_timeout(10)
    await real_sleep(0)
    await service.reset_timeout(10)
    await real_sleep(0)

    assert len(sleep_gates) == 2
    service.shutdown.assert_not_awaited()

    _ClockDateTime.current = service.shutdown_time
    sleep_gates[1][1].set()
    await real_sleep(0)
    await real_sleep(0)

    service.shutdown.assert_awaited_once()


async def test_current_generation_waking_early_reschedules_remaining_window(
    service: SupervisorService,
) -> None:
    await service.reset_timeout(10)
    generation = service._timeout_generation
    service._setup_timer.reset_mock()
    service.shutdown = AsyncMock()
    _ClockDateTime.current += timedelta(minutes=4)

    await service._handle_timeout_expiry(generation)

    service.shutdown.assert_not_awaited()
    service._setup_timer.assert_called_once()
    delay_minutes = service._setup_timer.call_args.args[0]
    assert delay_minutes == pytest.approx(6)
    assert service._setup_timer.call_args.kwargs == {"generation": generation}


async def test_cancel_invalidates_even_an_already_waking_timer(
    service: SupervisorService,
) -> None:
    await service.reset_timeout(10)
    cancelled_generation = service._timeout_generation
    service.shutdown = AsyncMock()

    await service.cancel_timeout()
    await service._handle_timeout_expiry(cancelled_generation)

    assert service._timeout_generation > cancelled_generation
    service.shutdown.assert_not_awaited()


async def test_manual_extend_remains_additive_to_existing_deadline(
    service: SupervisorService,
) -> None:
    await service.reset_timeout(10)
    original_deadline = service.shutdown_time
    _ClockDateTime.current += timedelta(minutes=4)

    await service.extend_timeout(3)

    assert service.shutdown_time == original_deadline + timedelta(minutes=3)


@pytest.mark.parametrize("minutes", [0, -1, -10, True])
async def test_reset_rejects_non_positive_or_boolean_minutes(
    service: SupervisorService,
    minutes,
) -> None:
    with pytest.raises(BadRequestException):
        await service.reset_timeout(minutes)


async def test_reset_none_uses_default_window(monkeypatch, service) -> None:
    monkeypatch.setattr(
        supervisor_module,
        "get_settings",
        lambda: SimpleNamespace(server_timeout_minutes=60),
    )

    result = await service.reset_timeout()

    assert result.timeout_minutes == 60
    assert service.shutdown_time == _ClockDateTime.current + timedelta(minutes=60)


def test_timer_uses_thread_fallback_when_no_event_loop_is_running(
    monkeypatch,
    service: SupervisorService,
) -> None:
    class _DormantLoop:
        def create_task(self, coroutine):
            coroutine.close()
            return MagicMock()

    started = []

    class _FakeTimer:
        def __init__(self, interval, callback):
            self.interval = interval
            self.callback = callback
            self.daemon = False

        def start(self):
            started.append(self.interval)

        def cancel(self):
            return None

    monkeypatch.setattr(supervisor_module.asyncio, "get_event_loop", _DormantLoop)
    monkeypatch.setattr(
        supervisor_module.asyncio,
        "get_running_loop",
        MagicMock(side_effect=RuntimeError("no running loop")),
    )
    monkeypatch.setattr(supervisor_module.threading, "Timer", _FakeTimer)
    service.timeout_active = True
    service.shutdown_time = _ClockDateTime.current + timedelta(minutes=5)
    service._timeout_generation = 1
    service.shutdown_task = None
    service.shutdown_timer = None

    SupervisorService._setup_timer(service, 5, generation=1)

    assert started == [300]


def test_thread_timer_waking_early_reschedules_on_thread_path_and_fires_once(
    monkeypatch,
    service: SupervisorService,
) -> None:
    timers = []

    class _FakeTimer:
        def __init__(self, interval, callback):
            self.interval = interval
            self.callback = callback
            self.daemon = False
            self.cancelled = False

        def start(self):
            timers.append(self)

        def cancel(self):
            self.cancelled = True

    monkeypatch.setattr(supervisor_module.threading, "Timer", _FakeTimer)
    service._setup_timer = SupervisorService._setup_timer.__get__(service)
    service.timeout_active = True
    service._timeout_generation = 1
    service.shutdown_time = _ClockDateTime.current + timedelta(minutes=5)
    service.shutdown = AsyncMock()

    service._setup_timer(5, generation=1)
    assert len(timers) == 1

    _ClockDateTime.current += timedelta(minutes=2)
    timers[0].callback()

    assert len(timers) == 2
    assert timers[1].interval == pytest.approx(180)
    service.shutdown.assert_not_awaited()

    _ClockDateTime.current = service.shutdown_time
    timers[1].callback()
    timers[1].callback()

    service.shutdown.assert_awaited_once()
