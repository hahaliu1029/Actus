from __future__ import annotations

import inspect

from app.main import lifespan


def test_task8_disconnect_workflows_drain_before_runtime_dependencies_close() -> None:
    source = inspect.getsource(lifespan)

    idle_stop = source.index("await idle_watchdog.stop()")
    auto_degrade_drain = source.index("await drain_auto_degrade_tasks()")
    fence_drain = source.index("await drain_mode_transition_fence_cleanups()")
    agent_shutdown = source.index("agent_svc.shutdown()")
    redis_shutdown = source.index("await redis_client.shutdown()")
    postgres_shutdown = source.index("await postgres_client.shutdown()")

    assert idle_stop < auto_degrade_drain < fence_drain < agent_shutdown
    assert agent_shutdown < redis_shutdown < postgres_shutdown

