"""Test that lifespan contains stale FINISHING cleanup logic."""
import inspect


def test_lifespan_has_finishing_cleanup():
    """Lifespan must contain SQL update for stale FINISHING sessions."""
    from app.main import lifespan
    source = inspect.getsource(lifespan)
    assert "finishing" in source.lower(), \
        "lifespan must contain FINISHING session cleanup"
    assert "postprocess_skipped_on_restart" in source, \
        "lifespan must log postprocess_skipped_on_restart for audit"


def test_lifespan_stale_finishing_cleanup_stops_supervisor():
    """codex r7 [MEDIUM TEST] / C3 PR-3c — the stale-FINISHING cleanup
    block in lifespan is a non-runner terminal-write path; after
    ``reconcile_orphans`` may have spawned mailbox supervisors for these
    same roots, this cleanup MUST also call ``supervisor_registry.stop``
    so the slot table stays consistent with the freshly-written
    terminal status.

    Structural assert via ``inspect.getsource`` mirrors the existing
    test pattern (no full lifespan boot required, no DB).
    """
    from app.main import lifespan

    source = inspect.getsource(lifespan)
    # The cleanup block must reference both the supervisor registry AND
    # the stop call so a future refactor that drops the stop hook fails
    # this CI gate (matches AgentService/_maybe_stop_supervisor_for_session
    # pattern from R5 and ExecutionSupervisor from R6).
    assert "supervisor_registry" in source, (
        "lifespan stale-FINISHING cleanup must reference supervisor_registry "
        "to stop per-pod MailboxSupervisor tasks after terminal write"
    )
    assert "supervisor_registry.stop" in source, (
        "lifespan stale-FINISHING cleanup must call "
        "supervisor_registry.stop(session_id) per codex r7 [HIGH CONTRACT] "
        "— PR-3c §11.3 mailbox-plane non-runner terminal contract"
    )
