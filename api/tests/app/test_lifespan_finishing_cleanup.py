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
