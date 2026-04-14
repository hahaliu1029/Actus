"""Test that agent_service.chat() has a FINISHING branch that calls _get_task."""
import pytest
import inspect


def test_chat_has_finishing_branch():
    """chat() must check SessionStatus.FINISHING before creating new task."""
    from app.application.services.agent_service import AgentService
    source = inspect.getsource(AgentService.chat)
    assert "FINISHING" in source, \
        "chat() must have a FINISHING status branch"


def test_chat_finishing_branch_calls_get_task():
    """The FINISHING branch must call _get_task (reuse), not _create_task."""
    from app.application.services.agent_service import AgentService
    source = inspect.getsource(AgentService.chat)
    finishing_idx = source.index("FINISHING")
    after_finishing = source[finishing_idx:]
    assert "_get_task" in after_finishing, \
        "FINISHING branch must call _get_task to reuse running task"
