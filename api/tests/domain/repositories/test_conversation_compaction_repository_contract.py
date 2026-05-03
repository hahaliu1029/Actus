"""Compile-time contract test: ABC must define the three required methods."""
import inspect

from app.domain.repositories.conversation_compaction_repository import (
    ConversationCompactionRepository,
)


def test_abc_has_required_methods():
    abstract_methods = ConversationCompactionRepository.__abstractmethods__
    assert {"create_or_get", "list_for_session", "get_by_id"}.issubset(abstract_methods)


def test_create_or_get_is_async():
    method = ConversationCompactionRepository.create_or_get
    assert inspect.iscoroutinefunction(method)
